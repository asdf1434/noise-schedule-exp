import argparse
import functools
import json
import math
import os
import time
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray, install_import_hook

# !! this  codeblock is directly from claude uhh figure out what exactly beartype.beartype is
# Enforce jaxtyping shape/dtype annotations at runtime for everything in src/,
# so mismatched array shapes fail loudly instead of silently propagating.
with install_import_hook("src", "beartype.beartype"):
    from src.conditioning import CONDITIONING
    from src.loss import compute_loss_cond
    from src.model import UNet
    from src.schedules import (
        get_logit_normal_cdf_steps,
        get_shifted_steps,
        get_uniform_steps,
        sample_t_logit_normal,
        sample_t_plateau_logit_normal,
        sample_t_uniform,
    )
    from src.utils import (
        get_mnist_dataloaders,
        sample_batch_cond,
        sample_batch_x,
        save_images,
    )


LEARNING_RATE = 2e-4
optim = optax.adam(LEARNING_RATE)


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    opt_state: optax.OptState,
    clean_images: Float[Array, "batch 1 28 28"],
    key: PRNGKeyArray,
    sample_t_fn: Callable,  # training schedule
    conditioning: str,
    labels: Optional[Int[Array, " batch"]] = None,
) -> tuple[eqx.Module, optax.OptState, Float[Array, ""]]:
    """
    single batch
    compute gradients and update model
    """

    key_noise, key_time = jax.random.split(key)
    batch_size = clean_images.shape[0]

    # generate noise and timesteps using sample_t_fn
    noise = jax.random.normal(key_noise, clean_images.shape)
    t = sample_t_fn(key_time, batch_size)

    loss_fn = lambda m: compute_loss_cond(
        m, conditioning, clean_images, noise, t, labels
    )
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    updates, new_opt_state = optim.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_opt_state, loss


def export_evaluation_images(
    model: eqx.Module,
    key: PRNGKeyArray,
    eval_samples: int,
    exp_name: str,
    epoch: int,
    conditioning: str,
    num_steps: int,
    eval_ref_images: Optional[Float[Array, "n 1 28 28"]] = None,
):
    """
    generate and save samples for different inference schedules
    """
    print(f"\nExporting eval images for epoch {epoch}")

    schedules_to_test = {
        "uniform": get_uniform_steps(num_steps=num_steps),
        # more steps at high noise
        "shifted_coarse": get_shifted_steps(num_steps=num_steps, shift=0.3),
        # more steps at low noise
        "shifted_fine": get_shifted_steps(num_steps=num_steps, shift=3.0),
        "logit_normal": get_logit_normal_cdf_steps(num_steps=num_steps),
    }

    eval_batch_size = min(200, eval_samples)
    num_batches = max(1, math.ceil(eval_samples / eval_batch_size))

    for schedule_name, timesteps in schedules_to_test.items():
        # eval_runs/experiment_name/epoch_###/inference_schedule
        out_dir = os.path.join("eval_runs", exp_name, f"epoch_{epoch}", schedule_name)
        os.makedirs(out_dir, exist_ok=True)

        all_samples = []
        for b in range(num_batches):
            key, sample_key = jax.random.split(key)

            cond_images = None
            labels = None
            if conditioning in ("lowres", "inpaint"):
                # cycle through refernece images
                idx = (
                    jnp.arange(eval_batch_size) + b * eval_batch_size
                ) % eval_ref_images.shape[0]
                cond_images = eval_ref_images[idx]
            elif conditioning == "class":
                labels = jnp.arange(eval_batch_size, dtype=jnp.int32) % 10

            if conditioning == "none":
                batch_samples = sample_batch_x(
                    model, sample_key, timesteps, eval_batch_size
                )
            else:
                batch_samples = sample_batch_cond(
                    model,
                    sample_key,
                    timesteps,
                    conditioning,
                    eval_batch_size,
                    cond_images=cond_images,
                    labels=labels,
                )
            all_samples.append(np.array(batch_samples))

        # save without running eval
        # do all the eval afterwards for speed/efficiency
        save_images(np.concatenate(all_samples, axis=0)[:eval_samples], out_dir)
        print(f"Exported samples for {schedule_name} into {out_dir}")

    print("-----------------------------------------")


def save_checkpoint(model: eqx.Module, epoch: int, checkpoint_dir: str, prefix: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{prefix}_epoch_{epoch}.eqx")
    eqx.tree_serialise_leaves(path, model)
    print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dist",
        type=str,
        default="uniform",
        choices=["uniform", "logit_normal", "plateau_logit_normal"],
    )
    parser.add_argument(
        "--conditioning",
        type=str,
        default="none",
        choices=list(CONDITIONING.keys()),
        help="none, class (digit label), lowres (7x7), inpaint (left half given)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="# of sampling steps per eval sched",
    )

    parser.add_argument("--dist_params", type=str, default="{}", help="schedule params")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for init/data order/training+eval noise; also appended to "
        "the experiment name so repeated trials with different seeds don't "
        "collide on checkpoints/eval_runs/metrics",
    )
    parser.add_argument(
        "--model_params",
        type=str,
        default='{"hidden_channels": 256, "num_channels": 64}',
        help="model architecture params",
    )

    # checkpoing/data arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help=".eqx file path")
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    dist_kwargs = json.loads(args.dist_params)
    model_kwargs = json.loads(args.model_params)

    cond_spec = CONDITIONING[args.conditioning]

    # this part is taken directly from claude to generate names
    param_str = "_".join([f"{k}_{v}" for k, v in dist_kwargs.items()])
    base_name = f"{args.train_dist}_{param_str}" if param_str else args.train_dist
    exp_name = (
        f"{args.conditioning}_{base_name}" if args.conditioning != "none" else base_name
    )
    exp_name = f"{exp_name}_seed{args.seed}"

    os.makedirs(os.path.join("logs", "metrics"), exist_ok=True)

    # Assign log filename based on experiment name if not explicitly passed
    if args.log_file is None:
        # Prepend the logs/metrics/ path here!
        args.log_file = os.path.join("logs", "metrics", f"metrics_{exp_name}.jsonl")
    else:
        # If a custom log file was passed via command line, make sure it goes there too
        args.log_file = os.path.join("logs", "metrics", args.log_file)

    batch_size = 128

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    print("init model")
    model = UNet(
        **model_kwargs,
        key=init_key,
        in_channels=cond_spec["in_channels"],
        num_classes=cond_spec["num_classes"],
    )

    if args.resume_from is not None:
        print(f"load checkpoing from {args.resume_from}")
        model = eqx.tree_deserialise_leaves(args.resume_from, model)

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    print("load MNIST")
    if cond_spec["needs_labels"]:
        all_images, all_labels = get_mnist_dataloaders(
            batch_size=batch_size, with_labels=True
        )
    else:
        all_images = get_mnist_dataloaders(batch_size=batch_size, with_labels=False)
        all_labels = None

    # lowres/inpaint needs the actual images at eval time
    # keep some of it away from the training
    needs_eval_ref = args.conditioning in ("lowres", "inpaint")
    if needs_eval_ref:
        num_eval_ref = min(1000, all_images.shape[0] // 10)
        eval_ref_images = all_images[:num_eval_ref]
        dataloader = all_images[num_eval_ref:]
        train_labels = all_labels[num_eval_ref:] if all_labels is not None else None
    else:
        eval_ref_images = None
        dataloader = all_images
        train_labels = all_labels

    num_batches = len(dataloader) // batch_size

    dist_fn_map = {
        "uniform": sample_t_uniform,
        "logit_normal": functools.partial(sample_t_logit_normal, **dist_kwargs),
        "plateau_logit_normal": functools.partial(
            sample_t_plateau_logit_normal, **dist_kwargs
        ),
    }
    train_dist_fn = dist_fn_map[args.train_dist]

    print(
        f"training for {args.epochs} epochs using {args.train_dist} distribution with conditioning={args.conditioning}"
    )
    print(f"log training metrics to: {args.log_file}")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        epoch_loss = jnp.array(0.0)

        key, shuffle_key = jax.random.split(key)
        indices = jax.random.permutation(shuffle_key, len(dataloader))

        shuffled_data = dataloader[indices][: num_batches * batch_size]
        batched_data = shuffled_data.reshape(num_batches, batch_size, 1, 28, 28)

        if train_labels is not None:
            shuffled_labels = train_labels[indices][: num_batches * batch_size]
            batched_labels = shuffled_labels.reshape(num_batches, batch_size)
        else:
            batched_labels = None

        for i in range(num_batches):
            key, step_key = jax.random.split(key)
            batch_labels = batched_labels[i] if batched_labels is not None else None
            model, opt_state, loss = make_step(
                model,
                opt_state,
                batched_data[i],
                step_key,
                train_dist_fn,
                args.conditioning,
                batch_labels,
            )
            epoch_loss += loss

        avg_loss = (epoch_loss / num_batches).item()
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s"
        )

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            key, eval_key = jax.random.split(key)

            export_evaluation_images(
                model,
                eval_key,
                args.eval_samples,
                exp_name,
                epoch,
                args.conditioning,
                args.num_steps,
                eval_ref_images,
            )

            # save metrics and checkpoint
            save_checkpoint(model, epoch, args.checkpoint_dir, exp_name)

            data = {"epoch": epoch, "loss": avg_loss}
            with open(args.log_file, "a") as f:
                f.write(json.dumps(data) + "\n\n\n")


if __name__ == "__main__":
    main()
