import argparse
import functools
import json
import math
import os
import sys
import time
from typing import Optional

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
    from src.datasets import DATASETS
    from src.infonoise import InfoNoiseSampler
    from src.loss import compute_loss_cond
    from src.model import UNet
    from src.naming import make_exp_name
    from src.schedules import (
        get_logit_normal_cdf_steps,
        get_shifted_steps,
        get_uniform_steps,
        sample_t_logit_normal,
        sample_t_plateau_logit_normal,
        sample_t_uniform,
    )
    from src.utils import (
        get_dataloaders,
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
    clean_images: Float[Array, "batch 1 h w"],
    key_noise: PRNGKeyArray,
    t: Float[Array, "batch 1 1 1"],
    conditioning: str,
    labels: Optional[Int[Array, " batch"]] = None,
    cond_params: tuple = (),
) -> tuple[eqx.Module, optax.OptState, Float[Array, ""], Float[Array, " batch"]]:
    """
    single batch
    compute gradients and update model

    ``t`` is drawn by the caller rather than here: InfoNoise's training
    distribution changes shape every refresh, so it can't be a jit-static
    callable. The per-sample unweighted loss comes back out as aux because
    that's the statistic InfoNoise's profile estimator consumes; it plays no
    part in the gradient.
    """

    noise = jax.random.normal(key_noise, clean_images.shape)

    loss_fn = lambda m: compute_loss_cond(
        m, conditioning, clean_images, noise, t, labels, cond_params
    )
    (loss, unweighted), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)

    updates, new_opt_state = optim.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_opt_state, loss, unweighted


def export_evaluation_images(
    model: eqx.Module,
    key: PRNGKeyArray,
    eval_samples: int,
    exp_name: str,
    epoch: int,
    conditioning: str,
    num_steps: int,
    image_shape: tuple,
    eval_ref_images: Optional[Float[Array, "n 1 h w"]] = None,
    cond_params: tuple = (),
):
    """
    generate and save samples for different inference schedules
    """
    print(f"\nExporting eval images for epoch {epoch}")

    # shift < 1 puts more steps at high noise, shift > 1 at low noise, shift == 1
    # is the same as uniform. Only 0.3 and 3.0 were ever evaluated, which gives
    # two points on what is really a continuous curve -- the extra values below
    # fill it in so the *best* shift per dataset is visible, not just the sign of
    # the difference. "shifted_coarse"/"shifted_fine" keep their original names
    # and shifts so existing results and analysis code stay valid.
    shifts = {
        "shifted_s0.15": 0.15,
        "shifted_coarse": 0.3,
        "shifted_s0.5": 0.5,
        "shifted_s0.7": 0.7,
        "shifted_s1.5": 1.5,
        "shifted_fine": 3.0,
        "shifted_s5.0": 5.0,
    }
    schedules_to_test = {
        "uniform": get_uniform_steps(num_steps=num_steps),
        "logit_normal": get_logit_normal_cdf_steps(num_steps=num_steps),
        **{
            name: get_shifted_steps(num_steps=num_steps, shift=shift)
            for name, shift in shifts.items()
        },
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
                    model, sample_key, timesteps, eval_batch_size, image_shape
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
                    image_shape=image_shape,
                    cond_params=cond_params,
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
    # Fail fast if JAX can't see a GPU (e.g. a node's CUDA driver doesn't
    # support this jaxlib build) instead of silently training on CPU for
    # the full run. Nonzero exit lets Slurm mark the task failed so it can
    # be resubmitted and land on a working node.
    gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpu_devices:
        print(
            f"ERROR: no GPU visible to JAX (jax.devices()={jax.devices()}). "
            "Refusing to train on CPU -- exiting so this task can be resubmitted "
            "onto a working node.",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dist",
        type=str,
        default="uniform",
        choices=["uniform", "logit_normal", "plateau_logit_normal", "infonoise"],
        help="Fixed training noise distributions, or 'infonoise' for the online "
        "information-guided allocation of arXiv:2602.18647 (see src/infonoise.py). "
        "InfoNoise knobs go through --dist_params too, e.g. "
        '\'{"warmup_steps": 2000, "refresh_every": 1000, "gate_c": 0.15}\'.',
    )
    parser.add_argument(
        "--conditioning",
        type=str,
        default="none",
        choices=list(CONDITIONING.keys()),
        help="none, class (digit label), lowres (7x7), inpaint (left half given)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=list(DATASETS),
        help="mnist/eurosat are 28x28; eurosat64 keeps native 64x64. Image "
        "geometry (size + channels) is read from the dataset registry in "
        "src/datasets.py -- add an entry there to support a new dataset.",
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
        "--cond_params",
        type=str,
        default="{}",
        help='JSON tuning how much help the conditioning gives, e.g. \'{"known_fraction": 0.75}\' '
        'for inpaint or \'{"factor": 2}\' for lowres. Becomes part of the experiment name, so '
        "different settings never overwrite each other. See CONDITIONING_PARAMS in "
        "src/conditioning.py.",
    )
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
    cond_kwargs = json.loads(args.cond_params)
    # Passed into jitted code, so it has to be hashable -- and `factor` decides
    # an array shape, so it can't be traced either. Sorted so the same settings
    # always produce the same experiment name.
    cond_items = tuple(sorted(cond_kwargs.items()))

    cond_spec = CONDITIONING[args.conditioning]
    ds_spec = DATASETS[args.dataset]
    # image geometry is derived from the dataset registry, not hardcoded to
    # 28x28 -- this is what lets a 64x64 dataset (e.g. eurosat64) train/sample.
    image_shape = (ds_spec.channels, ds_spec.image_size, ds_spec.image_size)

    exp_name = make_exp_name(
        args.dataset,
        args.conditioning,
        args.train_dist,
        dist_kwargs,
        args.seed,
        cond_params=cond_kwargs,
    )

    os.makedirs(os.path.join("logs", "metrics", exp_name), exist_ok=True)

    # Assign log filename based on experiment name if not explicitly passed
    if args.log_file is None:
        # Prepend the logs/metrics/<exp_name>/ path here!
        args.log_file = os.path.join("logs", "metrics", exp_name, "metrics.jsonl")
    else:
        # If a custom log file was passed via command line, make sure it goes there too
        args.log_file = os.path.join("logs", "metrics", exp_name, args.log_file)

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

    print(f"load {args.dataset}")
    if cond_spec["needs_labels"]:
        all_images, all_labels = get_dataloaders(
            args.dataset, batch_size=batch_size, with_labels=True
        )
    else:
        all_images = get_dataloaders(
            args.dataset, batch_size=batch_size, with_labels=False
        )
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

    infonoise = None
    if args.train_dist == "infonoise":
        # Everything in --dist_params prefixed "warmup_" configures the fixed
        # prior pi_0 InfoNoise samples from until its first refresh; the rest
        # configures the estimator itself.
        info_kwargs = dict(dist_kwargs)
        warmup_prior = info_kwargs.pop("warmup_prior", "logit_normal")
        warmup_prior_params = {
            k[len("warmup_prior_") :]: v
            for k, v in list(info_kwargs.items())
            if k.startswith("warmup_prior_")
        }
        for k in list(info_kwargs):
            if k.startswith("warmup_prior_"):
                info_kwargs.pop(k)
        base_dists = {
            "uniform": sample_t_uniform,
            "logit_normal": sample_t_logit_normal,
            "plateau_logit_normal": sample_t_plateau_logit_normal,
        }
        if warmup_prior not in base_dists:
            raise ValueError(
                f"--dist_params warmup_prior must be one of {sorted(base_dists)}, "
                f"got {warmup_prior!r}"
            )
        # built from the raw sample_t_* function, not dist_fn_map: the entries
        # there are already bound to the *whole* --dist_params dict, which for
        # infonoise is estimator config the prior knows nothing about
        warmup_fn = base_dists[warmup_prior]
        if warmup_prior_params:
            warmup_fn = functools.partial(warmup_fn, **warmup_prior_params)
        infonoise = InfoNoiseSampler(
            warmup_sample_fn=warmup_fn,
            log_path=os.path.join(
                "logs", "metrics", exp_name, "infonoise_profile.jsonl"
            ),
            **info_kwargs,
        )
        train_dist_fn = infonoise.sample_t
    else:
        train_dist_fn = dist_fn_map[args.train_dist]

    print(
        f"training for {args.epochs} epochs using {args.train_dist} distribution with conditioning={args.conditioning}"
    )
    print(f"log training metrics to: {args.log_file}")

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        epoch_loss = jnp.array(0.0)

        key, shuffle_key = jax.random.split(key)
        indices = jax.random.permutation(shuffle_key, len(dataloader))

        shuffled_data = dataloader[indices][: num_batches * batch_size]
        batched_data = shuffled_data.reshape(
            num_batches, batch_size, *image_shape
        )

        if train_labels is not None:
            shuffled_labels = train_labels[indices][: num_batches * batch_size]
            batched_labels = shuffled_labels.reshape(num_batches, batch_size)
        else:
            batched_labels = None

        for i in range(num_batches):
            key, step_key = jax.random.split(key)
            # split exactly as make_step used to internally, so the noise stream
            # is unchanged for a given --seed and old runs stay reproducible
            key_noise, key_time = jax.random.split(step_key)
            t = train_dist_fn(key_time, batch_size)

            batch_labels = batched_labels[i] if batched_labels is not None else None
            model, opt_state, loss, unweighted = make_step(
                model,
                opt_state,
                batched_data[i],
                key_noise,
                t,
                args.conditioning,
                batch_labels,
                cond_items,
            )
            epoch_loss += loss

            if infonoise is not None:
                infonoise.observe(t, unweighted)
                if infonoise.maybe_refresh(global_step):
                    print(
                        f"  infonoise refresh @ step {global_step}: "
                        f"{json.dumps(infonoise.profile_summary())}"
                    )
            global_step += 1

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
                image_shape,
                eval_ref_images,
                cond_items,
            )

            # save metrics and checkpoint
            save_checkpoint(model, epoch, args.checkpoint_dir, exp_name)

            data = {"epoch": epoch, "loss": avg_loss}
            if infonoise is not None:
                data["infonoise"] = infonoise.profile_summary()
            with open(args.log_file, "a") as f:
                f.write(json.dumps(data) + "\n\n\n")


if __name__ == "__main__":
    main()
