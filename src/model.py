from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


TIME_EMBED_SCALE = 1000.0  # t is continuous in [0, 1]; scale up so all frequency
# bands actually sweep a meaningful arc (freqs below were tuned for integer
# timesteps in the hundreds/thousands, e.g. DDPM's 0..999 range)


class TimeEmbedding(eqx.Module):
    linear1: eqx.nn.Linear  # ig this is how it works in jax
    linear2: eqx.nn.Linear

    def __init__(self, hidden_channels: int, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key, 2)
        self.linear1 = eqx.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels, key=key1
        )
        self.linear2 = eqx.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels, key=key2
        )

    def __call__(self, t: Float[Array, ""]) -> Float[Array, " hidden_channels"]:
        half = int(self.linear1.in_features) // 2
        indices = jnp.arange(half)  # [0, 1, ..., 127]
        freqs = jnp.exp(
            -indices * jnp.log(10000) / half
        )  # 10000 ^ (-indices/127) -> [1, 10000^-1/127, 10000^-2/127, ..., 10000^-1=0.0001]
        angles = (t * TIME_EMBED_SCALE) * freqs

        sin_part = jnp.sin(angles)
        cos_part = jnp.cos(angles)
        time_embed = jnp.concatenate([sin_part, cos_part])

        return self.linear2(jax.nn.silu(self.linear1(time_embed)))


class AdaLN(eqx.Module):
    linear1: eqx.nn.Linear
    # def __init__(self, hidden_channels, num_features, key):
    #   self.linear1 = eqx.nn.Linear(in_features=hidden_channels, out_features=2*num_features, key=key)

    def __init__(self, hidden_channels: int, num_features: int, key: PRNGKeyArray):
        self.linear1 = eqx.nn.Linear(
            in_features=hidden_channels, out_features=2 * num_features, key=key
        )
        # zero-init so gamma≈0, beta≈0 at start → residual blocks start as identity

        bias = self.linear1.bias
        assert bias is not None

        self.linear1 = eqx.tree_at(
            lambda layer: layer.weight,
            self.linear1,
            jnp.zeros_like(self.linear1.weight),
        )
        self.linear1 = eqx.tree_at(
            lambda layer: layer.bias, self.linear1, jnp.zeros_like(bias)
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        time_emb: Float[Array, " hidden_channels"],
    ) -> Float[Array, "channels height width"]:
        # first we normalize x to have 0 mean and std 1
        x_normed = jax.nn.standardize(x, axis=0)  # axis 0 is the first axis (channels)

        # then use linear layer and time_emb to get gamma (scale) and beta (shift)
        layer_pass = self.linear1(time_emb)
        gamma, beta = jnp.split(layer_pass, 2)

        # issue: shape of gamma is (64,)
        # shape of x_nomred is (64, 28, 28)
        # gamma needs to be reshaped to (64, 1, 1) befoer we can mult

        gamma = gamma.reshape(-1, 1, 1)
        beta = beta.reshape(-1, 1, 1)

        return (1 + gamma) * x_normed + beta
        # now the return type is the proper (64, 28, 28)


class ResBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    adaln1: AdaLN
    adaln2: AdaLN
    gate_proj: eqx.nn.Linear

    def __init__(self, hidden_channels: int, num_channels: int, key: PRNGKeyArray):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        self.conv1 = eqx.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
            key=key1,
        )
        self.conv2 = eqx.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
            key=key2,
        )
        self.adaln1 = AdaLN(
            hidden_channels=hidden_channels, num_features=num_channels, key=key3
        )
        self.adaln2 = AdaLN(
            hidden_channels=hidden_channels, num_features=num_channels, key=key4
        )

        # per-channel gate on the residual branch, conditioned on time_emb.
        # zero-init (weight AND bias) so gate==0 at init regardless of time_emb,
        # making the block a true identity map (x + 0*h == x) at the start of
        # training -- adaln zero-init alone only zeroes gamma/beta, which makes
        # adaln reduce to plain normalization (not zero), so h was never
        # actually ~0 without this.
        self.gate_proj = eqx.nn.Linear(
            in_features=hidden_channels, out_features=num_channels, key=key5
        )
        self.gate_proj = eqx.tree_at(
            lambda layer: layer.weight,
            self.gate_proj,
            jnp.zeros_like(self.gate_proj.weight),
        )
        self.gate_proj = eqx.tree_at(
            lambda layer: layer.bias,
            self.gate_proj,
            jnp.zeros_like(self.gate_proj.bias),
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        time_emb: Float[Array, " hidden_channels"],
    ) -> Float[Array, "channels height width"]:
        h = self.conv1(x)
        h = self.adaln1(h, time_emb)
        h = jax.nn.silu(h)
        h = self.conv2(h)
        h = self.adaln2(h, time_emb)
        h = jax.nn.silu(h)
        gate = self.gate_proj(time_emb).reshape(-1, 1, 1)
        return x + gate * h


class UNet(eqx.Module):
    time_emb: TimeEmbedding
    label_emb: Optional[eqx.nn.Embedding]
    input_conv: eqx.nn.Conv2d
    res1: ResBlock
    res2: ResBlock
    down_conv: eqx.nn.Conv2d
    res3: ResBlock
    res4: ResBlock
    up_conv: eqx.nn.Conv2d
    skip_conv: eqx.nn.Conv2d
    res5: ResBlock
    res6: ResBlock
    output_conv: eqx.nn.Conv2d

    def __init__(
        self,
        hidden_channels: int,
        num_channels: int,
        key: PRNGKeyArray,
        in_channels: int = 1,
        num_classes: Optional[int] = None,
    ):
        """
        in_channels lets model take in extra conditioning channels as input
        gets concatenated onto noisy image
        num_classes adds label embedding which gets summed with time embedding
            this is for the class conditioning
        """
        keys = jax.random.split(key, 13)
        ch = num_channels  # 64
        ch2 = num_channels * 2  # 128

        self.time_emb = TimeEmbedding(hidden_channels, keys[0])
        self.label_emb = (
            eqx.nn.Embedding(num_classes, hidden_channels, key=keys[12])
            if num_classes is not None
            else None
        )
        self.input_conv = eqx.nn.Conv2d(
            in_channels, ch, kernel_size=3, padding=1, key=keys[1]
        )

        # encoder (28x28, 64 channels)
        self.res1 = ResBlock(hidden_channels, ch, key=keys[2])
        self.res2 = ResBlock(hidden_channels, ch, key=keys[3])

        # downsample 28->14, 64->128
        self.down_conv = eqx.nn.Conv2d(
            ch, ch2, kernel_size=3, stride=2, padding=1, key=keys[4]
        )

        # bottleneck (14x14, 128 channels)
        self.res3 = ResBlock(hidden_channels, ch2, key=keys[5])
        self.res4 = ResBlock(hidden_channels, ch2, key=keys[6])

        # upsample conv: 128 -> 64
        self.up_conv = eqx.nn.Conv2d(ch2, ch, kernel_size=3, padding=1, key=keys[7])

        # after concatenating skip (64+64=128) -> 64
        self.skip_conv = eqx.nn.Conv2d(
            ch * 2, ch, kernel_size=3, padding=1, key=keys[8]
        )

        # decoder (28x28, 64 channels)
        self.res5 = ResBlock(hidden_channels, ch, key=keys[9])
        self.res6 = ResBlock(hidden_channels, ch, key=keys[10])

        self.output_conv = eqx.nn.Conv2d(ch, 1, kernel_size=3, padding=1, key=keys[11])

    def __call__(
        self,
        x: Float[Array, "in_channels 28 28"],
        t: Float[Array, ""],
        y: Optional[Int[Array, ""]] = None,
    ) -> Float[Array, "1 28 28"]:
        time_embedding = self.time_emb(t)
        if self.label_emb is not None:
            assert y is not None
            time_embedding = time_embedding + self.label_emb(y)

        # encoder
        h = self.input_conv(x)
        h = self.res1(h, time_embedding)
        h = self.res2(h, time_embedding)
        skip = h  # save for later (64, 28, 28)

        # downsample
        h = self.down_conv(h)  # (128, 14, 14)

        # bottleneck
        h = self.res3(h, time_embedding)
        h = self.res4(h, time_embedding)

        # upsample: nearest-neighbor 2x then conv
        h = jax.image.resize(
            h, (h.shape[0], h.shape[1] * 2, h.shape[2] * 2), method="nearest"
        )
        h = self.up_conv(h)  # (128, 28, 28) -> (64, 28, 28)

        # concatenate skip and squeeze
        h = jnp.concatenate([h, skip], axis=0)  # (128, 28, 28)
        h = self.skip_conv(h)  # (64, 28, 28)

        # decoder
        h = self.res5(h, time_embedding)
        h = self.res6(h, time_embedding)
        h = self.output_conv(h)
        return h
