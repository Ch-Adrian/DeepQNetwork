from typing import Sequence
from jax import jit, lax, random, vmap, debug
import haiku as hk
import jax
import jax.numpy as jnp

class CustomNetwork(hk.Module):
    def __init__(self, output_dim , name=None):
        super().__init__(name=name)
        self.dropout_rate = 0.2
        
        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=(2,2), stride=1, padding="SAME")
        self.conv2 = hk.Conv2D(output_channels=64, kernel_shape=(2,2), stride=1, padding="SAME")
        self.flatten = hk.Flatten()

        self.keys = vmap(random.PRNGKey)(jnp.arange(4))
        self.head = hk.Sequential([
            hk.Linear(128),
            lambda x: hk.dropout(self.keys[0], self.dropout_rate, x),
            jax.nn.leaky_relu,
            hk.Linear(256),
            lambda x: hk.dropout(self.keys[1], self.dropout_rate, x),
            jax.nn.leaky_relu,
            hk.Linear(128),
            lambda x: hk.dropout(self.keys[2], self.dropout_rate, x),
            jax.nn.leaky_relu,
            hk.Linear(64),
            lambda x: hk.dropout(self.keys[3], self.dropout_rate, x),
            jax.nn.leaky_relu,
            hk.Linear(output_dim),
        ])

    def __call__(self, x_batch):
        x = jnp.reshape(x_batch, (-1, 4, 4, 1))
        x = jnp.asarray(x, dtype=jnp.float32)
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)

        x = self.flatten(x)

        x = self.head(x)

        return x