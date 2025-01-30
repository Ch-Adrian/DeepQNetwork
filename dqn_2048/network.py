from typing import Sequence
from jax import jit, lax, random, vmap, debug
import haiku as hk
import jax
import jax.numpy as jnp

class CustomNetwork(hk.Module):
    def __init__(self, output_dim , name=None):
        super().__init__(name=name)
        self.dropout_rate = 0.2
        
        # self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=(2,2), stride=1, padding="SAME")
        # self.conv2 = hk.Conv2D(output_channels=64, kernel_shape=(2,2), stride=1, padding="SAME")
        self.conv1 = hk.Conv2D(output_channels=16, kernel_shape=(2,2), stride=1, padding="SAME")
        self.conv2 = hk.Conv2D(output_channels=32, kernel_shape=(2,2), stride=1, padding="SAME")
        self.conv3 = hk.Conv2D(output_channels=64, kernel_shape=(2,2), stride=1, padding="SAME")
        self.conv4 = hk.Conv2D(output_channels=128, kernel_shape=(2,2), stride=1, padding="SAME")
        self.conv5 = hk.Conv2D(output_channels=256, kernel_shape=(2,2), stride=1, padding="SAME")
        self.conv6 = hk.Conv2D(output_channels=512, kernel_shape=(2,2), stride=1, padding="SAME")
        self.conv7 = hk.Conv2D(output_channels=1024, kernel_shape=(2,2), stride=1, padding="SAME")
        self.conv8 = hk.Conv2D(output_channels=2048, kernel_shape=(2,2), stride=1, padding="SAME")
        self.flatten = hk.Flatten()

        self.keys = vmap(random.PRNGKey)(jnp.arange(4))
        self.head = hk.Sequential([
            hk.Linear(2048),
            lambda x: hk.dropout(self.keys[0], self.dropout_rate, x),
            jax.nn.leaky_relu,
            hk.Linear(1024),
            lambda x: hk.dropout(self.keys[0], self.dropout_rate, x),
            jax.nn.leaky_relu,
            hk.Linear(512),
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
        x = self.conv3(x)
        x = jax.nn.relu(x)
        x = self.conv4(x)
        x = jax.nn.relu(x)
        x = self.conv5(x)
        x = jax.nn.relu(x)
        x = self.conv6(x)
        x = jax.nn.relu(x)
        x = self.conv7(x)
        x = jax.nn.relu(x)
        x = self.conv8(x)
        x = jax.nn.relu(x)

        x = self.flatten(x)

        x = self.head(x)

        return x