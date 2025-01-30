from functools import partial

import haiku as hk
import jax.numpy as jnp
import optax
from jax import jit, lax, random, value_and_grad, vmap, debug
import numpy as np
import random as rand
from base_agent import BaseAgent

BATCH_SIZE = 32

class DQN(BaseAgent):
    def __init__(self, model: hk.Transformed, discount: float, n_actions: int) -> None:
        super(DQN, self).__init__(discount)
        self.model = model
        self.n_actions = n_actions

    @partial(jit, static_argnums=(0))
    def act(self, key: random.PRNGKey, online_net_params: dict, state: jnp.ndarray, epsilon: float):
        def _random_action(subkey):
            action = random.choice(subkey, jnp.where(state.action_mask, size=4)[0])
            # debug.print("_random_action: {x}", x=action)
            return action

        def _forward_pass(_):
            q_values = self.model.apply(online_net_params, None, jnp.resize(state.board, (16,)))[0]
            # debug.print("_forward_pass(q_values): {x}", x=q_values)
            where = jnp.where(state.action_mask, q_values, -jnp.inf)
            # debug.print("_forward_pass(where): {x}", x=where)
            result = jnp.argmax(where)
            # debug.print("_forward_pass(result): {x}", x=result)
            return result

        explore = random.uniform(key) < epsilon
        key, subkey = random.split(key)
        action = lax.cond(explore, _random_action, _forward_pass, operand=subkey)
        return action, subkey

    @partial(jit, static_argnames=("self", "optimizer"))
    def update(
        self,
        online_net_params: dict,
        target_net_params: dict,
        optimizer: optax.GradientTransformation,
        optimizer_state: jnp.ndarray,
        experiences: dict[str : jnp.ndarray],
    ):
        @jit
        def _batch_loss_fn(
            online_net_params: dict,
            target_net_params: dict,
            states: jnp.ndarray,
            action_masks: jnp.ndarray,
            actions: jnp.ndarray,
            rewards: jnp.ndarray,
            next_states: jnp.ndarray,
            new_action_masks: jnp.ndarray,
            dones: jnp.ndarray,
        ):
            @partial(vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0))
            def _loss_fn(online_net_params, target_net_params, state, action, reward, next_state, new_action_mask, done):
                target = reward + (1 - done) * self.discount * jnp.max(
                    jnp.where(new_action_mask, self.model.apply(target_net_params, None, next_state)[0], -jnp.inf)
                )
                # debug.print("_loss_fn(target): {x}", x=target)
                # debug.print("_loss_fn(state): {x}", x=state)
                prediction = self.model.apply(online_net_params, None, state)[0][action]
                # debug.print("_loss_fn(prediction): {x}", x=prediction)
                return jnp.square(target - prediction)

            # debug.print("_batch_loss_fn(states): {x}", x=states)
            # debug.print("_batch_loss_fn(actions): {x}", x=actions)
            # debug.print("_batch_loss_fn(rewards): {x}", x=rewards)
            # debug.print("_batch_loss_fn(next_states): {x}", x=next_states)
            # debug.print("_batch_loss_fn(new_action_masks): {x}", x=new_action_masks)
            # debug.print("_batch_loss_fn(dones): {x}", x=dones)
            loss = _loss_fn(online_net_params, target_net_params, states, actions, rewards, next_states, new_action_masks, dones)
            # loss = jnp.array([2,3,4])
            # debug.print("_batch_loss_fn(loss): {x}", x=loss)
            mse = jnp.mean(loss)
            # debug.print("_batch_loss_fn(mse): {x}", x=mse)
            return mse

        loss, grads = value_and_grad(_batch_loss_fn)(online_net_params, target_net_params, **experiences)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        online_net_params = optax.apply_updates(online_net_params, updates)
        return online_net_params, optimizer_state, loss

    # below part is parrallelized for batch processing

    @partial(jit, static_argnums=(0))
    def batch_act(self, key: random.PRNGKey, online_net_params: dict, state: jnp.ndarray, epsilon: float):
        return vmap(DQN.act, in_axes=(0, None, 0, None))(key, online_net_params, state, epsilon)

    @partial(jit, static_argnames=("self", "optimizer"))
    def batch_update(
        self,
        online_net_params: dict,
        target_net_params: dict,
        optimizer: optax.GradientTransformation,
        optimizer_state: jnp.ndarray,
        experiences: dict[str : jnp.ndarray],
    ): 
        return vmap(DQN.update, in_axes=(None, None, None, None, 0))(
            online_net_params, target_net_params, optimizer, optimizer_state, experiences
        )
