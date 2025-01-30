from abc import ABC, abstractmethod
from functools import partial
from jax import jit, lax, random, vmap, debug


class BaseReplayBuffer(ABC):
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    @partial(jit, static_argnums=(0))
    def add(self, buffer_state: dict, experience: tuple, idx: int, batch_size: int):
        # state, action_mask, action, reward, next_state, new_action_mask, done = experience
        idx = idx*batch_size

        @jit
        def _update_buffer_state(i: int, exp_and_buff: tuple):
            experience, buffer_state, idx = exp_and_buff
            state, action_mask, action, reward, next_state, new_action_mask, done = experience
            idx = (1+idx) % self.buffer_size

            buffer_state["states"] = buffer_state["states"].at[idx].set(state[i])
            buffer_state["action_masks"] = buffer_state["action_masks"].at[idx].set(action_mask[i])
            buffer_state["actions"] = buffer_state["actions"].at[idx].set(action[i])
            buffer_state["rewards"] = buffer_state["rewards"].at[idx].set(reward[i])
            buffer_state["next_states"] = buffer_state["next_states"].at[idx].set(next_state[i])
            buffer_state["new_action_masks"] = buffer_state["new_action_masks"].at[idx].set(new_action_mask[i])
            buffer_state["dones"] = buffer_state["dones"].at[idx].set(done[i])
            return experience, buffer_state, idx

        _, buffer_state, _ = lax.fori_loop(0, batch_size, _update_buffer_state, (experience, buffer_state, idx))
        # for i in range(batch_size):
        #     idx = (i+idx) % self.buffer_size

        #     buffer_state["states"] = buffer_state["states"].at[idx].set(state[i])
        #     buffer_state["action_masks"] = buffer_state["action_masks"].at[idx].set(action_mask[i])
        #     buffer_state["actions"] = buffer_state["actions"].at[idx].set(action[i])
        #     buffer_state["rewards"] = buffer_state["rewards"].at[idx].set(reward[i])
        #     buffer_state["next_states"] = buffer_state["next_states"].at[idx].set(next_state[i])
        #     buffer_state["new_action_masks"] = buffer_state["new_action_masks"].at[idx].set(new_action_mask[i])
        #     buffer_state["dones"] = buffer_state["dones"].at[idx].set(done[i])

        return buffer_state

    @abstractmethod
    def sample(self):
        pass
