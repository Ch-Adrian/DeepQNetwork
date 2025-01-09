import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import gymnasium as gym
import optax  # for optimizers
import numpy as np
from jax import random

class PolicyNetwork(nn.Module):
    hidden_size: int  # Hidden layer size

    def setup(self):
        # Define a two-layer fully connected network
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(2)  # Output layer for 2 possible actions

    def __call__(self, x):
        x = jax.nn.relu(self.dense1(x))
        x = self.dense2(x)
        return jax.nn.softmax(x)  # Output probabilities over actions

def create_env():
    # Create the CartPole environment
    env = gym.make('CartPole-v1')
    return env

def create_train_state(rng, model, learning_rate):
    # Initialize the model parameters with the random key
    params = model.init(rng, jnp.ones([1, 4]))  # Assuming state space of (4,)
    tx = optax.adam(learning_rate)  # Adam optimizer
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.insert(0, R)
    return np.array(returns)

def reinforce_loss(params, state_batch, action_batch, return_batch, model):
    # Compute the action probabilities
    action_probs = model.apply({'params': params}, state_batch)
    
    # Compute the log probability of the taken actions
    log_probs = jnp.log(jnp.sum(action_probs * action_batch, axis=-1))

    # Compute the loss (negative log probability * return)
    loss = -jnp.mean(log_probs * return_batch)  # REINFORCE loss (negative sign because we minimize)
    return loss

def train_step(train_state, env, model, rng, gamma=0.99):
    # Rollout one episode to collect experiences
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    
    while not done:
        states.append(state)
        print(state)

        state_batch = jnp.array(state[0])[None, :] 
        
        # Sample an action from the policy network
        action_probs = model.apply(train_state.params, state_batch)
        action_probs = [np.round(action_probs[0][0],2), np.round(action_probs[0][1],2)]
        print(action_probs)
        action = np.random.choice(2, p=action_probs)  # Sample from action distribution
        
        # Step the environment
        next_state, reward, done, _, _ = env.step(action)
        
        # Store action and reward
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
    
    # Convert lists to JAX arrays
    states = jnp.array(states)
    actions = jnp.array(actions)
    rewards = jnp.array(rewards)

    # Compute the returns for the trajectory
    returns = compute_returns(rewards, gamma)

    # Convert returns to JAX array
    returns = jnp.array(returns)

    # Compute the loss and gradients
    loss, grads = jax.value_and_grad(reinforce_loss)(train_state['params'], states, actions, returns, model)

    # Update the model parameters using the gradients
    train_state = train_state.apply_gradients(grads=grads)
    
    return train_state, loss

def train():
    # Set the random seed and create the environment
    rng = random.PRNGKey(42)
    env = create_env()
    
    # Create the model
    model = PolicyNetwork(hidden_size=128)
    
    # Create the initial training state with model parameters
    state = create_train_state(rng, model, learning_rate=1e-3)
    
    num_episodes = 1000
    for episode in range(num_episodes):
        state, loss = train_step(state, env, model, rng)
        
        # Print the progress
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} | Loss: {loss:.4f}")
        
    return state

if __name__ == "__main__":
    trained_state = train()
