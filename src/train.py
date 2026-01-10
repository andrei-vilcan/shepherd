import os
import pickle
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

from rocket_env import RocketLander, EnvParams, EnvState
from vis import visualize_trajectory

# Create the environment
env = RocketLander()
env_params = env.default_params

# Visualization folder
VIS_FOLDER = os.path.join(os.path.dirname(__file__), "visualizations")


# setting up a basic neural network

def initialize_mlp(layer_sizes, key: PRNGKey, scale: float = 1e-2):
    """
    Inputs:
        layer_sizes (tuple) Tuple of shapes of the neural network layers. Includes the input shape, hidden layer shape, and output layer shape.
        key (PRNGKey)
        scale (float) standard deviation of initial weights and biases

    Return:
        params (List) Tuple of weights and biases - [ (weights_1, biases_1), ..., (weights_n, biases_n) ]
    """

    keys = jr.split(key, 2 * len(layer_sizes))
    params = []

    for i in range(len(layer_sizes) - 1):
        input_size, output_size = layer_sizes[i], layer_sizes[i + 1]
        W = jr.normal(keys[2 * i], (input_size, output_size)) * scale
        b = jr.normal(keys[2 * i + 1], (output_size,)) * scale
        params.append((W, b))

    return params


def policy(params, x):
    """
    Gaussian policy network.
    
    Inputs:
        params (list): Network parameters as list of (W, b) tuples.
        x (array): Input observation of shape (obs_dim,).
        
    Returns:
        mean (array): Mean of Gaussian for each action dimension.
        log_std (array): Log standard deviation for each action dimension.
    """
    # Shared layers
    for (W, b) in params[:-1]:
        x = jnp.dot(x, W) + b
        x = jax.nn.relu(x)

    # Output layer: first half is mean, second half is log_std
    W, b = params[-1]
    output = jnp.dot(x, W) + b
    
    action_dim = output.shape[-1] // 2
    mean = jnp.tanh(output[:action_dim])  # Bound mean to [-1, 1]
    log_std = output[action_dim:]
    log_std = jnp.clip(log_std, -2.0, 0.5)  # Bound std to reasonable range
    
    return mean, log_std


def get_action(params, x, key: PRNGKey):
    """
    Sample an action from the Gaussian policy.

    Inputs:
        params (list): Network parameters.
        x (array): Input observation.
        key (PRNGKey): Random key for sampling.
     
    Returns:
        action (array): Sampled action, clipped to [-1, 1].
        mean (array): Mean of the Gaussian.
        log_std (array): Log standard deviation.
    """
    mean, log_std = policy(params, x)
    std = jnp.exp(log_std)
    noise = jr.normal(key, shape=mean.shape)
    action = mean + std * noise
    action = jnp.clip(action, -1.0, 1.0)  # Clip to valid action range
    return action, mean, log_std


def get_log_prob(params, x, action):
    """
    Compute log probability of action under the Gaussian policy.

    Inputs:
        params (list): Network parameters.
        x (array): Input observation.
        action (array): Action taken.
     
    Returns:
        log_prob (float): Log probability of the action.
    """
    mean, log_std = policy(params, x)
    std = jnp.exp(log_std)
    
    # Gaussian log probability
    log_prob = -0.5 * jnp.sum(
        ((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)
    )
    return log_prob


@jax.jit
def update_delta(delta, grad_theta):
    """
    Update the parameter gradients delta with new gradient.

    Inputs:
        delta (list): Current accumulated gradients.
        grad_theta (list): New gradient to add.
    
    Returns:
        updated_delta (list): Updated gradients.
    """
    updated_delta = jax.tree.map(lambda x, y: x + y, delta, grad_theta)
    return updated_delta, None


# =============================================================================
# Rollout Function
# =============================================================================

def rollout(params, env_params, rng_input: PRNGKey, steps_in_episode: int):
    """
    Rollout an episode using the policy.
    
    Inputs:
        params: Network parameters.
        env_params: Environment parameters.
        rng_input: Random key.
        steps_in_episode: Number of steps to run.
        
    Returns:
        Tuple of (obs, state, action, reward, next_obs, done) arrays.
    """
    rng_reset, rng_episode = jr.split(rng_input)
    obs, state = env.reset_env(rng_reset, env_params)

    def policy_step(state_input, tmp):
        """Single step of the policy in the environment."""
        obs, state, rng = state_input
        rng, rng_action, rng_step = jr.split(rng, 3)
        
        action, mean, log_std = get_action(params, obs, rng_action)
        next_obs, next_state, reward, done, _ = env.step_env(
            rng_step, state, action, env_params
        )
        
        carry = [next_obs, next_state, rng]
        return carry, [obs, state, action, reward, next_obs, done]

    # Scan over episode steps
    _, scan_out = jax.lax.scan(
        policy_step,
        [obs, state, rng_episode],
        (),
        length=steps_in_episode,
    )
    
    return scan_out


jit_rollout = jax.jit(rollout, static_argnums=3)


# =============================================================================
# REINFORCE Loss
# =============================================================================

def loss_REINFORCE(params, obs, action, reward, baseline, gamma: float = 0.99):
    """
    Compute REINFORCE loss with baseline.

    Inputs:
        params: Network parameters.
        obs (array): Batch of observations (batch, time, obs_dim).
        action (array): Batch of actions (batch, time, action_dim).
        reward (array): Batch of rewards (batch, time).
        baseline (array): Baseline values (time,).
        gamma (float): Discount factor.

    Returns:
        delta: Accumulated gradients.
        Gt (array): Discounted returns.
    """

    def trajectory_gradients(reward, obs, action, baseline, delta):
        G_init = 0.0

        def step(carry, variables):
            G, delta = carry
            r, obs, action, b = variables

            # Calculate discounted return and advantage
            G = gamma * G + r
            advantage = G - b

            # Compute gradient of log probability
            def neg_log_prob(params):
                return -get_log_prob(params, obs, action)

            grad_delta = jax.grad(neg_log_prob)(params)
            grad_delta = jax.tree.map(lambda gd: gd * advantage, grad_delta)
            delta, _ = update_delta(delta, grad_delta)

            return (G, delta), G

        # Iterate backwards in time
        variables = (reward[::-1], obs[::-1], action[::-1], baseline[::-1])
        (_, delta), Gt = jax.lax.scan(step, (G_init, delta), variables)
        
        return delta, Gt

    # Parallelize over batch
    parallel_trajectory_gradients = jax.vmap(
        trajectory_gradients, in_axes=(0, 0, 0, None, None)
    )
    
    # Initialize delta to zeros
    delta = jax.tree.map(lambda t: jnp.zeros(t.shape), params)

    # Compute gradients in parallel and sum
    deltas, Gs = parallel_trajectory_gradients(reward, obs, action, baseline, delta)
    delta, _ = jax.lax.scan(update_delta, delta, deltas)

    return delta, jnp.array(Gs)


loss_REINFORCE = jax.jit(loss_REINFORCE)


# =============================================================================
# Baseline Functions
# =============================================================================

def mean_baseline(Gs):
    """Compute constant baseline as mean of discounted returns."""
    T = Gs.shape[1]
    mean_reward = jnp.mean(Gs)
    return jnp.ones((T,)) * mean_reward


def timedependent_baseline(Gs):
    """Compute time-dependent baseline."""
    mean_reward = jnp.mean(Gs, axis=0)  # Mean over batches
    cumulative_rewards = jnp.cumsum(mean_reward)
    baseline = cumulative_rewards / jnp.arange(1, Gs.shape[1] + 1)
    return mean_reward - baseline


# =============================================================================
# Training Loop
# =============================================================================

def train(
    num_iters: int = 5000,
    steps_in_episode: int = 2000,
    lr: float = 0.001,
    gamma: float = 0.99,
    n_batches: int = 32,
    hidden_size: int = 128,
    n_hidden_layers: int = 2,
    visualize_every: int = 500,
    save_model_every: int = 10000,
    seed: int = 42,
    grad_clip: float = 1.0,
):
    """
    Train the rocket landing policy using REINFORCE.
    
    Args:
        num_iters: Number of training iterations.
        steps_in_episode: Steps per episode (should be less than max_steps_in_episode).
        lr: Learning rate.
        gamma: Discount factor.
        n_batches: Number of parallel episodes per iteration.
        hidden_size: Size of hidden layers.
        n_hidden_layers: Number of hidden layers.
        visualize_every: Visualize trajectory every n iterations.
        save_model_every: Save model checkpoint every n iterations.
        seed: Random seed.
        grad_clip: Maximum gradient norm for clipping.
    """
    key = PRNGKey(seed)
    
    # Network architecture: obs_dim -> hidden layers -> 2*action_dim (mean + log_std)
    obs_dim = 9  # From rocket_env observation space
    action_dim = 2  # throttle and gimbal
    hidden_layers = tuple([hidden_size] * n_hidden_layers)
    layer_sizes = (obs_dim,) + hidden_layers + (2 * action_dim,)
    
    # Initialize network and optimizer
    key, subkey = jr.split(key)
    params = initialize_mlp(layer_sizes, key=subkey)
    
    # Adam optimizer with gradient clipping for stability
    optim = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate=lr),
    )
    opt_state = optim.init(params)
    
    # Initialize baseline storage
    Gs = jnp.zeros((n_batches, steps_in_episode))
    
    # Training keys
    key, subkey = jr.split(key)
    iter_keys = jr.split(subkey, num_iters)
    
    # Parallel rollout function
    parallel_rollout = jax.vmap(rollout, in_axes=(None, None, 0, None))
    
    # Training step function
    def step(carry, key):
        params, opt_state, Gs = carry
        
        # Generate rollouts
        keys = jr.split(key, n_batches)
        obs, state, action, reward, next_obs, done = parallel_rollout(
            params, env_params, keys, steps_in_episode
        )
        
        # Compute baseline from previous returns
        baseline = mean_baseline(Gs)
        
        # Compute gradients
        delta, Gs_new = loss_REINFORCE(params, obs, action, reward, baseline, gamma)
        
        # Update parameters
        updates, opt_state = optim.update(delta, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Mean episode reward for logging
        mean_reward = jnp.mean(jnp.sum(reward, axis=-1))
        
        carry = (new_params, opt_state, Gs_new)
        return carry, mean_reward

    # Compile the step function
    step = jax.jit(step)
    
    # Training loop with periodic visualization
    history = []
    current_params = params
    current_opt_state = opt_state
    current_Gs = Gs
    
    # Ensure visualization folder exists
    vis_counter = 0
    
    print(f"Starting training for {num_iters} iterations...")
    print(f"  Network architecture: {layer_sizes}")
    print(f"  Episodes per iteration: {n_batches}")
    print(f"  Steps per episode: {steps_in_episode}")
    print(f"  Learning rate: {lr}")
    print(f"  Discount factor: {gamma}")
    print(f"  Gradient clipping: {grad_clip}")
    print(f"  Saving visualizations to: {VIS_FOLDER}")
    print()
    
    pbar = tqdm(range(num_iters), desc="Training", unit="iter")
    for i in pbar:
        # Training step
        (current_params, current_opt_state, current_Gs), mean_reward = step(
            (current_params, current_opt_state, current_Gs), iter_keys[i]
        )
        history.append(float(mean_reward))
        
        # Update progress bar with current reward
        if (i + 1) % 10 == 0:
            avg_reward = jnp.mean(jnp.array(history[-100:])) if len(history) >= 100 else jnp.mean(jnp.array(history))
            pbar.set_postfix({"Avg R": f"{avg_reward:.1f}"})
        
        # Visualization - save to file
        if (i + 1) % visualize_every == 0:
            vis_counter += 1
            save_path = os.path.join(VIS_FOLDER, f"{vis_counter:03d}_iter_{i + 1}.png")
            tqdm.write(f"Saving trajectory at iteration {i + 1} to {save_path}...")
            key, vis_key = jr.split(key)
            visualize_policy(current_params, vis_key, title=f"Iteration {i + 1}", save_path=save_path)
        
        # Save model checkpoint
        if (i + 1) % save_model_every == 0:
            model_path = os.path.join(VIS_FOLDER, f"model_iter_{i + 1}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(current_params, f)
            tqdm.write(f"Saved model checkpoint to {model_path}")
    
    # Final visualization
    vis_counter += 1
    save_path = os.path.join(VIS_FOLDER, f"{vis_counter:03d}_final.png")
    print(f"\nTraining complete! Saving final trajectory to {save_path}...")
    key, vis_key = jr.split(key)
    visualize_policy(current_params, vis_key, title="Final Policy", save_path=save_path)
    
    # Plot and save training history
    plt.figure(figsize=(10, 5))
    plt.plot(history, alpha=0.3, label='Episode Reward')
    # Smoothed curve
    window = min(100, len(history) // 10)
    if window > 1:
        smoothed = jnp.convolve(jnp.array(history), jnp.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(history)), smoothed, label=f'Smoothed (window={window})')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Episode Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    history_path = os.path.join(VIS_FOLDER, f"{vis_counter + 1:03d}_training_progress.png")
    plt.savefig(history_path, dpi=150)
    plt.close()
    print(f"Saved training progress to {history_path}")
    
    return current_params, history


def visualize_policy(params, key, title="Policy Trajectory", save_path=None):
    """Run a single episode and visualize the trajectory."""
    # Run rollout
    obs, state, action, reward, next_obs, done = rollout(
        params, env_params, key, env_params.max_steps_in_episode
    )
    
    # Convert states to list for visualization
    # We need to reconstruct EnvState objects from the state arrays
    states = []
    for i in range(len(state.x)):
        s = EnvState(
            x=state.x[i],
            y=state.y[i],
            theta=state.theta[i],
            dx=state.dx[i],
            dy=state.dy[i],
            omega=state.omega[i],
            throttle=state.throttle[i],
            gimbal=state.gimbal[i],
            fuel=state.fuel[i],
            target_angle=state.target_angle[i],
            time=state.time[i],
        )
        states.append(s)
        
        # Stop if done
        if done[i]:
            break
    
    # Only sum rewards up to the terminal step (after that, rewards are garbage due to r=0)
    episode_length = len(states)
    total_reward = float(jnp.sum(reward[:episode_length]))
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Episode length: {episode_length} steps")
    
    visualize_trajectory(states, env_params, title=f"{title} (R={total_reward:.1f})", save_path=save_path)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Training hyperparameters
    params, history = train(
        num_iters=10000,
        steps_in_episode=1500,
        lr=0.0002,
        gamma=0.995,
        n_batches=128,
        hidden_size=256,
        n_hidden_layers=3,
        visualize_every=500,
        save_model_every=10000,
        seed=42,
        grad_clip=0.5,
    )
