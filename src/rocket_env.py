"""JAX implementation of rocket landing environment."""
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

# GRAVITATIONAL_CONSTANT
G = 6.67430e-11


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the rocket landing environment."""
    # position
    x: jnp.ndarray
    y: jnp.ndarray
    theta: jnp.ndarray

    # velocity
    dx: jnp.ndarray
    dy: jnp.ndarray
    omega: jnp.ndarray

    # engine
    throttle: jnp.ndarray
    gimbal: jnp.ndarray

    # fuel
    fuel: jnp.ndarray

    # target landing angle
    target_angle: jnp.ndarray

    # timestep
    time: jnp.ndarray


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters for the rocket landing environment."""
    dt: float = 0.05

    planet_radius: float = 50.
    planet_mass: float = 6.0e16

    rocket_max_thrust: float = 14000.0
    rocket_max_gimbal: float = 0.3
    rocket_mass: float = 2600.0
    rocket_moment_of_inertia: float = 250.0
    rocket_initial_fuel: float = 2000.0
    rocket_fuel_consumption_rate: float = 50.0

    init_min_orbit_radius: float = 70.0
    init_max_orbit_radius: float = 300.0
    init_orbit_velocity_noise: float = 0.1

    noise_pos: float = 0.1
    noise_vel: float = 0.5
    noise_angle: float = 0.1

    max_steps_in_episode: int = 1000

    landing_max_speed: float = 2.0
    landing_max_angle: float = 0.3
    landing_max_omega: float = 0.5
    landing_position_tolerance: float = jnp.pi/30

    reward_i_cl: float = 100.0
    reward_ni_cl : float = -50.0
    reward_ni_il: float = -50.0
    reward_out_of_fuel: float = -40.0
    reward_out_of_bounds: float = -30.0
    reward_altitude_factor: float = 0.5
    reward_angular_position_factor: float = 0.5
    reward_radial_velocity_factor: float = 0.5
    reward_tangential_velocity_factor: float = 0.5
    reward_angle_factor: float = 0.5
    reward_angular_velocity_factor: float = 0.05
    reward_fuel_penalty: float = 0.01


class RocketLander(environment.Environment[EnvState, EnvParams]):
    """JAX/Gymnax implementation of a 2D rocket landing environment."""

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
            self,
            key: jax.Array,
            state: EnvState,
            action: jax.Array,
            params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]:
        """Integrate"""

        ### deterministic step

        # gravity
        r = jnp.sqrt(state.x**2 + state.y**2)
        F_gravity_magnitude = G * params.planet_mass * params.rocket_mass / r**2
        # gravitational acceleration (a = F/m)
        a_gravity_magnitude = F_gravity_magnitude / params.rocket_mass

        a_g_x = -a_gravity_magnitude * state.x / r
        a_g_y = -a_gravity_magnitude * state.y / r

        ### control step

        # throttle and gimbal
        throttle_action = action[0]
        gimbal_action = action[1]

        throttle = jnp.clip(throttle_action, -1.0, 1.0)
        gimbal = jnp.clip(gimbal_action, -1.0, 1.0)

        throttle = (throttle + 1.0)/2.0
        gimbal *= params.rocket_max_gimbal

        # can only get as much throttle as we have fuel / consumption
        throttle = jnp.minimum(throttle, state.fuel / (params.rocket_fuel_consumption_rate * params.dt))

        # thrust
        thrust_angle = state.theta + gimbal
        F_thrust = throttle * params.rocket_max_thrust
        # acceleration (a = F/m)
        a_t_x = F_thrust * jnp.sin(thrust_angle) / params.rocket_mass
        a_t_y = F_thrust * jnp.cos(thrust_angle) / params.rocket_mass
        # torque
        F_torque = F_thrust * jnp.sin(gimbal)
        a_omega = F_torque / params.rocket_moment_of_inertia

        ### new state calculation
        # directional sum of gravitational and thrusticular accelerations
        a_x = a_g_x + a_t_x
        a_y = a_g_y + a_t_y

        dx = state.dx + a_x * params.dt
        dy = state.dy + a_y * params.dt
        omega = state.omega + a_omega * params.dt

        x = state.x + dx * params.dt
        y = state.y + dy * params.dt
        theta = state.theta + omega * params.dt

        fuel = state.fuel - throttle * params.rocket_fuel_consumption_rate * params.dt
        fuel = jnp.maximum(0.0, fuel)  #safety

        new_state = EnvState(
            x=x,
            y=y,
            theta=theta,
            dx=dx,
            dy=dy,
            omega=omega,
            throttle=throttle,
            gimbal=gimbal,
            fuel=fuel,
            target_angle=state.target_angle,
            time=state.time+params.dt
        )
        done = self.is_terminal(new_state, params)
        reward = self._compute_reward(state, new_state, params)
        key, noise_key = jax.random.split(key)
        obs = self.get_obs(new_state, params, noise_key)
        return obs, new_state, reward, done, {}

    def reset_env(
            self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """Reset environment by sampling."""
        key, angle_key, orbit_radius_key, orbit_vel_noise_key, target_angle_key = jax.random.split(key, 5)

        ### rocket
        # initial position and velocity
        angle_init = jax.random.uniform(angle_key, minval=0.0, maxval=2.0 * jnp.pi)

        orbit_radius_init = jax.random.uniform(orbit_radius_key, minval=params.init_min_orbit_radius, maxval=params.init_max_orbit_radius)

        x_init = orbit_radius_init * jnp.cos(angle_init)
        y_init = orbit_radius_init * jnp.sin(angle_init)

        orbit_velocity_magnitude_init = jnp.sqrt(G * params.planet_mass / orbit_radius_init)
        # Add noise scaled by the noise parameter
        velocity_noise = params.init_orbit_velocity_noise * (2.0 * jax.random.uniform(orbit_vel_noise_key) - 1.0)
        orbit_velocity_magnitude_init *= (1.0 + velocity_noise)

        dx_init = -orbit_velocity_magnitude_init * jnp.sin(angle_init)  # negative for counterclockwise orbit
        dy_init = orbit_velocity_magnitude_init * jnp.cos(angle_init)

        # initial angle and angular velocity
        theta_init = angle_init + (3/2)*jnp.pi # rocket always orbiting "backwards" (thruster facing direction of motion)
        omega_init = jnp.array(0.0)  # no initial angular velocity

        target_angle = jax.random.uniform(target_angle_key, minval=0.0, maxval=2.0 * jnp.pi)

        state = EnvState(
            x=x_init,
            y=y_init,
            theta=theta_init,
            dx=dx_init,
            dy=dy_init,
            omega=omega_init,
            throttle=jnp.array(0.0),
            gimbal=jnp.array(0.0),
            fuel=jnp.array(params.rocket_initial_fuel),
            target_angle=target_angle,
            time=jnp.array(0)
        )

        return self.get_obs(state, params, key), state

    def get_obs(self, state: EnvState, params: EnvParams=None, key: jax.Array=None) -> jax.Array:
        """
        Return
        """

        if params is None:  # safety - always provide params
            params = self.default_params

        r = jnp.sqrt(state.x ** 2 + state.y ** 2)
        altitude = r - params.planet_radius

        positional_angle = jnp.arctan2(state.y, state.x)

        delta_angle = state.target_angle - positional_angle
        delta_angle = angle_normalize(delta_angle)

        radial_vel = (state.x * state.dx + state.y * state.dy) / r
        tangential_vel = (state.x * state.dy - state.y * state.dx) / r

        theta_relative = state.theta - positional_angle
        theta_relative = angle_normalize(theta_relative)

        # observation
        obs = jnp.array(
            [
                # polar coordinates (position)
                altitude,
                delta_angle,
                # polar coordinates (velocity)
                radial_vel,
                tangential_vel,
                # rotational values
                theta_relative,
                state.omega,
                # engine
                state.throttle,
                state.gimbal,
                state.fuel
            ]
        )

        # Add observation noise to simulate sensor uncertainty
        # if key is not None:
        #     noise = jax.random.normal(key, shape=(9,))
        #     noise_scales = jnp.array([
        #         params.noise_pos,      # altitude
        #         params.noise_angle,    # delta_angle
        #         params.noise_vel,      # radial_vel
        #         params.noise_vel,      # tangential_vel
        #         params.noise_angle,    # theta_relative
        #         params.noise_angle,    # omega
        #         0.0,                   # throttle (known exactly)
        #         0.0,                   # gimbal (known exactly)
        #         0.0,                   # fuel (known exactly)
        #     ])
        #     obs = obs + noise * noise_scales

        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""

        # timeout based on time in seconds (max_steps * dt)
        max_time = params.max_steps_in_episode * params.dt
        timeout = state.time >= max_time

        r = jnp.sqrt(state.x ** 2 + state.y ** 2)

        landed = r <= params.planet_radius

        escaped = r > params.planet_radius * 6

        done = landed | timeout | escaped
        return jnp.array(done)

    def _compute_reward(self, old_state: EnvState, new_state: EnvState, params: EnvParams) -> jax.Array:
        """Compute reward."""

        reward = jnp.array(0.0)

        ### preliminary calculations

        ## new state
        r = jnp.sqrt(new_state.x**2 + new_state.y**2)
        altitude = r - params.planet_radius
        normalized_altitude = altitude / params.planet_radius

        landed = r <= params.planet_radius
        escaped = r > params.planet_radius * 6

        # get angular distance from target landing spot
        positional_angle = jnp.arctan2(new_state.y, new_state.x)
        delta_angle = jnp.abs(angle_normalize(new_state.target_angle - positional_angle))

        # various velocities
        v_landing = jnp.sqrt(new_state.dx**2 + new_state.dy**2)
        v_radial = (new_state.x * new_state.dx + new_state.y * new_state.dy) / r
        v_tangential = (new_state.x * new_state.dy - new_state.y * new_state.dx) / r

        # get rocket orientation relative to surface
        theta_relative = angle_normalize(new_state.theta - positional_angle)

        ## old state
        r_old = jnp.sqrt(old_state.x ** 2 + old_state.y ** 2)
        old_altitude = r_old - params.planet_radius

        old_positional_angle = jnp.arctan2(old_state.y, old_state.x)
        old_delta_angle = jnp.abs(angle_normalize(old_state.target_angle - old_positional_angle))

        old_v_radial = (old_state.x * old_state.dx + old_state.y * old_state.dy) / r_old
        old_v_tangential = (old_state.x * old_state.dy - old_state.y * old_state.dx) / r_old

        old_theta_relative = angle_normalize(old_state.theta - old_positional_angle)

        ## landing traits
        slow = v_landing < params.landing_max_speed
        correct_position = delta_angle <= params.landing_position_tolerance
        correct_orientation = jnp.abs(theta_relative) <= params.landing_max_angle
        low_spin = jnp.abs(new_state.omega) <= params.landing_max_omega

        ### reward calculation

        ## rewards for termination by landing

        # ideal landing at correct location
        i_cl = landed & slow & correct_position & correct_orientation & low_spin
        reward = jnp.where(i_cl, params.reward_i_cl, reward)

        # non-ideal landing at correct location
        ni_cl = landed & correct_position & ~(slow & correct_orientation & low_spin)
        reward = jnp.where(ni_cl, params.reward_ni_cl, reward)

        # non-ideal landing at incorrect location
        ni_il = landed & ~correct_position # & ~(slow & correct_orientation & low_spin)
        reward = jnp.where(ni_il, params.reward_ni_il, reward)

        # penalty for escaping
        reward = jnp.where(escaped, params.reward_out_of_bounds, reward)

        ## rewards for non-terminal states
        non_terminal = ~landed & ~escaped

        # altitude -> 0
        delta_altitude = old_altitude - altitude
        reward += non_terminal * delta_altitude * params.reward_altitude_factor

        # if altitude -> 0
        # delta_angle -> 0
        altitude_factor = jnp.exp(-normalized_altitude)
        delta_delta_angle = old_delta_angle - delta_angle
        reward += non_terminal * delta_delta_angle * params.reward_angular_position_factor * altitude_factor

        # if altitude -> 0 and delta_angle -> 0
        # rocket radial velocity -> 0
        altitude_delta_angle_factor = jnp.exp(-normalized_altitude) * jnp.exp(-delta_angle)
        delta_radial_velocity = jnp.abs(old_v_radial) - jnp.abs(v_radial)
        reward += non_terminal * delta_radial_velocity * params.reward_radial_velocity_factor * altitude_delta_angle_factor

        # if altitude -> 0 and delta_angle -> 0 and rocket radial velocity -> 0
        # rocket tangential velocity -> 0
        altitude_delta_angle_radial_factor = jnp.exp(-normalized_altitude) * jnp.exp(-delta_angle) * jnp.exp(-jnp.abs(v_radial))
        delta_tangential_velocity = jnp.abs(old_v_tangential) - jnp.abs(v_tangential)
        reward += non_terminal * delta_tangential_velocity * params.reward_tangential_velocity_factor * altitude_delta_angle_radial_factor

        # if altitude -> 0 and delta_angle -> 0 and rocket radial velocity -> 0 and rocket tangential velocity -> 0
        # theta_relative -> 0
        altitude_delta_angle_radial_tangential_factor = jnp.exp(-normalized_altitude) * jnp.exp(-delta_angle) * jnp.exp(-jnp.abs(v_radial)) * jnp.exp(-jnp.abs(v_tangential))
        delta_theta_relative = jnp.abs(old_theta_relative) - jnp.abs(theta_relative)
        reward += non_terminal * delta_theta_relative * params.reward_angle_factor * altitude_delta_angle_radial_tangential_factor

        # if altitude -> 0 and delta_angle -> 0 and rocket radial velocity -> 0 and rocket tangential velocity -> 0 and theta_relative -> 0
        # angular_velocity -> 0
        altitude_delta_angle_radial_tangential_theta_factor = jnp.exp(-normalized_altitude) * jnp.exp(-delta_angle) * jnp.exp(-jnp.abs(v_radial)) * jnp.exp(-jnp.abs(v_tangential)) * jnp.exp(-jnp.abs(theta_relative))
        delta_omega = jnp.abs(old_state.omega) - jnp.abs(new_state.omega)
        reward += non_terminal * delta_omega * params.reward_angular_velocity_factor * altitude_delta_angle_radial_tangential_theta_factor

        # very small fuel usage penalty
        fuel_used = old_state.fuel - new_state.fuel
        reward -= params.reward_fuel_penalty * fuel_used

        # penalty if no fuel while far away from planet
        no_fuel = new_state.fuel <= 0
        reward = jnp.where(no_fuel, reward + params.reward_out_of_fuel * normalized_altitude, reward)

        return reward

    @property
    def name(self) -> str:
        """Environment name."""
        return "RocketLander"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array([-10.0, -jnp.pi, -100.0, -100.0, -jnp.pi, -10.0, 0.0, -0.5, 0.0])
        high = jnp.array([500.0, jnp.pi, 100.0, 100.0, jnp.pi, 10.0, 1.0, 0.5, 2500.0])
        return spaces.Box(low, high, shape=(9,), dtype=jnp.float32)


def angle_normalize(x: jax.Array) -> jax.Array:
    """Normalize the angle - radians."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def visualize_trajectory(states: list, params: EnvParams, title: str = "Rocket Landing Trajectory", save_path: Optional[str] = None):
    """
    Visualize the rocket trajectory around the planet.

    Args:
        states: List of EnvState objects from an episode
        params: Environment parameters
        title: Plot title
        save_path: If provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw planet
    theta = np.linspace(0, 2 * np.pi, 100)
    planet_x = params.planet_radius * np.cos(theta)
    planet_y = params.planet_radius * np.sin(theta)
    ax.fill(planet_x, planet_y, color='#3d5a80', alpha=0.8, label='Planet')
    ax.plot(planet_x, planet_y, color='#1d3557', linewidth=2)

    # Extract trajectory
    xs = np.array([float(s.x) for s in states])
    ys = np.array([float(s.y) for s in states])
    throttles = np.array([float(s.throttle) for s in states])

    # Draw trajectory colored by throttle
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    from matplotlib.collections import LineCollection
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='YlOrRd', norm=norm, linewidth=2, alpha=0.8)
    lc.set_array(throttles[:-1])
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label='Throttle')

    # Mark start and end
    ax.scatter(xs[0], ys[0], s=100, c='green', marker='o', zorder=5, label='Start')
    ax.scatter(xs[-1], ys[-1], s=100, c='red', marker='x', zorder=5, label='End')

    # Draw target landing site
    target_angle = float(states[0].target_angle)
    target_x = params.planet_radius * np.cos(target_angle)
    target_y = params.planet_radius * np.sin(target_angle)
    ax.scatter(target_x, target_y, s=200, c='yellow', marker='*', zorder=5,
               edgecolors='black', label='Target')

    # Draw rocket orientation at a few points
    n_arrows = min(10, len(states))
    arrow_indices = np.linspace(0, len(states) - 1, n_arrows, dtype=int)
    for idx in arrow_indices:
        s = states[idx]
        x, y = float(s.x), float(s.y)
        theta = float(s.theta)
        # Arrow shows rocket "up" direction
        dx = np.sin(theta) * 5
        dy = np.cos(theta) * 5
        ax.arrow(x, y, dx, dy, head_width=2, head_length=1, fc='white', ec='black', alpha=0.7)

    ax.set_xlim(-params.planet_radius * 3, params.planet_radius * 3)
    ax.set_ylim(-params.planet_radius * 3, params.planet_radius * 3)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Dark background for space
    ax.set_facecolor('#0d1b2a')
    fig.patch.set_facecolor('#0d1b2a')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.legend(facecolor='#1b263b', edgecolor='white', labelcolor='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()

    return fig


# Test the environment
if __name__ == "__main__":
    # Create environment
    env = RocketLander()
    params = env.default_params

    # Test reset and step
    key = jax.random.PRNGKey(42)

    # JIT compile for speed
    reset_fn = jax.jit(env.reset_env)
    step_fn = jax.jit(env.step_env)

    # Reset environment
    key, reset_key = jax.random.split(key)
    obs, state = reset_fn(reset_key, params)

    print("Initial observation shape:", obs.shape)
    print("Initial observation:", obs)
    print(f"Initial position: ({state.x:.2f}, {state.y:.2f})")
    print(f"Initial velocity: ({state.dx:.2f}, {state.dy:.2f})")
    print(f"Orbit radius: {jnp.sqrt(state.x ** 2 + state.y ** 2):.2f}")
    print(f"Target angle: {state.target_angle:.2f} rad")

    # Run a few steps with random actions
    states = [state]
    for i in range(100):
        key, action_key, step_key = jax.random.split(key, 3)
        # Random action: slight thrust, no gimbal
        action = jax.random.uniform(action_key, shape=(2,), minval=-1.0, maxval=1.0)
        action = action.at[0].set(-0.5)  # Low throttle

        obs, state, reward, done, info = step_fn(step_key, state, action, params)
        states.append(state)

        if done:
            print(f"Episode ended at step {i + 1}")
            break

    print(f"\nFinal position: ({state.x:.2f}, {state.y:.2f})")
    print(f"Final velocity: ({state.dx:.2f}, {state.dy:.2f})")
    print(f"Final altitude: {jnp.sqrt(state.x ** 2 + state.y ** 2) - params.planet_radius:.2f}")
    print(f"Fuel remaining: {state.fuel:.2f}")

    # Visualize
    visualize_trajectory(states, params, "Random Policy Test")
