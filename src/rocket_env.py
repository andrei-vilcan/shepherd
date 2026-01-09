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

    planet_radius: float = 50.0
    planet_mass: float = 5.0e14

    rocket_max_thrust: float = 38000.0
    rocket_max_gimbal: float = 0.3
    rocket_mass: float = 2600.0
    rocket_moment_of_inertia: float = 250.0
    rocket_initial_fuel: float = 5000.0
    rocket_fuel_consumption_rate: float = 100.

    init_min_orbit_radius: float = 80.0
    init_max_orbit_radius: float = 150.0
    init_orbit_velocity_noise: float = 0.1

    noise_pos: float = 0.1
    noise_vel: float = 0.5
    noise_angle: float = 0.1

    max_steps_in_episode: int = 1500

    landing_max_speed: float = 2.0
    landing_max_angle: float = 0.3
    landing_max_omega: float = 0.5
    landing_position_tolerance: float = jnp.pi/30

    # terminal rewards
    reward_landed: float = 100.0
    reward_sl_cp_co_ls: float = 1000.0
    reward_nsl_cp : float = 200.0
    reward_nsl_ncp: float = 100.0
    reward_sl_ncp_co_ls: float = 500.0
    reward_out_of_fuel: float = -50.0
    reward_out_of_bounds: float = -1000.0
    reward_timeout: float = -2000.0  # Much higher penalty to discourage staying in orbit
    # shaping rewards
    reward_altitude_factor: float = 0.001
    reward_angular_position_factor: float = 0.0003
    reward_radial_velocity_factor: float = 0.0003
    reward_tangential_velocity_factor: float = 0.0005
    reward_angle_factor: float = 0.0003
    reward_angular_velocity_factor: float = 0.00005
    reward_fuel_penalty: float = 0.0  # Disabled to encourage engine use
    reward_low_velocity_near_surface: float = 0.0005
    reward_deceleration_with_fuel: float = 0.0003
    reward_retrograde_orientation: float = 0.0005
    reward_retrograde_burn: float = 0.02  # Reward for firing engine while aligned for braking


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

        a_gravity_magnitude = G * params.planet_mass / r**2

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

        escaped = r > params.planet_radius * 4

        done = landed | timeout | escaped
        return jnp.array(done)

    def _compute_reward(self, old_state: EnvState, new_state: EnvState, params: EnvParams) -> jax.Array:
        """Compute reward."""
        
        ### preliminary calculations

        ## new state
        r = jnp.sqrt(new_state.x ** 2 + new_state.y ** 2)
        altitude = r - params.planet_radius
        normalized_altitude = jnp.clip(altitude / (params.init_max_orbit_radius - params.planet_radius), 0.0, 2.0)

        # terminal conditions
        landed = r <= params.planet_radius
        escaped = r > params.planet_radius * 4
        max_time = params.max_steps_in_episode * params.dt
        timeout = new_state.time >= max_time
        is_terminal = landed | escaped | timeout

        # angular distance from target landing spot
        positional_angle = jnp.arctan2(new_state.y, new_state.x)
        delta_angle = jnp.abs(angle_normalize(new_state.target_angle - positional_angle))

        # various velocities
        v_total = jnp.sqrt(new_state.dx ** 2 + new_state.dy ** 2)
        v_radial = (new_state.x * new_state.dx + new_state.y * new_state.dy) / r
        v_tangential = (new_state.x * new_state.dy - new_state.y * new_state.dx) / r

        # get rocket orientation relative to surface
        theta_relative = angle_normalize(new_state.theta - positional_angle)

        ## old state (same calculations)
        r_old = jnp.sqrt(old_state.x ** 2 + old_state.y ** 2)
        altitude_old = r_old - params.planet_radius
        normalized_altitude_old = jnp.clip(altitude_old / (params.init_max_orbit_radius - params.planet_radius), 0.0, 2.0)

        positional_angle_old = jnp.arctan2(old_state.y, old_state.x)
        delta_angle_old = jnp.abs(angle_normalize(old_state.target_angle - positional_angle_old))

        v_total_old = jnp.sqrt(old_state.dx ** 2 + old_state.dy ** 2)
        v_radial_old = (old_state.x * old_state.dx + old_state.y * old_state.dy) / r_old
        v_tangential_old = (old_state.x * old_state.dy - old_state.y * old_state.dx) / r_old

        theta_relative_old = jnp.abs(angle_normalize(old_state.theta - positional_angle_old))

        ## landing conditions
        slow = v_total < params.landing_max_speed
        correct_position = delta_angle <= params.landing_position_tolerance
        correct_orientation = jnp.abs(theta_relative) <= params.landing_max_angle
        low_spin = jnp.abs(new_state.omega) <= params.landing_max_omega

        ### terminal rewards

        terminal_reward = jnp.array(0.0)

        # ideal landing: slow, correct position, correct orientation, low spin
        sl_cp_co_ls = landed & slow & correct_position & correct_orientation & low_spin
        terminal_reward = jnp.where(sl_cp_co_ls, params.reward_sl_cp_co_ls, terminal_reward)

        # ideal landing at incorrect position
        sl_ncp_co_ls = landed & slow & ~correct_position & correct_orientation & low_spin
        terminal_reward = jnp.where(sl_ncp_co_ls, params.reward_sl_ncp_co_ls * jnp.exp(-delta_angle * 2.0), terminal_reward)

        # slow landing but wrong orientation/spin
        sl_nco_nls = landed & slow & ~(correct_orientation & low_spin)
        slow_reward = 100.0 * jnp.exp(-delta_angle)
        terminal_reward = jnp.where(sl_nco_nls, slow_reward, terminal_reward)

        # non-ideal (fast) landing at correct position
        nsl_cp = landed & ~slow & correct_position
        terminal_reward = jnp.where(nsl_cp, params.reward_nsl_cp, terminal_reward)

        # non-ideal (fast) landing at incorrect position
        nsl_ncp = landed & ~slow & ~correct_position
        terminal_reward = jnp.where(nsl_ncp, params.reward_nsl_ncp, terminal_reward)

        # escaped bounds
        terminal_reward = jnp.where(escaped, params.reward_out_of_bounds, terminal_reward)

        # timeout - heavily penalized to discourage staying in orbit
        terminal_reward = jnp.where(timeout & ~landed & ~escaped, params.reward_timeout, terminal_reward)

        ### non-terminal rewards
        
        shaping_reward = jnp.array(0.0)
        gamma = 0.99

        # reward retrograde burn
        thrust_angle = new_state.theta + new_state.gimbal
        velocity_angle = jnp.arctan2(new_state.dy, new_state.dx)
        alignment = jnp.cos(thrust_angle - velocity_angle)
        retrograde_alignment = (1.0 - alignment) / 2.0
        throttle_used = new_state.throttle > 0.1
        velocity_factor = jnp.minimum(v_total / 10.0, 1.0)
        retrograde_burn_reward = params.reward_retrograde_burn * retrograde_alignment * throttle_used * velocity_factor
        shaping_reward += retrograde_burn_reward

        # reward tangential velocity -> 0
        tangential_shaping = -0.3 * (gamma * jnp.abs(v_tangential) - jnp.abs(v_tangential_old))
        shaping_reward += tangential_shaping

        # if delta_angle -> 0, reward altitude -> 0
        delta_angle_factor = jnp.exp(-delta_angle * 3)
        altutude_reward = -delta_angle_factor * (gamma * normalized_altitude - normalized_altitude_old)

        shaping_reward += altutude_reward

        # if landing (low altitude and low velocity), reward theta_relative -> 0
        near_landing_factor = jnp.exp(-normalized_altitude * 3.0) * jnp.exp(-v_total)
        theta_relative_reward = -1.0 * near_landing_factor * (gamma * jnp.abs(theta_relative) - jnp.abs(theta_relative_old))
        shaping_reward += theta_relative_reward

        # if landing (same), reward angular velocity -> 0
        angular_velocity_reward = -0.1 * near_landing_factor * (gamma * jnp.abs(new_state.omega) - jnp.abs(old_state.omega))
        shaping_reward += angular_velocity_reward

        # if altitude -> 0 and delta_angle -> 0, reward v_total -> 0
        low_altitude_factor = jnp.exp(-normalized_altitude * 2.0)
        low_velocity_factor = jnp.exp(-delta_angle * 2.0)
        low_delta_angle_factor = jnp.exp(-delta_angle * 2.0)
        low_velocity_reward = 0.1 * low_altitude_factor * low_velocity_factor * low_delta_angle_factor
        shaping_reward += low_velocity_reward

        # reward velocity -> 0
        delta_v = -(v_total - v_total_old)
        velocity_decrease = jnp.maximum(0.0, delta_v)
        velocity_decrease_reward = 0.1 * velocity_decrease
        shaping_reward += velocity_decrease_reward

        # punish high angular velocity
        high_omega = jnp.abs(new_state.omega) > 2.0
        angular_velocity_punishment = 0.1 * high_omega * (jnp.abs(new_state.omega) - 2.0)
        shaping_reward -= angular_velocity_punishment

        reward = jnp.where(is_terminal, terminal_reward, shaping_reward)

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
        # [altitude, delta_angle, radial_vel, tangential_vel, theta_relative, omega, throttle, gimbal, fuel]
        low = jnp.array([-10.0, -jnp.pi, -50.0, -50.0, -jnp.pi, -10.0, 0.0, -0.5, 0.0])
        high = jnp.array([150.0, jnp.pi, 50.0, 50.0, jnp.pi, 10.0, 1.0, 0.5, 5000.0])
        return spaces.Box(low, high, shape=(9,), dtype=jnp.float32)


def angle_normalize(x: jax.Array) -> jax.Array:
    """Normalize the angle - radians."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi
