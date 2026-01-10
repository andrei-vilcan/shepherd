"""Visualization utilities for the rocket landing environment."""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from rocket_env import EnvParams


def visualize_trajectory(
    states: list,
    params: EnvParams,
    title: str = "Rocket Landing Trajectory",
    save_path: Optional[str] = None
):
    """
    Visualize the rocket trajectory around the planet.

    Args:
        states: List of EnvState objects from an episode
        params: Environment parameters
        title: Plot title
        save_path: If provided, save figure to this path
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

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

    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='YlOrRd', norm=norm, linewidth=2, alpha=0.8)
    lc.set_array(throttles[:-1])
    ax.add_collection(lc)
    cbar = plt.colorbar(lc, ax=ax, label='Throttle')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Mark start and end
    ax.scatter(xs[0], ys[0], s=100, c='green', marker='o', zorder=5, label='Start')
    ax.scatter(xs[-1], ys[-1], s=100, c='red', marker='x', zorder=5, label='End')

    # Draw target landing site
    target_angle = float(states[0].target_angle)
    target_x = params.planet_radius * np.cos(target_angle)
    target_y = params.planet_radius * np.sin(target_angle)
    ax.scatter(target_x, target_y, s=200, c='yellow', marker='*', zorder=5,
               edgecolors='black', label='Target')

    # Draw rocket orientation arrows
    n_arrows = min(30, len(states))
    arrow_indices = np.linspace(0, len(states) - 1, n_arrows, dtype=int)
    for idx in arrow_indices:
        s = states[idx]
        x, y = float(s.x), float(s.y)
        theta = float(s.theta)
        dx = np.sin(theta) * 8
        dy = np.cos(theta) * 8
        ax.arrow(x, y, dx, dy, head_width=4, head_length=2, fc='#cccccc', ec='white', alpha=0.9)

    # Set field of view
    fov = params.init_max_orbit_radius * 1.2
    ax.set_xlim(-fov, fov)
    ax.set_ylim(-fov, fov)
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
