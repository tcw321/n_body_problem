"""
Phase 2 — Real-time 2D visualization of the n-body simulation.

The simulation/render loop are deliberately decoupled:
  - nbody_step owns the physics (runs as fast as it can)
  - FuncAnimation owns the display (fires at a fixed interval in ms)

Usage:
    python visualize.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from main import nbody_step, leapfrog_step


def create_animation(pos, vel, mass, G=1.0, dt=0.01, softening=0.3, steps_per_frame=1, step_fn=None):
    """
    Build and return a (fig, FuncAnimation) pair without calling plt.show().

    Keeping show() out of this function makes it testable with a non-interactive
    backend (Agg) — no display required.

    Parameters
    ----------
    pos, vel, mass  : initial conditions (arrays are copied so the originals
                      are not mutated)
    steps_per_frame : how many simulation steps to advance per rendered frame.
                      Increase this to speed up the apparent simulation rate.
    step_fn         : integrator to use. Defaults to nbody_step (symplectic Euler).
                      Pass leapfrog_step for better energy conservation.
    """
    if step_fn is None:
        step_fn = nbody_step

    pos = pos.copy()
    vel = vel.copy()

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')

    spread = float(np.max(np.abs(pos))) * 2.5
    ax.set_xlim(-spread, spread)
    ax.set_ylim(-spread, spread)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # Dot size scaled by mass so heavier bodies are visually larger
    sizes = (mass / mass.max()) * 20 + 2
    scatter = ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c='white', alpha=0.8)
    step_label = ax.text(0.02, 0.96, 'step 0', transform=ax.transAxes,
                         color='white', fontsize=10)

    state = {'step': 0}

    def update(_frame):
        nonlocal pos, vel
        for _ in range(steps_per_frame):
            pos, vel = step_fn(pos, vel, mass, G=G, dt=dt, softening=softening)
        state['step'] += steps_per_frame
        scatter.set_offsets(pos[:, :2])
        step_label.set_text(f"step {state['step']}")
        return scatter, step_label

    ani = animation.FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)
    return fig, ani


def run_visualization(N=100, dt=0.01, softening=0.3, G=1.0, steps_per_frame=1):
    """Entry point for interactive use."""
    np.random.seed(42)
    pos  = np.random.randn(N, 3) * 5.0
    vel  = np.random.randn(N, 3) * 0.1
    mass = np.random.uniform(0.5, 2.0, N)

    fig, ani = create_animation(pos, vel, mass, G=G, dt=dt,
                                softening=softening,
                                steps_per_frame=steps_per_frame)
    plt.show()
    return ani


def circular_orbits_ic(N=20, M_central=100.0, r_min=2.0, r_max=10.0, G=1.0, seed=42):
    """
    Initial conditions for one heavy central body with N-1 planets on
    circular orbits at evenly-spaced radii.

    Circular orbital speed at radius r: v = sqrt(G * M_central / r)
    """
    rng = np.random.default_rng(seed)

    pos  = np.zeros((N, 3))
    vel  = np.zeros((N, 3))
    mass = np.ones(N)

    mass[0] = M_central  # central body at origin

    radii  = np.linspace(r_min, r_max, N - 1)
    angles = rng.uniform(0, 2 * np.pi, N - 1)

    pos[1:, 0] = radii * np.cos(angles)
    pos[1:, 1] = radii * np.sin(angles)

    v_circ = np.sqrt(G * M_central / radii)
    vel[1:, 0] = -v_circ * np.sin(angles)
    vel[1:, 1] =  v_circ * np.cos(angles)

    return pos, vel, mass


def run_circular_orbits(N=20, dt=0.005, softening=0.1, G=1.0, steps_per_frame=3):
    """Visualize a central body with N-1 planets on circular orbits (leapfrog)."""
    pos, vel, mass = circular_orbits_ic(N=N, G=G)
    fig, ani = create_animation(pos, vel, mass, G=G, dt=dt,
                                softening=softening,
                                steps_per_frame=steps_per_frame,
                                step_fn=leapfrog_step)
    fig.axes[0].set_title('Circular Orbits', color='white')
    plt.show()
    return ani


if __name__ == '__main__':
    print("Select visualization:")
    print("  1. Random cluster")
    print("  2. Circular orbits")
    choice = input("Enter 1 or 2 [default: 1]: ").strip()
    if choice == '2':
        run_circular_orbits()
    else:
        run_visualization()
