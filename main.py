from multiprocessing import Pool
import numpy as np

def compute_forces_chunk_loop(args):
    """
    Original loop-based force computation. Kept as reference for regression tests.
    """
    i_start, i_end, pos, mass, G, softening = args
    N = len(mass)
    force_chunk = np.zeros((i_end - i_start, 3))

    for idx, i in enumerate(range(i_start, i_end)):
        for j in range(N):
            if i == j:
                continue
            diff = pos[j] - pos[i]
            dist = np.sqrt(np.dot(diff, diff) + softening**2)
            force_chunk[idx] += G * mass[j] / dist**3 * diff

    return i_start, force_chunk


def compute_forces_chunk(args):
    """
    Vectorized force computation for a subset of bodies (rows i_start..i_end).
    Each worker is completely independent — no shared state.

    For each body i in the chunk, computes the gravitational force from all N bodies
    using NumPy broadcasting instead of Python loops.

    diff[idx, j] = pos[j] - pos_chunk[idx]   shape: (chunk_size, N, 3)
    dist_sq[idx, j] = |diff|^2 + softening^2  shape: (chunk_size, N)

    Self-interaction (i == j) contributes diff = [0,0,0], so force is naturally zero.
    """
    i_start, i_end, pos, mass, G, softening = args
    pos_chunk = pos[i_start:i_end]                              # (chunk_size, 3)

    diff = pos[np.newaxis, :, :] - pos_chunk[:, np.newaxis, :] # (chunk_size, N, 3)
    dist_sq = np.sum(diff ** 2, axis=2) + softening ** 2       # (chunk_size, N)
    dist_cubed = dist_sq ** 1.5                                  # (chunk_size, N)

    # Zero out self-interaction: set dist_cubed diagonal to inf so mass/inf = 0.
    # This is necessary when softening=0 (otherwise 0/0 = NaN).
    chunk_size = i_end - i_start
    dist_cubed[np.arange(chunk_size), np.arange(i_start, i_end)] = np.inf

    # G * mass[j] / dist^3 * diff,  summed over all j
    force_chunk = G * np.sum(
        (mass[np.newaxis, :] / dist_cubed)[:, :, np.newaxis] * diff,
        axis=1
    )                                                            # (chunk_size, 3)

    return i_start, force_chunk

def _compute_acc(pos, mass, G=1.0, softening=0.1):
    """Full-array acceleration computed in a single process."""
    _, acc = compute_forces_chunk((0, len(mass), pos, mass, G, softening))
    return acc


def euler_step(pos, vel, mass, G=1.0, dt=0.01, softening=0.1):
    """
    Forward Euler integrator: pos += vel*dt (old vel), vel += acc*dt.
    Not symplectic — energy drifts over long runs.

    Note: nbody_step uses updated velocity for the position update (symplectic Euler),
    which is a different (better-behaved) method. This function is explicit forward
    Euler, used here to demonstrate why integrator choice matters.
    """
    acc = _compute_acc(pos, mass, G, softening)
    pos = pos + vel * dt        # advance position with OLD velocity
    vel = vel + acc * dt        # then update velocity
    return pos, vel


def leapfrog_step(pos, vel, mass, G=1.0, dt=0.01, softening=0.1):
    """
    Leapfrog (Velocity Verlet / kick-drift-kick) integrator.
    Symplectic — conserves energy far better than Euler for long runs.

    Algorithm (KDK):
      1. vel_half = vel + acc * (dt/2)   [half kick]
      2. pos_new  = pos + vel_half * dt  [full drift]
      3. acc_new  = forces(pos_new)
      4. vel_new  = vel_half + acc_new * (dt/2)  [half kick]
    """
    acc = _compute_acc(pos, mass, G, softening)
    vel_half = vel + acc * (dt / 2)
    pos_new = pos + vel_half * dt
    acc_new = _compute_acc(pos_new, mass, G, softening)
    vel_new = vel_half + acc_new * (dt / 2)
    return pos_new, vel_new


def compute_energy(pos, vel, mass, G=1.0, softening=0.1):
    """
    Total mechanical energy: kinetic + potential.

    KE = 0.5 * sum_i( m_i * |v_i|^2 )
    PE = -G * sum_{i<j}( m_i * m_j / sqrt(r_ij^2 + softening^2) )
    """
    KE = 0.5 * np.sum(mass * np.sum(vel ** 2, axis=1))

    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]          # (N, N, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + softening ** 2)    # (N, N)
    i_idx, j_idx = np.triu_indices(len(mass), k=1)
    PE = -G * np.sum(mass[i_idx] * mass[j_idx] / dist[i_idx, j_idx])

    return KE + PE


def nbody_step(pos, vel, mass, G=1.0, dt=0.01, softening=0.1):
    """
    Single-process vectorized step. Suitable for small N (e.g. visualization).
    Calls compute_forces_chunk for the full body array on one process.
    """
    _, acc = compute_forces_chunk((0, len(mass), pos, mass, G, softening))
    vel = vel + acc * dt
    pos = pos + vel * dt
    return pos, vel


def nbody_parallel_step_persistent(pos, vel, mass, pool, n_workers, G=1.0, dt=0.01, softening=0.1):
    """
    Multi-process step using a caller-supplied, already-running pool.

    The pool is created once outside the step loop and reused every step,
    eliminating the per-step process spawn overhead of nbody_parallel_step.
    This is the version used for benchmarking and production runs.
    """
    N = len(mass)
    chunk_size = N // n_workers

    chunks = []
    for w in range(n_workers):
        i_start = w * chunk_size
        i_end = N if w == n_workers - 1 else i_start + chunk_size
        chunks.append((i_start, i_end, pos, mass, G, softening))

    results = pool.map(compute_forces_chunk, chunks)

    acc = np.zeros((N, 3))
    for i_start, acc_chunk in results:
        acc[i_start:i_start + len(acc_chunk)] = acc_chunk

    vel = vel + acc * dt
    pos = pos + vel * dt
    return pos, vel


def nbody_parallel_step(pos, vel, mass, n_workers=4, G=1.0, dt=0.01, softening=0.1):
    N = len(mass)
    chunk_size = N // n_workers

    # Build independent work chunks — no shared state between workers
    chunks = []
    for w in range(n_workers):
        i_start = w * chunk_size
        i_end = N if w == n_workers - 1 else i_start + chunk_size
        chunks.append((i_start, i_end, pos, mass, G, softening))

    # Workers run in parallel — sync point here
    with Pool(n_workers) as pool:
        results = pool.map(compute_forces_chunk, chunks)

    # Reassemble accelerations from all workers
    acc = np.zeros((N, 3))
    for i_start, acc_chunk in results:
        acc[i_start:i_start + len(acc_chunk)] = acc_chunk

    vel = vel + acc * dt
    pos = pos + vel * dt
    return pos, vel

def run_simulation(N=500, steps=1000, n_workers=4):
    # Random initial conditions
    np.random.seed(42)
    pos  = np.random.randn(N, 3) * 10.0   # positions
    vel  = np.random.randn(N, 3) * 0.1    # velocities
    mass = np.random.uniform(0.5, 2.0, N) # masses

    for step in range(steps):
        pos, vel = nbody_parallel_step(pos, vel, mass, n_workers=n_workers)
        if step % 100 == 0:
            print(f"Step {step}/{steps}")

if __name__ == "__main__":
    run_simulation(N=1000, steps=500, n_workers=8)