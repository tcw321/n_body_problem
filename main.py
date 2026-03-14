from multiprocessing import Pool
import numpy as np

def compute_forces_chunk(args):
    """
    Compute forces for a subset of bodies (rows i_start..i_end).
    Each worker is completely independent — no shared state.
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

    # Reassemble forces from all workers
    force = np.zeros((N, 3))
    for i_start, force_chunk in results:
        force[i_start:i_start + len(force_chunk)] = force_chunk

    acc = force / mass[:, np.newaxis]
    vel += acc * dt
    pos += vel * dt
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