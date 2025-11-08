# 2D Jacobi Solver: OpenMP vs. MPI

This project demonstrates two parallel implementations of an iterative Jacobi solver for a 2D Poisson equation on a square grid.

1.  **`task_1.cc`**: A serial implementation parallelized using **OpenMP** threads.
2.  **`task_2.cc`**: A distributed-memory implementation using **MPI** (and designed to be hybrid-ready with OpenMP).

Both programs solve the equation on an `N x N` grid by iteratively updating a solution grid `u` based on a forcing function `f` until the change between iterations drops below a specified `TOLERANCE`.

## `task_1.cc`: OpenMP Jacobi Solver

This file implements the Jacobi solver within a single process, using OpenMP to parallelize the computationally expensive loops.

### How It Works

1.  **Initialization**:
    * Two `N x N` grids (represented as 1D vectors), `u` (current solution) and `unew` (next step solution), are created.
    * The exact solution `u_exact` and forcing function `f` are computed.
    * Dirichlet boundary conditions (fixed values on the edges) are applied to `u` and `unew`.
    * This initialization is parallelized with `#pragma omp parallel for collapse(2)`.

2.  **Iterative Solver Loop**:
    The solver runs in a loop until convergence or `MAX_ITER` is reached. Each iteration has three steps:
    * **A. Compute `unew`**: A parallel loop (`#pragma omp parallel for`) iterates over all *interior* grid points (i, j). The value for `unew[i, j]` is calculated using the 5-point stencil based on the *old* values from the `u` grid.
    * **B. Check Convergence**: A parallel reduction loop (`#pragma omp parallel for reduction(max:max_diff)`) calculates the maximum absolute difference between `unew[i, j]` and `u[i, j]` for all interior points.
    * **C. Swap Grids**: `std::swap(u, unew)` is called. The newly computed `unew` becomes the `u` for the next iteration, and the old `u` becomes the `unew` to be overwritten.

3.  **Final Check**:
    * After the loop terminates, a final parallel reduction (`check_result`) computes the maximum absolute error between the final solution `u` and the `u_exact` grid.

---

## `task_2.cc`: MPI Jacobi Solver

This file implements a distributed version of the same solver using MPI. It's designed to run across multiple processes, with each process handling a "tile" of the global grid.

### How It Works

1.  **Initialization**:
    * MPI is initialized with `MPI_Init_thread`, requesting `MPI_THREAD_FUNNELED`. This prepares the program for hybrid MPI + OpenMP use, where only the main thread in each process makes MPI calls. However, my machine does not have enough cores to run OMP efficiently -- running OMP with MPI leads to thread oversubscription on a M4 MacBook Air.
    * **Domain Decomposition**: The `N x N` global grid is divided into horizontal tiles using 1D decomposition. Each MPI process calculates how many rows it owns (`local_N`) and its starting row index in the global grid.
    * **Ghost Rows**: Each process allocates local grids (`u_local`, `unew_local`) that are padded with two extra rows:
        * `local_row[0]` is the **top ghost row** (to store data from `rank - 1`).
        * `local_row[local_N + 1]` is the **bottom ghost row** (to store data from `rank + 1`).
    * Each process initializes only its owned rows (from `local_row[1]` to `local_row[local_N]`).

2.  **Iterative Solver Loop**:
    The core of the MPI solver is designed to overlap communication and computation.
    * **A. Post Non-blocking Exchanges**: The loop begins by posting four non-blocking MPI calls:
        * `MPI_Irecv`: To receive data *into* its top ghost row (from `rank_up`).
        * `MPI_Irecv`: To receive data *into* its bottom ghost row (from `rank_down`).
        * `MPI_Isend`: To send its top *boundary* row (`local_row[N]` which is row 1) to `rank_up`.
        * `MPI_Isend`: To send its bottom *boundary* row (`local_row[local_N * N]`) to `rank_down`.
    * **B. Compute Interior**: While the MPI calls are in flight, the process computes the internal rows of its tile (from `local_row[2]` to `local_row[local_N - 1]`). These calculations do not depend on the ghost row data, so they can be performed immediately.
    * **C. Compute Boundaries**: The process enters a loop to wait for the `MPI_Irecv` calls to complete.
        * It calls `MPI_Waitany` on the two receive requests.
        * When the **top** ghost row arrives (`recv_index == 0`), it computes its top boundary row (`local_row[1]`).
        * When the **bottom** ghost row arrives (`recv_index == 1`), it computes its bottom boundary row (`local_row[local_N]`).
        * This overlaps computation with the *other* pending communication.
    * **D. Check Convergence**: Each process finds its `local_max_diff`. Then, `MPI_Allreduce` is used to find the `global_max_diff` and share it with all processes.
    * **E. Swap Grids**: `std::swap(u_local, unew_local)` is called locally on each process.
    * **F. Wait for Sends**: `MPI_Waitall` is called on the *Isend* requests to ensure they have completed before the next iteration begins (which would modify the buffer they were reading from).

3.  **Final Check**:
    * Each process computes its `local_max_error` using `check_result`.
    * `MPI_Reduce` is used to find the global maximum error from all processes and report it on `rank 0`.


### Time to Solution
 **task_1.cc (OpenMP Jacobi Solver)**: 2.64095s
 **task_2.cc (MPI Jacobi Solver)**: 0.46594s, run with 10 MPI processes