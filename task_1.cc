#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>    
#include <iomanip>  
#include <time.h>  

using namespace std;

/**
 * @brief inits grids for problem. grids are actually 1d vectors
 *
 * Sets up forcing function f, exact solution u_exact, init state of u and unew
 * Applies Dirichlet BCs to u and unew based on exact solution (exact solution has zero boundaries on all sides except y=0 and y=1).
 *
 * @param u     main solution grid (u_old)
 * @param unew      grid for the next iteration
 * @param f     forcing function grid
 * @param u_exact       grid w/ exact solution
 * @param N     number points in one dimension (grid size is N x N)
 * @param h     grid spacing (delta x or delta y)
 */
void initialize_grids(vector<double>& u, vector<double>& unew,
                        vector<double>& f, vector<double>& u_exact,
                        int N, double h) {
    
    // parallelize grid initialization
    // can use parallel for loop since each grid point is independent
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = i * h;
            double y = j * h;
            int idx = i * N + j;

            u_exact[idx] = sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
            f[idx] = -8.0 * M_PI * M_PI * sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y);

            // Dirichlet (fixed) BCs
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                u[idx] = u_exact[idx];
                unew[idx] = u_exact[idx];
            } else {
                // init guess for interior points
                u[idx] = 0.0;
                unew[idx] = 0.0;
            }
        }
    }
}

/**
 * @brief checks the final computed solution against exact solution
 *
 * Calculates the max absolute error between the computed and exact solutions
 *
 * @param u     final computed solution grid
 * @param u_exact       exact solution
 * @param N     num points in one dimension (grid is N x N)
 * @return      max absolute error
 */
double check_result(const vector<double>& u, const vector<double>& u_exact, int N) {
    
    double max_error = 0.0;

    // reduction lets us safely find max error across threads: ensures each thread calculates its local max, then local maxes combined to find global max stored in max_error
    // collapse since we can split work among threads. since N=201, collapsing allows a more balanced workload for each thread. my computer has 10 logical cores (max 10 threads)
    #pragma omp parallel for collapse(2) reduction(max:max_error)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            double error = fabs(u[idx] - u_exact[idx]);
            if (error > max_error) {
                max_error = error;
            }
        }
    }
    return max_error;
}

/**
 * @brief run Jacobi/iterative solver
 */
int main() {
    // 1. setup
    const int N = 201;                // grid dimension (we have N-1 intervals)
    const double h = 1.0 / (N - 1);   // grid spacing (delta x == delta y since we are working in a square)
    const int MAX_ITER = 100000;       
    const double TOLERANCE = 1.0e-8;  // convergence tolerance
    const int PRINT_FREQ = 500;       // how often to print iteration status

    // allocate memory
    vector<double> u(N * N);
    vector<double> unew(N * N);
    vector<double> f(N * N);
    vector<double> u_exact(N * N);

    struct timespec start_total, end_total, start_iter, end_iter;
    double total_time, iter_time;

    initialize_grids(u, unew, f, u_exact, N, h);
    double h2 = h * h;

    // 2. iterative (jacobi) solver
    int iter = 0;
    double max_diff = 0.0;

    cout << "Starting Jacobi solver..." << endl;

    clock_gettime(CLOCK_MONOTONIC, &start_total);
    for (iter = 0; iter < MAX_ITER; ++iter) {
        // clock_gettime(CLOCK_MONOTONIC, &start_iter);
        max_diff = 0.0;

        // A. Compute unew from u
        // Only loop over interior points (1 : N-2 inclusive) since we are given boundary conditions
        #pragma omp parallel for // collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                int idx = i * N + j;
                int idx_im1 = (i - 1) * N + j; // i-1, j
                int idx_ip1 = (i + 1) * N + j; // i+1, j
                int idx_jm1 = i * N + (j - 1); // i, j-1
                int idx_jp1 = i * N + (j + 1); // i, j+1

                unew[idx] = 0.25 * (u[idx_im1] + u[idx_ip1] + u[idx_jm1] + u[idx_jp1] - h2 * f[idx]);
            }
        }

        // B. Check for convergence by finding max difference between u and unew
        #pragma omp parallel for collapse(2) reduction(max:max_diff)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                int idx = i * N + j;
                double diff = fabs(unew[idx] - u[idx]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }

        // C. Update u for next iteration. swaps ptrs
        swap(u, unew);

        clock_gettime(CLOCK_MONOTONIC, &end_iter);
        iter_time = (end_iter.tv_sec - start_iter.tv_sec) + 
                    (end_iter.tv_nsec - start_iter.tv_nsec) / 1e9;

        // print stats, check exit cond.
        if (iter % PRINT_FREQ == 0) {
            cout << "Iter: " << setw(5) << iter 
                      << ", Max Diff: " << scientific << max_diff 
                      << ", Iter Time: " << fixed << setprecision(10) << iter_time << "s" << endl;
        }

        if (max_diff < TOLERANCE) {
            cout << "Convergence reached at iteration " << iter << "!" << endl;
            cout << "Final Max Diff: " << scientific << max_diff << endl;
            break;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end_total);
    total_time = (end_total.tv_sec - start_total.tv_sec) + 
                 (end_total.tv_nsec - start_total.tv_nsec) / 1e9;

    if (iter == MAX_ITER) {
        cout << "Solver stopped after reaching max iterations (" << MAX_ITER << ")." << endl;
        cout << "Final Max Diff: " << scientific << max_diff << endl;
    }

    cout << "Total time to solution: " << fixed << setprecision(5) << total_time << "s" << endl;

    // 3. check solution correctness
    double max_error = check_result(u, u_exact, N);

    cout << fixed << setprecision(10);
    cout << "Maximum Absolute Error: " << max_error << endl;

    if (max_error > 1.0e-4) { // NOTE: tolerance for numerical error (arbitrary M_PIck, to be honest)
        cout << "Warning: Error is larger than expected." << endl;
    } else {
        cout << "Result is correct within expected numerical error." << endl;
    }

    return 0;
}