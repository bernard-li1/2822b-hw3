#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <time.h>

using namespace std;

/**
 * @brief inits local grids for single MPI process
 * 
 * each process inits 1d tile of forcing function f, exact sol'n u_exact, and 
 * init guesses for u and unew.
 * 
 * Dirichlet BCs based on given exact sol'n
 * 
 * @param u_local       local soln grid, padded with ghost rows
 * @param unew_local        local grid for next iteration, padded with ghost rows
 * @param f_local       local forcing function, padded
 * @param u_exact_local     local exact soln grid, padded
 * @param N     global grid dimension (N x N)
 * @param h     grid spacing (delta x or delta y)
 * @param local_N       num rows process owns
 * @param global_start_row      global row index of this process's local row 1
 */
void initialize_local_grids(vector<double>& u_local,
                            vector<double>& unew_local,
                            vector<double>& f_local,
                            vector<double>& u_exact_local,
                            int N, double h, int local_N,
                            int global_start_row) {
    
    // note: local N -> num of *owned* rows
    // total size of local vectors == (local_N + 2) * N
    // owned rows @ local indices [1, local_N]
    // ghost rows @ local indices 0 & local_N + 1
    for (int i_local = 1; i_local <= local_N; ++i_local) {
        int i_global = global_start_row + i_local - 1;

        for (int j=0; j < N; ++j) {
            double x = i_global * h;
            double y = j * h;

            // 1d index into padded array
            int idx_local = i_local * N + j;

            u_exact_local[idx_local] = sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
            f_local[idx_local] = -8.0 * M_PI * M_PI * sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y);

            // apply Dirichlet (fixed) BCs
            if (i_global == 0 || i_global == N-1 || j == 0 || j == N-1) {
                u_local[idx_local] = u_exact_local[idx_local];
                unew_local[idx_local] = u_exact_local[idx_local];
            } else {
                // init interior pts
                u_local[idx_local] = 0.0;
                unew_local[idx_local] = 0.0;
            }
        }
    }
}

/**
 * @brief checks computed soln against exact soln **locally**
 * 
 * checks max abs error *only on rows owned by process*
 * 
 * @param u_local     final computed local soln grid, padded
 * @param u_exact_local       local exact soln grid, padded
 * @param N     global grid dim (N x N)
 * @param local_N       num rows this proc owns
 * @return      max abs error *for this process*
*/
double check_result(const vector<double>& u_local,
                    const vector<double>& u_exact_local, int N, int local_N) {
    double max_error = 0.0;
    // proc owns [1,local_N]; ghost rows @ 0 and local_N + 1
    for (int i=1; i <= local_N; ++i) {
        for (int j=0; j < N; ++j) {
            int idx = i * N + j;
            double error = fabs(u_local[idx] - u_exact_local[idx]);
            if (error > max_error) {
                max_error = error;
            }
        }
    }
    return max_error;
}

/**
 * @brief main function for MPI Jacobi/iterative solver
 */
int main(int argc, char* argv[]) {
    // 1. MPI and problem setup
    // MPI_Init(&argc, &argv);
    int provided_thread_level;
    MPI_Init_thread( &argc , &argv , MPI_THREAD_FUNNELED , &provided_thread_level);
    cout<<"PROVIDED THREAD LEVEL: "<<provided_thread_level<<endl;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 201; // global grid dim is N x N
    const double h = 1.0 / (N-1); // delta x == delta y
    const int MAX_ITER = 100000;
    const double TOLERANCE = 1.0e-8;
    const int PRINT_FREQ = 500;
    double h2 = h * h;

    // 2. 1d tiling / ghost rows
    int rows_per_proc = N / size;
    int remainder_rows = N % size;
    int local_N = (rank < remainder_rows) ? (rows_per_proc + 1) : rows_per_proc;
    int global_start_row = 0;
    if (rank < remainder_rows) {
        global_start_row = rank * local_N;
    } else {
        global_start_row = (rows_per_proc + 1) * remainder_rows + (rank - remainder_rows) * rows_per_proc;
    }
    int rank_up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int rank_down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // 3. allocate local grids
    int local_padded_size = (local_N + 2) * N;
    vector<double> u_local(local_padded_size);
    vector<double> unew_local(local_padded_size);
    vector<double> f_local(local_padded_size);
    vector<double> u_exact_local(local_padded_size);

    initialize_local_grids(u_local, unew_local, f_local, u_exact_local, N, h, local_N, global_start_row);

    // MPI Request handles for non-blocking communication
    // request[0,1,2,3] = Recv from UP, Recv from DOWN, Send to UP, Send to Down
    MPI_Request requests[4];

    if (rank == 0) {
        cout << "Starting MPI Jacobi solver with " << size << " processes." << endl;
        cout << "Global grid size: " << N << " x " << N << endl;
    }

    // 4. iterative/jacobi solver 
    int iter = 0;
    double global_max_diff = 0.0;

    struct timespec start_total, end_total;
    double total_time;

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &start_total);
    }

    for (iter = 0; iter < MAX_ITER; ++iter) {
        double local_max_diff = 0.0;

        // A. post non-blocking ghost row exchanges
        MPI_Irecv(&u_local[0], N, MPI_DOUBLE, rank_up, 1, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&u_local[(local_N+1) * N], N, MPI_DOUBLE, rank_down, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(&u_local[N], N, MPI_DOUBLE, rank_up, 0, MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(&u_local[local_N * N], N, MPI_DOUBLE, rank_down, 1, MPI_COMM_WORLD, &requests[3]);

        // B. compute interior (local) pts; rows 2 to local_N -1
        for (int i=2; i<local_N; ++i) {
            for (int j=1; j<N - 1; ++j) {
                int idx = i * N + j;
                int idx_im1 = (i-1) * N + j;
                int idx_ip1 = (i+1) * N + j;
                int idx_jm1 = i*N + (j-1);
                int idx_jp1 = i*N + (j+1);

                unew_local[idx] = 0.25 * (u_local[idx_im1] + u_local[idx_ip1] +
                                          u_local[idx_jm1] + u_local[idx_jp1] -
                                          h2 * f_local[idx]);
            }
        }

        // C. Waitany for recvs + compute boundary pts as data arrives
        // MPI_Waitany to compute boundary rows as soon as ghost data arrives (overlap computation + communication!)

        // if local_N == 1, proc owns only 1 row. -> must wait for both UP and DOWN ghost rows
        if (local_N == 1) {
            MPI_Waitall(2, &requests[0], MPI_STATUS_IGNORE);

            int i = 1;
            // only compute if we are not a global boundary (fixed BCs)
            int i_global = global_start_row + i - 1;
            if (i_global != 0 && i_global != N - 1) {
                for (int j = 1; j < N - 1; ++j) {
                    int idx = i * N + j;
                    int idx_im1 = (i - 1) * N + j; // row 0 (UP ghost)
                    int idx_ip1 = (i + 1) * N + j; // row 2 (DOWN ghost)
                    int idx_jm1 = i * N + (j - 1);
                    int idx_jp1 = i * N + (j + 1);
                    
                    unew_local[idx] = 0.25 * (u_local[idx_im1] + u_local[idx_ip1] +
                                              u_local[idx_jm1] + u_local[idx_jp1] -
                                              h2 * f_local[idx]);
                }
            }

        } else {
            // standard case: local_N > 1
            
            int recv_index;
            // loop twice (for each recv)
            for (int n=0; n<2; ++n) {
                // wait for any recv to complete
                MPI_Waitany(2, &requests[0], &recv_index, MPI_STATUS_IGNORE);

                if (recv_index == 0) {
                    // data from UP (requests[0]) arrived. compute local row 1.
                    int i = 1;
                    int i_global = global_start_row + i - 1;
                    if (i_global != 0) {
                        for (int j=1; j<N-1;++j) {
                            int idx = i * N + j;
                            int idx_im1 = (i - 1) * N + j; // ghost row 0
                            int idx_ip1 = (i + 1) * N + j; // local row 2
                            int idx_jm1 = i * N + (j - 1); 
                            int idx_jp1 = i * N + (j + 1); 
                            
                            unew_local[idx] = 0.25 * (u_local[idx_im1] + u_local[idx_ip1] +
                                                    u_local[idx_jm1] + u_local[idx_jp1] -
                                                    h2 * f_local[idx]);
                        }
                    }
                } else if (recv_index == 1) {
                    // data from DOWN (requests[1]) arrived. compute local row local_N.
                    int i = local_N;
                    int i_global = global_start_row + i - 1;
                    if (i_global != N - 1) {
                        for (int j = 1; j < N - 1; ++j) {
                            int idx = i * N + j;
                            int idx_im1 = (i - 1) * N + j; // local row local_N-1
                            int idx_ip1 = (i + 1) * N + j; // ghost row local_N+1
                            int idx_jm1 = i * N + (j - 1);
                            int idx_jp1 = i * N + (j + 1); 
                            
                            unew_local[idx] = 0.25 * (u_local[idx_im1] + u_local[idx_ip1] +
                                                    u_local[idx_jm1] + u_local[idx_jp1] -
                                                    h2 * f_local[idx]);
                        }
                    }
                }
            }
        }

        // D. check for convergence
        for (int i=1; i<=local_N; ++i) {
            int i_global = global_start_row + i - 1;
            if (i_global != 0 && i_global != N -1) {
                for (int j = 1; j < N-1; ++j) {
                    int idx = i * N + j;
                    double diff = fabs(unew_local[idx] - u_local[idx]);
                    if (diff > local_max_diff) {
                        local_max_diff = diff;
                    }
                }
            }
        }

        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // E. update u for next iter
        swap(u_local, unew_local);

        // F. wait for sends to complete (avoids overwrite while sending)
        MPI_Waitall( 2 , &requests[2] , MPI_STATUS_IGNORE);

        // G. print stats, check exit
        if (rank == 0 && iter % PRINT_FREQ == 0) {
            cout << "Iter: " << setw(5) << iter 
                      << ", Max Diff: " << scientific << global_max_diff << endl;
        }

        if (global_max_diff < TOLERANCE) {
            if (rank == 0) {
                cout << "Convergence reached at iteration " << iter << "!" << endl;
                cout << "Final Max Diff: " << scientific << global_max_diff << endl;
            }
            break;
        }
    }

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &end_total);
        total_time = (end_total.tv_sec - start_total.tv_sec) + 
                     (end_total.tv_nsec - start_total.tv_nsec) / 1e9;
    }

    if (iter == MAX_ITER) {
        if (rank == 0) {
            cout << "Solver stopped after reaching max iterations (" << MAX_ITER << ")." << endl;
            cout << "Final Max Diff: " << scientific << global_max_diff << endl;
        }
    }

    // 5. check soln locally, then reduce to global max error on rank 0
    double local_max_error = check_result(u_local, u_exact_local, N, local_N);
    double global_max_error = 0.0;
    MPI_Reduce( &local_max_error , &global_max_error , 1 , MPI_DOUBLE , MPI_MAX , 0 , MPI_COMM_WORLD);
    

    if (rank == 0) {
        cout << "Total time to solution: " << fixed << setprecision(5) << total_time << "s" << endl;
        cout << fixed << setprecision(10);
        cout << "Maximum Absolute Error: " << global_max_error << endl;

        if (global_max_error > 1.0e-4) {
            cout << "Warning: Error is larger than expected." << endl;
        } else {
            cout << "Result is correct within expected numerical error." << endl;
        }
    }

    MPI_Finalize();
    return 0;
}

