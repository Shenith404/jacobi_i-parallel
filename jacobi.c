#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

// Define constants
#define N 1000         // Grid size (N x N)
#define MAX_ITER 10000 // Maximum number of iterations
#define TOL 1e-6       // Convergence tolerance

int main(int argc, char *argv[])
{
    int rank, size, i, j, iter;
    double **u, **u_new, diff, global_diff;
    int rows_per_proc, start_row, end_row;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    // Determine the number of rows per process
    rows_per_proc = N / size;
    start_row = rank * rows_per_proc;
    end_row = start_row + rows_per_proc - 1;

    // Allocate memory for local grids with ghost rows
    // Each process allocates rows_per_proc + 2 rows to include top and bottom ghost rows
    u = malloc((rows_per_proc + 2) * sizeof(double *));
    u_new = malloc((rows_per_proc + 2) * sizeof(double *));
    for (i = 0; i < rows_per_proc + 2; i++)
    {
        u[i] = malloc(N * sizeof(double));
        u_new[i] = malloc(N * sizeof(double));
    }

    // Initialize the local grids to zero
    for (i = 1; i <= rows_per_proc; i++)
    {
        for (j = 0; j < N; j++)
        {
            u[i][j] = 0.0;
            u_new[i][j] = 0.0;
        }
    }

    iter = 0;
    do
    {
        // Exchange boundary rows with neighboring processes

        // Send the first internal row to the previous process and receive the ghost row
        if (rank > 0)
        {
            MPI_Sendrecv(u[1], N, MPI_DOUBLE, rank - 1, 0,
                         u[0], N, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Send the last internal row to the next process and receive the ghost row
        if (rank < size - 1)
        {
            MPI_Sendrecv(u[rows_per_proc], N, MPI_DOUBLE, rank + 1, 0,
                         u[rows_per_proc + 1], N, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        diff = 0.0;

// Perform the Jacobi iteration using OpenMP for parallelization
#pragma omp parallel for private(j) reduction(max : diff)
        for (i = 1; i <= rows_per_proc; i++)
        {
            for (j = 1; j < N - 1; j++)
            {
                // Update the grid point based on the average of its four neighbors
                u_new[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] +
                                      u[i][j - 1] + u[i][j + 1]);

                // Compute the maximum difference for convergence check
                if (fabs(u_new[i][j] - u[i][j]) > diff)
                {
                    diff = fabs(u_new[i][j] - u[i][j]);
                }
            }
        }

        // Swap the pointers for the next iteration
        double **temp = u;
        u = u_new;
        u_new = temp;

        // Compute the global maximum difference across all processes
        MPI_Allreduce(&diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        iter++;
    } while (global_diff > TOL && iter < MAX_ITER);

    // Free allocated memory
    for (i = 0; i < rows_per_proc + 2; i++)
    {
        free(u[i]);
        free(u_new[i]);
    }
    free(u);
    free(u_new);

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}
// MPI Initialization: Sets up the MPI environment and determines the rank and size of the processes.

// Domain Decomposition: The grid is divided among the MPI processes, each handling a block of rows. Ghost rows are added to facilitate boundary exchanges.

// Memory Allocation: Each process allocates memory for its portion of the grid, including ghost rows.

// Boundary Exchange: Processes exchange their boundary rows with neighboring processes to ensure data consistency for the Jacobi update.

// Jacobi Update: Each grid point is updated based on the average of its four neighbors. OpenMP is used to parallelize this computation within each process.

// Convergence Check: The maximum difference between the new and old grid values is computed locally and then globally across all processes to check for convergence.

// Pointer Swapping: After each iteration, the pointers to the current and new grids are swapped to prepare for the next iteration.

// Finalization: Once convergence is achieved or the maximum number of iterations is reached, the MPI environment is finalized, and memory is freed.