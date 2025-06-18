#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define ITER_MAX 100000
#define EPSILON 1e-6
#define MAX_SIZE 150

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Only rank 0 reads the input file
    int n = 0;
    double A[MAX_SIZE][MAX_SIZE] = {0};
    double b[MAX_SIZE] = {0};

    if (rank == 0)
    {
        FILE *file = fopen("../matrix_output.txt", "r");
        if (file == NULL)
        {
            printf("Error opening file.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        fscanf(file, "%d", &n);

        // Read matrix A
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                fscanf(file, "%lf", &A[i][j]);

        // Read vector b
        for (int i = 0; i < n; i++)
            fscanf(file, "%lf", &b[i]);

        fclose(file);
    }

    // Broadcast the problem size to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast matrix A and vector b
    MPI_Bcast(A, MAX_SIZE * MAX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, MAX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Determine the workload for each process
    int chunk_size = n / size;
    int remainder = n % size;
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        counts[i] = chunk_size;
        if (i == size - 1)
            counts[i] += remainder;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    int start_row = displs[rank];
    int end_row = start_row + counts[rank];

    double x_old[MAX_SIZE] = {0};
    double x_new[MAX_SIZE] = {0};

    // Start measuring time
    double start_time = MPI_Wtime();

    // Jacobi iteration
    for (int k = 0; k < ITER_MAX; k++)
    {

        for (int i = start_row; i < end_row; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
            {
                if (j != i)
                {
                    sum += A[i][j] * x_old[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
        }

        // Synchronize x_new across all processes using Allgatherv
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       x_new, counts, displs, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Update x_old for next iteration
        for (int i = 0; i < n; i++)
        {
            x_old[i] = x_new[i];
            // printf("thread %d updated x_old[%d]", omp_get_thread_num(), x_old[i]);
        }
    }

    // Stop measuring time
    double end_time = MPI_Wtime();
    double time_spent = end_time - start_time;

    // Only rank 0 prints the result
    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            printf("x[%d] = %.6f\n", i, x_new[i]);
        }
        printf("Time taken: %.6f seconds\n", time_spent);
    }

    free(counts);
    free(displs);
    MPI_Finalize();
    return 0;
}