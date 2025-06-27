#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define ITER_MAX 10000
#define EPSILON 1e-6

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;
    double **A = NULL;
    double *b = NULL;

    if (rank == 0)
    {
        FILE *file = fopen("../matrix_output.txt", "r");
        if (file == NULL)
        {
            printf("Error opening file.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fscanf(file, "%d", &n);

        A = malloc(n * sizeof(double *));
        b = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++)
            A[i] = malloc(n * sizeof(double));

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                fscanf(file, "%lf", &A[i][j]);

        for (int i = 0; i < n; i++)
            fscanf(file, "%lf", &b[i]);

        fclose(file);
    }

    // Broadcast problem size
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        A = malloc(n * sizeof(double *));
        b = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++)
            A[i] = malloc(n * sizeof(double));
    }

    // Flatten matrix for broadcast
    double *flat_A = malloc(n * n * sizeof(double));
    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                flat_A[i * n + j] = A[i][j];
    }

    MPI_Bcast(flat_A, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i][j] = flat_A[i * n + j];
    }
    free(flat_A);

    // Determine workload
    int chunk_size = n / size;
    int remainder = n % size;
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        counts[i] = chunk_size + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    int start_row = displs[rank];
    int end_row = start_row + counts[rank];

    double *x_old = calloc(n, sizeof(double));
    double *x_new = calloc(n, sizeof(double));

    double start_time = MPI_Wtime();

    for (int k = 0; k < ITER_MAX; k++)
    {
        double local_max_error = 0.0;
        double global_max_error = 0.0;

        // Parallelize row computation
#pragma omp parallel for reduction(max : local_max_error)
        for (int i = start_row; i < end_row; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
            {
                if (j != i)
                    sum += A[i][j] * x_old[j];
            }
            x_new[i] = (b[i] - sum) / A[i][i];

            double error = fabs(x_new[i] - x_old[i]);
            if (error > local_max_error)
                local_max_error = error;
        }

        // Synchronize x_new across all processes
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       x_new, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        // Find global max error for convergence
        MPI_Allreduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (global_max_error < EPSILON)
        {
            if (rank == 0)
                printf("Converged after %d iterations\n", k + 1);
            break;
        }

#pragma omp parallel for
        for (int i = 0; i < n; i++)
            x_old[i] = x_new[i];
    }

    double end_time = MPI_Wtime();

    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
            printf("x[%d] = %.6f\n", i, x_new[i]);
        printf("Time taken: %.6f seconds\n", end_time - start_time);
    }

    
    MPI_Finalize();
    return 0;
}
