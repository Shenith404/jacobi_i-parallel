#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define ITER_MAX 10000
#define EPSILON 1e-6

int main()
{
    omp_set_num_threads(8); // Set the number of OpenMP threads
    printf("Number of threads: %d\n", omp_get_max_threads());

    FILE *file = fopen("../matrix_output.txt", "r");
    if (file == NULL)
    {
        printf("Error opening file.\n");
        return 1;
    }

    int n;
    fscanf(file, "%d", &n);

    // Dynamic memory allocation
    double **A = malloc(n * sizeof(double *));
    double *b = malloc(n * sizeof(double));
    double *x_old = calloc(n, sizeof(double));
    double *x_new = calloc(n, sizeof(double));

    for (int i = 0; i < n; i++)
        A[i] = malloc(n * sizeof(double));

    // Read matrix A
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            fscanf(file, "%lf", &A[i][j]);

    // Read vector b
    for (int i = 0; i < n; i++)
        fscanf(file, "%lf", &b[i]);

    fclose(file);

    // Start measuring time
    double start_time = omp_get_wtime();

    // Jacobi iteration
    for (int k = 0; k < ITER_MAX; k++)
    {
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
            {
                if (j != i)
                    sum += A[i][j] * x_old[j];
            }
            x_new[i] = (b[i] - sum) / A[i][i];
        }

#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            x_old[i] = x_new[i];
        }
    }

    // Stop measuring time
    double end_time = omp_get_wtime();

    // Print results
    for (int i = 0; i < n; i++)
    {
        printf("x[%d] = %.6f\n", i, x_new[i]);
    }

    printf("Time taken: %.6f seconds\n", end_time - start_time);
    printf("Number of iterations: %d\n", ITER_MAX);

    // Free memory
    for (int i = 0; i < n; i++)
        free(A[i]);
    free(A);
    free(b);
    free(x_old);
    free(x_new);

    return 0;
}
