#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ITER_MAX 10000
#define EPSILON 1e-6

int main()
{
    FILE *file = fopen("../matrix_output.txt", "r");
    if (file == NULL)
    {
        printf("Error opening file.\n");
        return 1;
    }

    clock_t start = clock();

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

    // Jacobi iteration
    for (int k = 0; k < ITER_MAX; k++)
    {
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

        // Copy x_new to x_old
        for (int i = 0; i < n; i++)
            x_old[i] = x_new[i];
    }

    // Print results
    for (int i = 0; i < n; i++)
        printf("x[%d] = %.6f\n", i, x_new[i]);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken: %.6f seconds\n", time_spent);
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
