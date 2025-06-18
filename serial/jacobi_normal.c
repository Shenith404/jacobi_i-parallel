#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ITER_MAX 100000
#define EPSILON 1e-6
#define MAX_SIZE 150

int main()
{
    FILE *file = fopen("../matrix_output.txt", "r");
    if (file == NULL)
    {
        printf("Error opening file.\n");
        return 1;
    }

    // start measuring time
    clock_t start = clock();

    int n;
    fscanf(file, "%d", &n);

    double max_error = 0.0;
    double A[MAX_SIZE][MAX_SIZE];
    double b[MAX_SIZE];
    double x_old[MAX_SIZE] = {0};
    double x_new[MAX_SIZE] = {0};

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

        // Check for convergence
        for (int i = 0; i < n; i++)
        {
            double error = fabs(x_new[i] - x_old[i]);
            if (error > max_error)
                max_error = error;
            x_old[i] = x_new[i];
        }

        if (max_error < EPSILON)
        {
            printf("Converged in %d iterations\n", k + 1);
            break;
        }
    }

    // Print result
    for (int i = 0; i < n; i++)
    {
        printf("x[%d] = %.6f\n", i, x_new[i]);
    }

    // Stop measuring time
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken: %.6f seconds\n", time_spent);

    return 0;
}
