#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define ITER_MAX 100000
#define EPSILON 1e-6
#define MAX_SIZE 150

int main()
{
    omp_set_num_threads(8); // Set the number of OpenMP threads
    printf("number of threads: %d\n", omp_get_max_threads());
    FILE *file = fopen("../matrix_output.txt", "r");
    if (file == NULL)
    {
        printf("Error opening file.\n");
        return 1;
    }

    int n;
    fscanf(file, "%d", &n);

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
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;

            // #pragma omp parallel for reduction(+ : sum)
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

    // Print result
    for (int i = 0; i < n; i++)
    {
        printf("x[%d] = %.6f\n", i, x_new[i]);
    }

    return 0;
}
