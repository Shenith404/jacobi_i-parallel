#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generate_test_matrix(int n, int diagonal_value, int off_diagonal, int b_value)
{
    // Print matrix size
    printf("%d\n", n);

    // Print matrix A (diagonally dominant)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                printf("%d", diagonal_value);
            }
            else
            {
                printf("%d", off_diagonal);
            }
            if (j < n - 1)
                printf(" ");
        }
        printf("\n");
    }

    // Print vector b (all equal values)
    for (int i = 0; i < n; i++)
    {
        printf("%d", b_value);
        if (i < n - 1)
            printf(" ");
    }
    printf("\n");
}

void generate_random_matrix(int n, int min_val, int max_val, double diagonal_dominance)
{
    srand(time(NULL));

    // Print matrix size
    printf("%d\n", n);

    // Generate and print matrix A
    for (int i = 0; i < n; i++)
    {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            int value;
            if (i == j)
            {
                // Make diagonal dominant
                value = (int)((max_val - min_val) * diagonal_dominance) + min_val;
            }
            else
            {
                value = rand() % (max_val - min_val + 1) + min_val;
            }
            printf("%d", value);
            if (j < n - 1)
                printf(" ");
            row_sum += value;
        }
        printf("\n");
    }

    // Generate and print vector b (scaled based on matrix values)
    for (int i = 0; i < n; i++)
    {
        printf("%d", rand() % (max_val - min_val + 1) + min_val);
        if (i < n - 1)
            printf(" ");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s <matrix_size> [mode]\n", argv[0]);
        printf("Modes:\n");
        printf("  0 - Simple test pattern (default)\n");
        printf("  1 - Random diagonally dominant matrix\n");
        return 1;
    }

    int n = atoi(argv[1]);
    int mode = (argc > 2) ? atoi(argv[2]) : 0;

    if (mode == 0)
    {
        // Simple test pattern (like your 5x5 example)
        generate_test_matrix(n, n + 1, 1, 2 * n);
    }
    else
    {
        // Random diagonally dominant matrix
        generate_random_matrix(n, 1, 10, 1.5);
    }

    return 0;
}
// ./matrix_gen 100 1 > input_random100.txt