#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generate_test_matrix(FILE *file, int n, int diagonal_value, int off_diagonal, int b_value)
{
    // Print matrix size to file
    fprintf(file, "%d\n", n);

    // Print matrix A (diagonally dominant) to file
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                fprintf(file, "%d", diagonal_value);
            }
            else
            {
                fprintf(file, "%d", off_diagonal);
            }
            if (j < n - 1)
                fprintf(file, " ");
        }
        fprintf(file, "\n");
    }

    // Print vector b (all equal values) to file
    for (int i = 0; i < n; i++)
    {
        fprintf(file, "%d", b_value);
        if (i < n - 1)
            fprintf(file, " ");
    }
    fprintf(file, "\n");
}

void generate_random_matrix(FILE *file, int n, int min_val, int max_val, double diagonal_dominance)
{
    srand(time(NULL));

    // Print matrix size to file
    fprintf(file, "%d\n", n);

    // Generate and print matrix A to file
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
            fprintf(file, "%d", value);
            if (j < n - 1)
                fprintf(file, " ");
            row_sum += value;
        }
        fprintf(file, "\n");
    }

    // Generate and print vector b (scaled based on matrix values) to file
    for (int i = 0; i < n; i++)
    {
        fprintf(file, "%d", rand() % (max_val - min_val + 1) + min_val);
        if (i < n - 1)
            fprintf(file, " ");
    }
    fprintf(file, "\n");
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s <matrix_size> [mode] [output_filename]\n", argv[0]);
        printf("Modes:\n");
        printf("  0 - Simple test pattern (default)\n");
        printf("  1 - Random diagonally dominant matrix\n");
        printf("If output_filename is not provided, defaults to 'matrix_output.txt'\n");
        return 1;
    }

    int n = atoi(argv[1]);
    int mode = (argc > 2) ? atoi(argv[2]) : 0;
    const char *filename = (argc > 3) ? argv[3] : "matrix_output.txt";

    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file for writing.\n");
        return 1;
    }

    if (mode == 0)
    {
        // Simple test pattern (like your 5x5 example)
        generate_test_matrix(file, n, n + 1, 1, 2 * n);
    }
    else
    {
        // Random diagonally dominant matrix
        generate_random_matrix(file, n, 1, 10, 1.5);
    }

    fclose(file);
    printf("Matrix successfully written to %s\n", filename);

    return 0;
}