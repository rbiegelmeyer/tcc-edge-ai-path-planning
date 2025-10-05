#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define WIDTH 20
#define HEIGHT 20
#define DIFFICULTY 1

void printArray(int16_t array[HEIGHT][WIDTH])
{
    for (int16_t x = 0; x < (WIDTH + 2); x++)
    {
        printf("X");
    }
    printf("\n");
    for (int16_t y = 0; y < HEIGHT; y++)
    {
        printf("X");
        for (int16_t x = 0; x < WIDTH; x++)
        {
            printf("%c", array[y][x] == 1   ? 'X'
                         : array[y][x] == 2 ? '.'
                                            : ' ');
        }
        printf("X\n");
    }
    for (int16_t x = 0; x < (WIDTH + 2); x++)
    {
        printf("X");
    }
    printf("\n");
}
void printArrayNum(int16_t array[HEIGHT][WIDTH])
{
    for (int16_t y = 0; y < HEIGHT; y++)
    {
        for (int16_t x = 0; x < WIDTH; x++)
        {
            printf("%d ", array[y][x]);
        }
        printf("\n");
    }
}

int16_t printPrettyMatrix(float array[HEIGHT][WIDTH])
{
    for (int16_t y = 0; y < HEIGHT; y++)
    {
        for (int16_t x = 0; x < WIDTH; x++)
        {
            printf("%4.1f ", array[y][x]);
        }
        printf("\n");
    }
    return 0;
}

void printCostArray(float array[HEIGHT][WIDTH])
{
    for (int16_t y = 0; y < HEIGHT; y++)
    {
        for (int16_t x = 0; x < WIDTH; x++)
        {
            printf("%f ", array[y][x]);
        }
        printf("\n");
    }
}

void insertObstacles(int16_t *array, int16_t height, int16_t width, int16_t difficulty)
{
    // Set random seed
    // srand(time(0));
    srand(10);

    // Add obstacles
    for (int16_t y = 0; y < height; y++)
    {
        for (int16_t x = 0; x < width; x++)
        {
            int16_t value = rand() % difficulty;
            if (value != 1)
                value = 0;
            // array[y][x] = value;

            *(array + y * width + x) = value;
        }
    }
}

void calculateHeuristics(float h[HEIGHT][WIDTH], float g[HEIGHT][WIDTH], float f[HEIGHT][WIDTH], int16_t end[2], int16_t height, int16_t width)
{
    for (int16_t y = 0; y < height; y++)
    {
        for (int16_t x = 0; x < width; x++)
        {
            h[y][x] = sqrt(pow(end[1] - x, 2) + pow(end[0] - y, 2));
            f[y][x] = g[y][x] + h[y][x];
        }
    }
}

int16_t *getNewCurrent(float f[HEIGHT][WIDTH], int16_t array[HEIGHT][WIDTH], int16_t open[HEIGHT][WIDTH])
{
    static int16_t lowest[2] = {-1, -1};
    float lowestValue = 99999.9f;
    for (int16_t y = 0; y < HEIGHT; y++)
    {
        for (int16_t x = 0; x < WIDTH; x++)
        {
            if (f[y][x] < lowestValue && open[y][x] == 1)
            {
                lowest[0] = y;
                lowest[1] = x;
                lowestValue = f[lowest[0]][lowest[1]];
            }
        }
    }
    return lowest;
}

void reconstruct(int16_t *current, int16_t parent[HEIGHT][WIDTH][2], int16_t array[HEIGHT][WIDTH])
{
    int16_t temp_y = 0, temp_x = 0;
    while (*current != -1)
    {
        array[current[0]][current[1]] = 2;

        temp_x = parent[current[0]][current[1]][1];
        temp_y = parent[current[0]][current[1]][0];

        current[0] = temp_y;
        current[1] = temp_x;
    }
    printf("\n");
    printArray(array);
    // printArrayNum(array);
    printf("\n");
}

int16_t steps[4][2] = {
    {0, 1},  // Direita
    {1, 0},  // Baixo
    {0, -1}, // Esquerda
    {-1, 0}, // Cima
};

int main(int argc, char **argv)
{
    int16_t array[HEIGHT][WIDTH] = {{0}};
    int16_t parent[HEIGHT][WIDTH][2];
    memset(parent, -1, sizeof(parent));

    float f[HEIGHT][WIDTH]; // = {0.0f};
    float g[HEIGHT][WIDTH]; // = {0.0f};
    float h[HEIGHT][WIDTH]; // = {0.0f};
    memset(f, 0.0f, sizeof(f));
    memset(g, 0.0f, sizeof(f));
    memset(h, 0.0f, sizeof(f));

    int16_t *current;
    int16_t done = 0;
    int16_t numOpen = 0;
    int16_t open[HEIGHT][WIDTH] = {0};
    int16_t closed[HEIGHT][WIDTH] = {0};

    // Defini posição de inicio e final
    // Levar em consideração o wall
    int16_t start[2] = {0, 0};
    int16_t end[2] = {HEIGHT - 1, WIDTH - 1};

    array[start[0]][start[1]] = 2;
    array[end[0]][end[1]] = 2;

    insertObstacles((int16_t *)array, HEIGHT, WIDTH, DIFFICULTY);
    calculateHeuristics(h, g, f, end, HEIGHT, WIDTH);

    open[start[0]][start[1]] = 1;
    numOpen++;

    while (numOpen > 0 && done == 0)
    {
        current = getNewCurrent(f, array, open);

        if (current[0] == end[0] && current[1] == end[1])
        {
            done = 1;
            reconstruct(current, parent, array);
            return 0;
        }

        open[current[0]][current[1]] = 0;
        numOpen -= 1;
        closed[current[0]][current[1]] = 1;

        // Itera ao redor do current
        int16_t x = 0;
        int16_t y = 0;
        for (uint8_t step = 0; step < 4; step++)
        {
            x = current[1] + steps[step][1];
            y = current[0] + steps[step][0];

            if (x < 0 || y < 0 || x > WIDTH - 1 || y > HEIGHT - 1 || // Se for parede
                closed[y][x] == 1 ||                                 // Se a rota está fechada/já foi usada/calculada os vizinhos
                array[y][x] == 1)                                    // Se houver um obstaculo
            {
                continue;
            }

            if (open[y][x] != 1) // Se já foi calculado
            {
                open[y][x] = 1; // Defini como calculado
                numOpen += 1;
            }
            g[y][x] = g[current[0]][current[1]] + 1.0f; // Atribui g
            f[y][x] = g[y][x] + h[y][x];

            parent[y][x][0] = current[0];
            parent[y][x][1] = current[1];
        }
    }
    if (numOpen == 0 && done == 0)
    {
        done = 1;
        printf("\n");
        printArray(array);
        printf("\nNo solution!\n\n");
    }

    return 0;
}