#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>

#ifdef WIN32
#include <io.h>
#define F_OK 0
#define access _access
#else
#include <unistd.h>
#endif

#define WIDTH 32
#define HEIGHT 32
#define DIFFICULTY 1

FILE *fptr;
const char filename[] = "result.csv";

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
                         : array[y][x] == 3 ? '+'
                         : array[y][x] == 4 ? 'o'
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

void insertObstacles(int16_t *array, int16_t height, int16_t width, int16_t difficulty, uint32_t id)
{
    // Set random seed
    // srand(time(0));
    srand(id);

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

int16_t reconstruct(int16_t *current, int16_t parent[HEIGHT][WIDTH][2], int16_t array[HEIGHT][WIDTH])
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

    return 0;
}
/**
 *
 */
int16_t save_result(
    FILE *file, uint32_t id,
    int16_t start_x, int16_t start_y, int16_t end_x, int16_t end_y,
    int16_t height, int16_t width, int16_t array[HEIGHT][WIDTH])
{

    fprintf(file, "%03d,%03d,%03d,%03d,%03d,%03d,%03d,", id, start_x, start_y, end_x, end_y, height, width);
    for (int16_t y = 0; y < height; y++)
    {
        for (int16_t x = 0; x < width; x++)
        {
            fprintf(file, "%d", array[y][x]);
        }
    }
    fprintf(file, "\n");

    return 0;
}

int16_t steps[4][2] = {
    {0, 1},  // Direita
    {1, 0},  // Baixo
    {0, -1}, // Esquerda
    {-1, 0}, // Cima
};

int16_t array[HEIGHT][WIDTH];
int16_t parent[HEIGHT][WIDTH][2];

float f[HEIGHT][WIDTH]; // = {0.0f};
float g[HEIGHT][WIDTH]; // = {0.0f};
float h[HEIGHT][WIDTH]; // = {0.0f};

int16_t open[HEIGHT][WIDTH];
int16_t closed[HEIGHT][WIDTH];

int16_t *current;
int16_t done = 0;
int16_t numOpen = 0;

int16_t find_path(uint32_t id, int16_t start[2], int16_t end[2], int16_t array[HEIGHT][WIDTH], int16_t height, int16_t widht)
{
    int16_t ret = 0;
    done = 0;
    numOpen = 0;

    memset(array, 0, sizeof(array));
    memset(parent, -1, sizeof(parent));
    memset(f, 0.0f, sizeof(f));
    memset(g, 0.0f, sizeof(f));
    memset(h, 0.0f, sizeof(f));
    memset(open, 0, sizeof(open));
    memset(closed, 0, sizeof(closed));

    array[start[0]][start[1]] = 3;
    array[end[0]][end[1]] = 4;

    insertObstacles((int16_t *)array, HEIGHT, WIDTH, DIFFICULTY, id);
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

            array[start[0]][start[1]] = 3;
            array[end[0]][end[1]] = 4;
            // printArray(array);
            // printf("\n");

            save_result(fptr, id, start[1], start[0], end[1], end[0], HEIGHT, WIDTH, array);

            break;
        }

        open[current[0]][current[1]] = 0;
        numOpen -= 1;
        closed[current[0]][current[1]] = 1;

        // Itera ao redor do current
        int16_t x = 0, y = 0;
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
        // printf("\n");
        // printArray(array);
        ret = 1;
        // printf("No solution!\n\n");
    }

    return ret;
}

// time_t time;

int main(int argc, char **argv)
{
    // uint32_t start = time(0);
    // uint32_t end1 = 0;
    // uint32_t end2 = 0;
    // printf("start = %d\n", start);

    if (access(filename, F_OK) == 0)
    {
        fptr = fopen(filename, "w+");
        fprintf(fptr, "id,start_x,start_y,end_x,end_y,height,width,map\n");
    }
    else
    {
        printf("!!!!!!!!!!!!!!\n");
        fptr = fopen(filename, "a+");
    }

    // mingw_gettimeofday(&time, NULL);
    // int64_t s1 = (int64_t)(time.tv_sec) * 1000;
    // printf("start = %lli\n", s1);

    uint8_t ret = 0;
    uint32_t start_id = 0;
    uint32_t quant_id = 10000;
    uint32_t end_id = start_id + quant_id;

    for (uint32_t id = start_id; id < end_id; id++)
    {
        // Defini posição de inicio e final
        // Levar em consideração o wall
        srand(id);
        int16_t start[2] = {rand() % HEIGHT, rand() % WIDTH};
        int16_t end[2] = {rand() % HEIGHT, rand() % WIDTH};
        // printf("%5d\n", id);

        ret = find_path(id, start, end, array, HEIGHT, WIDTH);

        printf("%5d - (%2d,%2d) -> (%2d,%2d)%s\n", id, start[1], start[0], end[1], end[0], (ret) ? " - No Solution" : "");
    }

    // end1 = time(0);
    fclose(fptr);

    // end2 = time(0);
    // printf("end1 = %d\n", end1 - start);
    // printf("end2 = %d\n", end2 - end1);

    return 0;
}