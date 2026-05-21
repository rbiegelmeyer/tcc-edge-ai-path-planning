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

#define MAX_WIDTH  128
#define MAX_HEIGHT 128

// #define DEBUG_PRINT

FILE *fptr;

void printArray(int16_t array[MAX_HEIGHT][MAX_WIDTH], int16_t height, int16_t width)
{
    for (int16_t x = 0; x < (width + 2); x++)
        printf("X");
    printf("\n");
    for (int16_t y = 0; y < height; y++)
    {
        printf("X");
        for (int16_t x = 0; x < width; x++)
        {
            printf("%c", array[y][x] == 1   ? 'X'
                         : array[y][x] == 2 ? '.'
                         : array[y][x] == 3 ? '+'
                         : array[y][x] == 4 ? 'o'
                                            : ' ');
        }
        printf("X\n");
    }
    for (int16_t x = 0; x < (width + 2); x++)
        printf("X");
    printf("\n");
}

void printArrayNum(int16_t array[MAX_HEIGHT][MAX_WIDTH], int16_t height, int16_t width)
{
    for (int16_t y = 0; y < height; y++)
    {
        for (int16_t x = 0; x < width; x++)
            printf("%d ", array[y][x]);
        printf("\n");
    }
}

int16_t printPrettyMatrix(float array[MAX_HEIGHT][MAX_WIDTH], int16_t height, int16_t width)
{
    for (int16_t y = 0; y < height; y++)
    {
        for (int16_t x = 0; x < width; x++)
            printf("%4.1f ", array[y][x]);
        printf("\n");
    }
    return 0;
}

void printCostArray(float array[MAX_HEIGHT][MAX_WIDTH], int16_t height, int16_t width)
{
    for (int16_t y = 0; y < height; y++)
    {
        for (int16_t x = 0; x < width; x++)
            printf("%f ", array[y][x]);
        printf("\n");
    }
}

// stride = largura real da linha no array (MAX_WIDTH), não a largura do mapa
void insertObstacles(int16_t *array, int16_t height, int16_t width, int16_t stride, int16_t difficulty, uint32_t id)
{
    srand(id + 1);

    int16_t num_obstacles = (width * height * difficulty) / 500;

    for (int16_t i = 0; i < num_obstacles; i++)
    {
        int16_t cy   = rand() % height;
        int16_t cx   = rand() % width;
        int16_t half = 2 + rand() % 3;

        for (int16_t dy = -half; dy <= half; dy++)
        {
            for (int16_t dx = -half; dx <= half; dx++)
            {
                int16_t ny = cy + dy;
                int16_t nx = cx + dx;

                if (ny < 0 || ny >= height || nx < 0 || nx >= width)
                    continue;

                int16_t *cell = array + ny * stride + nx;
                if (*cell != 3 && *cell != 4)
                    *cell = 1;
            }
        }
    }
}

void calculateHeuristics(float h[MAX_HEIGHT][MAX_WIDTH], float g[MAX_HEIGHT][MAX_WIDTH], float f[MAX_HEIGHT][MAX_WIDTH], int16_t end[2], int16_t height, int16_t width)
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

int16_t *getNewCurrent(float f[MAX_HEIGHT][MAX_WIDTH], int16_t array[MAX_HEIGHT][MAX_WIDTH], int16_t open[MAX_HEIGHT][MAX_WIDTH], int16_t height, int16_t width)
{
    static int16_t lowest[2] = {-1, -1};
    float lowestValue = 99999.9f;
    for (int16_t y = 0; y < height; y++)
    {
        for (int16_t x = 0; x < width; x++)
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

int16_t reconstruct(int16_t *current, int16_t parent[MAX_HEIGHT][MAX_WIDTH][2], int16_t array[MAX_HEIGHT][MAX_WIDTH])
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

int16_t save_result(
    FILE *file, uint32_t id, uint8_t difficulty,
    int16_t start_x, int16_t start_y, int16_t end_x, int16_t end_y,
    int16_t height, int16_t width, int16_t array[MAX_HEIGHT][MAX_WIDTH])
{
    fprintf(file, "%03d,%02d,%03d,%03d,%03d,%03d,%03d,%03d,",
            id, difficulty, start_x, start_y, end_x, end_y, height, width);
    for (int16_t y = 0; y < height; y++)
    {
        for (int16_t x = 0; x < width; x++)
            fprintf(file, "%d", array[y][x]);
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

int16_t map[MAX_HEIGHT][MAX_WIDTH];
int16_t parent[MAX_HEIGHT][MAX_WIDTH][2];

float f[MAX_HEIGHT][MAX_WIDTH];
float g[MAX_HEIGHT][MAX_WIDTH];
float h[MAX_HEIGHT][MAX_WIDTH];

int16_t open_list[MAX_HEIGHT][MAX_WIDTH];
int16_t closed_list[MAX_HEIGHT][MAX_WIDTH];

int16_t *current;
int16_t done    = 0;
int16_t numOpen = 0;

int16_t find_path(uint32_t id, int16_t start[2], int16_t end[2],
                  int16_t height, int16_t width, int16_t difficulty)
{
    int16_t ret = 0;
    done    = 0;
    numOpen = 0;

    memset(map,         0,  sizeof(map));
    memset(parent,     -1,  sizeof(parent));
    memset(f,           0,  sizeof(f));
    memset(g,           0,  sizeof(g));
    memset(h,           0,  sizeof(h));
    memset(open_list,   0,  sizeof(open_list));
    memset(closed_list, 0,  sizeof(closed_list));

    map[start[0]][start[1]] = 3;
    map[end[0]][end[1]]     = 4;

    insertObstacles((int16_t *)map, height, width, MAX_WIDTH, difficulty, id);
    calculateHeuristics(h, g, f, end, height, width);

    open_list[start[0]][start[1]] = 1;
    numOpen++;

    while (numOpen > 0 && done == 0)
    {
        current = getNewCurrent(f, map, open_list, height, width);

        if (current[0] == end[0] && current[1] == end[1])
        {
            done = 1;
            reconstruct(current, parent, map);

            map[start[0]][start[1]] = 3;
            map[end[0]][end[1]]     = 4;
#ifdef DEBUG_PRINT
            printArray(map, height, width);
            printf("\n");
#endif

            save_result(fptr, id, difficulty, start[1], start[0], end[1], end[0], height, width, map);

            break;
        }

        open_list[current[0]][current[1]] = 0;
        numOpen -= 1;
        closed_list[current[0]][current[1]] = 1;

        int16_t x = 0, y = 0;
        for (uint8_t step = 0; step < 4; step++)
        {
            x = current[1] + steps[step][1];
            y = current[0] + steps[step][0];

            if (x < 0 || y < 0 || x > width - 1 || y > height - 1 ||
                closed_list[y][x] == 1 ||
                map[y][x] == 1)
            {
                continue;
            }

            if (open_list[y][x] != 1)
            {
                open_list[y][x] = 1;
                numOpen += 1;
            }
            g[y][x] = g[current[0]][current[1]] + 1.0f;
            f[y][x] = g[y][x] + h[y][x];

            parent[y][x][0] = current[0];
            parent[y][x][1] = current[1];
        }
    }
    if (numOpen == 0 && done == 0)
    {
        done = 1;
        ret  = 1;
    }

    return ret;
}

void generate_dataset(int16_t width, int16_t height, int16_t difficulty,
                      uint32_t start_id, uint32_t end_id)
{
    char filename[80];
    snprintf(filename, sizeof(filename), "W%03dxH%03d_D%02d_S%06d_E%06d.csv",
             width, height, difficulty, start_id, end_id);

    fptr = fopen(filename, "w+");
    fprintf(fptr, "id,difficulty,start_x,start_y,end_x,end_y,height,width,map\n");

    for (uint32_t id = start_id; id < end_id; id++)
    {
        srand(id);
        int16_t start[2] = {rand() % height, rand() % width};
        int16_t end[2]   = {rand() % height, rand() % width};

        uint8_t ret = find_path(id, start, end, height, width, difficulty);

        printf("%5d - (%3d,%3d) -> (%3d,%3d)%s\n",
               id, start[1], start[0], end[1], end[0], ret ? " - No Solution" : "");
    }

    fclose(fptr);
}

int main(int argc, char **argv)
{
    for (int i = 2; i <3; i++)
    {
        generate_dataset(64, 64, i, 0, 50000);
    }

    return 0;
}
