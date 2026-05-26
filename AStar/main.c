#define _DEFAULT_SOURCE  /* POSIX extensions no C11 strict mode */
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

// ============================================================
//  MODO DE SUPERVISAO VISUAL  (compile com -DVISUALIZE)
//
//  Linux / WSL2: usa SDL2 — instale com: sudo apt install libsdl2-dev
//
//  Controles:
//    +  /  UP    : mais rapido (reduz delay)
//    -  /  DOWN  : mais lento  (aumenta delay)
//    SPACE       : pausar / continuar
//    H           : mostrar / ocultar valor da heuristica h(n)
//    W           : aguardar N/ENTER ao fim de cada mapa (modo inspecao)
//    N / ENTER   : proximo mapa (ou liberar se modo inspecao ativo)
//    ESC / Q     : encerrar visualizador
// ============================================================
#ifdef VISUALIZE

#include <SDL2/SDL.h>

#define VIS_DELAY_MAX     2000
#define VIS_DELAY_DEFAULT   25
#define VIS_WIN_SIZE       768
#define VIS_INFO_H          44

static int      vis_delay     = VIS_DELAY_DEFAULT;
static int      vis_paused    = 0;
static int      vis_quit      = 0;
static int      vis_skip      = 0;
static int      vis_step      = 0;
static uint32_t vis_map_id    = 0;
static int      vis_show_h    = 1;
static int      vis_hold_done = 0;

extern float h[MAX_HEIGHT][MAX_WIDTH];

typedef struct { uint8_t r, g, b; } VisColor;

// Catppuccin Mocha — 0=empty 1=wall 2=path 3=start 4=end 5=open 6=closed 7=current
static const VisColor VIS_COLORS[8] = {
    { 30,  30,  46},
    { 69,  71,  90},
    {137, 220, 235},
    {137, 180, 250},
    {250, 179, 135},
    {166, 227, 161},
    {243, 139, 168},
    {249, 226, 175},
};
static const VisColor C_GRID   = { 10,  10,  18};
static const VisColor C_INFOBG = { 24,  24,  37};
static const VisColor C_TEXT   = { 10,  10,  20};

// 5x7 pixel font — digits 0-9 (each byte = 5-bit row, bit4=leftmost pixel)
static const uint8_t VIS_FONT[10][7] = {
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, // 0
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, // 1
    {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, // 2
    {0x0F,0x01,0x01,0x0E,0x01,0x01,0x0F}, // 3
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, // 4
    {0x1F,0x10,0x10,0x1E,0x01,0x11,0x0E}, // 5
    {0x0E,0x10,0x10,0x1E,0x11,0x11,0x0E}, // 6
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, // 7
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, // 8
    {0x0E,0x11,0x11,0x0F,0x01,0x11,0x0E}, // 9
};

static SDL_Window   *vis_window   = NULL;
static SDL_Renderer *vis_renderer = NULL;
static int           vis_mw = 0, vis_mh = 0;
static int           vis_cs = 8;
static int           vis_win_w = 0, vis_win_h = 0;

static void vis_fill_rect(int x, int y, int rw, int rh, VisColor c)
{
    SDL_SetRenderDrawColor(vis_renderer, c.r, c.g, c.b, 255);
    SDL_Rect r = {x, y, rw, rh};
    SDL_RenderFillRect(vis_renderer, &r);
}

static void vis_draw_number(int cx, int cy, int val)
{
    SDL_SetRenderDrawColor(vis_renderer, C_TEXT.r, C_TEXT.g, C_TEXT.b, 255);
    if (val < 10) {
        int dx = cx - 2, dy = cy - 3;
        for (int row = 0; row < 7; row++) {
            uint8_t bits = VIS_FONT[val][row];
            for (int col = 0; col < 5; col++)
                if (bits & (0x10 >> col))
                    SDL_RenderDrawPoint(vis_renderer, dx + col, dy + row);
        }
    } else {
        int dx = cx - 5, dy = cy - 3;
        int tens = val / 10, ones = val % 10;
        for (int row = 0; row < 7; row++) {
            uint8_t t = VIS_FONT[tens][row], o = VIS_FONT[ones][row];
            for (int col = 0; col < 5; col++) {
                if (t & (0x10 >> col))
                    SDL_RenderDrawPoint(vis_renderer, dx + col,     dy + row);
                if (o & (0x10 >> col))
                    SDL_RenderDrawPoint(vis_renderer, dx + 6 + col, dy + row);
            }
        }
    }
}

static void vis_pump(void)
{
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) { vis_quit = 1; return; }
        if (e.type == SDL_KEYDOWN) {
            switch (e.key.keysym.sym) {
                case SDLK_EQUALS: case SDLK_PLUS: case SDLK_KP_PLUS: case SDLK_UP:
                    vis_delay = vis_delay <= 1 ? 0 : vis_delay / 2; break;
                case SDLK_MINUS: case SDLK_KP_MINUS: case SDLK_DOWN:
                    vis_delay = vis_delay == 0 ? 1 : vis_delay * 2;
                    if (vis_delay > VIS_DELAY_MAX) vis_delay = VIS_DELAY_MAX;
                    break;
                case SDLK_SPACE:  vis_paused = !vis_paused; break;
                case SDLK_h:      vis_show_h    = !vis_show_h;    break;
                case SDLK_w:      vis_hold_done = !vis_hold_done; break;
                case SDLK_n: case SDLK_RETURN: case SDLK_KP_ENTER:
                    vis_skip = 1; vis_paused = 0; break;
                case SDLK_ESCAPE: case SDLK_q: vis_quit = 1; break;
                default: break;
            }
        }
    }
}

static int vis_init(int mw, int mh)
{
    if (vis_window) return 0;
    vis_mw = mw; vis_mh = mh;
    vis_cs = VIS_WIN_SIZE / (mw > mh ? mw : mh);
    if (vis_cs < 2) vis_cs = 2;
    vis_win_w = mw * vis_cs;
    vis_win_h = mh * vis_cs + VIS_INFO_H;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return -1;
    }
    vis_window = SDL_CreateWindow("A* Visualizer",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        vis_win_w, vis_win_h, 0);
    if (!vis_window) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        SDL_Quit(); return -1;
    }
    vis_renderer = SDL_CreateRenderer(vis_window, -1, SDL_RENDERER_ACCELERATED);
    if (!vis_renderer)
        vis_renderer = SDL_CreateRenderer(vis_window, -1, SDL_RENDERER_SOFTWARE);
    if (!vis_renderer) {
        fprintf(stderr, "SDL_CreateRenderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(vis_window); vis_window = NULL; SDL_Quit(); return -1;
    }
    vis_pump();
    return 0;
}

static void vis_cleanup(void)
{
    if (vis_renderer) { SDL_DestroyRenderer(vis_renderer); vis_renderer = NULL; }
    if (vis_window)   { SDL_DestroyWindow(vis_window);     vis_window   = NULL; }
    SDL_Quit();
}

static void vis_new_map(uint32_t id, int mw, int mh)
{
    if (!vis_window) vis_init(mw, mh);
    if (vis_quit) return;
    vis_map_id = id; vis_step = 0; vis_skip = 0;

    if (mw != vis_mw || mh != vis_mh) {
        vis_mw = mw; vis_mh = mh;
        vis_cs = VIS_WIN_SIZE / (mw > mh ? mw : mh);
        if (vis_cs < 2) vis_cs = 2;
        vis_win_w = mw * vis_cs;
        vis_win_h = mh * vis_cs + VIS_INFO_H;
        SDL_SetWindowSize(vis_window, vis_win_w, vis_win_h);
    }
}

static void vis_draw(
    int16_t map[MAX_HEIGHT][MAX_WIDTH],
    int16_t open_list[MAX_HEIGHT][MAX_WIDTH],
    int16_t closed_list[MAX_HEIGHT][MAX_WIDTH],
    int16_t *cur, int16_t end[2],
    int16_t height, int16_t width, int final_frame)
{
    if (!vis_window || vis_quit) return;
    int cs = vis_cs;

    SDL_SetRenderDrawColor(vis_renderer, C_GRID.r, C_GRID.g, C_GRID.b, 255);
    SDL_RenderClear(vis_renderer);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int ci;
            int16_t cell = map[y][x];
            if (!final_frame && cur && cur[0] == y && cur[1] == x) ci = 7;
            else if (cell == 3)              ci = 3;
            else if (cell == 4)              ci = 4;
            else if (cell == 2)              ci = 2;
            else if (cell == 1)              ci = 1;
            else if (closed_list[y][x] == 1) ci = 6;
            else if (open_list[y][x]   == 1) ci = 5;
            else                             ci = 0;

            vis_fill_rect(x*cs+1, y*cs+1, cs-1, cs-1, VIS_COLORS[ci]);

            if (vis_show_h && cs >= 12 && h[y][x] > 0 &&
                (ci == 5 || ci == 6 || ci == 7 || ci == 2)) {
                int val = (int)(h[y][x] + 0.5f);
                val = val < 0 ? 0 : val > 99 ? 99 : val;
                vis_draw_number(x*cs + cs/2, y*cs + cs/2, val);
            }
        }
    }
    vis_fill_rect(end[1]*cs+1, end[0]*cs+1, cs-1, cs-1, VIS_COLORS[4]);

    int iy = height * cs;
    vis_fill_rect(0, iy, vis_win_w, VIS_INFO_H, C_INFOBG);

    static const VisColor leg[] = {
        {166,227,161},{243,139,168},{249,226,175},{137,220,235},{137,180,250},{250,179,135}
    };
    for (int i = 0; i < 6; i++)
        vis_fill_rect(10 + i*20, iy + (VIS_INFO_H-14)/2, 14, 14, leg[i]);

    int bar_w = (int)((VIS_DELAY_MAX - vis_delay) / (float)VIS_DELAY_MAX * vis_win_w);
    if (bar_w > 0) vis_fill_rect(0, iy + VIS_INFO_H - 5, bar_w, 4, VIS_COLORS[5]);

    if (vis_paused) {
        int px = vis_win_w - 36;
        vis_fill_rect(px,      iy + 8, 8, VIS_INFO_H - 18, VIS_COLORS[7]);
        vis_fill_rect(px + 14, iy + 8, 8, VIS_INFO_H - 18, VIS_COLORS[7]);
    }

    SDL_RenderPresent(vis_renderer);

    char title[256];
    if (final_frame && vis_hold_done)
        snprintf(title, sizeof(title),
            "A* Visualizer  |  Mapa #%u  |  >>> N/ENTER para continuar <<<",
            vis_map_id);
    else
        snprintf(title, sizeof(title),
            "A* Visualizer  |  Mapa #%u  |  Passo: %d  |  Vel: %dms%s%s",
            vis_map_id, vis_step, vis_delay,
            vis_paused    ? "  |  [PAUSADO]"   : "",
            vis_hold_done ? "  |  [W:aguarda]" : "");
    SDL_SetWindowTitle(vis_window, title);
    vis_step++;

    vis_pump();
    while (vis_paused && !vis_quit && !vis_skip) { vis_pump(); SDL_Delay(16); }

    if (final_frame) {
        SDL_Delay(300);
        while (vis_hold_done && !vis_skip && !vis_quit) {
            vis_pump();
            SDL_Delay(16);
        }
    } else if (vis_delay > 0 && !vis_skip) {
        SDL_Delay(vis_delay);
    }
    vis_pump();
}

static int vis_should_quit(void) { return vis_quit; }
static int vis_should_skip(void) { return vis_skip; }

#endif // VISUALIZE
// ============================================================

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

// stride = largura real da linha no array (MAX_WIDTH), nao a largura do mapa
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
#ifdef VISUALIZE
            vis_draw(map, open_list, closed_list, NULL, end, height, width, 1);
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

#ifdef VISUALIZE
        vis_draw(map, open_list, closed_list, current, end, height, width, 0);
        if (vis_should_quit() || vis_should_skip()) { done = 1; break; }
#endif
    }

    if (numOpen == 0 && done == 0)
    {
        done = 1;
        ret  = 1;
#ifdef VISUALIZE
        vis_draw(map, open_list, closed_list, NULL, end, height, width, 1);
#endif
    }

    return ret;
}

// Garante que (*out_y, *out_x) seja uma celula livre no mapa de obstaculos.
// Se a posicao atual for parede ou igual a (avoid_y, avoid_x), busca a celula
// livre mais proxima em varredura row-major a partir da posicao original.
static void snap_to_free(int16_t obs[MAX_HEIGHT][MAX_WIDTH],
                          int16_t height, int16_t width,
                          int16_t avoid_y, int16_t avoid_x,
                          int16_t *out_y, int16_t *out_x)
{
    if (obs[*out_y][*out_x] == 0 &&
        !(*out_y == avoid_y && *out_x == avoid_x))
        return;

    for (int16_t dy = 0; dy < height; dy++) {
        for (int16_t dx = 0; dx < width; dx++) {
            int16_t ny = (int16_t)((*out_y + dy) % height);
            int16_t nx = (int16_t)((*out_x + dx) % width);
            if (obs[ny][nx] == 0 && !(ny == avoid_y && nx == avoid_x)) {
                *out_y = ny;
                *out_x = nx;
                return;
            }
        }
    }
}

void generate_dataset(int16_t width, int16_t height, int16_t difficulty,
                      uint32_t start_id, uint32_t end_id)
{
    char filename[80];
    snprintf(filename, sizeof(filename), "W%03dxH%03d_D%02d_S%06d_E%06d.csv",
             width, height, difficulty, start_id, end_id);

    fptr = fopen(filename, "w+");
    fprintf(fptr, "id,difficulty,start_x,start_y,end_x,end_y,height,width,map\n");

    static int16_t obs[MAX_HEIGHT][MAX_WIDTH];

    for (uint32_t id = start_id; id < end_id; id++)
    {
#ifdef VISUALIZE
        if (vis_should_quit()) break;
        vis_new_map(id, width, height);
#endif
        // Pre-gera obstaculos para escolher start/end em celulas livres.
        // insertObstacles usa srand(id+1) internamente.
        memset(obs, 0, sizeof(obs));
        insertObstacles((int16_t *)obs, height, width, MAX_WIDTH, difficulty, id);

        srand(id);
        int16_t start[2] = {rand() % height, rand() % width};
        int16_t end[2]   = {rand() % height, rand() % width};

        snap_to_free(obs, height, width, -1,       -1,       &start[0], &start[1]);
        snap_to_free(obs, height, width, start[0], start[1], &end[0],   &end[1]);

        uint8_t ret = find_path(id, start, end, height, width, difficulty);

        printf("%5d - (%3d,%3d) -> (%3d,%3d)%s\n",
               id, start[1], start[0], end[1], end[0], ret ? " - Sem Solucao" : "");
    }

    fclose(fptr);
}

int main(int argc, char **argv)
{
#ifdef VISUALIZE
    printf("Modo de visualizacao ativo — use um terminal largo (>= 128 colunas).\n");
    printf("  +/UP : rapido  |  -/DOWN : lento  |  SPACE : pausar  |  N : proximo  |  Q : sair\n\n");
    fflush(stdout);
#endif

    for (int i = 3; i < 4; i++)
    {
#ifdef VISUALIZE
        if (vis_should_quit()) break;
#endif
        generate_dataset(64, 64, i, 0, 10000);
        generate_dataset(64, 64, i, 0, 20000);
    }

#ifdef VISUALIZE
    vis_cleanup();
#endif

    return 0;
}
