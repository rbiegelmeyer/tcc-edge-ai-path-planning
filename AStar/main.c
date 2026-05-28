/**
 * @file main.c
 * @brief A* pathfinding dataset generator for machine learning path-planning.
 *
 * Procedurally generates labeled grid maps with rectangular obstacles and
 * solves them using the A* search algorithm. Solved maps are written to CSV
 * files suitable for training ML models on path-planning tasks.
 *
 * An optional real-time visualization mode (SDL2) can be enabled at compile
 * time with the @c -DVISUALIZE flag. All visualization code is excluded from
 * the production build, adding zero overhead.
 *
 * @section map_encoding Map cell encoding
 *   - 0 : empty
 *   - 1 : wall
 *   - 2 : path (solution)
 *   - 3 : start
 *   - 4 : end (goal)
 *
 * @section build Building
 *   - Production build:   @c make
 *   - Visualization build: @c make @c visualize  (requires libsdl2-dev)
 */

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

/** @brief Maximum supported map width in cells. */
#define MAX_WIDTH  128
/** @brief Maximum supported map height in cells. */
#define MAX_HEIGHT 128

// #define DEBUG_PRINT

// ============================================================
//  VISUALIZATION MODE  (compile with -DVISUALIZE)
//
//  Linux / WSL2: SDL2 window — install with: sudo apt install libsdl2-dev
//
//  Controls:
//    +  /  UP    : faster (halves delay)
//    -  /  DOWN  : slower (doubles delay)
//    SPACE       : pause / resume
//    H           : toggle h(n) heuristic values inside cells
//    W           : hold mode — freeze after each map until N/ENTER
//    N / ENTER   : next map (or release hold)
//    ESC / Q     : quit
// ============================================================
#ifdef VISUALIZE

#include <SDL2/SDL.h>

/** @brief Maximum step delay in milliseconds (slowest speed). */
#define VIS_DELAY_MAX     2000
/** @brief Default step delay in milliseconds. */
#define VIS_DELAY_DEFAULT   25
/** @brief Target window size (longest dimension) in pixels. */
#define VIS_WIN_SIZE       768
/** @brief Height of the info bar at the bottom of the window, in pixels. */
#define VIS_INFO_H          44

/* ---- Visualizer state ---------------------------------------- */
static int      vis_delay     = VIS_DELAY_DEFAULT; /**< Current step delay (ms). */
static int      vis_paused    = 0;                 /**< Non-zero when paused. */
static int      vis_quit      = 0;                 /**< Non-zero when the user requests exit. */
static int      vis_skip      = 0;                 /**< Non-zero when the user skips the current map. */
static int      vis_step      = 0;                 /**< A* step counter for the current map. */
static uint32_t vis_map_id    = 0;                 /**< ID of the map currently being displayed. */
static int      vis_show_h    = 1;                 /**< Non-zero to display h(n) values inside cells. */
static int      vis_hold_done = 0;                 /**< Non-zero to hold after each solved map. */

/* Forward declaration — arrays are defined after the VISUALIZE block. */
extern float h[MAX_HEIGHT][MAX_WIDTH];

/** @brief 24-bit RGB color for SDL2 rendering. */
typedef struct { uint8_t r, g, b; } VisColor;

/**
 * @brief Cell color palette (Catppuccin Mocha theme).
 *
 * Indexed by cell state:
 *   0=empty, 1=wall, 2=path, 3=start, 4=end, 5=open, 6=closed, 7=current
 */
static const VisColor VIS_COLORS[8] = {
    { 30,  30,  46},  /* 0 empty   */
    { 69,  71,  90},  /* 1 wall    */
    {137, 220, 235},  /* 2 path    */
    {137, 180, 250},  /* 3 start   */
    {250, 179, 135},  /* 4 end     */
    {166, 227, 161},  /* 5 open    */
    {243, 139, 168},  /* 6 closed  */
    {249, 226, 175},  /* 7 current */
};
static const VisColor C_GRID   = { 10,  10,  18}; /**< 1-px grid line color. */
static const VisColor C_INFOBG = { 24,  24,  37}; /**< Info bar background color. */
static const VisColor C_TEXT   = { 10,  10,  20}; /**< Heuristic digit color. */

/**
 * @brief 5×7 bitmap pixel font for digits 0–9.
 *
 * Each row is stored as one byte where bit 4 (MSB of the low nibble) is the
 * leftmost pixel and bit 0 is the rightmost. Only the lower 5 bits are used.
 */
static const uint8_t VIS_FONT[10][7] = {
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, /* 0 */
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, /* 1 */
    {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, /* 2 */
    {0x0F,0x01,0x01,0x0E,0x01,0x01,0x0F}, /* 3 */
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, /* 4 */
    {0x1F,0x10,0x10,0x1E,0x01,0x11,0x0E}, /* 5 */
    {0x0E,0x10,0x10,0x1E,0x11,0x11,0x0E}, /* 6 */
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, /* 7 */
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, /* 8 */
    {0x0E,0x11,0x11,0x0F,0x01,0x11,0x0E}, /* 9 */
};

/* ---- SDL2 handles -------------------------------------------- */
static SDL_Window   *vis_window   = NULL; /**< Main SDL2 window. */
static SDL_Renderer *vis_renderer = NULL; /**< Hardware-accelerated (or software fallback) renderer. */
static int           vis_mw = 0, vis_mh = 0; /**< Current map dimensions (columns, rows). */
static int           vis_cs = 8;             /**< Cell size in pixels (computed from map size). */
static int           vis_win_w = 0, vis_win_h = 0; /**< Window dimensions in pixels. */

/* ----------------------------------------------------------------
 * Internal SDL2 helpers
 * ---------------------------------------------------------------- */

/**
 * @brief Fills a rectangle on the SDL2 renderer with a solid color.
 *
 * @param x  Left edge in pixels.
 * @param y  Top edge in pixels.
 * @param rw Rectangle width in pixels.
 * @param rh Rectangle height in pixels.
 * @param c  Fill color.
 */
static void vis_fill_rect(int x, int y, int rw, int rh, VisColor c)
{
    SDL_SetRenderDrawColor(vis_renderer, c.r, c.g, c.b, 255);
    SDL_Rect r = {x, y, rw, rh};
    SDL_RenderFillRect(vis_renderer, &r);
}

/**
 * @brief Renders a 1 or 2-digit integer centered at (cx, cy) using the
 *        5×7 bitmap font, in the C_TEXT color.
 *
 * @param cx  Horizontal center of the target cell in pixels.
 * @param cy  Vertical center of the target cell in pixels.
 * @param val Value to render (clamped to [0, 99]).
 */
static void vis_draw_number(int cx, int cy, int val)
{
    SDL_SetRenderDrawColor(vis_renderer, C_TEXT.r, C_TEXT.g, C_TEXT.b, 255);

    if (val < 10) {
        /* Single digit — center the 5-wide glyph on cx. */
        int dx = cx - 2, dy = cy - 3;
        for (int row = 0; row < 7; row++) {
            uint8_t bits = VIS_FONT[val][row];
            for (int col = 0; col < 5; col++)
                if (bits & (0x10 >> col))
                    SDL_RenderDrawPoint(vis_renderer, dx + col, dy + row);
        }
    } else {
        /* Two digits — total glyph width = 5 + 1 (gap) + 5 = 11 px. */
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

/**
 * @brief Drains the SDL2 event queue and updates the visualizer state.
 *
 * Handles SDL_QUIT and SDL_KEYDOWN events, updating the global state
 * variables (vis_quit, vis_paused, vis_skip, vis_delay, etc.) accordingly.
 * Must be called regularly to keep the window responsive.
 */
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

/**
 * @brief Creates the SDL2 window and renderer for the given map dimensions.
 *
 * The cell size is computed so that the largest map dimension fits within
 * VIS_WIN_SIZE pixels. Hardware acceleration is preferred; falls back to
 * software rendering if unavailable. Does nothing if already initialized.
 *
 * @param mw Map width in cells.
 * @param mh Map height in cells.
 * @return 0 on success, -1 on SDL error.
 */
static int vis_init(int mw, int mh)
{
    if (vis_window) return 0;

    /* Compute cell size so the map fits within VIS_WIN_SIZE. */
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

    /* Try hardware-accelerated renderer; fall back to software. */
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

/**
 * @brief Destroys the SDL2 renderer and window, then calls SDL_Quit.
 *
 * Safe to call even if vis_init was never called (guards against NULL).
 */
static void vis_cleanup(void)
{
    if (vis_renderer) { SDL_DestroyRenderer(vis_renderer); vis_renderer = NULL; }
    if (vis_window)   { SDL_DestroyWindow(vis_window);     vis_window   = NULL; }
    SDL_Quit();
}

/**
 * @brief Prepares the visualizer for a new map.
 *
 * Initializes SDL2 on the first call. Resets the step counter and skip flag.
 * Resizes the window if the new map dimensions differ from the previous one.
 *
 * @param id Map identifier shown in the window title.
 * @param mw Map width in cells.
 * @param mh Map height in cells.
 */
static void vis_new_map(uint32_t id, int mw, int mh)
{
    if (!vis_window) vis_init(mw, mh);
    if (vis_quit) return;
    vis_map_id = id; vis_step = 0; vis_skip = 0;

    /* Resize window only when map dimensions change between maps. */
    if (mw != vis_mw || mh != vis_mh) {
        vis_mw = mw; vis_mh = mh;
        vis_cs = VIS_WIN_SIZE / (mw > mh ? mw : mh);
        if (vis_cs < 2) vis_cs = 2;
        vis_win_w = mw * vis_cs;
        vis_win_h = mh * vis_cs + VIS_INFO_H;
        SDL_SetWindowSize(vis_window, vis_win_w, vis_win_h);
    }
}

/**
 * @brief Renders one frame of the A* state and blocks for the configured delay.
 *
 * Draws every cell with its current state color, optionally overlaying the
 * h(n) heuristic value. Then renders the info bar (legend, speed bar, pause
 * indicator) and presents the frame. Afterwards:
 *   - Pumps events and blocks while paused.
 *   - On the final frame, delays 300 ms and enters hold mode if enabled.
 *   - On non-final frames, delays @c vis_delay ms unless skip is active.
 *
 * @param map         Grid with cell values (0–4).
 * @param open_list   Open set bitmap (1 = in open list).
 * @param closed_list Closed set bitmap (1 = visited).
 * @param cur         Current node coordinates {row, col}, or NULL on final frame.
 * @param end         Goal coordinates {row, col}.
 * @param height      Map height in cells.
 * @param width       Map width in cells.
 * @param final_frame Non-zero when the search is complete (path found or no solution).
 */
static void vis_draw(
    int16_t map[MAX_HEIGHT][MAX_WIDTH],
    int16_t open_list[MAX_HEIGHT][MAX_WIDTH],
    int16_t closed_list[MAX_HEIGHT][MAX_WIDTH],
    int16_t *cur, int16_t end[2],
    int16_t height, int16_t width, int final_frame)
{
    if (!vis_window || vis_quit) return;
    int cs = vis_cs;

    /* Clear the entire window with the grid background color.
     * The 1-px border between cells is created by drawing each cell
     * at (x*cs+1, y*cs+1) with size (cs-1, cs-1), leaving a 1-px gap. */
    SDL_SetRenderDrawColor(vis_renderer, C_GRID.r, C_GRID.g, C_GRID.b, 255);
    SDL_RenderClear(vis_renderer);

    /* ---- Draw map cells ---- */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            /* Determine the color index for this cell. */
            int ci;
            int16_t cell = map[y][x];
            if (!final_frame && cur && cur[0] == y && cur[1] == x) ci = 7; /* current */
            else if (cell == 3)              ci = 3; /* start   */
            else if (cell == 4)              ci = 4; /* end     */
            else if (cell == 2)              ci = 2; /* path    */
            else if (cell == 1)              ci = 1; /* wall    */
            else if (closed_list[y][x] == 1) ci = 6; /* closed  */
            else if (open_list[y][x]   == 1) ci = 5; /* open    */
            else                             ci = 0; /* empty   */

            vis_fill_rect(x*cs+1, y*cs+1, cs-1, cs-1, VIS_COLORS[ci]);

            /* Overlay h(n) value on open, closed, current, and path cells
             * when the cell is large enough to fit the 5x7 font. */
            if (vis_show_h && cs >= 12 && h[y][x] > 0 &&
                (ci == 5 || ci == 6 || ci == 7 || ci == 2)) {
                int val = (int)(h[y][x] + 0.5f);
                val = val < 0 ? 0 : val > 99 ? 99 : val;
                vis_draw_number(x*cs + cs/2, y*cs + cs/2, val);
            }
        }
    }

    /* Redraw the goal cell on top so it is never hidden by the path color. */
    vis_fill_rect(end[1]*cs+1, end[0]*cs+1, cs-1, cs-1, VIS_COLORS[4]);

    /* ---- Draw info bar ---- */
    int iy = height * cs;
    vis_fill_rect(0, iy, vis_win_w, VIS_INFO_H, C_INFOBG);

    /* Color legend: one 14×14 square per state (open, closed, current, path, start, end). */
    static const VisColor leg[] = {
        {166,227,161},{243,139,168},{249,226,175},{137,220,235},{137,180,250},{250,179,135}
    };
    for (int i = 0; i < 6; i++)
        vis_fill_rect(10 + i*20, iy + (VIS_INFO_H-14)/2, 14, 14, leg[i]);

    /* Speed bar: width proportional to (1 - delay/max_delay). */
    int bar_w = (int)((VIS_DELAY_MAX - vis_delay) / (float)VIS_DELAY_MAX * vis_win_w);
    if (bar_w > 0) vis_fill_rect(0, iy + VIS_INFO_H - 5, bar_w, 4, VIS_COLORS[5]);

    /* Pause indicator: two vertical bars at the right edge of the info bar. */
    if (vis_paused) {
        int px = vis_win_w - 36;
        vis_fill_rect(px,      iy + 8, 8, VIS_INFO_H - 18, VIS_COLORS[7]);
        vis_fill_rect(px + 14, iy + 8, 8, VIS_INFO_H - 18, VIS_COLORS[7]);
    }

    SDL_RenderPresent(vis_renderer);

    /* Update window title with map ID, step count, and current speed. */
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

    /* ---- Timing and flow control ---- */
    vis_pump();
    /* Spin here while paused, processing events every ~16 ms. */
    while (vis_paused && !vis_quit && !vis_skip) { vis_pump(); SDL_Delay(16); }

    if (final_frame) {
        SDL_Delay(300); /* Minimum display time so the solved map is visible. */
        /* Hold mode: wait for N/ENTER before proceeding to the next map. */
        while (vis_hold_done && !vis_skip && !vis_quit) {
            vis_pump();
            SDL_Delay(16);
        }
    } else if (vis_delay > 0 && !vis_skip) {
        SDL_Delay(vis_delay);
    }
    vis_pump();
}

/** @brief Returns non-zero if the user has requested the visualizer to quit. */
static int vis_should_quit(void) { return vis_quit; }

/** @brief Returns non-zero if the user has requested to skip the current map. */
static int vis_should_skip(void) { return vis_skip; }

#endif // VISUALIZE
// ============================================================

/** @brief Global file pointer for the current CSV output file. */
FILE *fptr;

/* ================================================================
 * Debug / inspection utilities
 * ================================================================ */

/**
 * @brief Prints the map to stdout as ASCII art.
 *
 * Cell rendering: wall='X', path='.', start='+', end='o', empty=' '.
 * Bordered by a row of 'X' characters on all four sides.
 *
 * @param array  Map grid to print.
 * @param height Map height in cells.
 * @param width  Map width in cells.
 */
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

/**
 * @brief Prints the raw integer values of a map grid to stdout.
 *
 * @param array  Map grid to print.
 * @param height Map height in cells.
 * @param width  Map width in cells.
 */
void printArrayNum(int16_t array[MAX_HEIGHT][MAX_WIDTH], int16_t height, int16_t width)
{
    for (int16_t y = 0; y < height; y++)
    {
        for (int16_t x = 0; x < width; x++)
            printf("%d ", array[y][x]);
        printf("\n");
    }
}

/**
 * @brief Prints a float cost matrix (f, g, or h) to stdout, one decimal place.
 *
 * @param array  Float matrix to print.
 * @param height Matrix height.
 * @param width  Matrix width.
 * @return Always 0.
 */
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

/**
 * @brief Prints a float cost matrix to stdout with full floating-point precision.
 *
 * @param array  Float matrix to print.
 * @param height Matrix height.
 * @param width  Matrix width.
 */
void printCostArray(float array[MAX_HEIGHT][MAX_WIDTH], int16_t height, int16_t width)
{
    for (int16_t y = 0; y < height; y++)
    {
        for (int16_t x = 0; x < width; x++)
            printf("%f ", array[y][x]);
        printf("\n");
    }
}

/* ================================================================
 * Map generation
 * ================================================================ */

/**
 * @brief Places rectangular wall blocks on a map grid.
 *
 * The number of blocks is proportional to the map area and the difficulty
 * parameter: @c num_blocks = (width * height * difficulty) / 500.
 * Each block is a square with half-size randomly chosen from [2, 4] cells.
 *
 * The random seed is always @c id+1, so the same ID produces the same
 * obstacle layout regardless of how many times this function is called.
 * Cells already marked as start (3) or end (4) are never overwritten.
 *
 * @param array      Flat map buffer (row-major, int16_t elements).
 * @param height     Map height in cells.
 * @param width      Map width in cells.
 * @param stride     Row stride of the backing array (should be MAX_WIDTH).
 * @param difficulty Obstacle density parameter (higher = more walls).
 * @param id         Map ID used as the random seed base.
 */
void insertObstacles(int16_t *array, int16_t height, int16_t width, int16_t stride, int16_t difficulty, uint32_t id)
{
    srand(id + 1);

    int16_t num_obstacles = (width * height * difficulty) / 500;

    for (int16_t i = 0; i < num_obstacles; i++)
    {
        /* Choose a random center and half-size for this obstacle block. */
        int16_t cy   = rand() % height;
        int16_t cx   = rand() % width;
        int16_t half = 2 + rand() % 3;

        /* Fill the square block, clipping at map boundaries. */
        for (int16_t dy = -half; dy <= half; dy++)
        {
            for (int16_t dx = -half; dx <= half; dx++)
            {
                int16_t ny = cy + dy;
                int16_t nx = cx + dx;

                if (ny < 0 || ny >= height || nx < 0 || nx >= width)
                    continue;

                int16_t *cell = array + ny * stride + nx;
                /* Preserve start (3) and end (4) markers. */
                if (*cell != 3 && *cell != 4)
                    *cell = 1;
            }
        }
    }
}

/* ================================================================
 * A* cost functions
 * ================================================================ */

/**
 * @brief Pre-computes h(n) and f(n) for every cell in the grid.
 *
 * h(n) is the Euclidean distance from cell (x, y) to the goal. g(n) is
 * assumed to be zero at this point (initial state), so f(n) = h(n).
 * During the search, g and f are updated incrementally by find_path().
 *
 * @param h      Output: heuristic cost matrix.
 * @param g      Input/output: movement cost matrix (read, not modified here).
 * @param f      Output: total estimated cost matrix (f = g + h).
 * @param end    Goal coordinates {row, col}.
 * @param height Map height in cells.
 * @param width  Map width in cells.
 */
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

/**
 * @brief Selects the open node with the lowest f value (the next node to expand).
 *
 * Performs a linear scan over the entire grid — O(width × height) per call.
 * Returns {-1, -1} if the open list is empty.
 *
 * @param f      Total estimated cost matrix.
 * @param array  Map grid (unused directly, kept for signature consistency).
 * @param open   Open set bitmap (1 = in open list).
 * @param height Map height in cells.
 * @param width  Map width in cells.
 * @return Pointer to a static 2-element array {row, col} of the best node.
 */
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

/**
 * @brief Traces the parent chain from the goal back to the start and marks the path.
 *
 * Walks the parent pointer array from the current node until a cell whose
 * parent is {-1, -1} (sentinel set by memset in find_path). Each visited
 * cell is marked with value 2 (path) in the map grid.
 *
 * @param current Pointer to the current node coordinates {row, col},
 *                modified in-place as the trace walks backwards.
 * @param parent  Parent pointer array indexed by [row][col][0=row/1=col].
 * @param array   Map grid; path cells are written with value 2.
 * @return Always 0.
 */
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

/**
 * @brief Appends one solved map row to the CSV output file.
 *
 * The row format is:
 * @code
 * id,difficulty,start_x,start_y,end_x,end_y,height,width,<map_string>
 * @endcode
 * where @c map_string is the flattened grid in row-major order, one digit
 * per cell, with no separator.
 *
 * @param file      Open FILE pointer for the CSV output.
 * @param id        Map identifier.
 * @param difficulty Obstacle density used for this map.
 * @param start_x   Start column (x = horizontal).
 * @param start_y   Start row    (y = vertical).
 * @param end_x     Goal column.
 * @param end_y     Goal row.
 * @param height    Map height in cells.
 * @param width     Map width in cells.
 * @param array     Solved map grid.
 * @return Always 0.
 */
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

/* ================================================================
 * Global state used by the A* search
 * ================================================================ */

/** @brief The four cardinal directions: right, down, left, up. */
int16_t steps[4][2] = {
    {0, 1},  /* Right */
    {1, 0},  /* Down  */
    {0, -1}, /* Left  */
    {-1, 0}, /* Up    */
};

int16_t map[MAX_HEIGHT][MAX_WIDTH];           /**< Current map grid (cell values 0–4). */
int16_t parent[MAX_HEIGHT][MAX_WIDTH][2];     /**< Parent pointer for each cell: {row, col}. */

float f[MAX_HEIGHT][MAX_WIDTH];               /**< Total estimated cost f(n) = g(n) + h(n). */
float g[MAX_HEIGHT][MAX_WIDTH];               /**< Movement cost from start g(n). */
float h[MAX_HEIGHT][MAX_WIDTH];               /**< Heuristic cost to goal h(n). */

int16_t open_list[MAX_HEIGHT][MAX_WIDTH];     /**< Open set: 1 if cell is in the open list. */
int16_t closed_list[MAX_HEIGHT][MAX_WIDTH];   /**< Closed set: 1 if cell has been expanded. */

int16_t *current;   /**< Pointer to the node being expanded in the current A* step. */
int16_t done    = 0; /**< Non-zero when the search has finished (success or failure). */
int16_t numOpen = 0; /**< Number of nodes currently in the open list. */

/* ================================================================
 * A* search
 * ================================================================ */

/**
 * @brief Runs the A* algorithm on a freshly generated map.
 *
 * Steps:
 *   1. Clears all global state arrays (map, parent, f, g, h, open/closed lists).
 *   2. Marks the start (3) and end (4) cells on the map.
 *   3. Calls insertObstacles() to populate walls (start/end are preserved).
 *   4. Pre-computes h(n) and f(n) with calculateHeuristics().
 *   5. Adds the start cell to the open list.
 *   6. Iterates: expands the best open node, stops when the goal is reached
 *      or no nodes remain.
 *   7. On success: reconstructs the path and writes the result to the CSV.
 *   8. On failure: returns 1 without writing to the CSV.
 *
 * @param id         Map ID, also used as the random seed for obstacle placement.
 * @param start      Start coordinates {row, col}.
 * @param end        Goal coordinates {row, col}.
 * @param height     Map height in cells.
 * @param width      Map width in cells.
 * @param difficulty Obstacle density parameter passed to insertObstacles().
 * @return 0 if a path was found and saved, 1 if no solution exists.
 */
int16_t find_path(uint32_t id, int16_t start[2], int16_t end[2],
                  int16_t height, int16_t width, int16_t difficulty)
{
    int16_t ret = 0;
    done    = 0;
    numOpen = 0;

    /* ---- Initialize all state arrays ---- */
    memset(map,         0,  sizeof(map));
    memset(parent,     -1,  sizeof(parent)); /* -1 sentinel = "no parent" */
    memset(f,           0,  sizeof(f));
    memset(g,           0,  sizeof(g));
    memset(h,           0,  sizeof(h));
    memset(open_list,   0,  sizeof(open_list));
    memset(closed_list, 0,  sizeof(closed_list));

    /* Mark start/end before inserting obstacles so they are never overwritten. */
    map[start[0]][start[1]] = 3;
    map[end[0]][end[1]]     = 4;

    insertObstacles((int16_t *)map, height, width, MAX_WIDTH, difficulty, id);
    calculateHeuristics(h, g, f, end, height, width);

    /* ---- Bootstrap: add the start node to the open list ---- */
    open_list[start[0]][start[1]] = 1;
    numOpen++;

    /* ---- Main A* loop ---- */
    while (numOpen > 0 && done == 0)
    {
        /* Select the open node with the lowest f value. */
        current = getNewCurrent(f, map, open_list, height, width);

        /* Goal check: if current node is the goal, reconstruct the path. */
        if (current[0] == end[0] && current[1] == end[1])
        {
            done = 1;
            reconstruct(current, parent, map);

            /* Re-mark start and end after reconstruct() may have overwritten them. */
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

        /* Move current from open to closed. */
        open_list[current[0]][current[1]] = 0;
        numOpen -= 1;
        closed_list[current[0]][current[1]] = 1;

        /* ---- Expand neighbors (4-directional) ---- */
        int16_t x = 0, y = 0;
        for (uint8_t step = 0; step < 4; step++)
        {
            x = current[1] + steps[step][1];
            y = current[0] + steps[step][0];

            /* Skip out-of-bounds, already-visited, and wall cells. */
            if (x < 0 || y < 0 || x > width - 1 || y > height - 1 ||
                closed_list[y][x] == 1 ||
                map[y][x] == 1)
            {
                continue;
            }

            /* Add neighbor to the open list if not already present. */
            if (open_list[y][x] != 1)
            {
                open_list[y][x] = 1;
                numOpen += 1;
            }

            /* Update costs: uniform step cost = 1. */
            g[y][x] = g[current[0]][current[1]] + 1.0f;
            f[y][x] = g[y][x] + h[y][x];

            /* Record how we arrived at this neighbor. */
            parent[y][x][0] = current[0];
            parent[y][x][1] = current[1];
        }

#ifdef VISUALIZE
        vis_draw(map, open_list, closed_list, current, end, height, width, 0);
        if (vis_should_quit() || vis_should_skip()) { done = 1; break; }
#endif
    }

    /* ---- No solution: open list exhausted without reaching the goal ---- */
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

/* ================================================================
 * Dataset generation
 * ================================================================ */

/**
 * @brief Ensures a map coordinate lands on a free (non-wall) cell.
 *
 * If the cell at (*out_y, *out_x) is a wall or equals the cell to avoid,
 * performs a row-major scan starting from the original position and moves
 * the coordinate to the first free cell that is not (avoid_y, avoid_x).
 *
 * Used to guarantee that both start and end are placed on traversable cells
 * before calling find_path(), eliminating one of the main sources of maps
 * that would otherwise have no solution.
 *
 * @param obs     Pre-generated obstacle map (0 = free, 1 = wall).
 * @param height  Map height in cells.
 * @param width   Map width in cells.
 * @param avoid_y Row of the cell to avoid (e.g. the start cell when placing end).
 *                Pass -1 to disable avoidance.
 * @param avoid_x Column of the cell to avoid.
 * @param out_y   In/out: row of the coordinate to validate and possibly move.
 * @param out_x   In/out: column of the coordinate to validate and possibly move.
 */
static void snap_to_free(int16_t obs[MAX_HEIGHT][MAX_WIDTH],
                          int16_t height, int16_t width,
                          int16_t avoid_y, int16_t avoid_x,
                          int16_t *out_y, int16_t *out_x)
{
    if (obs[*out_y][*out_x] == 0 &&
        !(*out_y == avoid_y && *out_x == avoid_x))
        return; /* Already a valid free cell — nothing to do. */

    /* Scan in row-major order, wrapping around from the original position. */
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

/**
 * @brief Generates a batch of solved A* maps and writes them to a CSV file.
 *
 * For each map ID in [start_id, end_id):
 *   1. Pre-generates the obstacle layout on a temporary grid to identify
 *      free cells before committing to start/end positions.
 *   2. Chooses start/end with srand(id), then calls snap_to_free() to
 *      relocate any position that fell inside a wall.
 *   3. Runs find_path(); only maps with a solution are written to the CSV.
 *
 * Output filename format: @c W<W>xH<H>_D<D>_S<start>_E<end>.csv
 *
 * @param width     Map width in cells.
 * @param height    Map height in cells.
 * @param difficulty Obstacle density (higher = more walls).
 * @param start_id  First map ID to generate (inclusive).
 * @param end_id    Last map ID to generate (exclusive).
 */
void generate_dataset(int16_t width, int16_t height, int16_t difficulty,
                      uint32_t start_id, uint32_t end_id)
{
    char filename[80];
    snprintf(filename, sizeof(filename), "W%03dxH%03d_D%02d_S%06d_E%06d.csv",
             width, height, difficulty, start_id, end_id);

    fptr = fopen(filename, "w+");
    fprintf(fptr, "id,difficulty,start_x,start_y,end_x,end_y,height,width,map\n");

    /* Static buffer reused across iterations to avoid 32 KB of stack per call. */
    static int16_t obs[MAX_HEIGHT][MAX_WIDTH];

    for (uint32_t id = start_id; id < end_id; id++)
    {
#ifdef VISUALIZE
        if (vis_should_quit()) break;
        vis_new_map(id, width, height);
#endif
        /* Pre-generate obstacles so start/end can be placed only on free cells.
         * insertObstacles() uses srand(id+1) internally; the same call is
         * repeated inside find_path() and produces the identical layout. */
        memset(obs, 0, sizeof(obs));
        insertObstacles((int16_t *)obs, height, width, MAX_WIDTH, difficulty, id);

        /* Choose start/end with srand(id), independent of the obstacle seed. */
        srand(id);
        int16_t start[2] = {rand() % height, rand() % width};
        int16_t end[2]   = {rand() % height, rand() % width};

        /* Relocate start/end if they landed inside a wall or at the same cell. */
        snap_to_free(obs, height, width, -1,       -1,       &start[0], &start[1]);
        snap_to_free(obs, height, width, start[0], start[1], &end[0],   &end[1]);

        uint8_t ret = find_path(id, start, end, height, width, difficulty);

        printf("%5d - (%3d,%3d) -> (%3d,%3d)%s\n",
               id, start[1], start[0], end[1], end[0], ret ? " - Sem Solucao" : "");
    }

    fclose(fptr);
}

/* ================================================================
 * Entry point
 * ================================================================ */

/**
 * @brief Program entry point.
 *
 * Iterates over the desired difficulty range and calls generate_dataset()
 * for each value. Cleans up the SDL2 visualizer on exit if -DVISUALIZE
 * was active.
 *
 * @param argc Argument count (unused).
 * @param argv Argument vector (unused).
 * @return 0 on success.
 */
int main(int argc, char **argv)
{
#ifdef VISUALIZE
    printf("Modo de visualizacao ativo — use um terminal largo (>= 128 colunas).\n");
    printf("  +/UP : rapido  |  -/DOWN : lento  |  SPACE : pausar  |  N : proximo  |  Q : sair\n\n");
    fflush(stdout);
#endif

    /* Generate datasets for difficulty levels 3 to 3 (single value here).
     * Adjust the loop bounds to produce multiple difficulty levels at once. */
    for (int i = 3; i < 4; i++)
    {
#ifdef VISUALIZE
        if (vis_should_quit()) break;
#endif
        // generate_dataset(64, 64, i, 0, 1000);
        generate_dataset(64, 64, i, 0, 30000);
    }

#ifdef VISUALIZE
    vis_cleanup();
#endif

    return 0;
}
