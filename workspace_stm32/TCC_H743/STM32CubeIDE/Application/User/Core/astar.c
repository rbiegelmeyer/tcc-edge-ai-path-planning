/**
 * @file astar.c
 * @brief A* pathfinding library — implementation.
 *
 * Ported from the dataset-generator (main.c) for use as a standalone
 * library on the STM32H743.  All SDL2 visualisation, CSV output, obstacle
 * generation, and main() have been removed; only the search core remains.
 *
 * Internal working buffers are declared static so they are placed in the
 * BSS/data segment rather than on the stack, avoiding ~88 KB of stack
 * pressure (64×64 default) that would overflow the STM32 call stack.
 *
 * To place the buffers in DTCMRAM (fastest SRAM bank on STM32H743), add
 * the following attribute before each static array declaration:
 *   __attribute__((section(".dtcmram")))
 * and ensure your linker script defines a .dtcmram section mapped to
 * 0x20000000–0x2001FFFF.
 */

#include "astar.h"
#include <string.h>
#include <math.h>

/* ----------------------------------------------------------------
 * Internal accessor macros  ('map'/'result'/'width' must be in scope)
 * IMAP — read from the const input map (may be in Flash)
 * OMAP — write to the mutable result buffer (always in RAM)
 * ---------------------------------------------------------------- */
#define IMAP(r, c)  (map[(r) * width + (c)])
#define OMAP(r, c)  (result[(r) * width + (c)])

/* ----------------------------------------------------------------
 * Static working buffers — never allocated on the call stack
 * ---------------------------------------------------------------- */

/** Movement cost from start g(n). */
static float   _g[ASTAR_MAX_HEIGHT][ASTAR_MAX_WIDTH];

/** Euclidean heuristic to goal h(n). */
static float   _h[ASTAR_MAX_HEIGHT][ASTAR_MAX_WIDTH];

/** Total estimated cost f(n) = g(n) + h(n). */
static float   _f[ASTAR_MAX_HEIGHT][ASTAR_MAX_WIDTH];

/** Parent pointer for path reconstruction: _parent[r][c] = {parent_row, parent_col}. */
static int16_t _parent[ASTAR_MAX_HEIGHT][ASTAR_MAX_WIDTH][2];

/** Open set: 1 if the cell is queued for expansion. */
static int16_t _open[ASTAR_MAX_HEIGHT][ASTAR_MAX_WIDTH];

/** Closed set: 1 if the cell has already been expanded. */
static int16_t _closed[ASTAR_MAX_HEIGHT][ASTAR_MAX_WIDTH];

/* ----------------------------------------------------------------
 * 4-directional movement offsets: right, down, left, up
 * ---------------------------------------------------------------- */
static const int16_t _steps[4][2] = {
    { 0,  1},   /* right */
    { 1,  0},   /* down  */
    { 0, -1},   /* left  */
    {-1,  0},   /* up    */
};

/* ================================================================
 * Internal helpers (static — not visible outside this translation unit)
 * ================================================================ */

/**
 * @brief Pre-computes h(n) and f(n) for every cell.
 *
 * h(n) is the Euclidean distance from (r, c) to the goal.
 * At initialisation g(n) = 0 everywhere, so f(n) = h(n).
 */
static void _calc_heuristics(int16_t end_row, int16_t end_col,
                              int16_t height,  int16_t width)
{
    for (int16_t r = 0; r < height; r++) {
        for (int16_t c = 0; c < width; c++) {
            float dr = (float)(end_row - r);
            float dc = (float)(end_col - c);
            _h[r][c] = sqrtf(dr * dr + dc * dc);
            _f[r][c] = _h[r][c];  /* g = 0 at init */
        }
    }
}

/**
 * @brief Linear scan to find the open node with the lowest f value.
 *
 * @return Pointer to a static {row, col} pair.  Returns {-1, -1} when
 *         the open list is empty (should not happen during normal operation).
 */
static int16_t *_get_best_node(int16_t height, int16_t width)
{
    static int16_t best[2];
    float best_f = 1e9f;
    best[0] = -1;
    best[1] = -1;

    for (int16_t r = 0; r < height; r++) {
        for (int16_t c = 0; c < width; c++) {
            if (_open[r][c] == 1 && _f[r][c] < best_f) {
                best_f   = _f[r][c];
                best[0]  = r;
                best[1]  = c;
            }
        }
    }
    return best;
}

/**
 * @brief Walks the parent chain from @p current back to the start and
 *        marks every cell on the path with ASTAR_PATH (2).
 *
 * The loop terminates when current[0] == -1, which is the sentinel value
 * written by memset(-1) into _parent for the start cell (never explicitly
 * assigned a parent during the search).
 *
 * @param current  In/out: starting coordinates {row, col}; modified in-place.
 * @param result   Output RAM buffer; path cells are written here.
 * @param width    Map width (stride of @p result).
 */
static void _reconstruct(int16_t *current, int8_t *result, int16_t width)
{
    while (current[0] != -1) {
        result[current[0] * width + current[1]] = (int8_t)ASTAR_PATH;

        int16_t pr = _parent[current[0]][current[1]][0];
        int16_t pc = _parent[current[0]][current[1]][1];

        current[0] = pr;
        current[1] = pc;
    }
}

/* ================================================================
 * Public API
 * ================================================================ */

AStarStatus astar_solve(const int8_t *map, int8_t *result,
                        int16_t height, int16_t width)
{
    /* ---- Validate inputs ---- */
    if (map == NULL || result == NULL) return ASTAR_INVALID;
    if (height <= 0 || width <= 0)     return ASTAR_INVALID;
    if (height > ASTAR_MAX_HEIGHT || width > ASTAR_MAX_WIDTH) return ASTAR_INVALID;

    /* ---- Zero the result buffer before any early return ---- */
    memset(result, 0, (size_t)height * (size_t)width * sizeof(int8_t));

    /* ---- Locate ASTAR_START (3) and ASTAR_END (4) inside the input map ---- */
    int16_t start_row = -1, start_col = -1;
    int16_t end_row   = -1, end_col   = -1;

    for (int16_t r = 0; r < height; r++) {
        for (int16_t c = 0; c < width; c++) {
            int8_t v = IMAP(r, c);
            if (v == ASTAR_START) { start_row = r; start_col = c; }
            if (v == ASTAR_END)   { end_row   = r; end_col   = c; }
        }
    }

    if (start_row < 0 || end_row < 0) return ASTAR_INVALID;

    /* Trivial case: start == end */
    if (start_row == end_row && start_col == end_col) {
        OMAP(start_row, start_col) = (int8_t)ASTAR_START;
        return ASTAR_OK;
    }

    /* ---- Clear all working buffers ---- */
    memset(_g,      0,  sizeof(_g));
    memset(_h,      0,  sizeof(_h));
    memset(_f,      0,  sizeof(_f));
    memset(_parent, -1, sizeof(_parent));  /* 0xFF → int16_t -1 sentinel */
    memset(_open,   0,  sizeof(_open));
    memset(_closed, 0,  sizeof(_closed));

    /* ---- Pre-compute heuristics ---- */
    _calc_heuristics(end_row, end_col, height, width);

    /* ---- Seed the open list with the start node ---- */
    _open[start_row][start_col] = 1;
    int num_open = 1;

    int16_t *current;

    /* ================================================================
     * Main A* loop
     * ================================================================ */
    while (num_open > 0) {

        /* Select the open node with the lowest f(n). */
        current = _get_best_node(height, width);

        /* Goal reached: write path to result and restore markers. */
        if (current[0] == end_row && current[1] == end_col) {
            _reconstruct(current, result, width);
            OMAP(start_row, start_col) = (int8_t)ASTAR_START;
            OMAP(end_row,   end_col)   = (int8_t)ASTAR_END;
            return ASTAR_OK;
        }

        /* Move current node from open to closed. */
        _open  [current[0]][current[1]] = 0;
        _closed[current[0]][current[1]] = 1;
        num_open--;

        /* ---- Expand 4 neighbours ---- */
        for (int s = 0; s < 4; s++) {
            int16_t nr = (int16_t)(current[0] + _steps[s][0]);
            int16_t nc = (int16_t)(current[1] + _steps[s][1]);

            /* Bounds check. */
            if (nr < 0 || nr >= height || nc < 0 || nc >= width)
                continue;

            /* Skip already-visited and wall cells (read from const input map). */
            if (_closed[nr][nc] == 1)        continue;
            if (IMAP(nr, nc) == ASTAR_WALL)  continue;

            /* Add to open list if not already there. */
            if (_open[nr][nc] != 1) {
                _open[nr][nc] = 1;
                num_open++;
            }

            /* Update costs and parent (uniform step cost = 1). */
            _g[nr][nc]         = _g[current[0]][current[1]] + 1.0f;
            _f[nr][nc]         = _g[nr][nc] + _h[nr][nc];
            _parent[nr][nc][0] = current[0];
            _parent[nr][nc][1] = current[1];
        }
    }

    /* Open list exhausted — no solution; result stays fully zeroed. */
    return ASTAR_NO_PATH;
}
