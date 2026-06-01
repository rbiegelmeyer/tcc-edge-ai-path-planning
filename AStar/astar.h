/**
 * @file astar.h
 * @brief A* pathfinding library for embedded targets (STM32H743).
 *
 * Accepts a pre-populated grid maze and solves it with the A* algorithm.
 * The search uses Euclidean distance as the heuristic h(n) and uniform
 * step cost 1 for g(n), with 4-directional movement.
 *
 * @section encoding Map cell encoding
 *   - ASTAR_EMPTY (0) : free traversable cell
 *   - ASTAR_WALL  (1) : impassable obstacle
 *   - ASTAR_PATH  (2) : solution path (written on success)
 *   - ASTAR_START (3) : start position (written by the solver)
 *   - ASTAR_END   (4) : goal position  (written by the solver)
 *
 * @section memory RAM usage (static internal buffers)
 *   - 64×64  default : ~88 KB  (fits in STM32H743 DTCMRAM)
 *   - 128×128        : ~352 KB (fits in AXI SRAM)
 *   Override ASTAR_MAX_WIDTH / ASTAR_MAX_HEIGHT before including this
 *   header to change the maximum supported map size.
 *
 * @note Not re-entrant. Do not call from multiple threads or ISRs
 *       concurrently.
 */

#ifndef ASTAR_H
#define ASTAR_H

#include <stdint.h>

/* ----------------------------------------------------------------
 * Configuration — override in your compiler flags or before #include
 * ---------------------------------------------------------------- */

/** Maximum supported map width in cells. */
#ifndef ASTAR_MAX_WIDTH
#define ASTAR_MAX_WIDTH  64
#endif

/** Maximum supported map height in cells. */
#ifndef ASTAR_MAX_HEIGHT
#define ASTAR_MAX_HEIGHT 64
#endif

/* ----------------------------------------------------------------
 * Cell value constants
 * ---------------------------------------------------------------- */

#define ASTAR_EMPTY  ((int16_t)0)   /**< Free traversable cell.          */
#define ASTAR_WALL   ((int16_t)1)   /**< Impassable obstacle.            */
#define ASTAR_PATH   ((int16_t)2)   /**< Solution path cell (output).    */
#define ASTAR_START  ((int16_t)3)   /**< Start position marker.          */
#define ASTAR_END    ((int16_t)4)   /**< Goal position marker.           */

/* ----------------------------------------------------------------
 * Return codes
 * ---------------------------------------------------------------- */

/** Return codes for astar_solve(). */
typedef enum {
    ASTAR_OK       = 0,  /**< Path found; path cells marked with ASTAR_PATH (2). */
    ASTAR_NO_PATH  = 1,  /**< No path exists between start and end.              */
    ASTAR_INVALID  = 2,  /**< Invalid arguments (NULL map, size out of range…).  */
} AStarStatus;

/* ----------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------- */

/**
 * @brief  Solve A* on a caller-supplied maze.
 *
 * The @p map buffer is a flat, row-major grid with logical stride = @p width.
 * Element access: @c map[row * width + col].
 *
 * The map must already contain exactly one ASTAR_START (3) cell and one
 * ASTAR_END (4) cell; the solver locates them by scanning the grid.
 * All other cells must be ASTAR_EMPTY (0) or ASTAR_WALL (1).
 *
 * On success (ASTAR_OK), every cell on the solution path is written as
 * ASTAR_PATH (2).  The start (3) and end (4) markers are preserved.
 *
 * On failure (ASTAR_NO_PATH), no path cells are marked.
 * On failure (ASTAR_INVALID), the map is left unmodified.
 *
 * @param  map     Caller-allocated int16_t buffer, at least
 *                 @p height × @p width elements, containing values 0–4.
 * @param  height  Map height in cells; must be in [1, ASTAR_MAX_HEIGHT].
 * @param  width   Map width  in cells; must be in [1, ASTAR_MAX_WIDTH].
 *
 * @return ASTAR_OK, ASTAR_NO_PATH, or ASTAR_INVALID.
 */
AStarStatus astar_solve(int16_t *map, int16_t height, int16_t width);

#endif /* ASTAR_H */
