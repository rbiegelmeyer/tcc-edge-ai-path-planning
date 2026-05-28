# A* Pathfinding Dataset Generator

A C program that generates labeled CSV datasets of solved grid maps using the A\* pathfinding algorithm. The datasets are intended for training machine learning models on path-planning tasks.

---

## Overview

The program procedurally creates random grid maps with rectangular obstacles, runs A\* to find the shortest path between a random start and end cell, and records the solved map to a CSV file. A real-time SDL2 visualization mode is available as an optional compile-time feature.

---

## Map Encoding

Each cell in the grid is represented by an integer value:

| Value | Meaning      |
|-------|--------------|
| `0`   | Empty cell   |
| `1`   | Wall         |
| `2`   | Path         |
| `3`   | Start        |
| `4`   | End (goal)   |

---

## Algorithm

### A* Search

The implementation is a standard grid-based A\* with 4-directional movement (up, down, left, right).

- **g(n)** — cost from start to node `n` (step count, uniform cost = 1 per move)
- **h(n)** — Euclidean distance heuristic from `n` to the goal
- **f(n) = g(n) + h(n)** — total estimated cost

At each iteration the open node with the lowest `f` value is expanded. The algorithm terminates when the goal is reached (success) or the open list is empty (no solution).

### Obstacle Generation

Obstacles are generated deterministically using the map ID as a random seed (`srand(id + 1)`), so the same ID always produces the same map layout. Each obstacle is a filled square block:

- Number of blocks: `(width × height × difficulty) / 500`
- Block half-size: random value in `[2, 4]`
- Cells already marked as start (3) or end (4) are never overwritten

The `difficulty` parameter controls obstacle density. Higher values produce more cluttered maps with longer or more complex paths.

### Start / End Placement

Start and end positions are chosen with `srand(id)` (different seed from obstacles). Before calling the pathfinder, the program pre-generates the obstacle layout on a temporary map and uses `snap_to_free` to relocate any position that lands inside a wall, ensuring both cells are always reachable. This eliminates the main source of unsolvable maps and maximizes dataset yield.

---

## Dataset Output

Each run of `generate_dataset` writes one CSV file named:

```
W<width>xH<height>_D<difficulty>_S<start_id>_E<end_id>.csv
```

**CSV header:**
```
id,difficulty,start_x,start_y,end_x,end_y,height,width,map
```

**Per-row columns:**

| Column       | Type     | Description                                      |
|--------------|----------|--------------------------------------------------|
| `id`         | uint     | Map ID (also the random seed for start/end)      |
| `difficulty` | uint     | Obstacle density parameter                       |
| `start_x`    | int      | Start column (x = horizontal axis)               |
| `start_y`    | int      | Start row    (y = vertical axis)                 |
| `end_x`      | int      | Goal column                                      |
| `end_y`      | int      | Goal row                                         |
| `height`     | int      | Map height in cells                              |
| `width`      | int      | Map width in cells                               |
| `map`        | string   | Flattened map, row-major, one digit per cell     |

The `map` field is a string of `height × width` digits (0–4) with no separator, e.g. `00013...2...40000`.

Only maps where a solution was found are written. Maps with no solution are printed to stdout with the `- Sem Solucao` suffix and discarded.

---

## Building

### Production build (no visualization)

```sh
make
```

Produces `astar`. No external dependencies beyond the C standard library and `libm`.

### Visualization build (SDL2, Linux / WSL2)

Install SDL2 first:

```sh
sudo apt install libsdl2-dev
```

Then build:

```sh
make visualize
```

Produces `astar_vis`. Requires a display server (WSLg on Windows 11, or an X server on older setups).

---

## Running

```sh
./astar          # production — writes CSV files
./astar_vis      # visualization mode
```

The `main` function currently generates maps with:
- Grid size: 64 × 64
- Difficulty: 3
- IDs: 0 – 4999

Edit `main.c` to change these parameters.

---

## Visualization Controls

Available only in the `astar_vis` build.

| Key              | Action                                          |
|------------------|-------------------------------------------------|
| `+` / `UP`       | Faster (halves delay)                           |
| `-` / `DOWN`     | Slower (doubles delay)                          |
| `SPACE`          | Pause / resume                                  |
| `H`              | Toggle h(n) heuristic values inside cells       |
| `W`              | Toggle hold mode (freeze after each solved map) |
| `N` / `ENTER`    | Next map (or release hold)                      |
| `ESC` / `Q`      | Quit                                            |

The info bar at the bottom of the window shows a color legend, a speed bar, and a pause indicator.

**Cell color palette (Catppuccin Mocha):**

| Color         | Meaning              |
|---------------|----------------------|
| Dark grey     | Empty                |
| Mid grey      | Wall                 |
| Green         | Open list            |
| Pink          | Closed list          |
| Yellow        | Current node         |
| Cyan          | Path (final)         |
| Blue          | Start                |
| Orange        | End (goal)           |

---

## Configuration

All compile-time limits and defaults are defined as macros at the top of `main.c`:

| Macro               | Default | Description                            |
|---------------------|---------|----------------------------------------|
| `MAX_WIDTH`         | 128     | Maximum supported map width            |
| `MAX_HEIGHT`        | 128     | Maximum supported map height           |
| `VIS_WIN_SIZE`      | 768     | Visualization window size in pixels    |
| `VIS_INFO_H`        | 44      | Height of the info bar in pixels       |
| `VIS_DELAY_DEFAULT` | 25      | Initial step delay in milliseconds     |
| `VIS_DELAY_MAX`     | 2000    | Maximum step delay in milliseconds     |

---

## Code Structure

| Symbol                | Kind     | Description                                                       |
|-----------------------|----------|-------------------------------------------------------------------|
| `insertObstacles`     | function | Places rectangular wall blocks, seeded by `id + 1`               |
| `calculateHeuristics` | function | Pre-computes h(n) and f(n) for every cell (Euclidean distance)    |
| `getNewCurrent`       | function | Selects the open node with the lowest f value                     |
| `reconstruct`         | function | Traces the parent chain and marks the path cells with value 2     |
| `save_result`         | function | Writes one solved map row to the output CSV                       |
| `snap_to_free`        | function | Relocates a coordinate to the nearest free cell in the obstacle map |
| `find_path`           | function | Runs A\* for one map; returns 0 on success, 1 on no solution      |
| `generate_dataset`    | function | Outer loop: generates N maps and writes results to a CSV file     |
| `map`                 | global   | Current map grid (cell values 0–4)                                |
| `open_list`           | global   | Open set (1 = in open list, 0 = not)                              |
| `closed_list`         | global   | Closed set (1 = visited, 0 = not)                                 |
| `f`, `g`, `h`         | globals  | Cost arrays indexed by `[row][col]`                               |
| `parent`              | global   | Parent pointer array for path reconstruction                      |

The `#ifdef VISUALIZE` block (SDL2 rendering code) is completely excluded from the production build — it adds zero overhead when not compiled in.
