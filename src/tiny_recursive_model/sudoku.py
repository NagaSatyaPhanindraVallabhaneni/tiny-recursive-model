"""Synthetic Sudoku generation and a backtracking solver.

All puzzles are generated synthetically (no external data): a full valid grid is
built from a shuffled pattern, then holes are dug one at a time, keeping each
dig only if the puzzle still has exactly one solution (verified by counting
solutions, capped at 2, with an MRV backtracking solver).
"""

from __future__ import annotations

import random

FULL_MASK = 0x3FE  # bits 1..9 set


def _base_grid() -> list[int]:
    """One canonical valid Sudoku grid, as a flat list of 81 ints."""
    return [(3 * (r % 3) + r // 3 + c) % 9 + 1 for r in range(9) for c in range(9)]


def random_full_grid(rng: random.Random) -> list[int]:
    """A uniformly-ish shuffled valid grid: relabel digits, permute rows/cols/bands."""
    digits = list(range(1, 10))
    rng.shuffle(digits)
    relabel = {i + 1: digits[i] for i in range(9)}

    def shuffled_bands() -> list[int]:
        bands = [0, 1, 2]
        rng.shuffle(bands)
        rows: list[int] = []
        for b in bands:
            inner = [0, 1, 2]
            rng.shuffle(inner)
            rows.extend(3 * b + i for i in inner)
        return rows

    row_perm, col_perm = shuffled_bands(), shuffled_bands()
    base = _base_grid()
    grid = [0] * 81
    for r in range(9):
        for c in range(9):
            grid[r * 9 + c] = relabel[base[row_perm[r] * 9 + col_perm[c]]]
    if rng.random() < 0.5:  # transpose
        grid = [grid[c * 9 + r] for r in range(9) for c in range(9)]
    return grid


def _candidates_mask(grid: list[int], i: int) -> int:
    r, c = divmod(i, 9)
    used = 0
    for k in range(9):
        used |= 1 << grid[r * 9 + k]
        used |= 1 << grid[k * 9 + c]
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for dr in range(3):
        for dc in range(3):
            used |= 1 << grid[(br + dr) * 9 + bc + dc]
    return (~used) & FULL_MASK


def _pick_mrv(grid: list[int]) -> tuple[int, int] | None:
    """Most-constrained empty cell -> (index, candidates mask). None if full."""
    best_i, best_mask, best_n = -1, 0, 10
    for i in range(81):
        if grid[i] == 0:
            m = _candidates_mask(grid, i)
            n = bin(m).count("1")
            if n == 0:
                return -2, 0  # dead end marker
            if n < best_n:
                best_i, best_mask, best_n = i, m, n
                if n == 1:
                    break
    if best_i == -1:
        return None
    return best_i, best_mask


def count_solutions(grid: list[int], limit: int = 2) -> int:
    """Count solutions up to `limit` using MRV backtracking."""
    grid = list(grid)
    pick = _pick_mrv(grid)
    if pick is None:
        return 1
    if pick[0] == -2:
        return 0
    i, mask = pick
    total = 0
    m = mask
    while m:
        bit = m & -m
        grid[i] = bit.bit_length() - 1
        total += count_solutions(grid, limit - total)
        if total >= limit:
            return total
        grid[i] = 0
        m ^= bit
    return total


def solve_one(grid: list[int]) -> list[int] | None:
    """Return the first solution, or None if unsolvable."""
    grid = list(grid)
    pick = _pick_mrv(grid)
    if pick is None:
        return grid
    if pick[0] == -2:
        return None
    i, mask = pick
    m = mask
    while m:
        bit = m & -m
        grid[i] = bit.bit_length() - 1
        sol = solve_one(grid)
        if sol is not None:
            return sol
        grid[i] = 0
        m ^= bit
    return None


def is_valid_solution(grid: list[int]) -> bool:
    """Check a completed grid: every row/col/box contains 1..9 exactly once."""
    if len(grid) != 81 or any(not isinstance(v, int) or v < 1 or v > 9 for v in grid):
        return False
    target = set(range(1, 10))
    for k in range(9):
        if set(grid[k * 9 : (k + 1) * 9]) != target:
            return False
        if {grid[r * 9 + k] for r in range(9)} != target:
            return False
    for br in range(3):
        for bc in range(3):
            box = {
                grid[(3 * br + dr) * 9 + 3 * bc + dc] for dr in range(3) for dc in range(3)
            }
            if box != target:
                return False
    return True


def generate_puzzle(
    n_holes: int = 30, seed: int | None = None, max_digs: int = 81
) -> tuple[list[int], list[int], int]:
    """Generate a puzzle with a unique solution.

    Returns (puzzle, solution, holes_dug). Digs holes in random order, reverting
    any dig that breaks uniqueness. Fewer requested holes = easier puzzle.
    """
    rng = random.Random(seed)
    solution = random_full_grid(rng)
    puzzle = solution[:]
    holes = 0
    for i in rng.sample(range(81), min(max_digs, 81)):
        if holes >= n_holes:
            break
        puzzle[i] = 0
        if count_solutions(puzzle, limit=2) != 1:
            puzzle[i] = solution[i]  # revert: uniqueness broken
        else:
            holes += 1
    return puzzle, solution, holes
