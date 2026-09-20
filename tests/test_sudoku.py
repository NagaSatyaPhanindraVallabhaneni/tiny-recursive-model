"""Tests for synthetic Sudoku generation and the solver."""

import random

from tiny_recursive_model.sudoku import (
    count_solutions,
    generate_puzzle,
    is_valid_solution,
    random_full_grid,
    solve_one,
)


def _groups_ok(grid: list[int]) -> bool:
    target = set(range(1, 10))
    for k in range(9):
        if set(grid[k * 9 : (k + 1) * 9]) != target:
            return False
        if {grid[r * 9 + k] for r in range(9)} != target:
            return False
    return True


def test_random_full_grid_is_valid():
    rng = random.Random(0)
    for _ in range(3):
        assert _groups_ok(random_full_grid(rng))


def test_generate_puzzle_shapes_and_holes():
    puzzle, solution, holes = generate_puzzle(n_holes=25, seed=1)
    assert len(puzzle) == 81 and len(solution) == 81
    assert holes == 25
    assert sum(1 for v in puzzle if v == 0) == 25
    assert _groups_ok(solution)


def test_puzzle_givens_match_solution():
    puzzle, solution, _ = generate_puzzle(n_holes=30, seed=2)
    for i in range(81):
        if puzzle[i] != 0:
            assert puzzle[i] == solution[i]


def test_generated_puzzles_have_unique_solution():
    for seed in (10, 11, 12):
        puzzle, solution, _ = generate_puzzle(n_holes=25, seed=seed)
        assert count_solutions(puzzle, limit=2) == 1
        assert solve_one(puzzle) == solution


def test_difficulty_control_more_holes_fewer_givens():
    easy, _, _ = generate_puzzle(n_holes=20, seed=5)
    hard, _, _ = generate_puzzle(n_holes=40, seed=5)
    givens_easy = sum(1 for v in easy if v != 0)
    givens_hard = sum(1 for v in hard if v != 0)
    assert givens_hard < givens_easy


def test_solver_on_known_puzzle():
    puzzle = [
        5, 3, 0, 0, 7, 0, 0, 0, 0,
        6, 0, 0, 1, 9, 5, 0, 0, 0,
        0, 9, 8, 0, 0, 0, 0, 6, 0,
        8, 0, 0, 0, 6, 0, 0, 0, 3,
        4, 0, 0, 8, 0, 3, 0, 0, 1,
        7, 0, 0, 0, 2, 0, 0, 0, 6,
        0, 6, 0, 0, 0, 0, 2, 8, 0,
        0, 0, 0, 4, 1, 9, 0, 0, 5,
        0, 0, 0, 0, 8, 0, 0, 7, 9,
    ]
    expected = [
        5, 3, 4, 6, 7, 8, 9, 1, 2,
        6, 7, 2, 1, 9, 5, 3, 4, 8,
        1, 9, 8, 3, 4, 2, 5, 6, 7,
        8, 5, 9, 7, 6, 1, 4, 2, 3,
        4, 2, 6, 8, 5, 3, 7, 9, 1,
        7, 1, 3, 9, 2, 4, 8, 5, 6,
        9, 6, 1, 5, 3, 7, 2, 8, 4,
        2, 8, 7, 4, 1, 9, 6, 3, 5,
        3, 4, 5, 2, 8, 6, 1, 7, 9,
    ]
    assert solve_one(puzzle) == expected


def test_solver_returns_none_for_unsolvable():
    bad = [1] * 9 + [0] * 72  # duplicate 1s in row 0
    assert solve_one(bad) is None
    assert count_solutions(bad, limit=2) == 0


def test_is_valid_solution():
    good = [(3 * (r % 3) + r // 3 + c) % 9 + 1 for r in range(9) for c in range(9)]
    assert is_valid_solution(good)
    bad = good[:]
    bad[0], bad[1] = bad[1], bad[0]  # swap breaks rows/cols
    assert not is_valid_solution(bad)
    assert not is_valid_solution([0] * 81)
    assert not is_valid_solution([1] * 80)
