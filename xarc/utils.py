import json
from typing import Any, List, Tuple

import numpy as np


def dfs(x, y, input, visited, island) -> bool:
    r, c = input.shape
    if x < 0 or x >= r or y < 0 or y >= c:
        return False
    elif visited[x, y] or input[x, y] != 0:
        return True
    else:
        visited[x, y] = True
        island.append((x, y))

    is_island = True
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        is_isle = dfs(x + dx, y + dy, input, visited, island)
        is_island = is_island and is_isle

    return is_island


def find_islands(input: np.ndarray) -> List[Any]:
    islands = []
    visited = np.zeros_like(input, dtype=bool)
    r, c = input.shape
    for i in range(r):
        for j in range(c):
            if input[i][j] == 0 and visited[i, j] == False:
                island = []
                is_island = dfs(i, j, input, visited, island)
                if is_island:
                    islands.append(island)
    return islands


def get_element_with_highest_frequency(input: np.ndarray, is_unique=False):
    unique, counts = np.unique(input, return_counts=True)
    if is_unique:
        max_count = np.max(counts)
        max_count_elements = np.sum(counts == max_count)
        return unique[np.argmax(counts)] if max_count_elements == 1 else None

    return unique[np.argmax(counts)]


def insert_pattern_into_canvas(
    pattern: np.ndarray,
    canvas: np.ndarray,
) -> np.ndarray:
    rp, cp = pattern.shape
    rc, cc = canvas.shape
    if rc < rp or cc < cp:
        raise ValueError("Pattern is larger than canvas")
    x, y = (rc - rp) // 2, (cc - cp) // 2
    canvas = np.copy(canvas)
    canvas[x : x + rp, y : y + cp] = pattern
    return canvas


def convert_to_json(input: List[np.ndarray], output: List[np.ndarray]) -> List[dict]:
    train = list(map(lambda x, y: {"input": x.tolist(), "output": y.tolist()}, input[:-1], output[:-1]))
    test = {"input": input[-1].tolist(), "output": output[-1].tolist()}
    return json.dumps({"train": train, "test": [test]})
