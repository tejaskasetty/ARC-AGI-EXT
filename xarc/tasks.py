import random as rd
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from beartype import beartype

from .utils import find_islands, get_element_with_highest_frequency, insert_pattern_into_canvas

TASK_LIST = [1, 2, 372, "x1", "x2", "x3", "x4", "x5", "x6", "x7"]


class Task(ABC):
    NUM_COLORS: int = 10
    EMPTY_COLOR = 0
    SEPERATOR_COLOR = 5
    FILL_COLORS = list(set(range(NUM_COLORS)) - {EMPTY_COLOR, SEPERATOR_COLOR})
    CANVAS_SIZE = 6

    def __init__(self, canvas_size: int = CANVAS_SIZE):
        self.canvas_size = canvas_size

    @beartype
    def run(
        self, n: int, include_options=False
    ) -> List[Tuple[np.ndarray, np.ndarray]] | List[Tuple[np.ndarray, np.ndarray, List[np.ndarray], int]]:
        last_idx = 4 if include_options else 2
        return [self.gen_io()[:last_idx] for _ in range(n)]

    @beartype
    def gen_io(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], int]:
        input = self.gen_input()
        output = self.solve(input)
        options, idx = self.gen_options(input, output)
        canvas = np.full((self.canvas_size, self.canvas_size), self.EMPTY_COLOR)
        input = insert_pattern_into_canvas(input, canvas)
        output = insert_pattern_into_canvas(output, canvas)
        options = [insert_pattern_into_canvas(option, canvas) for option in options]
        return input, output, options, idx

    @beartype
    @abstractmethod
    def gen_input(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], int]:
        pass

    @beartype
    @abstractmethod
    def solve(self, input: np.ndarray) -> np.ndarray:
        pass

    @beartype
    def gen_options(self, input: np.ndarray, output: np.ndarray) -> Tuple[List[np.ndarray], int]:
        return []


class Task1(Task):
    def __init__(self, canvas_size: int = 11):
        self.base_color = np.random.randint(1, self.NUM_COLORS)
        remaining_colors = list(set(range(self.NUM_COLORS)) - {self.EMPTY_COLOR, self.base_color})
        self.fill_color = np.random.choice(remaining_colors)
        super().__init__(canvas_size)

    @beartype
    def gen_input(self) -> np.ndarray:
        input_size = np.random.randint(3, self.canvas_size - 1)
        input = np.random.choice([0, self.base_color], size=(input_size, input_size), p=[0.6, 0.4])
        if not self.check_is_valid(input):
            if np.random.rand() < 0.90:
                input = self.insert_rand_box(input, input_size)
        return input

    @beartype
    def insert_rand_box(self, input: np.ndarray, input_size: int) -> np.ndarray:
        w = np.random.randint(3, input_size + 1)
        h = np.random.randint(3, input_size + 1)
        x = np.random.randint(0, input_size - w + 1)
        y = np.random.randint(0, input_size - h + 1)
        input[x + 1 : x + w - 1, y + 1 : y + h - 1] = self.EMPTY_COLOR
        input[x, y + 1 : y + h - 1] = self.base_color
        input[x + w - 1, y + 1 : y + h - 1] = self.base_color
        input[x + 1 : x + w - 1, y] = self.base_color
        input[x + 1 : x + w - 1, y + h - 1] = self.base_color
        return input

    @beartype
    def check_is_valid(self, input: np.ndarray) -> bool:
        return len(find_islands(input)) > 0

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        islands = find_islands(input)
        if len(islands) == 0:
            return input
        idx = tuple(map(list, zip(*sum(islands, []))))
        output = np.copy(input)
        output[idx[0], idx[1]] = self.fill_color
        return output

    @beartype
    def gen_options(self, input: np.ndarray, output: np.ndarray) -> Tuple[List[np.ndarray], int]:
        options = []
        for i in range(4):
            deviation = np.copy(output)
            indices = np.argwhere(output != self.base_color)
            indices = indices[np.random.choice(indices.shape[0], min(3, indices.shape[0]), replace=False)]
            deviation[indices[:, 0], indices[:, 1]] = self.base_color
            options.append(deviation)
        rd.shuffle(options)
        idx = np.random.randint(0, 5)
        options.insert(idx, output)
        return options, idx


class Task2(Task):
    def __init__(self, canvas_size: int = 11):
        self.input_color = np.random.randint(1, self.NUM_COLORS)
        remaining_colors = list(set(range(self.NUM_COLORS)) - {self.EMPTY_COLOR, self.input_color})
        self.output_color = np.random.choice(remaining_colors)
        super().__init__(canvas_size)

    @beartype
    def gen_input(self) -> np.ndarray:
        input_size = np.random.randint(3, self.canvas_size + 1)
        input = np.random.choice([0, self.input_color], size=(input_size, input_size))
        return input

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        output = np.copy(input)
        output[output == self.input_color] = self.output_color
        return output

    @beartype
    def gen_options(self, input: np.ndarray, output: np.ndarray) -> Tuple[List[np.ndarray], int]:
        options = []
        colors = list(set(self.FILL_COLORS) - {self.output_color})
        rd.shuffle(colors)
        for i in range(4):
            deviation = np.copy(output)
            deviation[deviation == self.output_color] = colors[i]
            options.append(deviation)
        rd.shuffle(options)
        idx = np.random.randint(0, 5)
        options.insert(idx, output)
        return options, idx


class Task5(Task):
    def __init__(self, canvas_size: int = 11):
        self.input_color = np.random.choice(self.FILL_COLORS)
        super().__init__(canvas_size)

    @beartype
    def gen_input(self) -> np.ndarray:  # need to comeback and fix this. This is not correct
        input_size = np.random.randint(3, self.canvas_size // 2)
        input = np.random.choice([0, self.input_color], size=(input_size, 2 * input_size + 1))
        input[:, input_size] = self.SEPERATOR_COLOR
        input = np.pad(input, 1, constant_values=self.SEPERATOR_COLOR)
        return input

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        input_size = input.shape[0]
        left_input, right_input = input[1:-1, 1 : input_size - 1], input[1:-1, input_size:-1]
        output = left_input & right_input
        output = np.pad(output, 1, constant_values=self.SEPERATOR_COLOR)
        return output


class Task372(Task):
    def __init__(self, canvas_size: int = 11):
        self.color_1 = np.random.randint(1, self.NUM_COLORS)
        remaining_colors = list(set(range(self.NUM_COLORS)) - {self.EMPTY_COLOR, self.color_1})
        self.color_2 = np.random.choice(remaining_colors)
        super().__init__(canvas_size)

    @beartype
    def gen_input(self) -> np.ndarray:  # comeback and fix it.
        input_size = np.random.randint(3, self.canvas_size + 1)
        input = np.full((2, input_size), self.color_1)
        input[1, :] = self.color_2
        return input

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        output = np.copy(input)
        output[0, 1::2] = self.color_2
        output[1, 1::2] = self.color_1
        return output

    @beartype
    def gen_options(self, input: np.ndarray, output: np.ndarray) -> Tuple[List[np.ndarray], int]:
        options = []
        colors = (self.color_1, self.color_2)
        for i in range(4):
            deviation = np.copy(output)
            idx = np.argwhere((output == colors[i % 2]))[i // 2]
            deviation[idx[0], idx[1]] = colors[(i + 1) % 2]
            options.append(deviation)
        rd.shuffle(options)
        idx = np.random.randint(0, 5)
        options.insert(idx, output)
        return options, idx


class Taskx1(Task):
    def __init__(self, canvas_size: int = 11):
        self.input_color = np.random.randint(1, self.NUM_COLORS)
        super().__init__(canvas_size)

    @beartype
    def gen_input(self) -> np.ndarray:
        # input_size = np.random.randint(3, self.canvas_size + 1)
        input_size = self.canvas_size
        input = np.random.choice([0, self.input_color], size=(input_size, input_size))
        return input

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        output = np.where(input == self.input_color, self.EMPTY_COLOR, self.input_color)
        return output

    @beartype
    def gen_options(self, input: np.ndarray, output: np.ndarray) -> Tuple[List[np.ndarray], int]:
        options = []
        colors = (self.EMPTY_COLOR, self.input_color)
        for i in range(4):
            deviation = np.copy(output)
            indices = np.argwhere((output == colors[i % 2]))
            indices = indices[
                np.random.choice(indices.shape[0], size=min(4, indices.shape[0]), replace=False)
            ]
            deviation[indices[:, 0], indices[:, 1]] = colors[(i + 1) % 2]
            options.append(deviation)
        rd.shuffle(options)
        idx = np.random.randint(0, 5)
        options.insert(idx, output)
        return options, idx


class Taskx2(Task):
    def __init__(self, canvas_size: int = 11):
        self.input_color = np.random.randint(1, self.NUM_COLORS)
        remaining_colors = list(set(range(self.NUM_COLORS)) - {self.EMPTY_COLOR, self.input_color})
        self.output_color = np.random.choice(remaining_colors)
        super().__init__(canvas_size)

    @beartype
    def gen_input(self) -> np.ndarray:
        # input_size = np.random.randint(3, self.canvas_size + 1)
        input_size = self.canvas_size
        input = np.random.choice([0, self.input_color], size=(input_size, input_size))
        return input

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        output = np.where(input == self.input_color, self.EMPTY_COLOR, self.output_color)
        return output

    @beartype
    def gen_options(self, input: np.ndarray, output: np.ndarray) -> Tuple[List[np.ndarray], int]:
        options = []
        colors = (self.EMPTY_COLOR, self.output_color)
        for i in range(4):
            deviation = np.copy(output)
            indices = np.argwhere((output == colors[i % 2]))
            indices = indices[
                np.random.choice(indices.shape[0], size=min(4, indices.shape[0]), replace=False)
            ]
            deviation[indices[:, 0], indices[:, 1]] = colors[(i + 1) % 2]
            options.append(deviation)
        rd.shuffle(options)
        idx = np.random.randint(0, 5)
        options.insert(idx, output)
        return options, idx


class Taskx3(Task):
    def __init__(self, canvas_size: int = 11):
        self.input_color = np.random.randint(1, self.NUM_COLORS)
        super().__init__(canvas_size)

    @beartype
    def gen_input(self) -> np.ndarray:
        # input_size = np.random.randint(3, self.canvas_size + 1)
        input_size = self.canvas_size
        input = np.random.choice([0, self.input_color], size=(input_size, input_size), p=[0.7, 0.3])
        return input

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        output_color = self.input_color if np.any(input == self.input_color) else self.EMPTY_COLOR
        output = np.full_like(input, output_color)
        return output

    @beartype
    def gen_options(self, input: np.ndarray, output: np.ndarray) -> Tuple[List[np.ndarray], int]:
        options = []
        colors = list((set(self.FILL_COLORS) | {self.EMPTY_COLOR}) - {self.input_color})
        rd.shuffle(colors)
        for i in range(4):
            deviation = np.copy(output)
            deviation[deviation == self.input_color] = colors[i]
            options.append(deviation)
        rd.shuffle(options)
        idx = np.random.randint(0, 5)
        options.insert(idx, output)
        return options, idx


class Taskx4(Task):
    def __init__(self, canvas_size: int = 11):
        self.input_colors = list(range(1, self.NUM_COLORS))
        super().__init__(canvas_size)

    @beartype
    def gen_input(self) -> np.ndarray:
        # input_size = np.random.randint(3, self.canvas_size + 1)
        input_size = self.canvas_size
        input = np.random.choice(self.input_colors, size=(input_size, input_size))
        if not self.check_if_valid(input):
            unique, counts = np.unique(input, return_counts=True)
            min_count_color = unique[np.argmin(counts)]
            max_count_color = unique[np.random.choice(np.flatnonzero(counts == counts.max()))]
            input[input == min_count_color] = max_count_color

        return input

    @beartype
    def check_if_valid(self, input: np.ndarray) -> bool:
        hf_elem = get_element_with_highest_frequency(input, is_unique=True)
        return hf_elem is not None

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        out_color = get_element_with_highest_frequency(input)
        output = np.full_like(input, out_color)
        return output

    @beartype
    def gen_options(self, input: np.ndarray, output: np.ndarray) -> Tuple[List[np.ndarray], int]:
        options = []
        colors = np.unique(input).tolist()
        rd.shuffle(colors)
        for i in range(4):
            deviation = np.copy(output)
            deviation[:, :] = colors[i]
            options.append(deviation)
        idx = np.random.randint(0, 5)
        options.insert(idx, output)
        return options, idx


class Taskx5(Task):
    def __init__(self, canvas_size: int = 11):
        self.color_1 = np.random.randint(1, self.NUM_COLORS)
        remaining_colors = list(set(range(self.NUM_COLORS)) - {self.EMPTY_COLOR, self.color_1})
        self.color_2 = np.random.choice(remaining_colors)
        super().__init__(canvas_size)

    @beartype
    def gen_input(self) -> np.ndarray:
        input_size = np.random.randint(3, self.canvas_size + 1)
        input = np.full((input_size, 2), self.color_1)
        input[:, 1] = self.color_2
        return input

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        output = np.copy(input)
        output[1::2, 0] = self.color_2
        output[1::2, 1] = self.color_1
        return output

    @beartype
    def gen_options(self, input: np.ndarray, output: np.ndarray) -> Tuple[List[np.ndarray], int]:
        options = []
        colors = (self.color_1, self.color_2)
        for i in range(4):
            deviation = np.copy(output)
            idx = np.argwhere((output == colors[i % 2]))[i // 2]
            deviation[idx[0], idx[1]] = colors[(i + 1) % 2]
            options.append(deviation)
        rd.shuffle(options)
        idx = np.random.randint(0, 5)
        options.insert(idx, output)
        return options, idx


class Taskx6(Task):
    def __init__(self, canvas_size: int = 11):
        self.input_color = np.random.randint(1, self.NUM_COLORS)
        super().__init__(canvas_size)

    @beartype
    def gen_input(self) -> np.ndarray:
        # input_size = np.random.randint(3, self.canvas_size + 1)
        input_size = self.canvas_size
        input = np.random.choice([0, self.input_color], size=(input_size, input_size))
        return input

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        output = np.rot90(input)
        return output

    @beartype
    def gen_options(self, input: np.ndarray, output: np.ndarray) -> Tuple[List[np.ndarray], int]:
        options = []
        deviation = np.copy(output)
        for i in range(3):
            deviation = np.rot90(deviation)
            options.append(deviation)
        options.append(input.T)
        rd.shuffle(options)
        idx = np.random.randint(0, 5)
        options.insert(idx, output)
        return options, idx


class Taskx7(Task):
    def __init__(self, canvas_size: int = 11):
        self.input_color = np.random.randint(1, self.NUM_COLORS)
        remaining_colors = list(set(range(self.NUM_COLORS)) - {self.EMPTY_COLOR, self.input_color})
        self.output_color = np.random.choice(remaining_colors)
        super().__init__(canvas_size)

    @beartype
    def gen_input(self) -> np.ndarray:
        # input_size = np.random.randint(3, self.canvas_size + 1)
        input_size = self.canvas_size
        input = np.random.choice([0, self.input_color], size=(input_size, input_size))
        return input

    @beartype
    def solve(self, input: np.ndarray) -> np.ndarray:
        output = np.rot90(input.copy())
        output[output == self.input_color] = self.output_color
        return output

    @beartype
    def gen_options(self, input: np.ndarray, output: np.ndarray) -> Tuple[List[np.ndarray], int]:
        options = []
        colors = list(set(self.FILL_COLORS) - {self.output_color})
        rd.shuffle(colors)
        deviation = np.copy(output)
        for i in range(4):
            deviation = np.rot90(deviation)
            deviation[deviation == self.output_color] = colors[i]
            options.append(deviation)
        rd.shuffle(options)
        idx = np.random.randint(0, 5)
        options.insert(idx, output)
        return options, idx
