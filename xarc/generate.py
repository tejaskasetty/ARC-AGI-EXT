import argparse
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from . import tasks as tk
from .utils import convert_to_json


def generate_data(
    tasks: List[str],
    num_samples: int,
    canvas_size: int = tk.Task.CANVAS_SIZE,
    write_to_file: bool = False,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = []
    y = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for task in tqdm(tasks, desc="No. of tasks"):
        verbose and print(f"generating for task{task}")
        task_cls = getattr(tk, f"Task{task}")
        input, output = tuple(zip(*task_cls(canvas_size).run(num_samples)))
        if write_to_file:
            r = convert_to_json(input, output)
            with open(f"data/extended/training/task{task}.json", "w+") as f:
                f.write(r)
        x.append(torch.from_numpy(np.array(input)).to(device))
        y.append(torch.from_numpy(np.array(output)).to(device))
    return torch.stack(x), torch.stack(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tasks", nargs="+", help="List of tasks to generate", required=True)
    parser.add_argument("-n", "--num_samples", type=int, default=6)
    args = parser.parse_args()
    data = generate_data(args.tasks, args.num_samples, write_to_file=True)
