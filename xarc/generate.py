import json
import os
import pickle
from time import time
from typing import List, Literal, Tuple

import numpy as np
import torch
from tqdm import tqdm

from . import tasks as tk
from .utils import convert_to_json, write_to_file


def generate_data(
    tasks: List[str],
    num_samples: int,
    canvas_size: int = tk.Task.CANVAS_SIZE,
    folder_name: str | None = None,
    store_path: str | None = None,
    is_write_to_file: bool = False,
    format: Literal["pickle", "json"] = "pickle",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, dict] | None:
    x = []
    y = []

    dest_folder_path = None
    print(f"Generating xarc data. Number of tasks: {len(tasks)}. Number of samples: {num_samples}")
    if is_write_to_file:
        if store_path is None or folder_name is None:
            raise ValueError("Ivalid input. Both 'store_path' and 'folder_name' is required.")
        elif not os.path.exists(store_path):
            raise FileNotFoundError("Store path doesn't exists: ", store_path)
        dest_folder_path = os.path.join(store_path, folder_name)
        if not os.path.exists(dest_folder_path):
            os.mkdir(dest_folder_path)
        print(f"Writing the data to folder: {folder_name}, format: {format}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, task in enumerate(tqdm(tasks, desc="Task number")):
        verbose and print(f"Generating for task{task}.")
        task_cls = getattr(tk, f"Task{task}")
        input_array, output_array = tuple(zip(*task_cls(canvas_size).run(num_samples)))
        input_tensor, output_tensor = (
            torch.from_numpy(np.array(input_array)).to(device),
            torch.from_numpy(np.array(output_array)).to(device),
        )
        if is_write_to_file:
            dest_file_path = os.path.join(
                dest_folder_path, f"task_{i}.json" if format == "json" else f"task_{i}.pkl"
            )
            data = (input_array, output_array) if format == "json" else (input_tensor, output_tensor)
            write_to_file(data, dest_file_path, format=format)
        else:
            x.append(input_tensor)
            y.append(output_tensor)

    task_params = {
        "task_names": tasks,
        "n_tasks": len(tasks),
        "n_samples": num_samples,
        "canvas_size": canvas_size,
        "dataset_path": dest_folder_path,
    }
    if is_write_to_file:
        with open(os.path.join(dest_folder_path, "task_params.json"), "w+") as f:
            f.write(json.dumps(task_params))
    else:
        return torch.stack(x), torch.stack(y), task_params

    return None
