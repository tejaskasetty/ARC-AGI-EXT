import argparse

import numpy as np

from xarc import TASK_LIST, generate_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--num_tasks", nargs="+", help="List of tasks to generate", required=True)
    parser.add_argument("-t", "--num_tasks", type=int, required=True)
    parser.add_argument("-n", "--num_samples", type=int, required=True)
    parser.add_argument("-f", "--format", type=str, default="json")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-p", "--path", type=str)
    args = parser.parse_args()
    tasks = np.random.choice(TASK_LIST, args.num_tasks).tolist()
    data = generate_data(
        tasks,
        args.num_samples,
        seed=args.seed,
        store_path=args.path,
        is_write_to_file=True,
        format=args.format,
    )
    print("Done!")
