import argparse

import numpy as np

from xarc import TASK_LIST, generate_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--num_tasks", nargs="+", help="List of tasks to generate", required=True)
    parser.add_argument("-tt", "--num_train_tasks", type=int, required=True)
    parser.add_argument("-vt", "--num_val_tasks", type=int, required=True)
    parser.add_argument("-n", "--num_samples", type=int, required=True)
    parser.add_argument("-f", "--format", type=str, default="json")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-p", "--path", type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    # train data
    train_tasks = np.random.choice(TASK_LIST, args.num_train_tasks).tolist()
    generate_data(
        train_tasks,
        args.num_samples,
        folder_name = f"xarc{args.format[0]}_train_{args.seed}",
        store_path=args.path,
        is_write_to_file=True,
        format=args.format,
    )
    # val data
    val_tasks = np.random.choice(TASK_LIST, args.num_val_tasks).tolist()
    data = generate_data(
        val_tasks,
        args.num_samples,
        folder_name=f"xarc{args.format[0]}_val_{args.seed}",
        store_path=args.path,
        is_write_to_file=True,
        format=args.format,
    )

    print("Done!")
