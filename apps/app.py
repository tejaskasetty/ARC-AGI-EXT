import json
import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(
    app,
    support_credentials=True,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", "Access-Control-Allow-Origin"],
)
directory_path = "data/extended"  # Path to the data directory


@app.route("/data/<subset>", methods=["GET"])
def load_file(subset):
    try:
        folder_path = os.path.join(directory_path, subset)
        files = os.listdir(folder_path)
        sorted_files = sorted(files, key=lambda x: x.split("task")[1].split(".")[0])
        task_id = request.args.get("id")
        print("here", task_id)
        task_id = int(task_id) if task_id != None else np.random.randint(len(sorted_files))
        file = os.path.join(folder_path, sorted_files[task_id])
        response = jsonify(
            {
                "task_index": task_id + 1,
                "name": sorted_files[task_id],
                "length": len(sorted_files),
                "data": json.load(open(file)),
            }
        )
        return response
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    app.run(debug=True, port=3000)
