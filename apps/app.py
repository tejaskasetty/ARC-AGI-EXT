import json
import os

from flask import Flask, jsonify, request

app = Flask(__name__)
directory_path = "data/extended"  # Replace with the path to your directory


@app.route("/data/<subset>", methods=["GET"])
def load_file(subset):
    try:
        task_id = int(request.args.get("id"))
        folder_path = os.path.join(directory_path, subset)
        files = os.listdir(folder_path)
        sorted_files = sorted(files, key=lambda x: x.split("task")[1].split(".")[0])
        file = os.path.join(folder_path, sorted_files[task_id])
        return jsonify(json.load(open(file)))
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
