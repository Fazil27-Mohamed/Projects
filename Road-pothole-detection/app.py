import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import shutil
import glob
import subprocess

import cv2
import torch
from flask import Flask, render_template, request
from ultralytics import YOLO

torch.set_num_threads(1)

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


model = YOLO("best.pt")
model.to("cpu")

IMG_SIZE = 416


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template("index.html")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html")

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    ext = os.path.splitext(file.filename)[1].lower()

    image_ext = [".jpg", ".jpeg", ".png"]
    video_ext = [".mp4", ".avi", ".mov", ".mkv"]


    if ext in image_ext:

        with torch.no_grad():
            results = model.predict(
                source=input_path,
                imgsz=IMG_SIZE,
                conf=0.5,
                verbose=False,
            )

        annotated = results[0].plot()

        output_name = "result_" + file.filename
        output_path = os.path.join(RESULT_FOLDER, output_name)

        cv2.imwrite(output_path, annotated)

        del results, annotated
        gc.collect()

        try:
            os.remove(input_path)
        except OSError:
            pass

        return render_template(
            "index.html",
            image="results/" + output_name
        )


    elif ext in video_ext:

        with torch.no_grad():
            results = model.predict(
                source=input_path,
                imgsz=IMG_SIZE,
                conf=0.5,
                save=True,
                stream=True,   # process frame-by-frame instead of buffering the whole video
                verbose=False,
            )
            # stream=True returns a generator - must be consumed to actually run/save
            save_dir = None
            for r in results:
                if save_dir is None:
                    save_dir = str(r.save_dir)
                del r

        gc.collect()

        if save_dir is None:
            return "Detection video not found."

        video_files = []
        for pattern in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            video_files.extend(glob.glob(os.path.join(save_dir, pattern)))

        if len(video_files) == 0:
            return "Detection video not found."

        detected_video = video_files[0]

        temp_video = os.path.join(RESULT_FOLDER, "temp_video.mp4")
        shutil.copy2(detected_video, temp_video)

        final_video = os.path.join(RESULT_FOLDER, "result_video.mp4")

        subprocess.run([
            "ffmpeg", "-y",
            "-i", temp_video,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            final_video
        ])

        os.remove(temp_video)
        try:
            os.remove(input_path)
            shutil.rmtree(save_dir, ignore_errors=True)
        except OSError:
            pass

        return render_template(
            "index.html",
            video="results/result_video.mp4"
        )

    else:

        return "Unsupported File"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
