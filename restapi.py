"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from PIL import Image

import torch
from flask import Flask, request

def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

app = Flask(__name__)

DETECTION_URL = "/api/decodeLaundryTag"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("file"):
        image_file = request.files["file"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img, size=640)
        data = results.pandas().xyxy[0].to_json(orient="records")
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'custom', './last.pt')  # force_reload = recache latest code
    # checkpoint_ = torch.load('./last.pt')['model']
    # model.load_state_dict(checkpoint_.state_dict())
    
    # copy_attr(model, checkpoint_, includeinclude=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())
    # model = model.fuse().autoshape()
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
