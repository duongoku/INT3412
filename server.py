from flask import Flask, json, render_template, request
from flask_cors import CORS, cross_origin
import api as backend
import threading

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/get-result", methods=["GET"])
def get_result():
    image_url = request.args.get("image")
    if image_url not in backend.IMAGES_CACHE:
        backend.IMAGES_CACHE[image_url] = {
            "result": None,
            "status": "processing",
        }
        threading.Thread(
            target=backend.process_image, args=(image_url,), kwargs={}
        ).start()
    return backend.IMAGES_CACHE[image_url]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    backend.init()
    app.run()
