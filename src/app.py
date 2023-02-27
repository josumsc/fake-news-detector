from flask import Flask, render_template, request, jsonify
from detector import DetectorPipeline

app = Flask(__name__)


def inference(text):
    pipeline = DetectorPipeline()
    tokenizer, model = pipeline.get_tokenizer_and_model("JosuMSC/fake-news-detector")
    prediction, logits = pipeline.predict(tokenizer, model, text)
    logits = logits.cpu().detach().numpy().tolist()
    result = "Fake" if prediction == 0 else "Real"
    return result, logits


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect_html", methods=["POST"])
def detect_html():
    if request.method == "POST":
        text = request.form["text"]
        result, logits = inference(text)
        return render_template("index.html", result=result, logits=logits)


@app.route("/detect_json", methods=["POST"])
def detect_json():
    if request.method == "POST":
        text = request.json["text"]
        result, logits = inference(text)
        return jsonify(result=result, logits=logits)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
