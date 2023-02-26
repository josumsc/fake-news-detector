from flask import Flask, render_template, request, jsonify
from detector import DetectorPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        text = request.form['text']
        pipeline = DetectorPipeline()
        tokenizer, model = pipeline.get_tokenizer_and_model("JosuMSC/fake-news-detector")
        prediction, logits = pipeline.predict(tokenizer, model, text)
        inference = 'Fake' if prediction == 0 else 'Real'

        return jsonify(inference=inference)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
