from flask import Flask, request, jsonify, render_template
from Predict import carbrand
from ai_utils.utils import decodeImage

app = Flask(__name__)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.model = carbrand(self.filename)

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.model.prediction()
    return jsonify(result)

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='localhost', port=8000, debug=True)
