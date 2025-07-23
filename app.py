import numpy as np 
from flask import Flask, request, render_template 
import pickle

app = Flask(__name__)
model = pickle.load(open('crop_model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    feature = [np.array(float_feature)]
    prediction = model.predict(feature)
    return render_template('index.html', prediction_text = 'The crop is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)