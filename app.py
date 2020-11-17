import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import dump, load

app = Flask(__name__, template_folder = 'template')
predict_model = load('Real_Estate.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = predict_model.predict(final_features)
    output = round(abs(prediction[0]))
    return render_template('index.html',prediction_text="Estimated Cost of house is {}K $".format(output))

if __name__ == '__main__':
    app.run(debug=True)
