from typing import final
from flask import Flask,render_template,request,url_for,redirect
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route('/Index.html')
@app.route('/')
def hello_world():
    return render_template('/Index.html')
@app.route('/form.html')
def form_return():
    return render_template('/form.html')
@app.route('/loading_page.html')
def form_factor():
    return render_template('/loading_page.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    features = [float(request.form[x]) for x in request.form.keys()]
    input_data = np.array([features])

    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    if prediction == 1:
        return render_template('form.html',pred = 0)
    elif prediction == 2:
        return render_template('form.html',pred = 1)
    elif prediction == 3:
        return render_template('form.html',pred = 2)

if __name__=='__main__':
    app.run(debug=True)