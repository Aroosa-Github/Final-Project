from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# model = pickle.load(open("svm.pkl", "rb"))
model = pickle.load(open('diabetes.pkl','rb'))


@app.route('/')
def diabetes():
    return render_template("indexx.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    Pregnancies = request.form['1']
    Glucose = request.form['2']
    BloodPressure = request.form['3']
    SkinThickness = request.form['4']
    Insulin = request.form['5']
    BMI = request.form['6']
    DiabetesPedigreeFunction = request.form['7']
    Age = request.form['8']
 
    row_df = pd.DataFrame([pd.Series([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI ,DiabetesPedigreeFunction ,Age])])
    print(row_df)
    prediction=model.predict_proba(row_df)[0][1]
    
    output = prediction[0][1]
    output = (output*100)+'%'
    if output > (0.5):
        return render_template('result.html',pred=f'You have chance of having diabetes.\nProbability of having Diabetes is {output}')
    else:
        return render_template('result.html',pred=f'You are safe.\n Probability of having diabetes is {output}')




if __name__ == '__main__':
    app.run(debug=True)
