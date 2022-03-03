from flask import Flask, render_template, request
from LiverClassifier import random_forest_test, random_forest_train, random_forest_predict
#from Random_forest_heart import random_forest_testh, random_forest_trainh, random_forest_predicth
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random_forest import accuracy
from sklearn.metrics import accuracy_score
from time import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST']) 
def login_user():
    data_points = list()
    data = []
    for i in range(20,30):
        data.append(float(request.form['values'+str(i)]))

    for i in range(30):
        data_points.append(data[i]) 
    print(data_points)
    data_np = np.asarray(data,dtype = float)
    data_np = data_np.reshape(1,-1)
    out,acc,t=random_forest_predict(clf,data_np)
    acc1=0.0
    if(out==1):
        output = 'Malignant'
    else:
        output = 'Benign'
    acc_x = acc[0][0]
    acc_y = acc[0][1]
    if(acc_x>acc_y):
        acc1 = acc_x
    else:
        acc1=acc_y
    return render_template('result.html', output=output,accuracy=acc1, time=time)
if __name__=='__main__':
    global clf 
    cl=random_forest_trainh()
    random_forest_testh(cl)
    clf = random_forest_train()
    random_forest_test(clf)
    app.run(port=1008,debug=False)

