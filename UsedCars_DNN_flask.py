from flask import Flask,render_template,request
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
app = Flask(__name__)

#預測車價函式
def PredictPrice(x1,x2,x3):
    dfCars = pd.read_csv('used_cars_dm3_0713-y1t8.csv', encoding='utf8', low_memory=False)

    #標準化
    # cols_to_norm = ['city_fuel_economy','highway_fuel_economy','power_hp']
    # dfCars[cols_to_norm] = dfCars[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min())*0.6+0.2)
    
    y=dfCars['price']
    dfCars =dfCars.drop(['price','Unnamed: 0'],axis=1)

    print(dfCars.head())

    X_train, X_test, y_train, y_test = train_test_split(dfCars, y,random_state=1)
    regr = MLPRegressor(hidden_layer_sizes=(15, 10, 15), learning_rate='adaptive',
                learning_rate_init=0.1, max_iter=2000, random_state=1).fit(X_train, y_train)

    # print(X_test[:2])
    # print(regr.predict(X_test[:2]))
    tmp = np.array([[x1,x2,x3]])
    # print("Test:",regr.predict(tmp))

    return np.exp(regr.predict(tmp))
# 函式結束

# # #
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ipt')
def ipt():
    mx = request.args.get('')
    return render_template('InputText.html',a=[mx])

@app.route('/pt')
def pt():
    #接收Input
    if ((str(request.args.get('num1')) =='' or str(request.args.get('num2'))=='' or str(request.args.get('num3'))=='')  or 
            (str(request.args.get('num1')) =='None' or str(request.args.get('num2'))=='None' or str(request.args.get('num3'))=='None')):
        print('None')
        return render_template('InputText.html')

    num1 = np.log(float(request.args.get('num1')))
    num2 = np.log(float(request.args.get('num2')))
    num3 = np.log(float(request.args.get('num3')))

    #呼叫PredictPrice函式
    pprice = int(PredictPrice(num1,num2,num3))
    return render_template('Output.html', tnum1=str(pprice)) 

# # #


if __name__ == '__main__':

   app.run(debug=True)