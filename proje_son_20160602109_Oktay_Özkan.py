# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:23:52 2018

@author: oktay
"""
#Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings(action = 'ignore',category = SettingWithCopyWarning)



if __name__ == "__main__":
    df = pd.read_csv('DATA.csv', sep = ',', na_values = [''])

    # First hundred rows (y values are known)
    df1 = df.head(100)
    #Rows that y values are not known
    df2 = df.tail(20)
    print("First Approach")
    x = np.asmatrix(df1[['x1', 'x2', 'x3', 'x4', 'x5']])
    y = np.asmatrix(df1[['Y']])

   
    # a.x1 + b.x2 + c.x3 + d.x4 + e.x5 = y
    abcde = np.linalg.pinv(x) * y

    print(df1)
    print('*******************************')

    # print (a,b,c,d,e) 
    print("a,b,c,d,e:")
    print(abcde)

    print('*******************************')

   
    df2['Y'] = df2['Y'].fillna(df2['x1'] * abcde.item(0) + df2['x2'] * abcde.item(1) + df2['x3'] * abcde.item(2) + df2['x4'] * abcde.item(3) + df2['x5'] * abcde.item(4))

    
    print(df2)


    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    
    x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size=1 / 3, random_state=3)

    regression = LinearRegression()
    regression.fit(x_training, y_training)
    #
    print('Variance: ', regression.score(x_testing, y_testing))

    
    
    
    print("******************************************")
    print("Second Approach")
    y2=df1.Y
    x2=df1[['x1','x2','x3','x4','x5']]
    
    linearRegression=LinearRegression()
    linearRegression.fit(x2, y2)
    y_predict = linearRegression.predict(x2)
    
    print("x1:",  linearRegression.coef_[0],"  x2:",linearRegression.coef_[1],"  x3:",linearRegression.coef_[2],"  x4:",linearRegression.coef_[3],"  x5:",linearRegression.coef_[4])
    
    #Backward Elimination to eliminate parameter which p value>sl 
    
    import statsmodels.formula.api as sm
    
    x2.insert(0, "x0", 1)

    x2_optimal_model = x2[['x0', 'x1', 'x2', 'x3', 'x4', 'x5']]
    Ordinary_Least_Squares = sm.OLS(endog = y2, exog = x2_optimal_model).fit()
    print(Ordinary_Least_Squares.summary())
    #x1 was eliminated p=0,967
    x2_optimal_model = x2[['x0', 'x2', 'x3', 'x4', 'x5']]
    Ordinary_Least_Squares= sm.OLS(endog = y2, exog = x2_optimal_model).fit()
    print(Ordinary_Least_Squares.summary())
    #x4 was eliminated p=0,872
    x2_optimal_model = x2[['x0', 'x2', 'x3', 'x5']]
    Ordinary_Least_Squares = sm.OLS(endog = y2, exog = x2_optimal_model).fit()
    print(Ordinary_Least_Squares.summary())
    #Predict y
    x2_train=df1[['x2','x3','x5']]
    y2_train=df1.Y
    x2_test=df2[['x2','x3','x5']]
    linearRegression.fit(x2_train, y2_train)
    df2['Y'] = linearRegression.predict(x2_test)
    print(df2)
    
    
    x3=df1[['x2','x3','x5']]

    from sklearn.model_selection import cross_val_score

    print("Mean of rmse [ 'x2','x3','x5' ]: ", np.sqrt(-cross_val_score(linearRegression, x3, y, cv=10, scoring='neg_mean_squared_error')).mean())
    print("Mean of rmse [ 'x1','x2','x3','x4','x5' ]: ", np.sqrt(-cross_val_score(linearRegression, x2, y, cv=10, scoring='neg_mean_squared_error')).mean())
   
    
    
