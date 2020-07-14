# encoding: utf-8
"""
@author: sunkai
@contact: moguizhiz@126.com
@time: 2020/7/14 10:34
@file: Multivariate_Linear_Regression.py
@desc: 
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

def load_data():
    data = load_boston()
    x = data.data
    y = data.target
    return x, y

def train(x, y):
    print('X的形状：',x.shape)
    model = LinearRegression()
    model.fit(x,y)
    print("权重为：",model.coef_)
    print("偏置为：",model.intercept_)
    print("第12个房屋的预测价格：",model.predict(x[12,:].reshape(1,-1)))
    print("第12个房屋的真实价格：",y[12])


if __name__ == '__main__':
    x,y = load_data()
    train(x,y)