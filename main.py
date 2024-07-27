#Importing libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv('./swedish_insurance.csv')
x = data['X'].values
y = data['Y'].values


# Displaying data (uncomment to view interactive graphs of data in the dataset)
'''
fig = px.box(data['X'], points = 'all')
fig.update_layout(title = f'Distribution of X',title_x=0.5, yaxis_title= "Number of Insurance Claims")
fig.show()

fig = px.box(data['Y'], points = 'all')
fig.update_layout(title = f'Distribution of Y',title_x=0.5, yaxis_title= "Amount of Insurance Paid")
fig.show() 

fig = px.scatter(x = data['X'], y = data['Y'])
fig.update_layout(title = "Swedish Automobiles data", title_x = 0.5, xaxis_title = "Number of claims",yaxis_title = "Payment in claims",height = 500, width = 700)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.show()
'''

#Calculating mean,variance,covariance

mean_x = np.mean(data['X'])
mean_y = np.mean(data['Y'])

var_x = np.var(data['X'])
var_y = np.var(data['Y'])

def covar(x,y):
    mean_x = np.mean(data['X'])
    mean_y = np.mean(data['Y'])
    cov = 0.0
    for i in range(len(x)):
        cov+= (x[i]-mean_x) * (y[i]-mean_y)
    return cov/len(x)

cov_xy = covar(data['X'],data['Y'])

b1 = cov_xy / var_x
b0 = mean_y - (b1 * mean_x)
print(f'Coefficents:\n b0: {b0}  b1: {b1} ')

x = data['X'].values.copy()
x.sort

y_prediction = b0 + b1*x

#Putting in graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['X'], y=data['Y'], name='train', mode='markers', marker_color='rgba(152, 0, 0, .8)'))
fig.add_trace(go.Scatter(x=data['X'], y=y_prediction, name='prediction', mode='lines+markers', marker_color='rgba(0, 152, 0, .8)'))
fig.update_layout(title = f'Swedish Automobiles Data\n (visual comparison for correctness)',title_x=0.5, xaxis_title= "Number of Claims", yaxis_title="Payment in Claims")
fig.show()