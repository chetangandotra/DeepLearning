# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:57:45 2017

@author: Chetan
"""

import plotly.plotly as py1
import plotly.graph_objs as go

def plotly_graphs(percent_correct_train, percent_correct_test):
    py1.sign_in('chetang', 'vil7vTAuCSWt2lEZvaH9')
    trace = []
    graph_y = []
    graph_y.append(percent_correct_train)
    graph_y.append(percent_correct_test)
    for i in range(2):
        name = "Training Data"
        if i == 1:
            name = "Test Data"
        y1 = graph_y[i]
        x1 = [j+1 for j in range(len(y1))]
        trace1 = go.Scatter(
            x=x1,
            y=y1,
            name = name,
            connectgaps=True
        )
        trace.append(trace1)
    data = trace
    fig = dict(data=data)
    py1.iplot(fig, filename='InterimSoftmaxAccuracyVsIteration')

def plotly_graphs_Xval(percent_correct_train, percent_correct_test, X_val):
    py1.sign_in('chetang', 'vil7vTAuCSWt2lEZvaH9')
    trace = []
    graph_y = []
    graph_y.append(percent_correct_train)
    graph_y.append(percent_correct_test)
    color1 = ['rgb(49,130,189)', 'rgb(90,180,149)']
    for i in range(2):
        name = "Training Data"
        if i == 1:
            name = "Test Data"
        y1 = graph_y[i]
        x1 = ['After Layer ' + str(i) for i in X_val]
        trace1 = go.Bar(
            x=x1,
            y=y1,
            name = name,
            marker = dict(color=color1[i])
        )
        trace.append(trace1)
    data = trace
    fig = dict(data=data)
    py1.iplot(fig, filename='AccuracyVsIterationForInterimSoftmax')

def plotly_graphs_multiple_plots(X1, X2, X3, X4, X5, name1):
    py1.sign_in('chetang', 'vil7vTAuCSWt2lEZvaH9')
    trace = []
    graph_y = [X1, X2, X3, X4, X5]
    color1 = ['rgb(49,130,189)', 
    'rgb(90,180,149)',
    'rgb(40,110,199)',
    'rgb(140,180,219)',
    'rgb(10,140,239)']
    for i in range(len(graph_y)):
        name = '2'
        if i == 1:
            name = '4'
        elif i == 2:
            name = '8'
        elif i == 3:
            name = '16'
        else:
            name = '25'
        name = name + ' Samples Per Class'
        y1 = graph_y[i]
        x1 = [j+1 for j in range(len(y1))]
        trace1 = go.Scatter(
            x=x1,
            y=y1,
            name = name,
            marker = dict(color=color1[i]),
            connectgaps=True
        )
        trace.append(trace1)
    data = trace
    fig = dict(data=data)
    py1.iplot(fig, filename=name1)

def plotly_graphs_final_accuracies(accuracies, x_axis):
    py1.sign_in('chetang', 'vil7vTAuCSWt2lEZvaH9')
    trace = []
    y1 = accuracies
    x1 = x_axis
    trace1 = go.Scatter(
            x=x1,
            y=y1,
            connectgaps=True
        )
    trace.append(trace1)
    data = trace
    fig = dict(data=data)
    py1.iplot(fig, filename='FinalTestAccuraciesPlot')

#X_train = [4.9376,0.9997,0.2992,0.1297,0.0817,0.0532,0.0486,0.0426,0.037,0.0369]
#X_test = [4.1732,3.4386,3.0918,2.7909,2.6451,2.6594,2.6376,2.641,2.6383,2.6369]

#X_train = [32.49, 80.47, 94.63, 98.39, 99.22, 99.59, 99.66, 99.73, 99.78, 99.78]
#X_test = [44.62, 48.51, 53.31, 54.47, 55.90, 56.55, 56.55, 56.29, 56.42, 56.55]

#X_train = [100, 100, 100, 99.78]
#X_test = [31.91, 46.69, 53.11, 56.55]
#examples = [2, 4, 8, 16]

#X_train = [0.49, 0.39, 0.39, 0.39, 0.68]
#X_test = [0, 0, 0.39, 0.39, 0.78]
#layers = [6, 10, 12, 14, 18]

#X_train = [0.29, 0.58, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68]
#X_test = [0.39, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78]
X_train = [2.2, 95.2, 99.87, 99.87, 99.87, 99.87, 99.87, 99.87, 99.87, 99.87]
X_test = [14.01, 21.4, 21.4, 22.96, 26.85, 26.46, 25.68, 28.02, 28.4, 29.96]
'''
X_train_2 = [1.95, 76.85, 98.33, 100, 100, 100, 100, 100, 100, 100]
X_train_4 = [10.31, 82, 99.32, 100, 100, 100, 100, 100, 100, 100]
X_train_8 = [21.35, 83.27, 98.69, 99.85, 100, 100, 100, 100, 100]
X_train_16 = [32.49, 80.47, 94.63, 98.39, 99.22, 99.59, 99.66, 99.73, 99.78, 100]
X_train_25 = [36.3, 72.86, 82.77, 86.21, 87.16, 87.61]

X_test_2 = [5.06, 14.4, 28.79, 29.96, 29.57, 30.35, 30.74, 31.13, 31.52, 31.91]
X_test_4 = [23.74, 38.52, 43.97, 43.58, 44.75, 46.30, 46.30, 45.91, 45.53, 46.69]
X_test_8 = [40.27, 47.08, 52.14, 54.09, 53.31, 53.11, 53.70, 53.70, 53.70, 53.70]
X_test_16 = [44.62, 48.51, 53.31, 54.47, 55.9, 56.55, 56.55, 56.29, 56.42, 56.55]
X_test_25 = [50.82, 55.18, 55.41, 56.34, 56.50, 57.04]

X_test = [31.91, 46.69, 53.70, 56.55, 57.04]
examples = [2, 4, 8, 16, 25]

plotly_graphs_multiple_plots(X_train_2, X_train_4, X_train_8, X_train_16, 
                             X_train_25, 'trainingAccuracies')
plotly_graphs_multiple_plots(X_test_2, X_test_4, X_test_8, X_test_16, 
                             X_test_25, 'testingAccuracies')

plotly_graphs_final_accuracies(X_test, examples)
'''
plotly_graphs(X_train, X_test)
#plotly_graphs_Xval(X_train, X_test, layers)