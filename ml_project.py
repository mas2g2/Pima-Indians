import numpy as np
import csv
import ast

with open("PimaIndians.csv","r") as f:
    data = list(csv.reader(f));

data = np.array(data)
print(data.shape)
data = data[1:,:]
data = data.astype(float)
training_x,training_y,testing_x,testing_y = data[:261,:8],data[:261,8],data[261:,:8],data[261:,8]
print("Training x shape : ",training_x.shape," Training y shape : ",training_y)

f = open("theta.txt","r")
theta = f.read()
theta = ast.literal_eval(theta)

theta = np.array(theta)

def g_x(theta, x):
    return np.dot(theta,x.T)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def cost(theta,x,y):
        m = len(training_x)
        total_cost =-(1/m)*np.sum(y*np.log(sigmoid(g_x(theta,x)))+(1-y)*np.log(1-sigmoid(g_x(theta,x))))
        return total_cost

def grad_desc(theta,x,y):
        m = len(x)
        return (1/m)*np.dot(x.transpose(),sigmoid(g_x(theta,x))-y)

def train(theta,x,y,iterations,learning_rate):
        print("Training model ...");
        cost_history = []
        theta_history = []
        for i in range(iterations):
            theta -= learning_rate*grad_desc(theta,x,y)
            err = cost(theta,x,y)
            theta_history.append(theta)
            cost_history.append(err)
            print(err)
        cost_history = np.array(cost_history)
        return theta,cost_history

def score(theta,x,y):
    error_count = 0
    pred_y = sigmoid(g_x(theta,x))
    for i in range(len(y)):
        if pred_y[i] >= 0.5:
            pred_y[i] = 1
        else:
            pred_y[i] = 0

        if pred_y[i] != y[i]:
            error_count += 1
    return 1 - float(error_count/len(y))

iterations =4000
learning_rate = 0.01
theta,cost_history = train(theta,training_x,training_y,iterations,learning_rate)
print("Accuracy : ",score(theta,testing_x,testing_y))
