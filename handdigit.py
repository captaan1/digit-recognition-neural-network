import numpy as np
import pandas as pd

data = pd.read_csv("train.csv")
data=np.array(data)
m,n=data.shape
np.random.shuffle(data)

data_dev=data[0:1000].T
y_dev=data_dev[0]
x_dev=data_dev[0:n]

train_dev=data[1000:m].T
Y_train=train_dev[0]
x_train=train_dev[1:n]
x_train=x_train/255
_,m_train=x_train.shape
def init():
    W1=np.random.rand(10,784)-0.5
    B1=np.random.rand(10,10)-0.5
    W2=np.random.rand(10,10)-0.5
    B2=np.random.rand(10,10)-0.5
    return W1,W2,B1,B2
def ReLU(Z):
    return np.maximum(Z,0)
def softmax(Z):
    A=np.exp(Z)/sum(np.exp(Z))
    return A
def forwardprop(W1,W2,B1,B2,X):
    Z1=W1.dot(X)+B1
    A1=ReLU(Z1)
    Z2=W2.dot(A1)+B2
    A2=softmax(Z2)
def ReLU(z):
    return z>0
def ReLU_deriv(Z):
    return Z > 0
def onehot_y(Y):
    one_hot_y=np.zeros(Y.size,Y.max()+1)
    one_hot_y[np.arange(Y.size)]=1
    one_hot_y=one_hot_y.T
    return one_hot_y
def bakword_prop(Z1,A1,Z2,A2,W1,W2,X,Y):
    onehot_y=onehot_y(Y)
    dZ2=A2-onehot_y
    dW2=1/m*dZ2.dot(A1.T)
    db2=1/m*np.sum(dZ2)
    dZ1=W2.T.dot(dZ2)*ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardprop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = bakword_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
W1, b1, W2, b2 = gradient_descent(x_train, Y_train, 0.10, 500)









