import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    
    point_wrong=[]
    y_pred=np.zeros(len(X))
    
    #print(X,y)
    for i in range(len(X)):
        
        y_pred[i]=prediction(X[i], W, b)
        #print(y_pred[i])
        if(y_pred[i]!=y[i]):
            point_wrong.append(i)
    #print(point_wrong)
    if (len(point_wrong)>0):
        idx=np.random.choice(point_wrong)  
    ##print('--')
    ##print('length error points list',len(point_wrong))
    ##print(idx) 
    ##print('start',W,b)
    #print('X[idx],y_pred[idx],y[idx]',X[idx],y_pred[idx],y[idx])   #np.matmul(y[idx], X[idx])
    #print(y[idx]*X[idx])
    #W=W.reshape(1,2)
    Xnew=np.zeros((2,1))
    for j in range(np.shape(X)[1]):
        Xnew[j]=X[idx][j]
    #print('**')    
    #print(X[idx])
    ##print('Xnew',Xnew)
    #X[idx]=X[idx].reshape(1,2)
    if(y[idx]>0):
        W= W+learn_rate*Xnew    #X[idx]                               #y[idx]*X[idx]    
        b= b+learn_rate
    else:
        W= W-learn_rate*Xnew       #X[idx]                               #y[idx]*X[idx]    
        b= b-learn_rate       
    ##print('Final',W,b)
    ##print('--')
    #print(y_pred)
    #print(y)
    # Fill in code
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 1.0, num_epochs = 25):
    ##print(len(y))
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    #W=np.random.randn(np.shape(X[1]))
    ##print(W)
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
