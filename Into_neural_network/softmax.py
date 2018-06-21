import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.

def softmax(L):
    #L=[1,2,3]
    A=np.exp(L)
    B=np.sum(A)
    return A/B    #softmax

def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)   # no need to do append, since A/B resturn list
    return result
