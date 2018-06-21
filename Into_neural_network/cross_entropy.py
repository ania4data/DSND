import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    a=0.0
    for i in range(len(Y)):
        
        a -=Y[i]*np.log(P[i])+(1.-Y[i])*np.log(1.-P[i])
        
    return a


# or
import numpy as np

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))