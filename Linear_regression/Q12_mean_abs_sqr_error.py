
import numpy as np 

X=[[2, -2], [5, 6], [-4, -4], [-7, 1], [8, 14]]
X=np.array(X)
#y=1.2*x + 2
print(X)
print(X[:,0])

mean_abs_error=0
mean_sqr_error=0
for i in range(len(X)):

    ypred=1.2*X[i,0] + 2.0
    y=X[i,1]
    mean_abs_error += abs(y-ypred)
    mean_sqr_error += (y-ypred)**2.0

mean_abs_error =mean_abs_error/len(X)
mean_sqr_error =mean_sqr_error/len(X)   #in lecture divided by 2*m but correct answer of quiz is by m

print(len(X))
print(mean_abs_error,mean_sqr_error)


   



