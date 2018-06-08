
#3x1+ 4x2 - 10 = 0
x1=1.0
x2=1.0
#(1, 1)  is a positive category y>0
alpha=0.1
w1=3.0
w2=4.0
b0=-10.0
for i in range(20):

    w1 =w1+alpha*x1
    w2 =w2+alpha*x2
    b0 =b0+alpha
    print(i,w1*x1+w2*x2+b0)
