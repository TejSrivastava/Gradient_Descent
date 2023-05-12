# Gradient descent for Lineat Regression
# y=wx+b
# loss=(y-yhat)**2 /N
import numpy as np
#Initialzize some parameters
x =np.random.rand(10,1)
y =2*x+np.random.rand()
#Paramteres
w =0.0
b =0.0
#HyperParameter
l_r=0.01


#Create gradient descend function
def descend(x,y,w,b,l_r):
    dldw=0.0
    dldb=0.0
    N=x.shape[0]
    #loss=(y-(wx+b))**2
    for xi, yi in zip(x,y):
        
        dldw=-2*xi*(yi-(w*xi+b))
        dldb=-2*(yi-(w*xi+b))
    
    w = w -l_r*(1/N)*dldw
    b = b -l_r*(1/N)*dldb  
    
    return w,b

#Iteratively make updates
for epoch in range(400):
    w,b= descend(x,y,w,b,l_r)
    yhat=w*x + b
    loss=np.divide(np.sum((y-yhat)**2,axis=0),x.shape[0])
    print(f'{epoch} loss is {loss}, parameter w: {w}, b:{b}')
    
    