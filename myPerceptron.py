import numpy as np
from numpy.random import randn

#This function is taken from Dr Fayyaz ul Amir Afsar Minhas (Github User: foxtrotmike)
def getExamples(n=100,d=2):
    """
    Generates n d-dimensional normally distributed examples of each class        
    The mean of the positive class is [1] and for the negative class it is [-1]
    DO NOT CHANGE THIS FUNCTION
    """
    Xp = randn(n,d)+1   #generate n examples of the positive class
    #Xp[:,0]=Xp[:,0]+1
    Xn = randn(n,d)-1   #generate n examples of the negative class
    #Xn[:,0]=Xn[:,0]-1
    X = np.vstack((Xp,Xn))  #Stack the examples together to a single matrix
    Y = np.array([+1]*n+[-1]*n) #Associate Labels
    return (X,Y) 



w=getExamples()

data=w[0]
label=w[1]
lr=0.01

#Checking the loop execution
count=0
ok=False
maxRun=0

#Seperating data into weights and bias
w=np.random.random(data.shape[1])
b=np.random.random()

#Perceptron Loop
while not ok and maxRun<=1000:
    maxRun+=1
    count=0
    for i in range(len(data)):
        
        #Counting on all the example if satisfied by y*f(x) >=1 is all true then we have found the weights of perceptron
        #The code then breaks
        if label[i]*(w.T.dot(data[i])+b) >=1:
            count+=1
            
        #Else weights are updated
        else:
            w=w+lr*label[i]*data[i]
            b=b+lr*label[i]
            
        if count==len(data):
            ok=True
            
#Printing weights and bias  
print(w,b)


#############################################################
print('NOW TESTING')
l=[]
for i in range(len(data)):
    l.append(np.sign(w.T.dot(data[i])+b)==np.sign(label[i]))

print('ACCURACY : ',l.count(True)/len(l))    
