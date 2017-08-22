import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


mean = [1, 1]
std = [[0.05,0], [0,0.05]]

def linearclassifier(X, Y, reg):

    #Create X Augmented Vector
    dummyX = np.array([1.0 for i in range(X.shape[0])])
    dummyX = np.reshape(dummyX, (dummyX.shape[0],1 ))
    X = np.concatenate([dummyX , X], axis = 1)
    
    ##Solution with regulatization
    Xpseudo = np.matmul(np.linalg.inv(reg*np.identity(np.matmul(X.T,X).shape[0]) + np.matmul(X.T, X)),X.T)
    W = np.matmul(Xpseudo, Y)


    #Separate data into classes for plotting
    Xdata = pd.DataFrame(np.concatenate([X, np.reshape(Y, (Y.shape[0],1))],axis =1))
    class1 = Xdata[Xdata[3]==1.0]
    class2 = Xdata[Xdata[3]==-1.0]
    class1 = class1[[1,2]].as_matrix()
    class2 = class2[[1,2]].as_matrix()
    plt.scatter(class1[:,0], class1[:, 1], s = 1)   
    plt.scatter(class2[:,0], class2[:, 1], s = 1)

    x = np.linspace(0,3, num = 100)

    #Line equation of the discriminant function
    y = (-1.0/W[2])*(W[0] + W[1]*x)

    plt.plot(x, y)
    plt.xlim((X[:,1].min()-0.2,X[:,1].max()+0.2))
    plt.ylim((X[:,2].min()-0.2,X[:,2].max()+0.2))
    plt.show()
    return (W,Xpseudo)


def linearMachine(X, T, reg):
    """
    Linear Machine for Three Classesi

    T needs to be in matrix form, where t_n  is the label vector of the sample
    t_n =[1, 0, 0] means that it belongs to the first class

    """
    
    dummyX = np.array([1.0 for i in range(X.shape[0])])
    dummyX = np.reshape(dummyX, (dummyX.shape[0],1 ))
    X = np.concatenate([dummyX , X], axis = 1)


    Xpseudo = np.matmul(np.linalg.inv(reg*np.identity(np.matmul(X.T,X).shape[0]) + np.matmul(X.T, X)),X.T)
    W = np.matmul(Xpseudo, T) 
    
    #LINE EQUATIONS OF DISCRIMINANT FUNCTIONS
    
    x = np.linspace(-10,10, num = 100)
    y1 = (-1.0/W[2,0])*(W[0,0] + W[1,0]*x)
    y2 = (-1.0/W[2,1])*(W[0,1] + W[1,1]*x)
    y3 = (-1.0/W[2,2])*(W[0,2] + W[1,2]*x)

    plt.scatter(X[:,1], X[:,2], s = 1)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.xlim((X[:,1].min()-0.2,X[:,1].max()+0.2))
    plt.ylim((X[:,2].min()-0.2,X[:,2].max()+0.2))
    plt.show()

    return (W, Xpseudo)



def makering(r, thickness):
    theta = []
    radius = []
    for i in range(1000):
        theta.append(random.uniform(0,2*np.pi))
        radius.append(random.uniform(r, r + thickness))
    
    xvals = []
    yvals = []
    for i in range(len(theta)):
        xvals.append(radius[i]*np.cos(theta[i]))
        yvals.append(radius[i]*np.sin(theta[i]))
    
    return (np.array(xvals), np.array(yvals))



#GENERATE LINEARLY SEPARABLE CLASSES

w1 = np.random.multivariate_normal([1, 2], std, 1000);
w2 = np.random.multivariate_normal([2, 1], std, 1000);
w3 = np.random.multivariate_normal([3,2], std, 1000);


# GENERATE NOT LINEARLY SEPARABLE CLASSES
xring1, yring1 = makering(0.5, 0.5)
xring2, yring2 = makering(1.5, 0.5)
xring3, yring3 = makering(2.5, 0.5)


xring1 = np.reshape(xring1, (xring1.shape[0],1))
xring2 = np.reshape(xring2, (xring2.shape[0],1))
xring3 = np.reshape(xring3, (xring3.shape[0],1))
yring1 = np.reshape(yring1, (yring1.shape[0],1))
yring2 = np.reshape(yring2, (yring2.shape[0],1))
yring3 = np.reshape(yring3, (yring3.shape[0],1))


ring1 = np.concatenate([xring1, yring1], axis = 1)
ring2 = np.concatenate([xring2, yring2], axis = 1)
ring3 = np.concatenate([xring3, yring3], axis = 1)


labelsw1 = np.array([1.0 for i in w1])
labelsw2 = np.array([-1.0 for i in w2])
labels = np.concatenate([labelsw1, labelsw2])


Xdata2 = np.concatenate([w1, w2])


#LINEAR CLASSIFIER linearclassifier(X, y, lambda)

W, Xpseudo = linearclassifier(Xdata2, labels, 1)
W, Xpseudo = linearclassifier(Xdata2, labels, 100)


Xdata3 = np.concatenate([w1, w2, w3])
labelsw1 = np.array([np.array([1.0,0,0]) for i in w1])
labelsw2 = np.array([np.array([0,1.0,0]) for i in w2])
labelsw3 = np.array([np.array([0,0,1.0]) for i in w3])
labels = np.concatenate([labelsw1, labelsw2, labelsw3])

#LINEAR MACHINE, SEPARABLE
W, Xpseudo = linearMachine(Xdata3, labels, 5)

#LINEAR CLASSIFIER, NO SEPRABLE

labelsr1 = np.array([1.0 for i in ring1])
labelsr2 = np.array([-1.0 for i in ring2])
#labelsr3 = np.array([1.0 for i in ring3])
labelsring = np.concatenate([labelsr1, labelsr2])

Xrings = np.concatenate([ring1, ring2])
W, Xpseudo = linearclassifier(Xrings, labelsring, 0)


"""
Esta matriz X.T*X se acerca a la singularidad, su determinante es cero 
debido a esto no se puede clasificar.
"""

#LINEAR MACHINE, NO SEPARABLE

Xring3 = np.concatenate([ring1, ring2, ring3])
labelsr1 = np.array([np.array([1.0,0,0]) for i in ring1])
labelsr2 = np.array([np.array([0,1.0,0]) for i in ring2])
labelsr3 = np.array([np.array([0,0,1.0]) for i in ring3])
labelsring = np.concatenate([labelsr1, labelsr2, labelsr3])

W, Xpseudo = linearMachine(Xring3, labelsring, 1)

