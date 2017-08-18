import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


mean = [1, 1]
std = [[0.05,0], [0,0.05]]

def linearclassifier(X, Y):
	tri = np.array([1.0 for i in range(X.shape[0])])
	tri = np.reshape(tri, (tri.shape[0],1 ))
	X = np.concatenate([tri , X], axis = 1)
	Xpseudo = np.matmul(np.linalg.inv(np.matmul(X.T, X)),X.T)
	W = np.matmul(Xpseudo, labels)
	
	Xdata = pd.DataFrame(np.concatenate([X, np.reshape(Y, (Y.shape[0],1))],axis =1))
	class1 = Xdata[Xdata[3]==1.0]
	class2 = Xdata[Xdata[3]==-1.0]
	class1 = class1[[1,2]].as_matrix()
	class2 = class2[[1,2]].as_matrix()
	plt.scatter(class1[:,0], class1[:, 1], s = 1)
	plt.scatter(class2[:,0], class2[:, 1], s = 1)
	
	x = np.linspace(0,3, num = 100)
	y = (-1.0/W[2])*(W[0] + W[1]*x)
	plt.plot(x, y)
	plt.xlim((0,5))
	plt.ylim((0,3))
	plt.show()
	return W


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
    
    return (xvals, yvals)



#generating linearly separable classes

w1 = np.random.multivariate_normal([1, 2], std, 1000);
w2 = np.random.multivariate_normal([2,1], std, 1000);
w3 =  np.random.multivariate_normal([3,2], std, 1000);

# GENERATE NOT LINEARLY SEPARATE CLASSES
xring1, yring1 = makering(0.5, 0.5)
xring2, yring2 = makering(1.5, 0.5)
xring3, yring3 = makering(2.5, 0.5)

labelsw1 = np.array([1.0 for i in w1])
labelsw2 = np.array([-1.0 for i in w2])
labels = np.concatenate([labelsw1, labelsw2])

Xdata = np.concatenate([w1, w2])

W = linearclassifier(Xdata, labels)

