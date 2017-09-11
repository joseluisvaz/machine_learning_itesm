#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Andres Mendez-Vazquez"
__copyright__ = "Copyright 2017 Sampler Project"
__credits__ = ["Andres Mendez-Vazquez"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Andres Mendez-Vazquez"
__email__ = "amendez@gdl.cinvestav.mx"
__status__ = "Development"

import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt

class Distribution(object):
  """
  Base Distribution for the Samplers
  """
  def __init__(self,precision = 0.001):
    """
    Default Constructor
    """
    # The handler for the samples 
    self.__samples = np.random.uniform(0,1,1000)
    # The precision of the Samplers
    self.__precision = precision
    # The values for the pdf
    self.__ran = None # Variable for x          
    self.__y   = None # Variable for p(x)    
    # Unifrom Distribution
    def uniform(x):
      return st.uniform.pdf(x)
    # Setting Default p, the List of tuples with 
    # First Element    = list of distributions
    # Second parameter = Range of the x min to x max     
    self.__p = [(uniform),(-1.0,2.0)] 

  def set_samples(self,samples):
    """
    Setter for samples
    """    
    self.__samples = samples
  
  def get_samples(self):
    """
    Return samples
    """
    return self.__samples

  def get_p(self):
    """
    Get the inner values of p
    """
    return self.__p[0], self.__p[1]
  
  def set_p(self,p,lh):
    """
    Set the parameters for the p
    """    
    self.__p = [p,lh]
  
  def get_precision(self):
    """
    The getter of __precison
    """
    return self.__precision
    
  def set_precision(self, precision):
    """
    The setter of __precison
    """
    self.__precision = precision
  
  def get_ran(self):
    return self.__ran

  def set_ran(self,x):
    self.__ran = x

  def get_y(self):
    return self.__y

  def set_y(self,y):
    self.__y = y     
  
  def DisplayDistribution(self):
    """
    Displaying Univariate Distribution 
    """
    plt.figure()
    p , lh = self.get_p()
    sns.distplot(self.get_samples(), kde=False, norm_hist=False)
    t1 = lh[0]
    t2 = lh[1]      
    x = np.arange(t1,t2,self.get_precision())
    plt.plot(x, p(x))
    plt.show(block=False)

  
  def DisplayMarkovChain(self):
    """
    Display the Markov Chain
    """
    samples = self.get_samples()
    plt.figure()
    N = len(samples)
    plt.plot(range(N),samples)
    plt.show(block=False)
    

class SliceSampler(Distribution):
  """
  Class Implementing Different Distributions
  """  
  def __init__(self,precision = 0.001):
    """
    The default constructor
    """
    super(SliceSampler,self).__init__(precision)
    np.random.seed()
    
  def DisplayDistribution(self):
    """
    Displaying Univariate Distribution 
    """
    area = np.trapz(self.get_y(),self.get_ran(),axis=0)
    fig , subplt = plt.subplots(nrows=2, figsize=(8, 9))
    subplt[0].hist(self.get_samples(),\
             bins = self.get_ran())        
    subplt[1].plot(self.get_ran(), (1.0/area)*self.get_y(),\
            lw=2,color='green')
    plt.show()

  def Display_List_Dist(self):
    """
    Display List of Distributions
    """
    # Get the neccessary distributions
    p,lh =  self.get_p()
    low  = lh[0]
    high = lh[1]
    N = len(p)
    clr = ['g','c','b','r']    
    fig , subplt = plt.subplots(nrows=N, figsize=(8, 9))
    x_grid = np.arange(low,high,self.get_precision())
    for i in range(N):
      subplt[i].plot(x_grid,p[i](x_grid),\
                     clr[i%4], linewidth=2.5,\
                     label = 'PDF {}'.format(i))
      subplt[i].legend()
    plt.show(block = False)
 
  
  def A_multiple_sample(self,p,w):
    """
    Multiple A for General Slice Sampler
    """
    ran = self.get_ran()
    first = set(ran)    
    for i,f in enumerate(list(p)):
      y_i   = f(self.get_ran())
      A_i   = ran[w[i]<y_i]
      s_A_i = set(A_i)
      first=first.intersection(s_A_i)
    A = list(first)    
    N = len(A)
    i = np.random.randint(0,N)  
    return A[i]
            
        
  def General_Slice_Sampler(self,itera=1000,showp = 10):
    """
    The General Slice Sampler
    INPUT: Number of iterations for the slice sampler
    """
    samples = np.zeros(itera)
    x=0.0
    # Get the neccessary distributions        
    p, lh = self.get_p() 
    low  =  lh[0]
    high =  lh[1] 
    self.set_ran(np.arange(low,high,self.get_precision()))
    fd = np.ones(len(self.get_ran()))
    for f in list(p):
      fd = fd*f(self.get_ran())
    self.set_y(fd)
    fN = len(p)
    # Loop for iter
    for i in range(itera):
      # Loop in case of an emprty intersection
      if itera > showp:      
        if i%(itera/showp) ==0:
          print ("Iteration General Slice Sampler" + str(i))
      while True:
        w = list()
        # Loop for the w
        for j in range(fN):
          w.append(np.random.uniform(0, p[j](x)))
        x = self.A_multiple_sample(p,w)
        # Handling empty case
        if x != None:
          samples[i] = x
          break
    self.set_samples(samples)
      
  def SampledExpectedValue(self):
    """
    Expected Value of the samples
    """
    samples = self.get_samples()
    N = len(samples)
    Sumf = np.sum(samples)
    return (1.0/float(N))*Sumf
    
  def Set_Distribution(self,mu=0.0,sigma=1.0):
    """
    The test distributions for the slice sampler decomposition
    """    
    def p1(x):
      return np.exp(-(x-mu)**2/(2*(sigma**2)))
    def p2(x):
      to_return = []
      if type(x) == np.float64 or type(x) == float :
        if x >= 0 and x <= 1:
          return 1
        else:
          return 0
      else:
        for i in x:
          if i >= 0 and i <= 1:
            to_return.append(1)
          else:
            to_return.append(0)
        return np.array(to_return)

    self.set_p((p1, p2),(0,mu+sigma))

if __name__ == '__main__':  

  F2 = SliceSampler(precision = 0.01)
  F2.Set_Distribution(mu=2.0,sigma=2.0)
  F2.Display_List_Dist()  
  F2.General_Slice_Sampler(itera=10000)
  F2.DisplayDistribution()
  print ('Expected Value {}'.format(F2.SampledExpectedValue()))

    
