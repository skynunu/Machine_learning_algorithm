#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


def plot_margin(X1_train, X2_train, clf):
    def f(x, w, b, c=0):
        # given x, return y such that [x,y] in on the line
        # w.x + b = c
        return (-w[0] * x - b + c) / w[1]
    plt.figure(figsize=(8,6))
    plt.plot(X1_train[:,0], X1_train[:,1], "ro")
    plt.plot(X2_train[:,0], X2_train[:,1], "bo")
    plt.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")
    
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    
    # w.x + b = 0
    y_vals = f(x_vals, clf.w, clf.b)
    plt.plot(x_vals, y_vals, 'k')
    
    # w.x + b = 1
    y_vals = f(x_vals, clf.w, clf.b,1)
    plt.plot(x_vals, y_vals, 'k--')

    # w.x + b = -1
    y_vals = f(x_vals, clf.w, clf.b,-1)
    plt.plot(x_vals, y_vals, 'k--')
    
    plt.axis("tight")
    plt.show()
    


# In[3]:


def plot_contour(X1_train, X2_train, clf):
    
    plt.figure(figsize=(8,6))
    
    #dataset plot 하기
    plt.plot(X1_train[:,0], X1_train[:,1], "ro")
    plt.plot(X2_train[:,0], X2_train[:,1], "bo")
    plt.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")
    
    X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    
    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')
    
    plt.axis("tight")
    plt.show()


# In[ ]:





# In[ ]:




