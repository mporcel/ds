import math
import numpy as np
 
def sigmoid(x):
    return 1./(1+math.exp(-x))
   
def xor(x):
    return x[0] ^ x[1]
 
def N(w,input):
    w02, w12, b2, w03, w13, b3, w24, w34, b4 = w
    X,Y = input
    N2 = sigmoid(X * w02 + Y * w12 + b2)
    N3 = sigmoid(X * w03 + Y * w13 + b3)
    N4 = sigmoid(N2 * w24 + N3 * w34 + b4) 
    return N4
   
def epsilon(w):
    return np.sum([(N(w, input) - xor(input))**2 for input in [[0,0],[0,1],[1,0],[1,1]]])  
    
def gradient_descent( f, x_start, h, eta, n_steps):
    n = len(x_start)
    x = x_start
    e = np.identity(n)
    for i in range(n_steps):
        x -= eta * np.array([(f(x + h * e[j]) - f(x))/h for j in range(n)])
    return x
   
 
def deep_learn(w_start, n_steps):
    return gradient_descent(epsilon, w_start, 0.00001, 0.5, n_steps)
