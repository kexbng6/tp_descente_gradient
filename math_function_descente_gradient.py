import numpy as np

h = 1e-9
epsilon = 1e-6
lambda_k = 0.15 #taille de pas λk > 0 (learning rate)
max_iter = 3000
point_init = np.array([12,69])

deriv = lambda f,x: (f(x+h) - f(x))/h
def deriv_num(f, x, y):
    return np.array(deriv(f,x),deriv(f,y))

gradient_1D = lambda f,x: np.array(deriv(f,x))
gradient_2D = lambda f,x,y: np.array(deriv(f,x), deriv(f,y))