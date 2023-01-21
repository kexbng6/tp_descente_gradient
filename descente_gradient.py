import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math_function_derivative as mfd
import math_function_descente_gradient as mfdg


# Algorithme: (Descente de Gradient)
# Choisir un point initial x0 (arbitraire)
# Répéter :
# 1. Déterminer une direction de descente : -∇f ( k )
# 2. Choisir une taille de pas λk > 0
# 3. Actualiser xk+1 = xk − λk · ∇f (xk)
# 4. Tester le critère d'arrêt (est-ce qu'on a trouvé un minimum ?)


def test_derive(x):
    fct = lambda x: x ** 3
    test2 = mfdg.deriv(fct, x)
    print(test2)
    # print(d_test1)
    # print(deriv_num)

def steepest_descend(f, x0, lambda_k, momentum, tolerance, maxIter):
    # gradient_descent2DMomentum_MaxNorm(f, gradient, pointInitial, learning, momentum, tolerance, maxIter,maxNormGradient) #TODO -> paramètre fonction exemple "maxNormGradient" ?
    df = pd.DataFrame(columns=["step", "x0", "df", "λk", "norm ∇f"])

    for k in range(maxIter):
        step = lambda_k * mfdg.gradient_1D(f, x0)
        x0 = x0 - step
        df.loc[k, :] = [step, x0, mfdg.deriv(f, x0), lambda_k, np.abs(mfdg.gradient_1D(f, x0))]

        if (np.abs(k + 1 - x0) < tolerance):  # TODO -> Doute sur la condition d'arret
            break
    df = (df.reset_index()
          .rename(columns={'index': 'nb_iter'}))

    print(df)
    # print(df.head(5))
    # print(df.tail(5))
    # print(df.describe())


def steepest_descend_2D(f, lambda_k, momentum, tolerance, maxIter):
    # gradient_descent2DMomentum_MaxNorm(f, gradient, pointInitial, learning, momentum, tolerance, maxIter,maxNormGradient) #TODO -> paramètre fonction exemple "maxNormGradient" ?
    df = pd.DataFrame(
        columns=["step", "x", "y", "df_x", "df_y", "λk", "norm ∇f"])
    vector = mfdg.point_init
    #test = mfdg.gradient_2D(f,vector[0],vector[1])
    #print(test)
    #normGrad = np.linalg.norm(mfdg.gradient_2D(f,vector[0],vector[1]))
    grad = mfdg.gradient_2D(f,vector[0],vector[1])
    normGrad = np.linalg.norm(grad)
    print(normGrad)
    print(grad)
    for k in range(maxIter):
        step = lambda_k * grad
        vector = vector - step
        df.loc[k, :] = [step, vector[0], vector[1], mfdg.deriv(f, vector[0]), mfdg.deriv(f, vector[1]), lambda_k,
                        np.abs(grad)]

        if (np.all(np.abs(normGrad) < tolerance)):  # TODO -> Doute sur la condition d'arret
            df.loc[k + 1, :] = [step, vector[0], vector[1], mfdg.deriv(f, vector[0]), mfdg.deriv(f, vector[1]),
                                lambda_k,
                                np.abs(grad)]
            break
    #df = (df.reset_index().rename(columns={'index': 'nb_iter'}))

    print(df)
    # print(df.head(5))
    # print(df.tail(5))
    # print(df.describe())

################################################################################

def numericalDerivative(f, x, y, eps):
    gx = (f(x + eps, y) - f(x - eps, y)) / (2 * eps)
    gy = (f(x, y + eps) - f(x, y - eps)) / (2 * eps)
    return np.array([gx, gy])

def gradient_descent2D(cost, start, learn_rate, tolerance, n_iter):
    df = pd.DataFrame(columns=["Iteration", "X", "Y", "Cost", "Stepx", "Stepy", 'NormGrad'])
    vector = start
    grad = numericalDerivative(cost, vector[0], vector[1], 1e-05)
    normGrad = np.linalg.norm(grad)
    step = np.array([0, 0])

    for k in range(n_iter):
        df.loc[k] = [k, vector[0], vector[1], cost(vector[0], vector[1]), step[0], step[1], normGrad]
        grad = numericalDerivative(cost, vector[0], vector[1], 1e-05)
        step = learn_rate * grad
        vector = vector - step
        normGrad = np.linalg.norm(grad)

        if np.all(np.abs(normGrad) <= tolerance):
            df.loc[k + 1] = [k + 1, vector[0], vector[1], cost(vector[0], vector[1]), step[0], step[1], normGrad]
            break
    print(df)
    #return df

def gradient_descent2DMomentum(cost, start, learn_rate, momentum, tolerance, n_iter):
    df = pd.DataFrame(columns=["Iteration", "X", "Y", "Cost", "Stepx", "Stepy", 'NormGrad'])
    vector = start
    grad = numericalDerivative(cost, vector[0], vector[1], 1e-05)
    normGrad = np.linalg.norm(grad)
    step = np.array([0, 0])

    for k in range(n_iter):

        df.loc[k] = [k, vector[0], vector[1], cost(vector[0], vector[1]), step[0], step[1], normGrad]
        grad = numericalDerivative(cost, vector[0], vector[1], 1e-05)
        step = learn_rate * grad + momentum * step
        vector = vector - step
        normGrad = np.linalg.norm(grad)

        if np.all(np.abs(normGrad) <= tolerance):
            df.loc[k + 1] = [k + 1, vector[0], vector[1], cost(vector[0], vector[1]), step[0], step[1], normGrad]
            break

    # sol = df
    # # tester le momentum 0.9
    # scale = 30
    # X, Y = np.mgrid[-scale:scale:30j, -scale:scale:30j]
    # Z = (X ** 2 + 2 * Y ** 2)
    # figure2 = plt.figure(1)
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='pink', rstride=1, cstride=1, alpha=0.8)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.contour(X, Y, Z, 20, lw=3, cmap='RdGy', offset=-1)
    # # ax.plot(sol['X'],sol['Y'],'-o', color='black')
    #
    # figure2 = plt.figure(2)
    # plt.contour(X, Y, Z, 20, lw=3, cmap='RdGy')
    # plt.plot(sol['X'], sol['Y'], '-o', color='black')
    # plt.xlabel('x')
    # plt.ylabel('y')
    #
    # figure2 = plt.figure(3)
    # plt.plot(sol['Iteration'], sol['Cost'], '-', color='black')

    return df

fct = lambda x, bg: x ** 3 + 2 * bg ** 2

pointInitial = [20, 30]
momentum = 0
learning = 0.03
maxIter = 3000
tolerance = 1e-05

# sol = gradient_descent2DMomentum(fct,pointInitial, learning, momentum, tolerance, maxIter)
# print(sol.head(10).to_markdown())
# print(sol.tail(10).to_markdown())

scale = 30
X, Y = np.mgrid[-scale:scale:30j, -scale:scale:30j]
Z = (X ** 2 + 2 * Y ** 2)
figure2 = plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='pink', rstride=1, cstride=1, alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.contour(X, Y, Z, 20, lw=3, cmap='RdGy', offset=-1)
#ax.plot(sol['X'],sol['Y'],'-o', color='black')



if __name__ == "__main__":
    # test_derive(3)
    #fct = lambda x: x**3

    boothFunction = lambda x,y: (x+2*y-7)**2 + (2*x+y-5)**2
    himmelblauFunction = lambda x,y: (x**2 + y -11)**2 + (x+y**2-7)**2

    #https: // en.wikipedia.org / wiki / Test_functions_for_optimization
    # steepest_descend(fct,2,mfdg.lambda_k, 0, mfd.epsilon, mfdg.max_iter)
    #steepest_descend_2D(fct, mfdg.lambda_k, 0, mfdg.epsilon, mfdg.max_iter)
    sol = gradient_descent2DMomentum(himmelblauFunction, mfdg.point_init, mfdg.lambda_k, momentum, tolerance, mfdg.max_iter)
    # tester le momentum 0.9
    scale = 30
    X, Y = np.mgrid[-scale:scale:30j, -scale:scale:30j]
    Z = (X ** 2 + 2 * Y ** 2)
    figure2 = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='pink', rstride=1, cstride=1, alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.contour(X, Y, Z, 20, lw=3, cmap='RdGy', offset=-1)
    # ax.plot(sol['X'],sol['Y'],'-o', color='black')

    figure2 = plt.figure(2)
    plt.contour(X, Y, Z, 20, lw=3, cmap='RdGy')
    plt.plot(sol['X'], sol['Y'], '-o', color='black')
    plt.xlabel('x')
    plt.ylabel('y')

    figure2 = plt.figure(3)
    plt.plot(sol['Iteration'], sol['Cost'], '-', color='black')
    plt.show()
    #gradient_descent2DMomentum(fct, mfdg.point_init, mfdg.lambda_k, momentum, tolerance, mfdg.max_iter)
