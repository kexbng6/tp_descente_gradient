#
#
# # Algorithme: (Descente de Gradient)
# # Choisir un point initial x0 (arbitraire)
# # Répéter :
# # 1. Déterminer une direction de descente : -∇f ( k )
# # 2. Choisir une taille de pas λk > 0
# # 3. Actualiser xk+1 = xk − λk · ∇f (xk)
# # 4. Tester le critère d'arrêt (est-ce qu'on a trouvé un minimum ?)
#
#
# def test_derive(x):
#     fct = lambda x: x ** 3
#     test2 = mfdg.deriv(fct, x)
#     print(test2)
#     # print(d_test1)
#     # print(deriv_num)
#
# def steepest_descend(f, x0, lambda_k, momentum, tolerance, maxIter):
#     # gradient_descent2DMomentum_MaxNorm(f, gradient, pointInitial, learning, momentum, tolerance, maxIter,maxNormGradient) #TODO -> paramètre fonction exemple "maxNormGradient" ?
#     df = pd.DataFrame(columns=["step", "x0", "df", "λk", "norm ∇f"])
#
#     for k in range(maxIter):
#         step = lambda_k * mfdg.gradient_1D(f, x0)
#         x0 = x0 - step
#         df.loc[k, :] = [step, x0, mfdg.deriv(f, x0), lambda_k, np.abs(mfdg.gradient_1D(f, x0))]
#
#         if (np.abs(k + 1 - x0) < tolerance):  # TODO -> Doute sur la condition d'arret
#             break
#     df = (df.reset_index()
#           .rename(columns={'index': 'nb_iter'}))
#
#     print(df)
#     # print(df.head(5))
#     # print(df.tail(5))
#     # print(df.describe())
#
#
# def steepest_descend_2D(f, lambda_k, momentum, tolerance, maxIter):
#     # gradient_descent2DMomentum_MaxNorm(f, gradient, pointInitial, learning, momentum, tolerance, maxIter,maxNormGradient) #TODO -> paramètre fonction exemple "maxNormGradient" ?
#     df = pd.DataFrame(
#         columns=["step", "x", "y", "df_x", "df_y", "λk", "norm ∇f"])
#     vector = mfdg.point_init
#     #test = mfdg.gradient_2D(f,vector[0],vector[1])
#     #print(test)
#     #normGrad = np.linalg.norm(mfdg.gradient_2D(f,vector[0],vector[1]))
#     grad = mfdg.gradient_2D(f,vector[0],vector[1])
#     normGrad = np.linalg.norm(grad)
#     print(normGrad)
#     print(grad)
#     for k in range(maxIter):
#         step = lambda_k * grad
#         vector = vector - step
#         df.loc[k, :] = [step, vector[0], vector[1], mfdg.deriv(f, vector[0]), mfdg.deriv(f, vector[1]), lambda_k,
#                         np.abs(grad)]
#
#         if (np.all(np.abs(normGrad) < tolerance)):  # TODO -> Doute sur la condition d'arret
#             df.loc[k + 1, :] = [step, vector[0], vector[1], mfdg.deriv(f, vector[0]), mfdg.deriv(f, vector[1]),
#                                 lambda_k,
#                                 np.abs(grad)]
#             break
#     #df = (df.reset_index().rename(columns={'index': 'nb_iter'}))
#
#     print(df)
#     # print(df.head(5))
#     # print(df.tail(5))
#     # print(df.describe())
#
# ################################################################################


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

"""
Données d'initisalisation pour les calculs
"""

"""
Fonction tester
"""
fct = lambda x, y: x ** 3 + 2 * y ** 2
boothFunction = lambda x, y: (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2  # min global -> f(1,3) = 0
himmelblauFunction = lambda x, y: (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2  # min global -> f(3.0,2.0) = 0
McCormickFunction = lambda x, y: np.sin(x + y) + (
            x - y) ** 2 - 1.5 * x + 2.5 * y + 1  # min global -> f(-0.54719, -1.54719 = -1.9133


########################################################################################################################

def numericalDerivative(f, x, y, eps):
    """
    Fonction de dérivée numérique
    :param f: Fonction
    :param x: Point x
    :param y: Point Y
    :param eps: epsilon
    :return: Array de la dérivée de X et Y
    """
    gx = (f(x + eps, y) - f(x - eps, y)) / (2 * eps)
    gy = (f(x, y + eps) - f(x, y - eps)) / (2 * eps)
    return np.array([gx, gy])


def gradient_descent2D(cost, start, learn_rate, tolerance, n_iter):
    """
    Fonction de descente de gradient simple
    :param cost: Coût
    :param start: Point de départ X et Y
    :param learn_rate: Fréquence d'apprentissage
    :param tolerance: Conditions d'arrêt de la boucle
    :param n_iter: Nombre maximum d'itération
    :return: tableau contentenant toutes les données du calculs
    """
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
    # print(df)
    return df


def gradient_descent2DMomentum(cost, start, learn_rate, momentum, tolerance, n_iter):
    """
    Fonction de descente de gradient avec Momentum
    :param cost: Coût
    :param start: Point de départ
    :param learn_rate: Fréquence d'apprentissage
    :param momentum: Momentum
    :param tolerance: Condition d'arrêt
    :param n_iter: Nombre d'itération maxmimum
    :return: tableau contentenant toutes les données du calculs
    """
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
    return df


def gradient_descent2D_AdAM(cost, start, learn_rate, momentum, tolerance, n_iter):
    """
    Fonction de descente de gradient avec Momentum
    :param cost: Coût
    :param start: Point de départ
    :param learn_rate: Fréquence d'apprentissage
    :param momentum: Momentum
    :param tolerance: Condition d'arrêt
    :param n_iter: Nombre d'itération maxmimum
    :return: tableau contentenant toutes les données du calculs
    """
    df = pd.DataFrame(columns=["Iteration", "X", "Y", "Cost", "Stepx", "Stepy", 'NormGrad'])
    vector = start
    grad = numericalDerivative(cost, vector[0], vector[1], 1e-05)
    normGrad = np.linalg.norm(grad)
    step = np.array([0, 0])
    # mk = 0  # moyenne mobile
    mk = np.array([0, 0])
    vk = np.array([0, 0])
    t = 0
    eps = 1e-8
    # vk = 0  # moyenne mobile

    for k in range(n_iter):
        t += 1
        # grad = numericalDerivative(cost, vector[0], vector[1], 1e-05)
        grad = numericalDerivative(cost, vector[0], vector[1], eps)
        mk = b1 * mk + (1 - b1) * grad  # β1mk−1 + (1 − β1)∇fk
        vk = b2 * vk + (1 - b2) * grad ** 2  # β2vk−1 + (1 − β2)∇fk ⊙ ∇fk
        m_corr = mk / (1 - b1 ** t)
        v_corr = vk / (1 - b2 ** t)
        step = learn_rate * m_corr / (np.sqrt(v_corr) + eps)
        # step = step - learn_rate * mk #learn_rate * grad + momentum * step
        vector = vector - step
        normGrad = np.linalg.norm(grad)
        df.loc[k] = [k, vector[0], vector[1], cost(vector[0], vector[1]), step[0], step[1], normGrad]
        if np.all(np.abs(normGrad) <= tolerance):
            df.loc[k + 1] = [k + 1, vector[0], vector[1], cost(vector[0], vector[1]), step[0], step[1], normGrad]
            break
    return df


########################################################################################################################

def affichage(resultDF, function):
    """
    Fonction de génération des graphique
    :param resultDF: Tableau générer DF par les fonctions de calculs
    """
    scale = 15
    print(resultDF)
    X, Y = np.mgrid[-scale:scale:30j, -scale:scale:30j]
    Z = function(X, Y)
    plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='jet', norm=LogNorm(), rstride=1, cstride=1, alpha=0.99)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.contour(X, Y, Z, 20, lw=3, cmap='RdGy', offset=-1)
    # ax.plot(sol['X'],sol['Y'],'-o', color='black')

    plt.figure(2)
    plt.contour(X, Y, Z, 20, lw=3, cmap='RdGy')
    plt.plot(resultDF['X'], resultDF['Y'], '-o', color='black')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.figure(3)
    plt.plot(resultDF['Iteration'], resultDF['Cost'], '-', color='black')
    plt.show()


########################################################################################################################


pointInitial = [10, 10]
momentum = 0.3
learning = 0.03
maxIter = 30000
tolerance = 1e-05
b1 = 0.9
b2 = 0.999

if __name__ == "__main__":
    # df1 = gradient_descent2DMomentum(fct, mfdg.point_init, mfdg.lambda_k, momentum, tolerance,
    #                                  mfdg.max_iter)
    # #affichage(df1,fct)

    # df2 = gradient_descent2DMomentum(himmelblauFunction, mfdg.point_init, mfdg.lambda_k, momentum, tolerance,
    #                                  mfdg.max_iter)
    # #affichage(df2,himmelblauFunction)

    # df3 = gradient_descent2DMomentum(boothFunction, mfdg.point_init, mfdg.lambda_k, momentum, tolerance,
    #                                  mfdg.max_iter)
    # #affichage(df3,boothFunction)

    df4 = gradient_descent2D_AdAM(himmelblauFunction, pointInitial, learning, momentum, tolerance, maxIter)

    affichage(df4, himmelblauFunction)