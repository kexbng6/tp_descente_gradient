

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