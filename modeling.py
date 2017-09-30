import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


# k = X_train.shape[1]

# params = np.zeros((len(alphas), k))
# for i,a in enumerate(alphas):
#     X_data = preprocessing.scale(X)
#     fit = Lasso(alpha=a, normalize=True).fit(X_data, y)
#     params[i] = fit.coef_
#
# fig = plt.figure(figsize=(14,6))
# for param in params.T:
#     plt.plot(alphas, param)
# plt.show()

# def rmlse(true, predicted):
#     epsilon = np.sqrt( np.mean( (np.log(predicted+1)-np.log(true + 1) )**2 ) )
#     return epsilon
#
# def cross_validation(x_data, y_data, alpha, k):
#     """
#     Parameters: x and y training data as numpy array
#     Return: Error for each iteration of cross validation
#     """
#     test_error = []
#     kfold = KFold(n_splits = k, shuffle = True)
#     #kfold.get_n_splits(x_data)
#     for train_index, test_index in kfold.split(x_data):
#         cvx_train, cvx_test = x_data[train_index], x_data[test_index]
#         cvy_train, cvy_test = y_data[train_index], y_data[test_index]
#
#         linear = LinearRegression()
#         linear.fit(cvx_train, cvy_train)
#         cvtest_predicted = linear.predict(cvx_test)
#
#         test_error.append(mean_squared_error(cvy_test, cvtest_predicted))
#
#         #coeff = linear._coeffs()
#
#     return np.mean(test_error)

def model_generator(X_train, y_train, alphas, vebest, X_test):
    alpha_best = 0
    validation_error_best = vebest
    X_data = preprocessing.scale(X_train)
    for a in alphas:
        validation_error = []
        training_error = []
        kfold = KFold(n_splits = 5)
        for train_index, test_index in kfold.split(X_data):
            cvx_train, cvx_test = X_data[train_index], X_data[test_index]
            cvy_train, cvy_test = y_train[train_index], y_train[test_index]
            lasso_fit = Lasso(alpha = a).fit(cvx_train, cvy_train)
            cvtest_predicted = lasso_fit.predict(cvx_test)
            validation_error.append(mean_squared_error(cvy_test, cvtest_predicted))

        if np.mean(validation_error) < validation_error_best:
            validation_error_best = np.mean(validation_error)
            alpha_best = a

    current_best = Lasso(alpha = alpha_best).fit(X_data, y_train)
    coeffs = current_best.coef_

    zero_betas = []
    for i, coeff in enumerate(coeffs):
        if coeff == 0:
            zero_betas.append(i)

    print zero_betas

    for index in zero_betas:
        X_train = np.delete(X_train, [index], axis = 1)
        X_test = np.delete(X_test, [index], axis = 1)

    return X_train, X_test, validation_error_best, alpha_best



if __name__ == "__main__":

    data = pd.read_csv('data/cleaned_train.csv')
    test_data = pd.read_csv('data/cleaned_test.csv')

    y_train = np.array(data['SalePrice'])
    X_train = np.array(data[['sale_age', 'UsageBand_High','MachineHoursCurrentMeter',
                        'UsageBand_Low', 'UsageBand_Medium', 'ProdGroup_BL', 'ProdGroup_MG',
                        'ProdGroup_SSL', 'ProdGroup_TEX', 'ProdGroup_TTT', 'ProdGroup_WL']])

    y_train = y_train[:100000]
    X_train = X_train[:100000]

    # y_test = np.array(test_data['SalePrice'])
    X_test = np.array(test_data[['sale_age', 'UsageBand_High','MachineHoursCurrentMeter',
                        'UsageBand_Low', 'UsageBand_Medium', 'ProdGroup_BL', 'ProdGroup_MG',
                        'ProdGroup_SSL', 'ProdGroup_TEX', 'ProdGroup_TTT', 'ProdGroup_WL']])
    alphas = np.logspace(-2, 2)
    validation_error_best = 9999999999999999999999
    alpha_best = 100
    loop1 =  model_generator(X_train, y_train, alphas,
                            validation_error_best, X_test)
    loop1xtrain, loop1xtest, validation_error_best, alpha_best = loop1
    print validation_error_best, alpha_best

    loop2 = model_generator(loop1xtrain, y_train, alphas,
                            validation_error_best, loop1xtest)
    loop2xtrain, loop2xtest, validation_error_best, alpha_best = loop2
    print validation_error_best, alpha_best

    loop3 = model_generator(loop2xtrain, y_train, alphas,
                            validation_error_best, loop2xtest)
    loop3xtrain, loop3xtest, validation_error_best, alpha_best = loop3
    print validation_error_best, alpha_best
    cv_error = cross_validation(X_train, y_train, 10)
    print np.sqrt(cv_error)
    print alpha_best
    print np.sqrt(validation_error_best)
    X_data = preprocessing.scale(X_train)
    linear = Lasso(alpha = alpha_best)
    linear.fit(X_train, y_train)
    test_predicted = linear.predict(X_test)
    #set all the negative predictions to an arbitrary min
    # for i, j in enumerate(test_predicted):
    #     if j < 0:
    #         test_predicted[i] = 1
    sales_id = np.array(test_data['SalesID'])
    test_predicted = np.column_stack((sales_id, test_predicted))
    test_predicted = pd.DataFrame({'SalesID': test_predicted[:,0], 'SalePrice': test_predicted[:,1]})
    test_predicted.to_csv('data/test_predicted.csv')
