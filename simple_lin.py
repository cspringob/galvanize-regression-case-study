import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf



data = pd.read_csv('data/cleaned.csv')

#data['sale_age'] = data['saledate'].dt.year - data['YearMade']

y = np.array(data['SalePrice'])
x = np.array(data[['sale_age','MachineID']])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# first_pass_est_sale = smf.ols(formula = 'SalePrice ~ sale_age', data = x_train[x_train['sale_age']< 200]).fit()
# print first_pass_est_sale.summary()

linear = LinearRegression()
linear.fit(x_train, y_train)

train_predicted = linear.predict(x_train)
test_predicted = linear.predict(x_test)

def rmlse(true, predicted):
    epsilon = np.sqrt( np.mean( (np.log(predicted+1)-np.log(true + 1) )**2 ) )
    return epsilon
#
def cross_validation(x_data, y_data, k):
    """
    Parameters: x and y training data as numpy array
    Return: Error for each iteration of cross validation
    """
    test_error = []
    kfold = KFold(n_splits = k, shuffle = True)
    #kfold.get_n_splits(x_data)
    for train_index, test_index in kfold.split(x_data):
        cvx_train, cvx_test = x_data[train_index], x_data[test_index]
        cvy_train, cvy_test = y_data[train_index], y_data[test_index]

        linear = LinearRegression()
        linear.fit(cvx_train, cvy_train)
        cvtest_predicted = linear.predict(cvx_test)

        test_error.append(rmlse(cvy_test, cvtest_predicted))

        #coeff = linear._coeffs()

    return np.mean(test_error)

if __name__ == "__main__":
    cross_val = cross_validation(x_train, y_train, 10)
    print cross_val

    #test_one = rmlse(y_test, test_predicted)
    #print test_one
















"""
bottom of page
"""
