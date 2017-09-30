import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
filename = 'data/Train.zip'


def clean_data(filename, fileout):

    def dateparse(dates): return [pd.datetime.strptime(
        d, '%m/%d/%Y %H:%M') for d in dates]
    data = pd.read_csv(filename, parse_dates=[
                       'saledate'], date_parser=dateparse, compression='zip')

    # make sale_age category that is difference b/t the year it is sold and the year it is made
    data['sale_age'] = data['saledate'].dt.year - data['YearMade']

    # get rid of any zero or NaN values in MachineHoursCurrentMeter
    #commented out below to allow for just removing NaN values

    data_mean = np.mean(
        data[data['MachineHoursCurrentMeter'] > 0]['MachineHoursCurrentMeter'])

    data['MachineHoursCurrentMeter'] = data['MachineHoursCurrentMeter'].fillna(
        0)

    data[data['MachineHoursCurrentMeter'] ==
         0]['MachineHoursCurrentMeter'] = data_mean


    #data = data[data['MachineHoursCurrentMeter'] > 0] #inserted to only remove NaN
    data = data[data['MachineHoursCurrentMeter'] < 40000]

    # make a dummy of the usage band and append it to our data
    dummies = pd.get_dummies(data['UsageBand']).rename(
        columns=lambda x: 'UsageBand_' + str(x))
    data = data.join(dummies)

    # make a dummy of the product group types (general tractor type) and append it to our data
    data['ProductGroup'].unique()
    ProductGroup_dummies = pd.get_dummies(data['ProductGroup']).rename(
        columns=lambda x: 'ProdGroup_' + str(x))
    data = data.join(ProductGroup_dummies)
    data.info()
    #pd.tools.plotting.scatter_matrix(data, alpha=0.2, figsize=(40,40), diagonal = 'kde')
    # plt.show()
    # data.columns
    #fig, ax  = plt.subplots()
    # ax.scatter(data[data['UsageBand_Low']==1]['UsageBand_Low'], data[data['UsageBand_Low']==1]['SalePrice'], c = 'r')
    # data[data['UsageBand_Low']==1].boxplot(column = ['SalePrice'])
    # plt.show()
    # data[data['UsageBand_Medium']==1].boxplot(column = ['SalePrice'])
    # plt.show()
    # data[data['UsageBand_High']==1].boxplot(column = ['SalePrice'])
    # plt.show()

    # spit out cleaned data as a csv
    data.to_csv(fileout)


if __name__ == '__main__':
    clean_data('data/Train.zip', './data/cleaned_train.csv')
    clean_data('data/Test.zip', './data/cleaned_test.csv')
