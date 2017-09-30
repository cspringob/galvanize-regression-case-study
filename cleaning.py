import pandas as pd
import matplotlib.pyplot as plt


dateparse = lambda dates: [pd.datetime.strptime(d, '%m/%d/%Y %H:%M') for d in dates]
data = pd.read_csv('data/Train.zip', parse_dates=['saledate'], date_parser=dateparse, compression = 'zip')

# make sale_age category that is difference b/t the year it is sold and the year it is made
data['sale_age'] = data['saledate'].dt.year - data['YearMade']


# get rid of any zero or NaN values in MachineHoursCurrentMeter
data = data[(data['MachineHoursCurrentMeter']>0) & (data['MachineHoursCurrentMeter']<40000)]
test = data[data['MachineHoursCurrentMeter']<50000]['MachineHoursCurrentMeter']
#test.hist()
#plt.show()

# make a dummy of the usage band and append it to our data
dummies = pd.get_dummies(data['UsageBand']).rename(columns = lambda x: 'UsageBand_'+str(x))
data  = data.join(dummies)

<<<<<<< HEAD
# make a dummy of the product group types (general tractor type) and append it to our data
len(data['ProductGroup'].unique())
ProductGroup_dummies = pd.get_dummies(data['ProductGroup']).rename(columns = lambda x: 'ProdGroup_'+str(x))
data = data.join(ProductGroup_dummies)

#pd.tools.plotting.scatter_matrix(data, alpha=0.2, figsize=(40,40), diagonal = 'kde')
#plt.show()
# data.columns
#fig, ax  = plt.subplots()
# ax.scatter(data[data['UsageBand_Low']==1]['UsageBand_Low'], data[data['UsageBand_Low']==1]['SalePrice'], c = 'r')
# data[data['UsageBand_Low']==1].boxplot(column = ['SalePrice'])
# plt.show()
# data[data['UsageBand_Medium']==1].boxplot(column = ['SalePrice'])
# plt.show()
# data[data['UsageBand_High']==1].boxplot(column = ['SalePrice'])
# plt.show()
=======
print(data.columns)

>>>>>>> d9561d076c39f1b536d3901540cdbd18187e5407

# spit out cleaned data as a csv
data.to_csv('./data/cleaned.csv')
