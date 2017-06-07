import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
#print forecast_out

df['label'] = df[forecast_col].shift(-forecast_out)


#X is for features, y is for label
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X) 
X_lately = X[-forecast_out:] #this is the last forecast_out days
X = X[:-forecast_out]


df.dropna(inplace=True)
y = np.array(df['label'])




# below call basically shuffles all the data, keeping integrity and then outputs
# the 0.2 is for the split 80% train, 20% test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train) #fit is for training data -- this func is literally training the data
accuracy = clf.score(X_test, y_test) # score is for testing -- like it sounds

# print (len(X), len(X_lately))
forecast_set = clf.predict(X_lately) # predicts based on most recent data

print (forecast_set, accuracy, forecast_out)


df['Forecast'] = np.nan

last_date = df.iloc[-1].name # very last date
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print (df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()




