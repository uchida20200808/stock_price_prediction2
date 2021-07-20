#!/usr/bin/env python
# coding: utf-8

# In[68]:


'''
Stock Price Prediction – Machine Learning Project in Python
https://data-flair.training/blogs/stock-price-prediction-machine-learning-project-in-python/
'''


# In[69]:


'''
Machine learning has significant applications in the stock price prediction. 
In this machine learning project, 
we will be talking about predicting the returns on stocks. 

This is a very complex task and has uncertainties. 
We will develop this project into two parts:

1.First, we will learn how to predict stock price using the LSTM neural network.
2.Then we will build a dashboard using Plotly dash for stock analysis.
'''


# In[70]:


'''
Datasets
To build the stock price prediction model,
we will use the NSE TATA GLOBAL dataset. 

This is a dataset of Tata Beverages from Tata Global Beverages Limited, National Stock Exchange of India
:Tata Global Dataset

To develop the dashboard for stock analysis 
we will use another stock dataset with multiple stocks like Apple, Microsoft, Facebook
: Stocks Dataset
'''


# In[71]:


#import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

from sklearn.preprocessing import MinMaxScaler


# In[72]:


#Read the dataset:
#https://data-flair.training/blogs/download-tata-global-beverages-stocks-data/
df=pd.read_csv("NSE-TATA.csv")
df.head()


# In[73]:


#Analyze the closing prices from dataframe:
df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']


# In[74]:


plt.figure(figsize=(16,8))
plt.plot(df["Close"],label='Close Price history')


# In[75]:


#Sort the dataset on date time and filter “Date” and “Close” columns:
#https://note.nkmk.me/python-pandas-sort-values-sort-index/
data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])


# In[76]:


new_dataset.tail()


# In[77]:


for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Close"][i]=data["Close"][i]


# In[78]:


new_dataset.tail()


# In[79]:



scaler=MinMaxScaler(feature_range=(0,1))


# In[80]:


#https://note.nkmk.me/python-pandas-dataframe-values-columns-index/
#df['columns']の他、df.columnsでもデータを選択できる
new_dataset.Close


# In[81]:


new_dataset.index=new_dataset.Date
#https://note.nkmk.me/python-pandas-drop/
#axis=1は列を削除
new_dataset.drop("Date",axis=1,inplace=True)


# In[82]:


#データフレーム内の値をndarrayの形で取り出す
#https://note.nkmk.me/python-pandas-dataframe-values-columns-index/
final_dataset=new_dataset.values


# In[83]:


final_dataset


# In[84]:


#80% of new_dataset is used for a train_data
train_data=final_dataset[0:987,:]
#the rest 20% of new_dataset is used for a valid_data
valid_data=final_dataset[987:,:]


# In[85]:


#Normalize the new filtered dataset:
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)


# In[86]:


x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])


# In[87]:


x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)


# In[88]:


x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))


# In[89]:


#Build and train the LSTM model:
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))


# In[90]:


lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)


# In[91]:


inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)


# In[92]:


#Take a sample of a dataset to make stock price predictions using the LSTM model:
X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=lstm_model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

#Save the LSTM model:
lstm_model.save("saved_lstm_model.h5")


# In[93]:


#Visualize the predicted stock costs with actual stock costs:
train_data=new_dataset[:987]
valid_data=new_dataset[987:]
valid_data['Predictions']=closing_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')


# In[ ]:




