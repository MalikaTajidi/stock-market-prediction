#MY FIRST PROJECT HEHE

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[2]:


apple = pd.read_csv('path to AAPL.csv')


# In[3]:


apple.info()


# In[4]:


apple['Date'] = pd.to_datetime(apple['Date'])


# In[5]:


print(f'dataframe contains stock prices between {apple.Date.min()} {apple.Date.max()}')
print(f'total days {(apple.Date.max() - apple.Date.min()).days}')


# In[6]:


apple.describe()


# In[7]:


apple[['Open','High','Low','Close','Adj Close']].plot(kind='box')


# In[8]:


layout = go.Layout(
    title='Stock Prices of Apple',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color="#1f77b4" 
        )
    ),
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color="#1f77b4"  
        )
    )
)
apple_data = [{'x': apple['Date'], 'y': apple['Close']}]
plot = go.Figure(data=apple_data, layout=layout)


# In[20]:


iplot(plot)


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[11]:


X= np.array(apple.index).reshape(-1,1)
Y= apple['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


# In[12]:


scaler = StandardScaler().fit(X_train)


# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[15]:


trace0 = go.Scatter(
  x=X_train.T[0],
  y=Y_train,
  mode= 'markers',
  name= 'Actual'
)
trace1= go.Scatter(
   x=X_train.T[0],
   y=lm.predict(X_train).T,
    mode= 'lines',
    name= 'Predicted'
)
apple_data= [trace0,trace1]
layout.xaxis.title.text= 'Day'
plot2= go.Figure(data=apple_data, layout=layout)


# In[21]:


iplot(plot2)


# In[17]:


scores = f"""
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test, lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test, lm.predict(X_test))}
"""

print(scores)

