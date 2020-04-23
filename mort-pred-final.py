#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required packages


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Read the files


# In[4]:


df = pd.read_csv('train.csv')
labels = pd.read_csv('labels.csv')


# In[5]:


#EDA


# In[6]:


df.head()


# In[8]:


temp = pd.concat([df,labels],axis=1)


# In[10]:


temp.corr()['In-hospital_death'].sort_values()


# In[11]:


temp = temp.drop(['MechVent','RecordID','Gender','Height'],axis=1)


# In[12]:


temp.head()


# In[16]:


temp = temp.drop('ICUType',axis=1)


# In[30]:


from sklearn.preprocessing import MinMaxScaler


# In[31]:


scaler=MinMaxScaler()


# In[17]:


#Splitting the data


# In[19]:


X = temp.drop('In-hospital_death',axis=1).values


# In[20]:


y = temp['In-hospital_death'].values


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=150)


# In[24]:


#Building the model


# In[25]:


from tensorflow.keras.models import Sequential


# In[52]:


from tensorflow.keras.layers import Dense, Dropout


# In[27]:


model = Sequential()


# In[53]:


model.add(Dense(units=37,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=18,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=9,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='Adam',loss='binary_crossentropy')


# In[32]:


X_train = scaler.fit_transform(X_train)


# In[33]:


X_test = scaler.transform(X_test)


# In[46]:


from tensorflow.keras.callbacks import EarlyStopping


# In[47]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)


# In[54]:


model.fit(x=X_train,y=y_train,epochs=10,verbose=1,validation_data=(X_test,y_test),callbacks=[early_stop])


# In[57]:


preds = model.predict_classes(X_test)


# In[36]:


#Check


# In[37]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[58]:


preds


# In[59]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[60]:


print(accuracy_score(y_test, preds))

