#!/usr/bin/env python
# coding: utf-8

# # 1. Import Packages

# In[1]:


import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from time import perf_counter

from mlrose_local import mlrose

import warnings
warnings.filterwarnings("ignore")


# # 2. Load and clean the datasets

# Data are downloaded from Kaggle:<br>
# nba games https://www.kaggle.com/nathanlauga/nba-games<br>
# nba_games.csv : All games from 2004 season to last update with the date, teams and some details like number of points, etc.

# In[2]:


# Change the path into your current working directory
os.chdir(r"C:\Users\13102\Desktop\CS7641 Machine Learning Randomized Optimization")

nba_games = pd.read_csv('nba_games.csv')


# We will calculate the different between home team and away team in five aspects, including FG_PCT(field goals percentage), FT_PCT(field throws percentage), FG3_PCT(three-point field goals percentage),AST(assists) and REB(rebounds). HOME_TEAM_WINS represents that if home team wins, the value is 1 and if home team loses, the value is 0.

# In[3]:


data1 = nba_games[['GAME_DATE_EST','HOME_TEAM_WINS',]]
pd.options.mode.chained_assignment = None

# Select the data for 2019-2020 season from 2019-10-4 to 2020-3-1
start_date='2019-10-4'
end_date='2020-3-1'

data1['GAME_DATE_EST'] = pd.to_datetime(data1['GAME_DATE_EST'])  
mask = (data1['GAME_DATE_EST'] >= start_date) & (data1['GAME_DATE_EST'] <= end_date)
data1 = data1.loc[mask]

# Drop useless columns
data1 = data1.reset_index().drop(columns=['index', 'GAME_DATE_EST'])

cols1 = ['FG_PCT','FT_PCT','FG3_PCT','AST','REB']
for col in cols1:
    data1[col+'_diff'] = nba_games[col+'_home'].sub(nba_games[col+'_away'], axis = 0)

# Change datatype from float32 into int64
X1 = np.array(data1.values[:,1:-1],dtype='int64')
Y1 = np.array(data1.values[:,0],dtype='int64')
data1.describe(include='all')


# # 3. Get Train and test datasets

# In[4]:


np.random.seed(10)
X_train, X_test, y_train, y_test = train_test_split( X1, Y1, test_size=0.30, random_state=4)

# Normalize feature data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # One hot encode target values
one_hot = OneHotEncoder()
y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()


# # 4. Run Neural Networks model

# In this model, we will compare the fitting time for a neural networks model. Therefore, we only calculate the fitting time.

# In[5]:


#Train Model using MLPClassifier function on Relu Activation Function

clf = MLPClassifier(hidden_layer_sizes=(9,9), activation = 'relu', max_iter=1000, solver = 'lbfgs' )
time_start = perf_counter()
clf.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'Train fitting time for NN model = {fit_time}')

time_start = perf_counter()
yhat = clf.predict(X_test_scaled)
fit_time = perf_counter() - time_start
print (f'Test fitting time for NN model = {fit_time}')

# Evaulation
jaccard = jaccard_score(y_test_hot, yhat, average='micro')
print("jaccard index: ",jaccard)
print (classification_report(y_test_hot, yhat))


# In[6]:


#Train Model using MLPClassifier function on Sigmoid Activation Function

clf_sig = MLPClassifier(hidden_layer_sizes=(9,9), activation = 'logistic', max_iter=1000, solver = 'lbfgs' )
time_start = perf_counter()
clf_sig.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'Train fitting time for NN model = {fit_time}')

time_start = perf_counter()
yhat_sig = clf_sig.predict(X_test_scaled)
fit_time = perf_counter() - time_start
print (f'Test fitting time for NN model = {fit_time}')

# Evaulation
jaccard_sig = jaccard_score(y_test_hot, yhat_sig, average='micro')
print("jaccard index: ",jaccard)
print (classification_report(y_test_hot, yhat_sig))


# # 5. Use randomized hill climbing for optimization

# Activation Function Relu

# In[7]:


np.random.seed(7)
clf_hill = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', 
                                algorithm = 'random_hill_climb', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.5, early_stopping = True, clip_max = 5, 
                                max_attempts = 100)
time_start = perf_counter()
clf_hill.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'Train fitting time for NN model = {fit_time}')


# In[8]:


#Predict Labels for test set and assess accuracy
time_start = perf_counter()
yhat_hill_test = clf_hill.predict(X_test_scaled)
fit_time = perf_counter() - time_start
test_accuracy1 = accuracy_score(y_test_hot, yhat_hill_test)
f1 = f1_score(y_test_hot, yhat_hill_test, average='weighted') 
jaccard1 = jaccard_score(y_test_hot, yhat_hill_test, average='micro')
print (f'Test fitting time for NN model = {fit_time}')
print (f'accuracy score = {test_accuracy1}')
print("f1 score: ", f1)
print("jaccard index: ",jaccard1)
print (classification_report(y_test_hot, yhat_hill_test))


# Activation Function Sigmoid

# In[9]:


np.random.seed(7)
clf_hill_sig = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'sigmoid', 
                                algorithm = 'random_hill_climb', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.5, early_stopping = True, clip_max = 5, 
                                max_attempts = 100)
time_start = perf_counter()
clf_hill_sig.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'Train fitting time for NN model = {fit_time}')


# In[10]:


#Predict Labels for test set and assess accuracy
time_start = perf_counter()
yhat_hill_test_sig = clf_hill_sig.predict(X_test_scaled)
fit_time = perf_counter() - time_start
test_accuracy2 = accuracy_score(y_test_hot, yhat_hill_test_sig)
f2 = f1_score(y_test_hot, yhat_hill_test_sig, average='weighted') 
jaccard2 = jaccard_score(y_test_hot, yhat_hill_test_sig, average='micro')
print (f'Test fitting time for NN model = {fit_time}')
print (f'accuracy score = {test_accuracy2}')
print("f1 score: ", f2)
print("jaccard index: ",jaccard2)
print (classification_report(y_test_hot, yhat_hill_test_sig))


# # 6. Use simulated annealing for optimization

# Activation Function Relu

# In[11]:


np.random.seed(7)
clf_sim = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', 
                                algorithm = 'simulated_annealing', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.5, early_stopping = True, clip_max = 5, 
                                max_attempts = 100)
time_start = perf_counter()
clf_sim.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'Train fitting time for NN model = {fit_time}')


# In[12]:


#Predict Labels for test set and assess accuracy
time_start = perf_counter()
yhat_sim_test = clf_sim.predict(X_test_scaled)
fit_time = perf_counter() - time_start
test_accuracy3 = accuracy_score(y_test_hot, yhat_sim_test)
f3 = f1_score(y_test_hot, yhat_sim_test, average='weighted') 
jaccard3 = jaccard_score(y_test_hot, yhat_sim_test, average='micro')
print (f'Test fitting time for NN model = {fit_time}')
print (f'test accuracy score = {test_accuracy3}')
print("f1 score: ", f3)
print("jaccard index: ",jaccard3)
print (classification_report(y_test_hot, yhat_sim_test))


# Activation Function Sigmoid

# In[13]:


#np.random.seed(7)
clf_sim_sig = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'sigmoid', 
                                algorithm = 'simulated_annealing', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.5, early_stopping = True, clip_max = 5, 
                                max_attempts = 100)
time_start = perf_counter()
clf_sim_sig.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'Train fitting time for NN model = {fit_time}')


# In[14]:


#Predict Labels for test set and assess accuracy
time_start = perf_counter()
yhat_sim_test_sig = clf_sim_sig.predict(X_test_scaled)
fit_time = perf_counter() - time_start
test_accuracy4 = accuracy_score(y_test_hot, yhat_sim_test_sig)
f4 = f1_score(y_test_hot, yhat_sim_test_sig, average='weighted') 
jaccard4 = jaccard_score(y_test_hot, yhat_sim_test_sig, average='micro')
print (f'Test fitting time for NN model = {fit_time}')
print (f'test accuracy score = {test_accuracy4}')
print("f1 score: ", f4)
print("jaccard index: ",jaccard4)
print (classification_report(y_test_hot, yhat_sim_test_sig))


# # 7. Use a genetic algorithm for optimization

# Activation Function Relu

# In[15]:


np.random.seed(7)
clf_sim = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', 
                                algorithm = 'simulated_annealing', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.5, early_stopping = True, clip_max = 5, 
                                max_attempts = 100)
time_start = perf_counter()
clf_sim.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'Train fitting time for NN model = {fit_time}')


# In[16]:


#Predict Labels for test set and assess accuracy
time_start = perf_counter()
yhat_sim_test = clf_sim.predict(X_test_scaled)
fit_time = perf_counter() - time_start
test_accuracy3 = accuracy_score(y_test_hot, yhat_sim_test)
f3 = f1_score(y_test_hot, yhat_sim_test, average='weighted') 
jaccard3 = jaccard_score(y_test_hot, yhat_sim_test, average='micro')
print (f'Test fitting time for NN model = {fit_time}')
print (f'test accuracy score = {test_accuracy3}')
print("f1 score: ", f3)
print("jaccard index: ",jaccard3)
print (classification_report(y_test_hot, yhat_sim_test))


# Activation Function Sigmoid

# In[17]:


#np.random.seed(7)
clf_sim_sig = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'sigmoid', 
                                algorithm = 'simulated_annealing', 
                                max_iters=1000, bias = True, is_classifier = True, 
                                learning_rate = 0.5, early_stopping = True, clip_max = 5, 
                                max_attempts = 100)
time_start = perf_counter()
clf_sim_sig.fit(X_train_scaled, y_train_hot)
fit_time = perf_counter() - time_start
print(f'Train fitting time for NN model = {fit_time}')


# In[18]:


#Predict Labels for test set and assess accuracy
time_start = perf_counter()
yhat_sim_test_sig = clf_sim_sig.predict(X_test_scaled)
fit_time = perf_counter() - time_start
test_accuracy4 = accuracy_score(y_test_hot, yhat_sim_test_sig)
f4 = f1_score(y_test_hot, yhat_sim_test_sig, average='weighted') 
jaccard4 = jaccard_score(y_test_hot, yhat_sim_test_sig, average='micro')
print (f'Test fitting time for NN model = {fit_time}')
print (f'test accuracy score = {test_accuracy4}')
print("f1 score: ", f4)
print("jaccard index: ",jaccard4)
print (classification_report(y_test_hot, yhat_sim_test_sig))

