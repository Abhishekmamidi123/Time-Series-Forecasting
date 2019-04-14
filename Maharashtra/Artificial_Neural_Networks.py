
# coding: utf-8

# # <center>Time Series Analysis on Pune precipitation data from 1965 to 2002.</center>

# In[8]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import itertools
import warnings
warnings.filterwarnings('ignore')


# In[73]:


#filename = 'pune_1965_to_2002.csv'
#STORAGE_FOLDER = 'output/'


# In[78]:


def preprocess_data(filename):
    rainfall_data_matrix = pd.read_csv(filename, delimiter='\t')
    rainfall_data_matrix.set_index('Year', inplace=True)
    rainfall_data_matrix = rainfall_data_matrix.transpose()
    dates = pd.date_range(start='1901-01', freq='MS', periods=len(rainfall_data_matrix.columns)*12)
    
    rainfall_data_matrix_np = rainfall_data_matrix.transpose().as_matrix()
    shape = rainfall_data_matrix_np.shape
    rainfall_data_matrix_np = rainfall_data_matrix_np.reshape((shape[0] * shape[1], 1))
    
    rainfall_data = pd.DataFrame({'Precipitation': rainfall_data_matrix_np[:,0]})
    rainfall_data.set_index(dates, inplace=True)

    test_rainfall_data = rainfall_data.ix['1998': '2002']
    rainfall_data = rainfall_data.ix[: '1998']
    rainfall_data = rainfall_data.round(5)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(rainfall_data)
    
    return rainfall_data, test_rainfall_data, scaler


# In[72]:


# FILENAME = 'pune_1965_to_2002.csv'
# rainfall_data, test_rainfall_data, scaler = preprocess_data(FILENAME)


# ## <center> Artificial Neural Networks </center>

# In[12]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    return smape

def correlation_coefficient(y_true, y_pred):
    corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
    return corr_coef
    
# In[13]:

def calculate_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    corr_coef = correlation_coefficient(y_true, y_pred)
    return round(mse, 3), round(mae, 3), round(mape, 3), round(rmse, 3), round(corr_coef, 3)


# In[14]:


def plot_keras_model(model, show_shapes=True, show_layer_names=True):
    return SVG(model_to_dot(model, show_shapes=show_shapes, show_layer_names=show_layer_names).create(prog='dot',format='svg'))


# In[15]:


def get_combinations(parameters):
    return list(itertools.product(*parameters))


# In[16]:


def create_NN(input_nodes, hidden_nodes, output_nodes):
    model = Sequential()
    model.add(Dense(int(hidden_nodes), input_dim=int(input_nodes)))
    model.add(Dense(int(output_nodes)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[17]:


def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True)
    return model


# In[18]:


def reshape_arrays(X_train, y_train):
    X_train = np.array(X_train)
    y_train = np.reshape(y_train, (len(y_train), 1))
    return X_train, y_train


# In[19]:


def preprocess_FNN(data, look_back):
    data = np.array(data)[:, 0]
    X_train = []
    y_train = []
    for i in range(data.shape[0]-look_back):
        x = data[i:look_back+i][::-1]
        y = data[look_back+i]
        X_train.append(list(x))
        y_train.append(y)
    input_seq_for_test = data[i+1:look_back+i+1][::-1]
    return X_train, y_train, input_seq_for_test


# In[20]:


def forecast_FNN(model, input_sequence, future_steps):
    forecasted_values = []
    for i in range(future_steps):
        forecasted_value = model.predict(input_sequence)
        if forecasted_value < 0:
            forecasted_value = forecasted_value - forecasted_value
        forecasted_values.append(forecasted_value[0][0])
        input_sequence[0] = np.append(forecasted_value, input_sequence[0][:-1])
    return forecasted_values


# In[21]:


def FNN(data, look_back, hidden_nodes, output_nodes, epochs, batch_size, future_steps, scaler):
    data = scaler.transform(data)
    X_train, y_train, input_seq_for_test_FNN = preprocess_FNN(data, look_back)
    X_train, y_train = reshape_arrays(X_train, y_train)

    model_FNN = create_NN(input_nodes=look_back, hidden_nodes=hidden_nodes, output_nodes=output_nodes)
    model_FNN = train_model(model_FNN, X_train, y_train, epochs, batch_size)

    input_seq_for_test_FNN = np.reshape(input_seq_for_test_FNN, (1, len(input_seq_for_test_FNN)))
    forecasted_values_FNN = forecast_FNN(model_FNN, input_sequence=input_seq_for_test_FNN, future_steps=future_steps)
    
    forecasted_values_FNN = list(scaler.inverse_transform([forecasted_values_FNN])[0])
    print(forecasted_values_FNN)
    return model_FNN, forecasted_values_FNN


# In[22]:


def get_accuracies_FNN(rainfall_data, test_rainfall_data, parameters, scaler):
    combination_of_params = get_combinations(parameters)
    information_FNN = []
    iterator = 0
    print('FNN - Number of combinations: ' + str(len(combination_of_params)))
    
    for param in combination_of_params:
        if (iterator+1) != len(combination_of_params):
            print(iterator+1, end=' -> ')
        else:
            print(iterator+1)
        iterator = iterator+1

        look_back = param[0]
        hidden_nodes = param[1]
        output_nodes = param[2]
        epochs = param[3]
        batch_size = param[4]
        future_steps = param[5]

        model_FNN, forecasted_values_FNN = FNN(rainfall_data, look_back, hidden_nodes, output_nodes, epochs, batch_size, future_steps, scaler)
        
        y_true = test_rainfall_data.ix[:future_steps].Precipitation
        mse, mae, mape, rmse, corr_coef = calculate_performance(y_true, forecasted_values_FNN)
        
        info = list(param) + [mse, mae, mape, rmse, corr_coef] + forecasted_values_FNN
        information_FNN.append(info)

    information_FNN_df = pd.DataFrame(information_FNN)
    indexes = [str(i) for i in list(range(1, future_steps+1))]
    information_FNN_df.columns = ['look_back', 'hidden_nodes', 'output_nodes', 'epochs', 'batch_size', 'future_steps', 'MSE', 'MAE', 'MAPE', 'RMSE', 'R'] + indexes
    return information_FNN_df


# In[23]:


def preprocess_TLNN(data, time_lagged_points):
    data = np.array(data)[:, 0]
    X_train = []
    y_train = []
    for i in range(max(time_lagged_points), data.shape[0]):
        x = [data[i-p] for p in time_lagged_points]
        y = data[i]
        X_train.append(list(x))
        y_train.append(y)
    input_seq_for_test = [data[i+1-p] for p in time_lagged_points]
    return X_train, y_train, input_seq_for_test


# In[24]:


def forecast_TLNN(model, time_lagged_points, last_sequence, future_steps):
    forecasted_values = []
    max_lag = max(time_lagged_points)
    for i in range(future_steps):
        input_sequence = [last_sequence[max_lag - p] for p in time_lagged_points]
        forecasted_value = model.predict(np.reshape(input_sequence, (1, len(input_sequence))))
        if forecasted_value < 0:
            forecasted_value = forecasted_value - forecasted_value
        forecasted_values.append(forecasted_value[0][0])
        last_sequence = last_sequence[1:] + [forecasted_value[0][0]]
    return forecasted_values


# In[25]:


def TLNN(data, time_lagged_points, hidden_nodes, output_nodes, epochs, batch_size, future_steps, scaler):
    data = scaler.transform(data)
    X_train, y_train, input_seq_for_test_TLNN = preprocess_TLNN(data, time_lagged_points)
    X_train, y_train = reshape_arrays(X_train, y_train)
    model_TLNN = create_NN(input_nodes=len(time_lagged_points), hidden_nodes=hidden_nodes, output_nodes=output_nodes)
    model_TLNN = train_model(model_TLNN, X_train, y_train, epochs, batch_size)

    max_lag = max(time_lagged_points)
    forecasted_values_TLNN = forecast_TLNN(model_TLNN, time_lagged_points, 
                                      list(data[-max_lag:]), future_steps=future_steps)
    forecasted_values_TLNN = list(scaler.inverse_transform([forecasted_values_TLNN])[0])
    print(forecasted_values_TLNN)
    return model_TLNN, forecasted_values_TLNN


# In[26]:


def get_accuracies_TLNN(rainfall_data, test_rainfall_data, parameters, scaler):
    combination_of_params = get_combinations(parameters)
    information_TLNN = []
    iterator = 0
    print('TLNN - Number of combinations: ' + str(len(combination_of_params)))
    
    for param in combination_of_params:
        if (iterator+1) != len(combination_of_params):
            print(iterator+1, end=' -> ')
        else:
            print(iterator+1)
        iterator = iterator+1

        time_lagged_points = param[0]
        hidden_nodes = param[1]
        output_nodes = param[2]
        epochs = param[3]
        batch_size = param[4]
        future_steps = param[5]

        model_TLNN, forecasted_values_TLNN = TLNN(rainfall_data, time_lagged_points, hidden_nodes, output_nodes, epochs, batch_size, future_steps, scaler)
        
        y_true = test_rainfall_data.ix[:future_steps].Precipitation
        mse, mae, mape, rmse, corr_coef = calculate_performance(y_true, forecasted_values_TLNN)
        
        info = list(param) + [mse, mae, mape, rmse, corr_coef] + forecasted_values_TLNN
        information_TLNN.append(info)

    information_TLNN_df = pd.DataFrame(information_TLNN)
    indexes = [str(i) for i in list(range(1, future_steps+1))]
    information_TLNN_df.columns = ['look_back_lags', 'hidden_nodes', 'output_nodes', 'epochs', 'batch_size', 'future_steps', 'MSE', 'MAE', 'MAPE', 'RMSE', 'R'] + indexes
    return information_TLNN_df


# In[27]:


def preprocess_SANN(data, seasonal_period):
    data = np.array(data)[:, 0]
    X_train = []
    y_train = []
    for i in range(seasonal_period, data.shape[0]-seasonal_period+1):
        x = data[i-seasonal_period:i][::-1]
        y = data[i:i+seasonal_period]
        X_train.append(list(x))
        y_train.append(list(y))
    input_seq_for_test = data[i+1-seasonal_period:i+1][::-1]
    return X_train, y_train, input_seq_for_test


# In[28]:


def forecast_SANN(model, input_sequence, seasonal_period, future_steps):
    iterations = future_steps/seasonal_period
    forecasted_values = []
    for i in range(int(iterations) + 1):
        next_forecasted_values = model.predict(input_sequence)
        forecasted_values += list(next_forecasted_values[0])
        input_sequence = next_forecasted_values
    return forecasted_values[:future_steps]


# In[29]:


def SANN(data, seasonal_period, hidden_nodes, epochs, batch_size, future_steps, scaler):
    data = scaler.transform(data)
    X_train, y_train, input_seq_for_test_SANN = preprocess_SANN(data, seasonal_period)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    input_seq_for_test_SANN = np.reshape(input_seq_for_test_SANN, (1, len(input_seq_for_test_SANN)))
    model_SANN = create_NN(input_nodes=seasonal_period, hidden_nodes=hidden_nodes, output_nodes=seasonal_period)
    model_SANN = train_model(model_SANN, X_train, y_train, epochs, batch_size)
    
    forecasted_values_SANN = forecast_SANN(model_SANN, input_seq_for_test_SANN, seasonal_period, future_steps=future_steps)
    forecasted_values_SANN = list(scaler.inverse_transform([forecasted_values_SANN])[0])
    print(forecasted_values_SANN)
    return model_SANN, forecasted_values_SANN


# In[30]:


def get_accuracies_SANN(rainfall_data, test_rainfall_data, parameters, scaler):
    combination_of_params = get_combinations(parameters)
    information_SANN = []
    iterator = 0
    print('SANN - Number of combinations: ' + str(len(combination_of_params)))
    
    for param in combination_of_params:
        if (iterator+1) != len(combination_of_params):
            print(iterator+1, end=' -> ')
        else:
            print(iterator+1)
        iterator = iterator+1

        seasonal_period = param[0]
        hidden_nodes = param[1]
        epochs = param[2]
        batch_size = param[3]
        future_steps = param[4]

        model_SANN, forecasted_values_SANN = SANN(rainfall_data, seasonal_period, hidden_nodes, epochs, batch_size, future_steps, scaler)
        
        y_true = test_rainfall_data.ix[:future_steps].Precipitation
        mse, mae, mape, rmse, corr_coef = calculate_performance(y_true, forecasted_values_SANN)
        
        info = list(param) + [mse, mae, mape, rmse, corr_coef] + forecasted_values_SANN
        information_SANN.append(info)

    information_SANN_df = pd.DataFrame(information_SANN)
    indexes = [str(i) for i in list(range(1, future_steps+1))]
    information_SANN_df.columns = ['seasonal_period', 'hidden_nodes', 'epochs', 'batch_size', 'future_steps', 'MSE', 'MAE', 'MAPE', 'RMSE', 'R'] + indexes
    return information_SANN_df

# ===============

def create_RNN(input_nodes, hidden_nodes, output_nodes):
    model = Sequential()
    model.add(SimpleRNN(int(hidden_nodes), input_shape=(int(input_nodes), 1)))
    model.add(Dense(int(output_nodes)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[32]:


def preprocess_RNN(data, look_back):
    data = np.array(data)[:, 0]
    X_train = []
    y_train = []
    for i in range(data.shape[0]-look_back):
        x = data[i:look_back+i][::-1]
        y = data[look_back+i]
        X_train.append(list(x))
        y_train.append(y)
    input_seq_for_test = data[i+1:look_back+i+1][::-1]
    return X_train, y_train, input_seq_for_test


# In[33]:


def forecast_RNN(model, input_sequence, future_steps):
    forecasted_values = []
    print(input_sequence)
    for i in range(future_steps):
        forecasted_value = model.predict(np.reshape(input_sequence, (1, len(input_sequence[0][0]), 1)))
        if forecasted_value < 0:
            forecasted_value = forecasted_value - forecasted_value
        forecasted_values.append(forecasted_value[0][0])
        input_sequence[0][0] = np.append(forecasted_value, input_sequence[0][0][:-1])
    return forecasted_values


# In[34]:


def Recurrent_Neural_Network(data, look_back, hidden_nodes, output_nodes, epochs, batch_size, future_steps, scaler):
    data = scaler.transform(data)
    X_train, y_train, input_seq_for_test_RNN = preprocess_RNN(data, look_back)
    X_train = np.reshape(X_train, (len(X_train), look_back, 1))

    model_RNN = create_RNN(input_nodes=look_back, hidden_nodes=hidden_nodes, output_nodes=output_nodes)
    plot_keras_model(model_RNN)
    model_RNN = train_model(model_RNN, X_train, y_train, epochs, batch_size)

    input_seq_for_test_RNN = np.reshape(input_seq_for_test_RNN, (1, 1, len(input_seq_for_test_RNN)))
    forecasted_values_RNN = forecast_RNN(model_RNN, input_sequence=input_seq_for_test_RNN, future_steps=future_steps)
    forecasted_values_RNN = list(scaler.inverse_transform([forecasted_values_RNN])[0])
    print(forecasted_values_RNN)
    return model_RNN, forecasted_values_RNN


# In[35]:


def get_accuracies_RNN(rainfall_data, test_rainfall_data, parameters, scaler):
    combination_of_params = get_combinations(parameters)
    information_RNN = []
    iterator = 0
    print('RNN - Number of combinations: ' + str(len(combination_of_params)))
    
    for param in combination_of_params:
        if (iterator+1) != len(combination_of_params):
            print(iterator+1, end=' -> ')
        else:
            print(iterator+1)
        iterator = iterator+1

        input_nodes = param[0]
        hidden_nodes = param[1]
        output_nodes = param[2]
        epochs = param[3]
        batch_size = param[4]
        future_steps = param[5]

        model_RNN, forecasted_values_RNN = Recurrent_Neural_Network(rainfall_data, input_nodes, hidden_nodes, output_nodes, epochs, batch_size, future_steps, scaler)
        
        y_true = test_rainfall_data.ix[:future_steps].Precipitation
        mse, mae, mape, rmse, corr_coef = calculate_performance(y_true, forecasted_values_RNN)
        
        info = list(param) + [mse, mae, mape, rmse, corr_coef] + forecasted_values_RNN
        information_RNN.append(info)

    information_RNN_df = pd.DataFrame(information_RNN)
    indexes = [str(i) for i in list(range(1, future_steps+1))]
    information_RNN_df.columns = ['look_back', 'hidden_nodes', 'output_nodes', 'epochs', 'batch_size', 'future_steps', 'MSE', 'MAE', 'MAPE', 'RMSE', 'R'] + indexes
    return information_RNN_df

# ===============

# In[31]:

def create_LSTM(input_nodes, hidden_nodes, output_nodes):
    model = Sequential()
#     model.add(LSTM(int(hidden_nodes), input_shape=(1, int(input_nodes))))
    model.add(LSTM(int(hidden_nodes), input_shape=(int(input_nodes), 1)))
    model.add(Dense(int(output_nodes)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[32]:


def preprocess_LSTM(data, look_back):
    data = np.array(data)[:, 0]
    X_train = []
    y_train = []
    for i in range(data.shape[0]-look_back):
        x = data[i:look_back+i][::-1]
        y = data[look_back+i]
        X_train.append(list(x))
        y_train.append(y)
    input_seq_for_test = data[i+1:look_back+i+1][::-1]
    return X_train, y_train, input_seq_for_test


# In[33]:


def forecast_LSTM(model, input_sequence, future_steps):
    forecasted_values = []
    print(input_sequence)
    for i in range(future_steps):
        forecasted_value = model.predict(np.reshape(input_sequence, (1, len(input_sequence[0][0]), 1)))
        if forecasted_value < 0:
            forecasted_value = forecasted_value - forecasted_value
        forecasted_values.append(forecasted_value[0][0])
        input_sequence[0][0] = np.append(forecasted_value, input_sequence[0][0][:-1])
    return forecasted_values


# In[34]:


def Long_Short_Term_Memory(data, look_back, hidden_nodes, output_nodes, epochs, batch_size, future_steps, scaler):
    data = scaler.transform(data)
    X_train, y_train, input_seq_for_test_LSTM = preprocess_LSTM(data, look_back)
    X_train = np.reshape(X_train, (len(X_train), look_back, 1))

    model_LSTM = create_LSTM(input_nodes=look_back, hidden_nodes=hidden_nodes, output_nodes=output_nodes)
    plot_keras_model(model_LSTM)
    model_LSTM = train_model(model_LSTM, X_train, y_train, epochs, batch_size)

    input_seq_for_test_LSTM = np.reshape(input_seq_for_test_LSTM, (1, 1, len(input_seq_for_test_LSTM)))
    forecasted_values_LSTM = forecast_LSTM(model_LSTM, input_sequence=input_seq_for_test_LSTM, future_steps=future_steps)
    forecasted_values_LSTM = list(scaler.inverse_transform([forecasted_values_LSTM])[0])
    print(forecasted_values_LSTM)
    return model_LSTM, forecasted_values_LSTM


# In[35]:


def get_accuracies_LSTM(rainfall_data, test_rainfall_data, parameters, scaler):
    combination_of_params = get_combinations(parameters)
    information_LSTM = []
    iterator = 0
    print('LSTM - Number of combinations: ' + str(len(combination_of_params)))
    
    for param in combination_of_params:
        if (iterator+1) != len(combination_of_params):
            print(iterator+1, end=' -> ')
        else:
            print(iterator+1)
        iterator = iterator+1

        input_nodes = param[0]
        hidden_nodes = param[1]
        output_nodes = param[2]
        epochs = param[3]
        batch_size = param[4]
        future_steps = param[5]

        model_LSTM, forecasted_values_LSTM = Long_Short_Term_Memory(rainfall_data, input_nodes, hidden_nodes, output_nodes, epochs, batch_size, future_steps, scaler)
        
        y_true = test_rainfall_data.ix[:future_steps].Precipitation
        mse, mae, mape, rmse, corr_coef = calculate_performance(y_true, forecasted_values_LSTM)
        
        info = list(param) + [mse, mae, mape, rmse, corr_coef] + forecasted_values_LSTM
        information_LSTM.append(info)

    information_LSTM_df = pd.DataFrame(information_LSTM)
    indexes = [str(i) for i in list(range(1, future_steps+1))]
    information_LSTM_df.columns = ['look_back', 'hidden_nodes', 'output_nodes', 'epochs', 'batch_size', 'future_steps', 'MSE', 'MAE', 'MAPE', 'RMSE', 'R'] + indexes
    return information_LSTM_df


# In[47]:


def analyze_results(data_frame, test_rainfall_data, name, STORAGE_FOLDER, flag=False):
    optimized_params = data_frame.iloc[data_frame.RMSE.argmin]
    future_steps = optimized_params.future_steps
    forecast_values = optimized_params[-1*int(future_steps):]
    y_true = test_rainfall_data.ix[:int(future_steps)]
    forecast_values.index = y_true.index
        
    print('=== Best parameters of ' + name + ' ===\n')
    if (name == 'FNN'):
        model = create_NN(optimized_params.look_back, 
                          optimized_params.hidden_nodes, 
                          optimized_params.output_nodes)
        print('Input nodes(p): ' + str(optimized_params.look_back))
        print('Hidden nodes: ' + str(optimized_params.hidden_nodes))
        print('Output nodes: ' + str(optimized_params.output_nodes))
    elif (name == 'RNN'):
        model = create_RNN(optimized_params.look_back,
                            optimized_params.hidden_nodes,
                            optimized_params.output_nodes)
        print('Input nodes(s): ' + str(optimized_params.look_back))
        print('Hidden nodes: ' + str(optimized_params.hidden_nodes))
        print('Output nodes: ' + str(optimized_params.output_nodes))
    elif (name == 'LSTM'):
        model = create_LSTM(optimized_params.look_back,
                            optimized_params.hidden_nodes,
                            optimized_params.output_nodes)
        print('Input nodes(s): ' + str(optimized_params.look_back))
        print('Hidden nodes: ' + str(optimized_params.hidden_nodes))
        print('Output nodes: ' + str(optimized_params.output_nodes))
    elif (name == 'TLNN'):
        model = create_NN(len(optimized_params.look_back_lags), 
                          optimized_params.hidden_nodes, 
                          optimized_params.output_nodes)
        s = ''
        for i in optimized_params.look_back_lags:
            s = s+' '+str(i)
        print('Look back lags: ' + s)
        print('Hidden nodes: ' + str(optimized_params.hidden_nodes))
        print('Output nodes: ' + str(optimized_params.output_nodes))
    elif (name == 'SANN'):
        model = create_NN(optimized_params.seasonal_period, 
                          optimized_params.hidden_nodes, 
                          optimized_params.seasonal_period)
        print('Input nodes(s): ' + str(optimized_params.seasonal_period))
        print('Hidden nodes: ' + str(optimized_params.hidden_nodes))
        print('Output nodes: ' + str(optimized_params.seasonal_period))
    elif (name == 'CNN'):
        model = create_CNN(optimized_params.look_back,
                           optimized_params.filters,
                           optimized_params.output_nodes)
        print('Input nodes(s): ' + str(optimized_params.look_back))
        print('Filters: ' + str(optimized_params.filters))
        print('Output nodes: ' + str(optimized_params.output_nodes))
        
    print('Number of epochs: ' + str(optimized_params.epochs))
    print('Batch size: ' + str(optimized_params.batch_size))
    print('Number of future steps forecasted: ' + str(optimized_params.future_steps))
    print('Mean Squared Error(MSE): ' + str(optimized_params.MSE))
    print('Mean Absolute Error(MAE): ' + str(optimized_params.MAE))
    print('Root Mean Squared Error(RMSE): ' + str(optimized_params.RMSE))
    print('\n\n')
    
    # Save model
    from keras.utils import plot_model
    plot_model(model, to_file = STORAGE_FOLDER + name + '_best_fit_model.png', show_shapes=True, show_layer_names=True)
    
    # Save data
    data_frame.to_csv(STORAGE_FOLDER + name + '_information.csv')
    optimized_params.to_csv(STORAGE_FOLDER + name + '_optimized_values.csv')
    
    # Actual and forecasted values
    errors = test_rainfall_data.Precipitation - forecast_values
    actual_forecast = pd.DataFrame({'Actual': test_rainfall_data.Precipitation, 'Forecasted': forecast_values, 
                                    'Errors': errors})
    actual_forecast.to_csv(STORAGE_FOLDER + name + '_actual_and_forecasted.csv')
    
    plt.figure(figsize=(10,5))
    plt.plot(actual_forecast.drop(columns=['Actual', 'Forecasted']), color='blue', label='Error: Actual - Forecasted')
    plt.xlabel('Year')
    plt.ylabel('Error')
    plt.legend(loc='best')
    plt.title(name + ' - Error: Actual - Forecasted')
    plt.savefig(STORAGE_FOLDER + name + '_error_plot'  + '.png')
    
    
    plt.figure(figsize=(10,5))
    plt.plot(y_true, color='green', label='Actual values')
    plt.plot(forecast_values, color='red', label='Forecasted values')
    plt.xlabel('Year')
    plt.ylabel('Monthly mean Precipitation')
    plt.legend(loc='best')
    if (flag==False):
        plt.title(name + ' - Comaprison: Actual vs Forecasted')
        plt.savefig(STORAGE_FOLDER + name + '_best_forecast'  + '.png')
    else:
        plt.title('Best of all: ' + name + ' - Comaprison: Actual vs Forecasted')
        plt.savefig(STORAGE_FOLDER + 'BEST_FORECAST_' + name + '.png')
    
    return optimized_params


# In[48]:


def best_of_all(list_of_methods):
    RMSE_values = [x.RMSE for x in list_of_methods]
    MAE_values = [x.MAE for x in list_of_methods]
    R_values = [x.R for x in list_of_methods]
    
    index = np.argmin(RMSE_values)
    if (index == 0):
        name = 'FNN'
    elif (index == 1):
        name = 'TLNN'
    elif (index == 2):
        name = 'SANN'
    elif (index == 3):
        name = 'RNN'
    elif (index == 4):
        name = 'LSTM'
    else:
        name = 'CNN'
    print(RMSE_values)
    
    names = ['FNN', 'TLNN', 'SANN', 'RNN', 'LSTM']
    RMSE_info = pd.Series(RMSE_values, index=names)
    MAE_info = pd.Series(MAE_values, index=names)
    R_info = pd.Series(R_values, index=names)
    
    print('Overall Best method on this data is ' + name)
    return index, name, RMSE_info, MAE_info, R_info


# In[49]:


def compare_ANN_methods(rainfall_data, test_rainfall_data, scaler, parameters_FNN, parameters_TLNN, 
                        parameters_SANN, parameters_RNN, parameters_LSTM, future_steps, STORAGE_FOLDER):
    
    information_FNN_df = get_accuracies_FNN(rainfall_data, test_rainfall_data, parameters_FNN, scaler)
    optimized_params_FNN = analyze_results(information_FNN_df, test_rainfall_data, 'FNN', STORAGE_FOLDER)
    
    information_TLNN_df = get_accuracies_TLNN(rainfall_data, test_rainfall_data, parameters_TLNN, scaler)
    optimized_params_TLNN = analyze_results(information_TLNN_df, test_rainfall_data, 'TLNN', STORAGE_FOLDER)
    
    information_SANN_df = get_accuracies_SANN(rainfall_data, test_rainfall_data, parameters_SANN, scaler)
    optimized_params_SANN = analyze_results(information_SANN_df, test_rainfall_data, 'SANN', STORAGE_FOLDER)
    
    information_RNN_df = get_accuracies_RNN(rainfall_data, test_rainfall_data, parameters_RNN, scaler)
    optimized_params_RNN = analyze_results(information_RNN_df, test_rainfall_data, 'RNN', STORAGE_FOLDER)
    
    information_LSTM_df = get_accuracies_LSTM(rainfall_data, test_rainfall_data, parameters_LSTM, scaler)
    optimized_params_LSTM = analyze_results(information_LSTM_df, test_rainfall_data, 'LSTM', STORAGE_FOLDER)    
    
    list_of_methods = [optimized_params_FNN, optimized_params_TLNN, optimized_params_SANN, optimized_params_RNN, optimized_params_LSTM]
    information = [information_FNN_df, information_TLNN_df, information_SANN_df, information_RNN_df, information_LSTM_df]
    index, name, RMSE_info, MAE_info, R_info = best_of_all(list_of_methods)
    best_optimized_params = analyze_results(information[index], test_rainfall_data, name, STORAGE_FOLDER, True)
    return RMSE_info, MAE_info, R_info


# In[69]:


def save_RMSE_info(STORAGE_FOLDER, RMSE_info, MAE_info, R_info):
    
    RMSE_df = pd.DataFrame({'RMSE': RMSE_info, 'MAE': MAE_info, 'R': R_info})
    RMSE_df.index = RMSE_info.index
    RMSE_df.to_csv(STORAGE_FOLDER + 'RMSE_score.csv')
    
    axis = RMSE_info.plot(kind='bar', figsize=(10,5), rot=0, title='RMSE scores')
    for p in axis.patches:
        axis.annotate(np.round(p.get_height(),decimals=2), 
                    (p.get_x()+p.get_width()/2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points', fontsize=14, color='black')

    fig = axis.get_figure()
    fig.savefig(STORAGE_FOLDER + 'RMSE.png')


# In[79]:


# STORAGE_FOLDER = 'output/'
future_steps = 60


# In[80]:


# look_back, hidden_nodes, output_nodes, epochs, batch_size, future_steps
parameters_FNN = [[1,2,3,6,8,10,12], [3,4,5,6], [1], [500], [20], [future_steps]]
parameters_FNN = [[12], [4], [1], [500], [20], [future_steps]]

# time_lagged_points, hidden_nodes, output_nodes, epochs, batch_size, future_steps
parameters_TLNN = [[[1,2,3,11,12], [1,2,3,4,11,12], [1,2,3,11,12,13], [1,2,3,4,5,6,10,11,12]], [3,4,5,6], [1], [300], [20], [future_steps]]
parameters_TLNN = [[[1,2,3,4,5,6,10,11,12]], [4], [1], [300], [20], [future_steps]]

# seasonal_period, hidden_nodes, epochs, batch_size, future_steps
parameters_SANN = [[12], [3,4,5,6,7,8,9,10], [500], [20], [future_steps]]
parameters_SANN = [[12], [3], [500], [20], [future_steps]]

# look_back, hidden_nodes, output_nodes, epochs, batch_size, future_steps
parameters_RNN = [[1,2,3,4,5,6,7,8,9,10,11,12,13], [3,4,5,6], [1], [300], [20], [future_steps]]
parameters_RNN = [[12], [4], [1], [300], [20], [future_steps]]

# look_back, hidden_nodes, output_nodes, epochs, batch_size, future_steps
parameters_LSTM = [[1,2,3,4,5,6,7,8,9,10,11,12,13], [3,4,5,6], [1], [300], [20], [future_steps]]
parameters_LSTM = [[12], [4], [1], [300], [20], [future_steps]]

# RMSE_info = compare_ANN_methods(rainfall_data, test_rainfall_data, scaler, 
#                    parameters_FNN, parameters_TLNN, parameters_SANN, parameters_LSTM, future_steps, STORAGE_FOLDER)


# In[76]:


# save_RMSE_info(STORAGE_FOLDER, RMSE_info)

