
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import math

import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import itertools
import warnings
warnings.filterwarnings('ignore')


# In[28]:


def get_ARIMA_best_parameters(rainfall_data):

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None
    temp_model = None
    AIC_scores_list = []
    for param in pdq:
        print(param)
        for param_seasonal in seasonal_pdq:        
            try:
                temp_model = sm.tsa.statespace.SARIMAX(rainfall_data,
                                                 order = param,
                                                 seasonal_order = param_seasonal,
                                                 enforce_stationarity=True,
                                                 enforce_invertibility=True)
                results = temp_model.fit()
                l = []
                l.append(param[0])
                l.append(param[1])
                l.append(param[2])
                l.append(param_seasonal[0])
                l.append(param_seasonal[1])
                l.append(param_seasonal[2])
                l.append(param_seasonal[3])
                print(results.aic)
                l.append(results.aic)
                AIC_scores_list.append(l)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
            except:
                continue
    print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
    return best_pdq, best_seasonal_pdq, best_aic, AIC_scores_list


# In[29]:


def get_best_model(rainfall_data, best_pdq, best_seasonal_pdq):
    best_model = sm.tsa.statespace.SARIMAX(rainfall_data,
                                          order=best_pdq,
                                          seasonal_order=best_seasonal_pdq,
                                          enforce_stationarity=True,
                                          enforce_invertibility=True)
    best_results = best_model.fit()
    return best_model, best_results


# In[30]:


def get_forcasted_values(best_results, n_steps, alpha):
    pred_uc = best_results.get_forecast(steps=n_steps, alpha=alpha)
    pred_ci = pred_uc.conf_int()
    forecasted_values = pred_uc.predicted_mean
    return forecasted_values


# In[43]:


def save_results(AIC_scores_list, SOURCE_FOLDER):
    AIC_scores = pd.DataFrame.from_records(AIC_scores_list)
    AIC_scores.columns=['p', 'd', 'q', 'P', 'D', 'Q', 's', 'AIC']
    AIC_scores.to_csv(SOURCE_FOLDER + '/' + 'ARIMA_information.csv')
    optimized_values_ARIMA = AIC_scores.iloc[AIC_scores['AIC'].argmin()]
    optimized_values_ARIMA.to_csv(SOURCE_FOLDER + '/' + 'ARIMA_optimized_values.csv')


# In[41]:


def ARIMA(rainfall_data, SOURCE_FOLDER, n_steps):
    alpha = 0.05

    best_pdq, best_seasonal_pdq, best_aic, AIC_scores_list = get_ARIMA_best_parameters(rainfall_data)
    best_pdq = (0, 1, 1)
    best_seasonal_pdq = (0, 1, 1, 12)
    best_model, best_results = get_best_model(rainfall_data, best_pdq, best_seasonal_pdq)
    forecasted_values = get_forcasted_values(best_results, n_steps, alpha)
    save_results(AIC_scores_list, SOURCE_FOLDER)
    return forecasted_values

# In[46]:


# filename = 'pred.csv'
# rainfall_data = pd.read_csv(filename, index_col=0)
# SOURCE_FOLDER = ''
# ARIMA(rainfall_data, SOURCE_FOLDER)

