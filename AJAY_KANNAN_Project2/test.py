# Import libraries
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import rfft
from joblib import load

# Load test data 
data=pd.read_csv('test.csv',header=None)

# Def function to create no meal data matrix
def create_no_meal_feature_matrix(no_meal_data):
    index_to_remove_non_meal=no_meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
    no_meal_data_cleaned=no_meal_data.drop(no_meal_data.index[index_to_remove_non_meal]).reset_index().drop(columns='index')
    no_meal_data_cleaned=no_meal_data_cleaned.interpolate(method='linear',axis=1)
    index_to_drop_again=no_meal_data_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    no_meal_data_cleaned=no_meal_data_cleaned.drop(no_meal_data_cleaned.index[index_to_drop_again]).reset_index().drop(columns='index')
    non_meal_feature_matrix=pd.DataFrame()  


    first_max=[]
    index_first_max=[]
    sec_max=[]
    index_second_max=[]
    third_max=[]
    for i in range(len(no_meal_data_cleaned)):
        array=abs(rfft(no_meal_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(no_meal_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        first_max.append(sorted_array[-2])
        sec_max.append(sorted_array[-3])
        third_max.append(sorted_array[-4])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    non_meal_feature_matrix['second_max']=sec_max
    non_meal_feature_matrix['third_max']=third_max
    first_differential_data=[]
    second_differential_data=[]
    standard_deviation=[]
    for i in range(len(no_meal_data_cleaned)):
        first_differential_data.append(np.diff(no_meal_data_cleaned.iloc[:,0:24].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(no_meal_data_cleaned.iloc[:,0:24].iloc[i].tolist())).max())
        standard_deviation.append(np.std(no_meal_data_cleaned.iloc[i]))
    non_meal_feature_matrix['2ndDifferential']=second_differential_data
    non_meal_feature_matrix['standard_deviation']=standard_deviation
    return non_meal_feature_matrix


# Define feature matrix for non-meal data
dataset = create_no_meal_feature_matrix(data)
print(dataset)

# Load model
with open('model.pkl', 'rb') as f:
    model = load(f)
    pred = model.predict(dataset)    
    f.close()

# Save prediction
res = pd.DataFrame(pred, columns=['Prediction'])
print(res)
res.to_csv('Result.csv',index=False,header=False)

