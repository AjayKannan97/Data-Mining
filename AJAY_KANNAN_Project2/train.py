# Import libraries
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from scipy.fftpack import rfft
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

# Load data
ins_df=pd.read_csv('InsulinData.csv',usecols=['Date','Time','BWZ Carb Input (grams)'])
ins_df_1=pd.read_csv('Insulin_patient2.csv',usecols=['Date','Time','BWZ Carb Input (grams)'])
cgm_df=pd.read_csv('CGMData.csv',usecols=['Date','Time','Sensor Glucose (mg/dL)'])
cgm_df_1=pd.read_csv('CGM_patient2.csv',usecols=['Date','Time','Sensor Glucose (mg/dL)'])

# Convert to datetime
ins_df['date_time_stamp']=pd.to_datetime(ins_df['Date'] + ' ' + ins_df['Time'])
ins_df_1['date_time_stamp']=pd.to_datetime(ins_df_1['Date'] + ' ' + ins_df_1['Time'])
cgm_df['date_time_stamp']=pd.to_datetime(cgm_df['Date'] + ' ' + cgm_df['Time'])
cgm_df_1['date_time_stamp']=pd.to_datetime(cgm_df_1['Date'] + ' ' + cgm_df_1['Time'])

# Create and define meal data
def create_meal_data_func(ins_df,cgm_df,dateidentifier):
    insulin_df=ins_df.copy()
    insulin_df=insulin_df.set_index('date_time_stamp')
    timestamp_30_df=insulin_df.sort_values(by='date_time_stamp',ascending=True).dropna().reset_index()
    timestamp_30_df['BWZ Carb Input (grams)'].replace(0.0,np.nan)
    timestamp_30_df=timestamp_30_df.dropna()
    timestamp_30_df=timestamp_30_df.reset_index().drop(columns='index')
    list_timestamp_valid=[]
    value=0
    for idx,i in enumerate(timestamp_30_df['date_time_stamp']):
        try:
            value=(timestamp_30_df['date_time_stamp'][idx+1]-i).seconds / 60.0
            if value >= 120:
                list_timestamp_valid.append(i)
        except KeyError:
            break
    
    list1=[]
    if dateidentifier==1:
        for idx,i in enumerate(list_timestamp_valid):
            start=pd.to_datetime(i - timedelta(minutes=30))
            end=pd.to_datetime(i + timedelta(minutes=120))
            get_date=i.date().strftime("%m/%d/%Y")
            cgm_list = cgm_df.loc[cgm_df['Date']==get_date].set_index('date_time_stamp').between_time(start_time=start.strftime('%H:%M:%S'),end_time=end.strftime('%H:%M:%S'))['Sensor Glucose (mg/dL)'].values.tolist()
            list1.append(cgm_list)
        return pd.DataFrame(list1)
    else:
        for idx,i in enumerate(list_timestamp_valid):
            start=pd.to_datetime(i - timedelta(minutes=30))
            end=pd.to_datetime(i + timedelta(minutes=120))
            get_date=i.date().strftime('%Y-%m-%d')
            list1.append(cgm_df.loc[cgm_df['Date']==get_date].set_index('date_time_stamp').between_time(start_time=start.strftime('%H:%M:%S'),end_time=end.strftime('%H:%M:%S'))['Sensor Glucose (mg/dL)'].values.tolist())
        return pd.DataFrame(list1)
    
# Create and define no meal data
def create_no_meal_data_func(ins_df,cgm_df):
    insulin_no_meal_df=ins_df.copy()
    test1_df=insulin_no_meal_df.sort_values(by='date_time_stamp',ascending=True).replace(0.0,np.nan).dropna().copy()
    test1_df=test1_df.reset_index().drop(columns='index')
    valid_timestamp=[]
    for idx,i in enumerate(test1_df['date_time_stamp']):
        try:
            value=(test1_df['date_time_stamp'][idx+1]-i).seconds//3600
            if value >=4:
                valid_timestamp.append(i)
        except KeyError:
            break
    dataset=[]
    for idx, i in enumerate(valid_timestamp):
        iteration_dataset=1
        try:
            length_of_24_dataset=len(cgm_df.loc[(cgm_df['date_time_stamp']>=valid_timestamp[idx]+pd.Timedelta(hours=2))&(cgm_df['date_time_stamp']<valid_timestamp[idx+1])])//24
            while (iteration_dataset<=length_of_24_dataset):
                if iteration_dataset==1:
                    dataset.append(cgm_df.loc[(cgm_df['date_time_stamp']>=valid_timestamp[idx]+pd.Timedelta(hours=2))&(cgm_df['date_time_stamp']<valid_timestamp[idx+1])]['Sensor Glucose (mg/dL)'][:iteration_dataset*24].values.tolist())
                    iteration_dataset+=1
                else:
                    dataset.append(cgm_df.loc[(cgm_df['date_time_stamp']>=valid_timestamp[idx]+pd.Timedelta(hours=2))&(cgm_df['date_time_stamp']<valid_timestamp[idx+1])]['Sensor Glucose (mg/dL)'][(iteration_dataset-1)*24:(iteration_dataset)*24].values.tolist())
                    iteration_dataset+=1
        except IndexError:
            break
    return pd.DataFrame(dataset)


# Def function to create meal meal data matrix
def create_meal_feature_matrix(meal_data):
    index=meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>6).dropna().index
    meal_data_cleaned=meal_data.drop(meal_data.index[index]).reset_index().drop(columns='index')
    meal_data_cleaned=meal_data_cleaned.interpolate(method='linear',axis=1)
    index_to_drop_again=meal_data_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    meal_data_cleaned=meal_data_cleaned.drop(meal_data.index[index_to_drop_again]).reset_index().drop(columns='index')
    meal_data_cleaned=meal_data_cleaned.dropna().reset_index().drop(columns='index')

    first_max=[]
    index_first_max=[]
    sec_max=[]
    index_second_max=[]
    third_max=[]
    
    for i in range(len(meal_data_cleaned)):
        array=abs(rfft(meal_data_cleaned.iloc[:,0:25].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(meal_data_cleaned.iloc[:,0:25].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        first_max.append(sorted_array[-2])
        sec_max.append(sorted_array[-3])
        third_max.append(sorted_array[-4])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    meal_feature_matrix=pd.DataFrame()
    meal_feature_matrix['second_max']=sec_max
    meal_feature_matrix['third_max']=third_max
    tm=meal_data_cleaned.iloc[:,22:25].idxmin(axis=1)
    maximum=meal_data_cleaned.iloc[:,5:19].idxmax(axis=1)
    list1=[]
    second_differential_data=[]
    standard_deviation=[]
    for i in range(len(meal_data_cleaned)):
        list1.append(np.diff(meal_data_cleaned.iloc[:,maximum[i]:tm[i]].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(meal_data_cleaned.iloc[:,maximum[i]:tm[i]].iloc[i].tolist())).max())
        standard_deviation.append(np.std(meal_data_cleaned.iloc[i]))
    meal_feature_matrix['2ndDifferential']=second_differential_data
    meal_feature_matrix['standard_deviation']=standard_deviation
    return meal_feature_matrix

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

# Create meal data
meal_data=create_meal_data_func(ins_df,cgm_df,1)
meal_data1=create_meal_data_func(ins_df_1,cgm_df_1,2)

# Truncate meal data to 2 hours
meal_data=meal_data.iloc[:,0:24]
meal_data1=meal_data1.iloc[:,0:24]

# Create no meal data
no_meal_data=create_no_meal_data_func(ins_df,cgm_df)
no_meal_data1=create_no_meal_data_func(ins_df_1,cgm_df_1)

# Create meal data matrix
meal_feature_matrix=create_meal_feature_matrix(meal_data)
meal_feature_matrix_1=create_meal_feature_matrix(meal_data1)
meal_feature_matrix=pd.concat([meal_feature_matrix,meal_feature_matrix_1]).reset_index().drop(columns='index')

# Create no meal data matrix
non_meal_feature_matrix=create_no_meal_feature_matrix(no_meal_data)
non_meal_feature_matrix_1=create_no_meal_feature_matrix(no_meal_data1)
non_meal_feature_matrix=pd.concat([non_meal_feature_matrix,non_meal_feature_matrix_1]).reset_index().drop(columns='index')

# Create label for meal and no meal data
meal_feature_matrix['label']=1
non_meal_feature_matrix['label']=0

# Create total meal and no meal data matrix
total_data=pd.concat([meal_feature_matrix,non_meal_feature_matrix]).reset_index().drop(columns='index')
dataset=shuffle(total_data,random_state=1).reset_index().drop(columns='index')

# Define KFold cross validation 
kfold = KFold(n_splits=6,shuffle=False)
              
# Copy dataset
copy_dataset = dataset.drop(columns='label')
print(copy_dataset)

# Define model
model = DecisionTreeClassifier(criterion="entropy")

# Define list to store accuracy
accuracy = []
f1 = []
recall = []
precision = []

i = 0
# Run KFold cross validation
for train_index, test_index in kfold.split(copy_dataset):
    X_train,X_test,y_train,y_test = copy_dataset.loc[train_index],copy_dataset.loc[test_index],dataset.label.loc[train_index],dataset.label.loc[test_index]
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred)) 
    f1.append(f1_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))

    # Print classification report
    print("\n Classification Report: ",str(i+1))
    i = i+1
    print(classification_report(y_pred, y_test, target_names=['0','1']))

# Mean accuracy, f1 score, recall and precision
print("\n Accuracy: ",str(np.mean(accuracy)*100))
print("\n F1 Score: ",str(np.mean(f1)*100))
print("\n Recall: ",str(np.mean(recall)*100))
print("\n Precision: ",str(np.mean(precision)*100))

# Fit model on entire dataset
classifier = DecisionTreeClassifier(criterion='entropy')
X, y = copy_dataset, dataset['label']
classifier.fit(X,y)

# Save model
pickle.dump(classifier, open('model.pkl','wb'))





