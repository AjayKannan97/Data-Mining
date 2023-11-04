# Libraries
import pandas as pd
import numpy as np
import math
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

from scipy.fftpack import fft
from scipy.stats import iqr
from scipy import signal


# Importing the dataset - Insulin Data and Glucose Data

# Insulin Data
insulin_data = pd.read_csv('InsulinData.csv', parse_dates=[['Date','Time']], keep_date_col=True, low_memory=False)
insulin_df = insulin_data[['Date_Time', 'Index', 'BWZ Carb Input (grams)']]
insulin_df.loc[:, 'Index']

# Glucose Data
glucose_data = pd.read_csv('CGMData.csv', parse_dates=[['Date','Time']], keep_date_col=True, low_memory=False)
glucose_df = glucose_data[['Date_Time', 'Sensor Glucose (mg/dL)']]

# Functions

# Choose bin for carb input
def choose_bin(x, min_carb, total_bins):
    partition = float((x - min_carb)/20)
    bin =  math.floor(partition)
    if bin == total_bins:
        bin = bin - 1
    return bin

# Calculate entropy of a row
def entropy(df_row):
    """Calculate the entropy of a pandas DataFrame row."""
    entropy = 0
    if len(df_row) <= 1:
        return 0
    else:
        values, counts = np.unique(df_row, return_counts=True)
        freqs = counts / len(df_row)
        non_zero_freqs = np.count_nonzero(freqs)
        if non_zero_freqs <= 1:
            return 0
        for f in freqs:
            entropy -= f * np.log2(f)
        return entropy

# Calulate power spectral density
def calculate_psd(row):
    f,power_values=signal.periodogram(row)
    psd1=power_values[:5].mean()
    psd2=power_values[5:10].mean()
    psd3=power_values[10:16].mean()
    return psd1,psd2,psd3

# Calculate FFT
def calculate_fft(row_values):
    # Compute the periodogram of row_values
    periodogram = signal.periodogram(row_values)
    # Compute the FFT of row_values
    fft_values = fft(row_values)
    # Compute the power spectrum of the FFT
    power_spectrum = np.abs(fft_values)**2
    # Sort the power spectrum in descending order
    sorted_indices = np.argsort(power_spectrum)[::-1]
    # Ignore the 1st peak
    sorted_indices = sorted_indices[1:]
    # Consider the 2nd to 7th peak's power only
    selected_indices = sorted_indices[:6]
    selected_power = []
    for ind in selected_indices:
        selected_power.append(power_spectrum[ind])
    return selected_power

# Define the features   
meals = []
meals_df = pd.DataFrame()
meal_matrix = pd.DataFrame()
two_hours = 60 * 60 * 2
thirty_min = 30 * 60
sensor_time_interval = 30

bin_matrix = []
bins = []
min_carb = 0
max_carb = 0
total_bins = 0

# Create a copy of the dataframes
processed_insulin_df = insulin_df.copy()
processed_glucose_df = glucose_df.copy()

# process insulin data
valid_carb_input = processed_insulin_df['BWZ Carb Input (grams)'].notna() & processed_insulin_df['BWZ Carb Input (grams)'] != 0.0
processed_insulin_df = processed_insulin_df.loc[valid_carb_input][['Date_Time', 'BWZ Carb Input (grams)']]
processed_insulin_df.set_index(['Date_Time'], inplace = True)
processed_insulin_df = processed_insulin_df.sort_index().reset_index()

valid_glucose = processed_glucose_df['Sensor Glucose (mg/dL)'].notna()
processed_glucose_df = processed_glucose_df.loc[valid_glucose][['Date_Time', 'Sensor Glucose (mg/dL)']]
processed_glucose_df.set_index(['Date_Time'], inplace = True)
processed_glucose_df = processed_glucose_df.sort_index().reset_index()

# Total number of bins
min_carb = processed_insulin_df['BWZ Carb Input (grams)'].min()
max_carb = processed_insulin_df['BWZ Carb Input (grams)'].max()
total_bins = math.ceil((max_carb - min_carb) / 20)

# Assign bins
for i in range(len(processed_insulin_df)):
    carb_input = processed_insulin_df['BWZ Carb Input (grams)'][i]
    selected_bin = choose_bin(carb_input, min_carb, total_bins)
    bins.append(selected_bin)

processed_insulin_df['bin'] = bins

# Create a list of meals
for i in range(0, len(processed_insulin_df)-1):
    time_diff_seconds = (processed_insulin_df.iloc[i + 1]['Date_Time'] - processed_insulin_df.iloc[i]['Date_Time']).total_seconds()
    if(time_diff_seconds > two_hours):
        meals.append(True)
    else:
        meals.append(False)
    
meals.append(True)
meals_df = processed_insulin_df[meals]

# Create a matrix of meals
for i in range(len(meals_df)):
    lower_bound = meals_df.iloc[i]['Date_Time'] - datetime.timedelta(seconds=thirty_min)
    upper_bound = meals_df.iloc[i]['Date_Time'] + datetime.timedelta(seconds=two_hours)
    is_within_bounds = (processed_glucose_df['Date_Time'] >= lower_bound) & (processed_glucose_df['Date_Time'] < upper_bound)
    bin = meals_df.iloc[i]['bin']
    filtered_glucose_df = processed_glucose_df[is_within_bounds]
    
    if len(filtered_glucose_df.index) == sensor_time_interval:
        filtered_glucose_df = filtered_glucose_df.T
        filtered_glucose_df.drop('Date_Time', inplace=True)
        
        filtered_glucose_df.reset_index(drop=True, inplace=True)
        filtered_glucose_df.columns = list(range(1, 31))
        
        meal_matrix = meal_matrix.append(filtered_glucose_df, ignore_index=True)
        bin_matrix.append(bin)

# Convert to numeric
meal_matrix = meal_matrix.apply(pd.to_numeric)
bin_matrix = np.array(bin_matrix)

# Create features
features = pd.DataFrame()

for i in range(0, meal_matrix.shape[0]):
    x = meal_matrix.iloc[i, :].tolist()
    
    fft_powerValues=calculate_fft(x)
    psd1,psd2,psd3=calculate_psd(x)
    features = features.append({

        "VelocityMin": np.diff(x).min(),
        "VelocityMax": np.diff(x).max(),
        "VelocityMean": np.diff(x).mean(),
        "AccelerationMin":np.diff(np.diff(x)).min(),
        "AccelerationMax":np.diff(np.diff(x)).max(),
        "AccelerationMean":np.diff(np.diff(x)).mean(),
        "Entropy": entropy(x),
        "iqr":iqr(x),
        "fft1": fft_powerValues[0],
        "fft2": fft_powerValues[1],
        "fft3": fft_powerValues[2],
        "fft4": fft_powerValues[3],
        "fft5": fft_powerValues[4],
        "fft6": fft_powerValues[5],
        "psd1":psd1,
        "psd2":psd2,
        "psd3":psd3
    },
    ignore_index=True
    )

# Convert to numeric
feature_matrix = features.to_numpy()

# Scale features to have zero mean and unit variance
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_matrix)
print('scaled_features: ', scaled_features)

# Define Cluster centers
def ground_truth_cm(k, predicted_labels, true_labels):
    confusion_matrix = np.zeros((k, k))
    for i, true_label in enumerate(true_labels):
        predicted_label = predicted_labels[i]
        confusion_matrix[predicted_label][true_label] += 1
    return confusion_matrix

# Calculate SSE for DBSCAN
def calc_dbscan_sse(labels, feature_matrix):
    total_cluster_variance = 0
    max_cluster_label = max(labels)
    for i in range(max_cluster_label + 1):
        cluster_points = feature_matrix[labels == i]
        centered_points = cluster_points - np.mean(cluster_points, axis=0)
        cluster_variance = np.sum(centered_points ** 2)
        total_cluster_variance += cluster_variance
    return total_cluster_variance

# Calculate SSE for KMeans 
def calc_cluster_entropy(confusion_matrix):
    total_sum = np.sum(confusion_matrix)
    num_bins = confusion_matrix.shape[0]
    cluster_entropies = []

    for i in range(num_bins):
        row_sum = np.sum(confusion_matrix[i])
        if row_sum == 0:
            continue
        cluster_entropy = 0
        for j in range(num_bins):
            if confusion_matrix[i,j] == 0:
                continue
            col_fraction = confusion_matrix[i,j] / row_sum
            entropy = -1 * col_fraction * np.log2(col_fraction)
            cluster_entropy += entropy
        cluster_entropies.append((row_sum / total_sum) * cluster_entropy)

    return np.sum(cluster_entropies)

# Calculate cluster purity for KMeans
def calc_cluster_purity(confusion_matrix):
    total_sum = np.sum(confusion_matrix)
    num_bins = confusion_matrix.shape[0]
    cluster_purities = []

    for i in range(num_bins):
        row_max = np.max(confusion_matrix[i])
        row_sum = np.sum(confusion_matrix[i])
        if row_sum == 0:
            continue
        cluster_purity = row_max / row_sum
        cluster_purities.append((row_sum / total_sum) * cluster_purity)

    return np.sum(cluster_purities)


# Calculate KMeans  
kmeans = KMeans(n_clusters=total_bins, random_state=0).fit(scaled_features)
kmeans_centroid_locations = kmeans.cluster_centers_
kmeans_labels = kmeans.labels_
kmeans_gtm = ground_truth_cm(int(total_bins), kmeans_labels, bin_matrix)
kmeans_sse = kmeans.inertia_
kmeans_entropy = calc_cluster_entropy(kmeans_gtm)
kmeans_purity = calc_cluster_purity(kmeans_gtm)

# Calculate DBSCAN 
default_epsilon = 50 # Radius of the neighborhood to be considered. 
dbscan = DBSCAN(eps=default_epsilon, min_samples=total_bins, metric="euclidean").fit(feature_matrix)
dbscan_labels = dbscan.labels_
dbscan_clusters = len(np.unique(dbscan_labels))
dbscan_outliers = np.sum(np.array(dbscan_labels) == -1, axis=0)
dbscan_gtm = ground_truth_cm(int(total_bins), dbscan_labels, bin_matrix)

# Calculate SSE, Entropy and Purity for DBSCAN
dbscan_sse = calc_dbscan_sse(dbscan_labels, scaled_features)
dbscan_entropy = calc_cluster_entropy(dbscan_gtm)
dbscan_purity = calc_cluster_purity(dbscan_gtm)

# Print results
output = pd.DataFrame(
    [
        [
            kmeans_sse,
            dbscan_sse,
            kmeans_entropy,
            dbscan_entropy,
            kmeans_purity,
            dbscan_purity,
        ]
    ],
    columns=[
        "SSE for KMeans",
        "SSE for DBSCAN",
        "Entropy for KMeans",
        "Entropy for DBSCAN",
        "Purity for KMeans",
        "Purity for DBSCAN",
    ],
)
output = output.fillna(0)

# Save results to CSV
output.to_csv("Results.csv", index=False, header=None)



