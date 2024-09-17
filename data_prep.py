
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from xgboost import XGBRegressor
from itertools import product
import geopandas as gpd
import time
import pyproj


from scipy.spatial.distance import cdist
from pykrige.ok import OrdinaryKriging
import pyproj
import pykrige.kriging_tools as kt
from shapely.geometry import LineString
from shapely.geometry import Point
import geopandas as gpd
from scipy.interpolate import NearestNDInterpolator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

#######################################################################################################################

directory = os.getcwd()
start_csv_filename = '20200101_Data_v2.csv' #Initial Dataset
start_csv_file= os.path.join(directory, start_csv_filename) 

start_data = pd.read_csv(start_csv_file)
print(f"Dataset with {len(start_data)} rows and {len(start_data.columns)} columns.")

start_data.head()

#######################################################################################################################
'''
CREATING BASELINE MODEL CALCULATIONS FOR LATER MODEL COMPARISON
'''
start_data['baseline'] = start_data['perf_WindPower'] + start_data['perf_wavepower'] + (start_data['perf_roughnessfactor'] * start_data['perf_powermodellogspeed'])

base_csv_filename = f'baseline_{start_csv_filename}'
base_csv_file = os.path.join(directory, base_csv_filename)
start_data.to_csv(base_csv_file, index=False)

print(f"New column 'baseline' added, and the DataFrame has been saved to {base_csv_filename} at {directory}")

#######################################################################################################################

'''
REMOVING SHIPS WITH SPEEED>5KM
'''

base2_csv_filename = f'baseline_{start_csv_filename}'
base2_csv_file= os.path.join(directory, base_csv_filename) 

data = pd.read_csv(base2_csv_file)

filtered_data = data[data['perf_gpsspeed'] > 5]
cleaned_csv_filename = f'cleaned_{start_csv_filename}'
cleaned_csv_file = os.path.join(directory, cleaned_csv_filename)
filtered_data.to_csv(cleaned_csv_file, index=False)

print(f"Rows with perf_gpsspeed < 5 have been removed. Cleaned file saved at {directory} under the name {cleaned_csv_filename}")

#######################################################################################################################

'''
CREATING MODEL SELECTION DATASETS |STANDARDIZATION K-FOLDS
'''
start_time = time.time()

def shuffle_and_kfold(csv_file,num, exclude_columns, k=5, output_dir='kfold_splits', random_state=None):
  
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_file)

    columns_to_standardize = [col for col in data.columns if col not in exclude_columns]

    means = data[columns_to_standardize].mean()
    stds = data[columns_to_standardize].std()

    standardized_data = (data[columns_to_standardize] - means) / stds

    for col in exclude_columns:
        standardized_data[col] = data[col]
        
    standardized_data_shuffled= standardized_data.sample(frac=1, random_state=random_state).reset_index(drop=True)


    shuffled_data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    shuffled_filename = os.path.join(output_dir, f'shuffled_data_{num}.csv')
    shuffled_data.to_csv(shuffled_filename, index=False)
    print(f"Shuffled dataset saved to: {shuffled_filename}")

    standardized_data_shuffled_filename = os.path.join(output_dir, f'standardized_shuffled_data_{num}.csv')
    standardized_data_shuffled.to_csv(standardized_data_shuffled_filename, index=False)
    print(f"Standardized Shuffled dataset saved to: {standardized_data_shuffled_filename}")

    training_dir = os.path.join(output_dir, 'kfolds_train')
    testing_dir = os.path.join(output_dir, 'kfolds_test')
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(testing_dir, exist_ok=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(kf.split(standardized_data_shuffled), 1):
        train_data = standardized_data_shuffled.iloc[train_idx]
        test_data =  standardized_data_shuffled.iloc[test_idx]

        train_filename = os.path.join(training_dir, f'train_shuffle{num}_fold{fold}_.csv')
        test_filename = os.path.join(testing_dir, f'test_shuffle{num}_fold{fold}.csv')

        train_data.to_csv(train_filename, index=False)
        test_data.to_csv(test_filename, index=False)

        print(f"Fold {fold} - Train data saved to: {train_filename}")
        print(f"Fold {fold} - Test data saved to: {test_filename}")

    print(f"k folds for shuffle {num} finished.")
    print("___________________________________________\n")


model_selection_csv = 'data_80%.csv'
exclude_columns = ["perf_mv_id",  "perf_date",  "perf_Lat",  "perf_Lon",  "perf_DaysFromDD", 
                   "perf_DaysFromDelivery", "perf_shaftpower", "baseline"] #the columns that we don't want to be standardized 

for num in range(1,11):
    print(f"SHUFFLE:{num}\n")
    random_state = np.random.default_rng().integers(0, 2**32 - 1)
    shuffle_and_kfold(model_selection_csv, num, exclude_columns, k=5, output_dir='kfold_splits',random_state=random_state)


end_time = time.time()
elapsed_time = end_time - start_time 
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print("End of the k fold process.")
print(f"Elapsed time: {hours} hours, {minutes} minutes, and {seconds} seconds")

#######################################################################################################################