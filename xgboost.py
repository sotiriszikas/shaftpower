'''
FUNCTION
'''
start_time = time.time()

def myxgboost(train_dir, test_dir, features, target_variable, n_est, lr, depth, shuffle, fold, output_dir):
    
    train_file = os.path.join(train_dir, f'train_shuffle{shuffle}_fold{fold}_.csv')
    test_file = os.path.join(test_dir, f'test_shuffle{shuffle}_fold{fold}.csv')

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    X_train, y_train = train_data[features], train_data[target_variable]
    X_test, y_test = test_data[features], test_data[target_variable]

    model = XGBRegressor(n_estimators=n_est, learning_rate=lr, max_depth=depth)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mseb = mean_squared_error(y_test, test_data['baseline'])
    rmse = np.sqrt(mse)
    rmseb = np.sqrt(mseb)
    r2 = r2_score(y_test, predictions)
    r2b = r2_score(y_test, test_data['baseline'])
    maeb = mean_absolute_error(y_test, test_data['baseline'])
    mae = mean_absolute_error(y_test, predictions)
    epsilon = 1e-10 
    mape = np.mean(np.abs((y_test - predictions) / (y_test + epsilon))) * 100 
    mapeb = np.mean(np.abs((y_test - test_data['baseline']) / (y_test + epsilon))) * 100 
    
    print(f"File: {os.path.basename(test_file)}")
    
    subfolder_name = f"XGBoost_Predictions_n{n_est}_lr{lr}_depth{depth}"
    xg_path = os.path.join(output_dir, subfolder_name)
    if not os.path.exists(xg_path):
        os.makedirs(xg_path)

    
    predictions_df = pd.DataFrame({'Prediction': predictions})


    metadata_df = pd.DataFrame({
        'Shuffle': [shuffle],
        'Fold': [fold],
        'n_estimators': [n_est],
        'lr': [lr],
        'depth': [depth],
        'MSE': [mse],
        'MSE_B': [mseb],
        'RMSE': [rmse],
        'RMSE_B': [rmseb],
        'R square': [r2],
        'R square_B': [r2b],
        'MAE': [mae],
        'MAE_B': [maeb],
        'MAPE': [mape],
        'MAPE_B': [mapeb],
    })

    output_df = pd.concat([metadata_df, test_data, predictions_df], axis=1)


    output_file = os.path.join(xg_path, f'predictions_metadata_shuffle{shuffle}_fold{fold}.csv')
    output_df.to_csv(output_file, index=False)
    print(f"Predictions and metadata saved to: {output_file}")


    metadata_only_file = os.path.join(xg_path, f'mse_r2_shuffle{shuffle}_fold{fold}.csv')
    metadata_df.to_csv(metadata_only_file, index=False)
    print(f"Metadata saved to: {metadata_only_file}\n")

    return model


train_dir = r'C:\Users\szika\Desktop\Maritime\kfold_splits\kfolds_train'
test_dir = r'C:\Users\szika\Desktop\Maritime\kfold_splits\kfolds_test'
output_dir = r'C:\Users\szika\Desktop\Maritime\kfold_splits\XGBoost'


features = [
        'perf_DaysFromDD', 'perf_DaysFromDelivery', 'perf_gpsspeed',
        'perf_draft', 'perf_waveheight', 'perf_waverelativedirection',
        'perf_windspeed', 'perf_windrelativedirection', 'perf_currentrelativedirection',
        'perf_currentvelocity']


target_variable = 'perf_shaftpower'

n_list = [5,10,15,20,25,30,35,40,45,50,100,200] 
l_list = [0.001,0.01,0.1,1]
d_list = [3,4,5,7,9,15,20]

count = 0
print(f"Starting XGBoost process for {len(n_list)* len(l_list) * len(d_list)} models.\n")

for n_est, lr, depth in product(n_list, l_list, d_list):
    count += 1
    print("______________________________________________________________________________")
    print(f"MODEL:{count} n_estimators:{n_est} | learning rate: {lr} | depth: {depth}")
    print("______________________________________________________________________________")
    for shuffle in range(1, 11):
        for fold in range(1, 6):
            myxgboost(train_dir, test_dir,features, target_variable, n_est, lr, depth, shuffle, fold, output_dir)

end_time = time.time()
elapsed_time = end_time - start_time 
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f"End of XGBoost Regression. Models created: {len(n_list)*len(l_list)*len(d_list)}")
print(f"Elapsed time: {hours} hours, {minutes} minutes, and {seconds} seconds")

#######################################################################################################################

'''
MERGING GENERATED MODEL FILES
'''

num = 0
save_path = r'C:\Users\szika\Desktop\Maritime\kfold_splits'

combined_data = pd.DataFrame()


for n_est, lr, depth in product(n_list, l_list, d_list):
    directory_path = f'C:\\Users\\szika\\Desktop\\Maritime\\kfold_splits\\XGBoost\\XGBoost_Predictions_n{n_est}_lr{lr}_depth{depth}'


    for shuffle in range(1, 11):
        for fold in range(1, 6):
            file_name = f'mse_r2_shuffle{shuffle}_fold{fold}.csv'
            file_path = os.path.join(directory_path, file_name)

            try:
                data = pd.read_csv(file_path)
                combined_data = pd.concat([combined_data, data], ignore_index=True)
                num += 1
            except FileNotFoundError:
                print(f"Model {n_est}|{lr}|{depth} not found, skipping...")

combined_file_path = os.path.join(save_path, f'XGB_combined_mse_r2_models{int(num/50)}.csv')
combined_data.to_csv(combined_file_path, index=False)

print(f"Combined data for all existing XGBoost models saved to {combined_file_path}")

#######################################################################################################################
