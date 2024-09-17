import pandas as pd
import matplotlib.pyplot as plt

'''
ERROR | HYPERPARAMETER
'''

error = 'R-squared' # R-squared | RMSE
hp = 'depth' #n_estimators, lr, depth

output_dir = r'C:\Users\USER\Desktop\Maritime'
file_path = os.path.join(output_dir, "XGB_combined_mse_r2_models336.csv") #Εδώ το όνομα του combined csv 
combined_data = pd.read_csv(file_path)

boxplots_folder = os.path.join(output_dir, "Boxplots")
os.makedirs(boxplots_folder, exist_ok=True)

values = combined_data[f"{hp}"].unique()

fig, ax = plt.subplots(figsize=(12, 8))

box_colors = ['#e94a47', '#18bbba', '#152237', '#f0bb0d', '#3789b1', '#2b485c', '#ff7676','#148783','#f9e3a2','#b17e77','#b5e0dc','#fac1c0']

boxplot_data = []

for i, value in enumerate(values):
    error_values = combined_data[combined_data[f"{hp}"] == value][f"{error}"]
    boxplot_data.append(error_values)
    ax.text(i + 1, error_values.mean(), f" Mean: {error_values.mean():.2f}\n Std: {error_values.std():.2f}",
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'),fontsize=5)


boxplot = ax.boxplot(boxplot_data, labels=values, patch_artist=True, notch=True)


for patch, color in zip(boxplot['boxes'], box_colors):
    patch.set_facecolor(color)

ax.set_xlabel(f"{hp}")
ax.set_ylabel(f"{error}")
ax.set_title(f"XGBoost: Boxplots of {error} with Mean and Std annotations for different {hp} values")

ax.set_facecolor('#fdfcf9') #background color
#D5F6F6
output_file_path = os.path.join(boxplots_folder, f"XGB_{error}_Boxplot_{hp}.png")
plt.savefig(output_file_path, dpi=400)
plt.show()

#######################################################################################################################

'''
MODELS VS BASELINE
'''

error = 'RMSE'  # Error column for XGBoost models
error_b = f'{error}_B'  # Error column for baseline model

data_path = r"C:\Users\USER\Desktop\Maritime\XGB_combined_mse_r2_models336.csv"
boxplots_folder = r"C:\Users\USER\Desktop\Maritime\Boxplots"
os.makedirs(boxplots_folder, exist_ok=True)

# Read data and filter for specific depth and hyperparameters
data = pd.read_csv(data_path)
filtered_data = data[
    (data["n_estimators"].isin([100, 200])) & (data["lr"].isin([0.1, 1.0])) & (data["depth"] == 9)
]

xgb_data_dict = {}
for n_estimators in [100, 200]:
    for lr in [0.1, 1.0]:
        model_data = filtered_data[
            (filtered_data["n_estimators"] == n_estimators) & (filtered_data["lr"] == lr)
        ][error].values
        xgb_data_dict[f"XGBoost (n={n_estimators}, lr={lr})"] = model_data

baseline_data = filtered_data[(filtered_data["n_estimators"] == 100) & (filtered_data["lr"] == 0.1)][
    error_b
].values


boxplot_data = list(xgb_data_dict.values()) + [baseline_data]
boxplot_labels = list(xgb_data_dict.keys()) + ["Baseline Model"]


fig, ax = plt.subplots(figsize=(14, 8))  # Adjust figure size for better clarity

bp = ax.boxplot(
    boxplot_data,
    patch_artist=True,
    notch=True,  # Add notches for better comparison
    medianprops={"linewidth": 2},
    labels=boxplot_labels,
)

box_colors = ['#e94a47', '#18bbba', '#152237', '#f0bb0d', '#3789b1', '#2b485c', '#ff7676','#148783','#f9e3a2','#b17e77','#b5e0dc','#fac1c0']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)


ax.set_xlabel("Models")
ax.set_ylabel(error)
ax.set_title(f"Comparison of {error} for XGBoost Models with Baseline Model (depth = 9)")


ax.set_facecolor('#fdfcf9') #background

output_file_path = os.path.join(boxplots_folder, f"XGBoost_Baseline_{error}_Comparison.png")
plt.savefig(output_file_path, dpi=400)

# Display the plot
plt.show()

#######################################################################################################################

