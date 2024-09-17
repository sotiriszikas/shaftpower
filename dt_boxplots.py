import pandas as pd
import matplotlib.pyplot as plt

'''
ERROR BOXPLOTS 
'''

boxplots_folder = os.path.join(output_dir, "Boxplots")
os.makedirs(boxplots_folder, exist_ok=True)

error = 'R-squared'

output_dir = r'C:\Users\USER\Desktop\Maritime'
file_path = os.path.join(output_dir, "DT_combined_mse_r2_models15.csv")
combined_data = pd.read_csv(file_path)

depth_values = combined_data["Depth"].unique()

fig, ax = plt.subplots(figsize=(12, 8))

box_colors = ['#e94a47', '#18bbba', '#152237', '#f0bb0d', '#3789b1', '#2b485c', '#ff7676','#148783','#f9e3a2','#b17e77','#b5e0dc','#fac1c0']

boxplot_data = []

for i, depth in enumerate(depth_values):
    error_values = combined_data[combined_data["Depth"] == depth][f"{error}"]
    boxplot_data.append(error_values)
    ax.text(i + 1, error_values.mean(), f" Mean: {error_values.mean():.2f}\n Std: {error_values.std():.2f}",
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='white'),fontsize=3)


boxplot = ax.boxplot(boxplot_data, labels=depth_values, patch_artist=True,notch=True)


for patch, color in zip(boxplot['boxes'], box_colors):
    patch.set_facecolor(color)

ax.set_xlabel("Depth")
ax.set_ylabel(f"{error}")
ax.set_title(f"DT: Boxplot of {error} with Mean and Std annotations for different depth values")

ax.set_facecolor('#fdfcf9') #background color


output_file_path = os.path.join(boxplots_folder, f"DT_{error}_Boxplot.png")
plt.savefig(output_file_path, dpi=400)
plt.show()

#######################################################################################################################

'''
DT MODELS VS BASELINE
'''

depth = 10
error = 'R-squared'

output_dir = r'C:\Users\USER\Desktop\Maritime'
boxplots_folder = os.path.join(output_dir, "Boxplots")
os.makedirs(boxplots_folder, exist_ok=True)

file_path = os.path.join(output_dir, "DT_combined_mse_r2_models15.csv")
combined_data = pd.read_csv(file_path)

depth_filtered_data = combined_data[combined_data["Depth"] == depth]

error_columns = [error, f'{error}_B']
boxplot_data = [depth_filtered_data[error].values for error in error_columns]

fig, ax = plt.subplots(figsize=(12, 8))
box_colors = ['#e94a47', '#18bbba']  

# Specify labels during boxplot creation
boxplot = ax.boxplot(boxplot_data, patch_artist=True,notch=True,
                     labels=["DT Model", "Baseline Model"])

for patch, color in zip(boxplot['boxes'], box_colors):
  patch.set_facecolor(color)


ax.set_xlabel("Models")
ax.set_ylabel(error)
ax.set_title(f"Comparison of {error} and {error}_B at Depth {depth}")

ax.set_facecolor('#fdfcf9') #background color


output_file_path = os.path.join(boxplots_folder, f"DT_Baseline_{error}_Comparison.png")
plt.savefig(output_file_path, dpi=400)

plt.show()

#######################################################################################################################