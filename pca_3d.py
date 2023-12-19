import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def zscore_norm(df, column_name):
    return (df[column_name] - df[column_name].mean()) / df[column_name].std()

def min_max_norm(df, column_name):
    return (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())    


df = pd.read_csv("dataset.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)




normalized_df = train_df.copy()
#Preprocessing
normalized_df['hemoglobin']          = zscore_norm(normalized_df, 'hemoglobin')
normalized_df['hearing(right)']      = normalized_df['hearing(right)'] - 1
normalized_df['fasting blood sugar'] = zscore_norm(normalized_df, 'fasting blood sugar')
normalized_df['LDL']                 = zscore_norm(normalized_df, 'LDL')
normalized_df['height(cm)']          = zscore_norm(normalized_df, 'height(cm)')
normalized_df['weight(kg)']          = zscore_norm(normalized_df, 'weight(kg)')
normalized_df['Cholesterol']         = zscore_norm(normalized_df, 'Cholesterol')
normalized_df['serum creatinine']    = zscore_norm(normalized_df, 'serum creatinine')
normalized_df['Gtp']                 = zscore_norm(normalized_df, 'Gtp')


data_dim_reduction = normalized_df.drop(columns=['Unnamed: 0', 'smoking']).to_numpy()

pca = PCA(3)
pca.fit(data_dim_reduction)
transformed_data = pca.transform(data_dim_reduction)
transformed_df = pd.DataFrame(data=transformed_data, columns=['PC1', 'PC2', 'PC3'])
transformed_df = pd.concat([transformed_df.reset_index(drop=True), normalized_df['smoking'].reset_index(drop=True)], axis=1, ignore_index=True)
transformed_df.columns = ['PC1', 'PC2', 'PC3', 'smoking']

sampled_df = transformed_df.sample(frac=0.01, random_state=42)



# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color representing 'smoking' labels
scatter = ax.scatter(sampled_df['PC1'], sampled_df['PC2'], sampled_df['PC3'], c=sampled_df['smoking'], cmap='viridis')

# Legend for 'smoking' labels
legend = ax.legend(*scatter.legend_elements(), title='Smoking')
ax.add_artist(legend)

# Set labels and title
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D Scatter Plot of Principal Components with Smoking Labels')
plt.show()