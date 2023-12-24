import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import useful_functions
from sklearn.cluster import KMeans


train_data = useful_functions.get_normalized_train()

data_dim_reduction = train_data.drop(columns=['smoking']).to_numpy()

pca = PCA(3)
pca.fit(data_dim_reduction)
transformed_data = pca.transform(data_dim_reduction)
transformed_df = pd.DataFrame(data=transformed_data, columns=['PC1', 'PC2', 'PC3'])
transformed_df = pd.concat([transformed_df.reset_index(drop=True), train_data['smoking'].reset_index(drop=True)], axis=1, ignore_index=True)
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