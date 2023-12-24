import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import useful_functions

train_data = useful_functions.get_normalized_train().sample(frac=0.01, random_state=42)

data_dim_reduction = train_data.drop(columns=['smoking']).to_numpy()

tsne = TSNE(n_components=3)

data_3d = tsne.fit_transform(data_dim_reduction)
tsne_df = pd.concat([pd.DataFrame(data=data_3d).reset_index(drop=True), train_data['smoking'].reset_index(drop=True)], axis=1, ignore_index=True)
tsne_df.columns = ['X1', 'X2', 'X3', 'smoking']


# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color representing 'smoking' labels
scatter = ax.scatter(tsne_df['X1'], tsne_df['X2'], tsne_df['X3'], c=tsne_df['smoking'], cmap='viridis')

# Legend for 'smoking' labels
legend = ax.legend(*scatter.legend_elements(), title='Smoking')
ax.add_artist(legend)

# Set labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('3D Scatter Plot of TSNE Components with Smoking Labels')
plt.show()
