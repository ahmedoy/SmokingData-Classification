import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

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


data_dim_reduction = normalized_df.drop(columns=['Unnamed: 0', 'smoking']).sample(frac=0.01, random_state=42).to_numpy()

tsne = TSNE(n_components=3)

data_3d = tsne.fit_transform(data_dim_reduction)
tsne_df = pd.concat([pd.DataFrame(data=data_3d).reset_index(drop=True), normalized_df['smoking'].reset_index(drop=True)], axis=1, ignore_index=True)
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