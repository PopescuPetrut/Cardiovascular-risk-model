import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df_train = pd.read_csv('data_training.csv')
df_test = pd.read_csv('data_testing.csv')

dataset = {'Training': df_train, 'Testing': df_test}

for name, df in dataset.items():
	missing_count = df.isnull().sum()
	missing_percent = (df.isnull().sum() / len(df)) * 100
	missing_table = pd.DataFrame({'Missing values': missing_count, 'Missing percentage': missing_percent})
	print(missing_table[missing_table['Missing values'] > 0])

	print(df.describe().round(5))
	print(df.describe(include = ['object', 'str']))

	sb.set_theme(style = "whitegrid")
	
	num_cols = df.select_dtypes(include = [np.number]).columns
	plt.figure(figsize = (15, 15))

	for i, col in enumerate(num_cols, 1):
		plt.subplot(2, 3, i)
		sb.histplot(df[col].dropna(), kde = True, color = 'blue')
		plt.title(f'Distribution for {col}, ({name})')

	plt.tight_layout()
	plt.savefig(f'plots/numerical_distribution_for_{name.lower()}.png')
	plt.close()

	cat_cols = df.select_dtypes(include = ['object', 'str']).columns
	plt.figure(figsize = (15, 15))
	for i, col in enumerate(cat_cols, 1):
		plt.subplot(1, 4, i)
		sb.countplot(data = df, x = col, hue = col, palette = 'Set2', legend = False)
		plt.title(f'Distribution for {col} ({name})')
		plt.xticks(rotation = 15)
	
	plt.tight_layout()
	plt.savefig(f'plots/categorical_distribution_for_{name.lower()}.png')
	plt.close()

	plt.figure(figsize = (15, 15))
	for i, col in enumerate(['Pulse', 'Corporal body fat', 'Colesterol'], 1):
		plt.subplot(1, 3, i)
		sb.boxplot(data = df, y = col, color = 'orange')
		plt.title(f'Outlier detection {col} ({name})')

	plt.tight_layout()
	plt.savefig(f'plots/boxplot_outliers_for_{name.lower()}.png')
	plt.close()

	plt.figure(figsize = (15, 15))
	heatmap = df[num_cols].corr()
	sb.heatmap(heatmap, annot = True, cmap = 'Blues', fmt = ".2f", linewidths = 1)
	plt.title(f'Corrrelation matrix ({name})')
	plt.tight_layout()
	plt.savefig(f'plots/heatmap_for_{name.lower()}.png')
	plt.close()

	plt.figure(figsize = (15, 15))
	for i, col in enumerate(['Age', 'Pulse', 'Colesterol'], 1):
		plt.subplot(1, 3, i)
		sb.violinplot(data = df, x = 'Cardiovascular risk', y = col, hue = 'Cardiovascular risk', palette = 'Pastel1', legend = False)
		plt.title(f'{col} vs cardiovascular risk ({name})')
	
	plt.tight_layout()
	plt.savefig(f'plots/target_relations_for_{name.lower()}.png')
	plt.close()
