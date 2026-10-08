import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df_train = pd.read_csv('data_training.csv')
df_test = pd.read_csv('data_testing.csv')

num_cols = df_train.select_dtypes(include = [np.number]).columns
cat_cols = df_train.select_dtypes(include = ['object', 'str']).columns

for col in num_cols:
	mean_val = df_train[col].mean()
	df_train[col] = df_train[col].fillna(mean_val)
	df_test[col] = df_test[col].fillna(mean_val)

for col in cat_cols:
	most_frq = df_train[col].mode()[0]
	df_train[col] = df_train[col].fillna(most_frq)
	df_test[col] = df_test[col].fillna(most_frq)

dataset = [df_train, df_test]

for df in dataset:
	df['Gender'] = df['Gender'].map({'Masculin': 1, 'Feminin': 0})
	df['Smoker'] = df['Smoker'].map({'Yes': 1, 'No': 0})
	df['Family history'] = df['Family history'].map({'Yes': 1, 'No': 0})
	df['Cardiovascular risk'] = df['Cardiovascular risk'].map({'Yes': 1, 'No': 0})

x_train = df_train.drop(columns = 'Cardiovascular risk')
y_train = df_train['Cardiovascular risk'].astype(int)

x_test = df_test.drop(columns = 'Cardiovascular risk')
y_test = df_test['Cardiovascular risk'].astype(int)

model = RandomForestClassifier(n_estimators = 100, random_state = 53)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Model accuracy is: {accuracy * 100: .2f}%\n')

print("Classification description:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (6, 5))
sb.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['Low risk (0)', 'High risk (1)'],
		   yticklabels = ['Low risk (0)', 'High risk (1)']);

plt.xlabel('Model prediction')
plt.ylabel('Real value')
plt.title('Confussion matrix for Random Forest model')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png')
plt.close()