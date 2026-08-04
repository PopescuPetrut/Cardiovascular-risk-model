import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(53)

number_of_instances = 1000

age = np.random.randint(16, 90, size = number_of_instances)
sex = np.random.choice(['Masculin', 'Feminin'], size = number_of_instances, p = [0.5, 0.5])
body_fat_percentage = np.round(np.random.normal(26, 5, size = number_of_instances), 5)
pulse = np.random.randint(60, 120, size = number_of_instances)
colesterol_level = np.random.randint(100, 300, size = number_of_instances)
smoker = np.random.choice(['Yes', 'No'], size = number_of_instances, p = [0.25, 0.75])
family_history = np.random.choice(['Yes', 'No'], size = number_of_instances, p = [0.2, 0.8])
sleep_hours = np.round(np.random.uniform(3, 10, size= number_of_instances) * 7, 1)

base_risk = ((age > 60) * 2.0 + (body_fat_percentage > 29) * 1.5 + (pulse > 85) * 1.25 +
			 (colesterol_level > 250) * 2.5 + (smoker == 'Yes') * 3 + (family_history == 'Yes')
			   * 2 + (sleep_hours < 35) * 1)

error = np.random.randint(0, 1.2, size = number_of_instances)
final_risk = base_risk + error

risk = np.where(final_risk > 5, 'Yes', 'No')

df = pd.DataFrame({'Age': age, 'Gender': sex, 'Corporal body fat': body_fat_percentage, 'Pulse': pulse,
				   'Colesterol': colesterol_level, 'Smoker': smoker, 'Sleeped hours in a week': sleep_hours,
				   'Family history': family_history, 'Cardiovascular risk': risk})

mesurement_error_number = 10

mesurement_errors_pulse = np.random.choice(df.index, size = mesurement_error_number, replace = False)
mesurement_errors_body_fat = np.random.choice(df.index, size = mesurement_error_number, replace = False)

df.loc[mesurement_errors_pulse, 'Pulse'] = np.random.randint(200, 270, mesurement_error_number)
df.loc[mesurement_errors_body_fat, 'Corporal body fat'] = np.random.randint(60, 85, mesurement_error_number)

anrmoality = 7

unhealty_teens = df[(df['Age'] < 28) & (df['Smoker'] == 'No')].index
if len(unhealty_teens) > 0:
	teens_genetics = np.random.choice(unhealty_teens, size = min(anrmoality, len(unhealty_teens)), replace = False)
	df.loc[teens_genetics, 'Cardiovascular risk'] = 'Yes'

healty_elders = df[(df['Age'] > 70) & (df['Smoker'] == 'Yes')].index
if len(healty_elders) > 0:
	elders_genetics = np.random.choice(healty_elders, size = min(anrmoality, len(healty_elders)), replace = False)
	df.loc[elders_genetics, 'Cardiovascular risk'] = 'No'

nan_zone_colesterol = np.random.rand(len(df)) < 0.05
nan_zone_body_fat = np.random.rand(len(df)) < 0.05

df.loc[nan_zone_body_fat, 'Corporal body fat'] = np.nan
df.loc[nan_zone_colesterol, 'Colesterol'] = np.nan

df_train, df_test = train_test_split(df, test_size = 0.3, random_state = 53, stratify=df['Cardiovascular risk'])

df_train.to_csv('data_training.csv', index = False)
df_test.to_csv('data_testing.csv', index = False)
