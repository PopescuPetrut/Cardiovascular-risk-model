import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv('data_training.csv')

for col in df_train.select_dtypes(include = [np.number]).columns:
	df_train[col] = df_train[col].fillna(df_train[col].mean())
for col in df_train.select_dtypes(include = ['object', 'str']).columns:
	df_train[col] = df_train[col].fillna(df_train[col].mode()[0])

df_train['Gender'] = df_train['Gender'].map({'Masculine': 1, 'Feminin': 0})
df_train['Smoker'] = df_train['Smoker'].map({'Yes': 1, 'No': 0})
df_train['Family history'] = df_train['Family history'].map({'Yes': 1, 'No': 0})
df_train['Cardiovascular risk'] = df_train['Cardiovascular risk'].map({'Yes': 1, 'No': 0})

x_train = df_train.drop(columns = 'Cardiovascular risk')
y_train = df_train['Cardiovascular risk'].astype(int)

model = RandomForestClassifier(n_estimators = 100, random_state = 53)
model.fit(x_train, y_train)

def risk_prediction(age, sex, imc, pulse, colesteorl, smoker, sleeping_hours, fam_history):
	if sex == 'Masculine':
		gen = 1
	else:
		gen = 0

	if smoker == 'Yes':
		smokeing = 1
	else:
		smokeing = 0

	if fam_history == 'Yes':
		fam = 1
	else:
		fam = 0

	pacient_data = [age, gen, imc, pulse, colesteorl, smokeing, sleeping_hours, fam]
	col = ['Age', 'Gender', 'Corporal body fat', 'Pulse', 'Colesterol', 'Smoker', 'Sleeped hours in a week', 'Family history']

	df_pacint = pd.DataFrame([pacient_data], columns = col)

	prediction = model.predict(df_pacint)[0]

	if prediction == 1:
		return "High cardiovascular risk"
	else:
		return "Small cardiovascular risk"
	

interface = gr.Interface(

	fn = risk_prediction,
	inputs = [
		gr.Slider(minimum=16, maximum=90, step=1, value=40, label="Vârstă"),
		gr.Radio(choices=["Masculin", "Feminin"], value="Masculin", label="Gen"),
		gr.Slider(minimum=5, maximum=50, step=1, value=25, label="Procent Grăsime Corporală"),
		gr.Slider(minimum=50, maximum=150, step=1, value=80, label="Puls (BPM)"),
		gr.Slider(minimum=100, maximum=400, step=1, value=200, label="Nivel Colesterol"),
		gr.Radio(choices=["Da", "Nu"], value="Nu", label="Fumător?"),
		gr.Slider(minimum=10, maximum=80, step=1, value=45, label="Ore de somn pe săptămână"),
		gr.Radio(choices=["Da", "Nu"], value="Nu", label="Istoric în familie?")
	],

	outputs = gr.Textbox(label = "Predictia modelului AI"),

	title = "Sistem Medical de Predicție - Risc Cardiovascular",
	
	description="Introduceți datele pacientului pentru a evalua riscul folosind modelul Random Forest."

)

interface.launch()
