from flask import Flask, render_template,url_for,request

app = Flask(__name__)


# EDA Pkgs
import pandas as pd
import numpy as np


# Utils Pkgs
import os
import joblib
from IPython.display import display, HTML



# Fxn Loading
def load_model(model_file):
	loaded_model = joblib.load(os.path.join(model_file),"w+")
	return loaded_model


def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key



# Routes
@app.route('/')
def index():
	return render_template('index.html')

# Routes
@app.route('/about')
def about():
	return render_template('about.html')


@app.route('/dataset')
def dataset():
	df = pd.read_csv("data/clean_hepatitis_dataset.csv")
	return render_template('dataset.html', df_table=df)


@app.route('/predict',methods=['GET','POST'])
def predict():
	# Receives the input query from form
	if request.method == 'POST':
		age = request.form['age']
		sex = request.form['sex']
		steroid= request.form['steroid']
		antivirals= request.form['antivirals']
		fatigue= request.form['fatigue']
		spiders= request.form['spiders']
		ascites= request.form['ascites']
		varices= request.form['varices']
		bilirubin= request.form['bilirubin']
		alk_phosphate= request.form['alk_phosphate']
		sgot= request.form['sgot']
		albumin= request.form['albumin']
		protime= request.form['protime']
		histology= request.form['histology']
        
    # include inputs into a varible for ml transformation and evaluation
		sample_result = {"age":age,"sex":sex,"steroid":steroid,"antivirals":antivirals,"fatigue":fatigue,"spiders":spiders,"ascites":ascites,"varices":varices,"bilirubin":bilirubin,"alk_phosphate":alk_phosphate,"sgot":sgot,"albumin":albumin,"protime":protime,"histolog":histology}
		single_data = [age,sex,steroid,antivirals,fatigue,spiders,ascites,varices,bilirubin,alk_phosphate,sgot,albumin,protime,histology]
	
		# print(single_data)
		# print(len(single_data))


	# encode data for machine learning model
		numerical_encoded_data = [ float(int(x)) for x in single_data ]

 	# load pickle model
		model = load_model('models/logistic_regression_hepB_model.pkl')
		# predict and print prediction
		prediction = model.predict(np.array(numerical_encoded_data).reshape(1,-1))
		print(prediction)
		prediction_label = {"Die":1,"Live":2}
		final_result = get_key(prediction[0],prediction_label)

	# get and print prediction probability
		pred_prob = model.predict_proba(np.array(numerical_encoded_data).reshape(1,-1))
		## meaning [0][0]first array first value 
		pred_probalility_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
		pred_probalility_df = pd.DataFrame(pred_probalility_score.items(),columns= ['Status','Probability']).round(1)
		pred_probalility_df = HTML(pred_probalility_df.to_html(index=False))
	


	return render_template("index.html",
		sample_result=sample_result,
		prediction=final_result,
		pred_probalility_score=pred_probalility_df,
		)



if __name__ == '__main__':
	app.run(debug=True)
