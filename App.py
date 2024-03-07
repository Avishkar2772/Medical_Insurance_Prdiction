
import numpy as np
import pandas as pd
import pickle as pkl 
import streamlit as st
import json
from streamlit_lottie import st_lottie
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

insurance_data = pd.read_csv('insurance.csv')

#encoding the sex column
insurance_data.replace({'sex':{'female':0, 'male':1}},inplace = True)

#encoding the smoker
insurance_data.replace({'smoker':{'no':0 ,'yes':1}},inplace = True)

#encoding the region
insurance_data.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2,'northwest':3}},inplace = True)

input_data = insurance_data.drop(columns ='charges')
output_data = insurance_data['charges']


input_train_data, input_test_data, output_train_data, output_test_data = train_test_split(input_data,output_data , test_size = 0.2, random_state= 2)



# Model creation 

model = RandomForestRegressor(n_estimators=100, max_depth=7)


#Training the model 

model.fit(input_train_data,output_train_data)


#prediction on training data
test_data_prediction = model.predict(input_test_data)

#age	sex	bmi	children	smoker	region	charges
#44	female	20.235	1	yes	northeast	19594.80965
#25	female	20.800	1	no	southwest	3208.78700

input_data = (25,0,20.800,1,1,1)
input_data_array = np.asarray(input_data)
input_data_array = input_data_array.reshape(1,-1)

insurance_premium = model.predict(input_data_array)
#insurance_premium[0]

pkl.dump(model ,open('MIPML.pkl','wb'))


model = pkl.load(open('MIPML.pkl','rb'))

st.title("MEDICAL INSURANCE PREMIUM PREDICTOR")

with st.sidebar:
    text = st.text("Please fill the details")
    gender = st.selectbox('choose Gender',['Female','Male'])
    smoker = st.selectbox('Are You Smoker ?',['Yes','No'])
    region = st.selectbox('choose Region',['SouthEast','SouthWest','NorthEast','NorthWest'])
    age = st.slider('Enter Age',5,100)
    bmi = st.slider('Enter BMI',5,100)
    children = st.slider('No of Children',0,10)

#gender
if gender == 'female':
    gender = 0
    
else:
    gender = 1
 
    
#smoker
if smoker == 'No':
    smoker = 0
    
else:
    smoker = 1
    
    
#region
if region == 'SouthEast':
    region = 0
    
if region == 'Southwest':
    region = 1
    
if region == 'NorthEast':
    region = 2
    
else:
    region = 3  
    
    
    
input_data = (age,gender,bmi,children,smoker,region)
input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)

if st.button('CALCULATE PREMIUM'):
    predict_prem = model.predict(input_data)
    display_string = 'Insurance Premium will be '+str(round(predict_prem[0],2))+ ' USD dollars'
    st.markdown(display_string)
        
        
def load_lottiefiles(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_coding = load_lottiefiles(r"C:\Users\admin\Desktop\Final_Project\coding.json")

st.lottie(
    lottie_coding,
    speed=2,
    reverse=True,
    loop=True,
    quality="low",
    height=300,
    width=650,
    key=None
    )










