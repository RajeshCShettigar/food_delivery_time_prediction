import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# load the saved models
with open('xgbr_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('rfr_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# load the feature importance data
feat_imp = pd.read_csv('newfile.csv')

# define the function to make predictions using the XGBoost model
def xgb_predict(age, distance, rating):
    data = np.array([[age, distance, rating]])
    return xgb_model.predict(data)[0]

# define the function to make predictions using the Random Forest model
def rf_predict(age, distance, rating):
    data = np.array([[age, distance, rating]])
    return rf_model.predict(data)[0]


# set up the Streamlit app
st.set_page_config(page_title='Food Delivery Time Prediction',layout='centered')

st.markdown(
    """
    <style>
    .reportview-container {
        width: 90%;
    }
    .gradient-text {
        background: linear-gradient(135deg, #FF6FD8 10%, #3813C2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <h1 style='text-align: center;'>
        🍕🍔🍟🍗Food Delivery Time Prediction🍗🍟🍔🍕
    </h1>
    """,
    unsafe_allow_html=True
)

st.write('Enter the following information to get an estimate of the delivery time:')

# create input fields for age, distance, and rating
age = st.number_input('Delivery Partner Age', min_value=18, max_value=80, value=30, step=1)
distance = st.number_input('Delivery Distance (km)', min_value=1, max_value=20, value=5, step=1)
rating = st.number_input('Delivery Partner Rating (out of 5)', min_value=1.0, max_value=5.0, value=4.0, step=0.1)

# make predictions using the XGBoost and Random Forest models
xgb_prediction = xgb_predict(age, distance, rating)
rf_prediction = rf_predict(age, distance, rating)

# display the predicted delivery time for each model
st.write('XGBoost Regressor Prediction:', round(xgb_prediction, 2), 'minutes')
st.write('Random Forest Regressor Prediction:', round(rf_prediction, 2), 'minutes')


df = pd.read_csv('newfile.csv')
print(df.head(3))


# create a graph to show feature importance
fig =   px.scatter(data_frame = df,x="Delivery_person_Ratings",y="Time_taken(min)",size="Time_taken(min)",color = "distance",
                    trendline="ols", 
                    title = "Relationship Between Time Taken and Ratings",width=800,height=600)
st.plotly_chart(fig)

fig = px.scatter(data_frame = df, 
                    x="Delivery_person_Age",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    color = "distance",
                    trendline="ols", 
                    title = "Relationship Between Time Taken and Age",width=800,height=600)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig)

fig = px.scatter(data_frame = df, 
                    x="distance",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    trendline="ols", 
                    title = "Relationship Between Distance and Time Taken")
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig)

# create a graph to compare the predicted and actual values
df['xgb_prediction'] = xgb_model.predict(df[['Delivery_person_Age', 'Delivery_person_Ratings','distance']])
df['rf_prediction'] = rf_model.predict(df[['Delivery_person_Age', 'Delivery_person_Ratings','distance']])
fig = px.scatter(data_frame=df, x='Time_taken(min)', y=['xgb_prediction', 'rf_prediction'],
                 labels={'value': 'Delivery Time (minutes)', 'variable': 'Model'},
                 title='Predicted vs Actual Delivery Time',width=800,height=600)
fig.update_traces(marker=dict(size=8))
st.plotly_chart(fig)
