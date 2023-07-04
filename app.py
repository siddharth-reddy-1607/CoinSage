import yfinance as yf
import os

import tensorflow as tf

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import StackLayer,BlockLayer


plt.style.use("dark_background")
mpl.rcParams['figure.facecolor'] = '#121212' 
mpl.rcParams['axes.facecolor'] = '#121212'
mpl.rcParams['grid.color'] = 'gray' 
mpl.rcParams['grid.linestyle'] = ':'

mpl.rcParams['text.color'] = 'white'  
mpl.rcParams['axes.labelcolor'] = 'white'
mpl.rcParams['axes.edgecolor'] = 'white'
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['ytick.color'] = 'white'

@st.cache_data
def load_results():
    app_directory = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(app_directory, 'Results.csv')
    results=pd.read_csv(results_path)
    results.rename(columns={"Unnamed: 0":"Model"},inplace=True)
    return results

def load_data():
  BTC_ticker=yf.Ticker("BTC-USD")
  BTC_info=BTC_ticker.info
  BT_data=BTC_ticker.history(period="1wk")
  df=pd.DataFrame({"Close":BT_data["Close"]})
  return df[1:] #Gives 7 Days Not Including Today, so drop the first row


@st.cache_resource(show_spinner=False)
def load_model():
    app_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(app_directory, 'final_n_beats_v1')
    model = tf.keras.models.load_model(model_path,custom_objects={"BlockLayer":BlockLayer,
                                                                  "StackLayer":StackLayer})
    return model


df=load_data()
info=st.container()

with info:
    st.header("CoinSage")
    st.subheader("How It Works ?🤔")
    st.markdown("CoinSage uses the [NBeats](https://arxiv.org/abs/1905.10437) , a pure Deep Learning Model created by ElementAI that won the M4 Competition. The model uses a Stack of Blocks with residual connections (to prevent overfitting), where each Block consists of Fully Connected Layers. I have trained the model on closing Bitcoin Prices of the Last Five Years, with a window of 7 (it predicts the price of the BitCoin tomorrow using the prices of BitCoin in the previous week) starting from July 2nd,2018 till July 1st 2023. The model acheived a stunning MAE of 400 USD of the Test Set. However, since forecasting the price of BitCoin is linked with Aleatory Uncertainty, the model is not expected to be 100% accurate and we recommed to you not heavily rely on the model's predictions but rather use it as a reference. I am constantly working on finding better models to deal with Aleatory Uncertainty and Im currently working on understanding Bayesian Neural Networks. I will update the app once I build and test the model.")
    st.subheader("Other Models I Trained")
    st.markdown("In the process of creating this app, I have trained other Deep Learning Models suchs as DNNs, CNNs, LSTMs and also an Ensemble of DNNs trained on different loss functions. However (although not by much), I have found that NBeats outperforms all of them. The below plot shows the MAE of the different models on the Test Set.")
    results=load_results()
    fig,ax=plt.subplots(figsize=(15,12))
    ax=plt.bar(results["Model"],results["MAE"],width=0.3,label="MAE",color=(0.1,0.6,0.6,0.9))
    for i in range(len(results["Model"])):
        plt.text(i,results["MAE"][i]+2,round(results["MAE"][i],2),ha="center",fontsize=13)
    plt.xticks(rotation=90,fontsize=15)
    plt.xlabel("Model",fontsize=15)
    plt.ylabel("MAE (in USD)",fontsize=15)
    plt.title("MAE of Different Models",fontsize=20)
    st.pyplot(fig)

col1,col2,col3=st.columns(3)

if col2.button("Predict Tomorrow's Bitcoin"):
    fig, ax = plt.subplots(figsize=(10,7))
    ax=plt.plot(df["Close"])
    ax=plt.title("BTC-USD Closing Prices For Last Week")
    ax=plt.xlabel("Date")
    ax=plt.ylabel("Price(USD)")
    st.pyplot(fig)
    with st.spinner("Loading Model..."):
        model=load_model()
    st.success("Model Loaded Successfully!!")
    with st.spinner("Predicting..."):
        prediction=model.predict(df["Close"].values.reshape(1,7))
    st.success("Prediction Done!!")
    st.subheader(f"Tomorrow's Bitcoin Price is expected to be {prediction[0][0]:.2f} USD")

    
    

