import yfinance as yf
import os

import tensorflow as tf

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# from utils import StackLayer,BlockLayer


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

class BlockLayer(tf.keras.layers.Layer):
 def __init__(self,lookback_period,horizon,n_layers,n_units,**kwargs):
    super().__init__(**kwargs)
    self.lookback_period=lookback_period
    self.horizon=horizon
    self.n_layers=n_layers
    self.n_units=n_units

    self.fully_connected=tf.keras.Sequential([tf.keras.layers.Dense(n_units,activation='relu') for _ in range(n_layers)],name="Fully_Connected_Layer")
    self.theta_layer=tf.keras.layers.Dense(lookback_period+horizon,activation='linear',name="Theta_Layer")

 def call(self,input):

    x=self.fully_connected(input)
    backcast_forecast=self.theta_layer(x)

    backcast=backcast_forecast[:,:-self.horizon]
    forecast=backcast_forecast[:,-self.horizon:]

    return backcast,forecast

class StackLayer(tf.keras.layers.Layer):
  def __init__(self,lookback_period,horizon,n_layers,n_units,num_blocks=4,**kwargs):
    super().__init__(**kwargs)
    self.num_blocks=num_blocks
    self.horizon=horizon
    self.first_block=BlockLayer(lookback_period=lookback_period,horizon=horizon,n_layers=n_layers,n_units=n_units,name="Initial_Block")
    self.block_list=[BlockLayer(lookback_period=lookback_period,horizon=horizon,n_layers=n_layers,n_units=n_units,name=f"Block_{i}") for i in range(1,num_blocks)]

  def call(self,input):

    block_backcast,block_forecast=self.first_block(input)
    stack_forecast_residual=tf.zeros(shape=(self.horizon),dtype=tf.float32)
    stack_forecast_residual=tf.expand_dims(stack_forecast_residual,axis=0)
    stack_forecast_residual=tf.keras.layers.Add()([stack_forecast_residual,block_forecast])
    stack_backcast_residual=tf.keras.layers.Subtract()([input,block_backcast])

    for block in self.block_list:
      block_backcast,block_forecast=block(stack_backcast_residual)
      stack_forecast_residual=tf.keras.layers.Add()([block_forecast,stack_forecast_residual])
      stack_backcast_residual=tf.keras.layers.Subtract()([stack_backcast_residual,block_backcast])

    return stack_backcast_residual,stack_forecast_residual

def initialize_session():
    if "yesterday" not in st.session_state:
        st.session_state.yesterday="Available"

def load_data():
  BTC_ticker=yf.Ticker("BTC-USD")
  BT_data=BTC_ticker.history(period="1wk")
  df=pd.DataFrame({"Close":BT_data["Close"]})
  if len(df)<=7:
    st.session_state.yesterday="Not Available"
    st.error("Yesterday's Price has not yet been updated. Please come back later to predict the price of Bitcoin tomorrow.")
  else:
    st.session_state.yesterday="Available"
    return df[-7:]

@st.cache_data
def load_results():
    app_directory = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(app_directory, 'Results.csv')
    results=pd.read_csv(results_path)
    results.rename(columns={"Unnamed: 0":"Model"},inplace=True)
    return results

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
if st.session_state.yesterday=="Available":
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

    
    

