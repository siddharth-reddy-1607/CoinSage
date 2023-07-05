# import tensorflow as tf

# class BlockLayer(tf.keras.layers.Layer):
#  def __init__(self,lookback_period,horizon,n_layers,n_units,**kwargs):
#     super().__init__(**kwargs)
#     self.lookback_period=lookback_period
#     self.horizon=horizon
#     self.n_layers=n_layers
#     self.n_units=n_units

#     self.fully_connected=tf.keras.Sequential([tf.keras.layers.Dense(n_units,activation='relu') for _ in range(n_layers)],name="Fully_Connected_Layer")
#     self.theta_layer=tf.keras.layers.Dense(lookback_period+horizon,activation='linear',name="Theta_Layer")

#  def call(self,input):

#     x=self.fully_connected(input)
#     backcast_forecast=self.theta_layer(x)

#     backcast=backcast_forecast[:,:-self.horizon]
#     forecast=backcast_forecast[:,-self.horizon:]

#     return backcast,forecast

# class StackLayer(tf.keras.layers.Layer):
#   def __init__(self,lookback_period,horizon,n_layers,n_units,num_blocks=4,**kwargs):
#     super().__init__(**kwargs)
#     self.num_blocks=num_blocks
#     self.horizon=horizon
#     self.first_block=BlockLayer(lookback_period=lookback_period,horizon=horizon,n_layers=n_layers,n_units=n_units,name="Initial_Block")
#     self.block_list=[BlockLayer(lookback_period=lookback_period,horizon=horizon,n_layers=n_layers,n_units=n_units,name=f"Block_{i}") for i in range(1,num_blocks)]

#   def call(self,input):

#     block_backcast,block_forecast=self.first_block(input)
#     stack_forecast_residual=tf.zeros(shape=(self.horizon),dtype=tf.float32)
#     stack_forecast_residual=tf.expand_dims(stack_forecast_residual,axis=0)
#     stack_forecast_residual=tf.keras.layers.Add()([stack_forecast_residual,block_forecast])
#     stack_backcast_residual=tf.keras.layers.Subtract()([input,block_backcast])

#     for block in self.block_list:
#       block_backcast,block_forecast=block(stack_backcast_residual)
#       stack_forecast_residual=tf.keras.layers.Add()([block_forecast,stack_forecast_residual])
#       stack_backcast_residual=tf.keras.layers.Subtract()([stack_backcast_residual,block_backcast])

#     return stack_backcast_residual,stack_forecast_residual
