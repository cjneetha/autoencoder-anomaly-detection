import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
sns.set()
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import keras.backend.tensorflow_backend as tb 
tb._SYMBOLIC_SCOPE.value = True



st.title('Anomaly Detector')
st.markdown('This app uses an Autoencoder Network to classify a data point as an Anomaly.')
st.markdown('1) On the left sidebar, configure the Network Architecture.')
st.markdown('2) Create a new data point.')
st.markdown('3) Hit \'Go\' to test if it\'s an Anomaly based on a threshold.')


@st.cache(suppress_st_warning=True)
def get_data(distribution='normal', seed=42):
	np.random.seed(seed)
	if distribution == 'gaussian':
		x = np.random.normal(0, 1, 100)
		y = np.random.normal(0, 1, 100)
	elif distribution == 'uniform':
		x = np.random.uniform(0, 1, 100)
		y = np.random.uniform(0, 1, 100)
	elif distribution == 'exponential':
		x = np.random.exponential(scale=1, size=100)
		y = np.random.exponential(scale=1, size=100)
	elif distribution == 'beta (α=β=0.5)':
		x = np.random.beta(a=0.5, b=0.5, size=100)
		y = np.random.beta(a=0.5, b=0.5, size=100)
	elif distribution == 'chisquare (k=1)':
		x = np.random.chisquare(1, size=100)
		y = np.random.chisquare(1, size=100)
	else:
		x = np.random.normal(0, 1, 100)
		y = np.random.normal(0, 1, 100)

	data = pd.DataFrame(zip(x, y), columns=['x', 'y'])
	return data

@st.cache(suppress_st_warning=True)
def plot_data(data):
	data = data.copy()
	data['size'] = 12
	fig = px.scatter(data, x='x', y='y', size='size')
	return fig

@st.cache(suppress_st_warning=True)
def autoencoder(data, num_units_h1, num_units_h2, act_func):

	train_x = data.copy()
	model=Sequential() 
	model.add(Dense(num_units_h1, activation=act_func,
	                kernel_initializer='glorot_uniform',
	                kernel_regularizer=regularizers.l2(0.0),
	                input_shape=(train_x.shape[1],)
	               )
	         )
	model.add(Dense(num_units_h2, activation=act_func,
	                kernel_initializer='glorot_uniform'))
	model.add(Dense(train_x.shape[1],
	                kernel_initializer='glorot_uniform'))
	model.compile(loss='mae',optimizer='adam')
	model.fit(train_x, train_x, verbose=0)
	pred_x = pd.DataFrame(model.predict(train_x), columns=train_x.columns)
	train_x['mae'] = (pred_x - train_x).abs().mean(axis=1).values
	return model, train_x

# Sidebar inputs
st.sidebar.header('Configure Data')
data_dist = st.sidebar.selectbox('Select Data Distribution', 
	['gaussian', 'uniform', 'exponential', 'beta (α=β=0.5)', 'chisquare (k=1)'])


data_seed = st.sidebar.slider('Choose a Random Seed to Resample Data', 0, 100)

st.sidebar.header('Configure Network Architecture')
num_units_h1 = st.sidebar.slider('Neurons in the 1st Hidden Layer', 
	1, 5, 5)
num_units_h2 = st.sidebar.slider('Neurons in the 2nd Hidden Layer', 
	1, 5, 1)
act_function = st.sidebar.selectbox('Activation Function', 
	['elu', 'relu', 'tanh', 'sigmoid'], index=0)

st.sidebar.header('Create New Point')
x = float(st.sidebar.text_input('Enter X:', -1))
y = float(st.sidebar.text_input('Enter Y:', 1))


# Plot data and get reconstruction mae for training data
data = get_data(data_dist, data_seed)
fig = plot_data(data)
fig_placeholder = st.empty()
fig_placeholder.plotly_chart(fig)

threshold_placeholder = st.empty()
rcloss_placeholder = st.empty()
st.write('	')
message = st.empty()
message.warning('Training Network...')
model, train_x = autoencoder(data, num_units_h1, num_units_h2, act_function)
message.success('Network Trained... Select a Threshold & Click \'Go\' on the Left Sidebar!')

st.sidebar.header('Select Anomaly Threshold')
# Display training mae distribution
sns.distplot(train_x.mae, bins=50)
plt.title('Reconstruction Loss Distribution for Training Points')
st.sidebar.pyplot()

# Select threshold
threshold = float(st.sidebar.text_input('Anomaly Threshold:', train_x.mae.quantile(0.95).round(2)))
#threshold = st.sidebar.slider('Anomaly Threshold', 
#	train_x.mae.min(), train_x.mae.max(), train_x.mae.quantile(0.95))
go_button = st.sidebar.button('Go')

# If Go button pressed
if go_button:
	message.warning('Calculating Reconstruction Loss...')
	# Create dataframe with entered data
	test_x = pd.DataFrame({'x': [x], 'y': [y]})
	# Get the prediction
	pred = pd.DataFrame(model.predict(test_x.values), 
		columns=test_x.columns)
	# Get the reconstruction mae
	test_x['mae'] = [(test_x - pred).abs().mean(axis=1).iloc[0].round(4)]
	to_plot = pd.concat([train_x, test_x], ignore_index=True, axis=0)
	to_plot['size'] = 12
	to_plot.at[to_plot.index[-1], 'size'] = 15	
	to_plot['color'] = 'Inliers'
	to_plot.at[to_plot.index[-1], 'color'] = 'New Point'
	fig = px.scatter(to_plot, x='x', y='y', color='color', size='size')
	fig_placeholder.plotly_chart(fig)
	rc_loss = test_x.mae.iloc[0]
	#rcloss_placeholder.subheader('Reconstruction Loss: ' + str(rc_loss))
	#threshold_placeholder.subheader('Threshold: ' + str(round(threshold, 4)))
	if rc_loss > threshold:
		text = 'The new point is an **Anomaly**, because its reconstruction loss is greater than the threshold'
	else:
		text = 'The new point is **not an Anomaly** since its reconstruction loss is less than the threshold.'

	result = pd.DataFrame({'value': [threshold, rc_loss],
		'type': ['Threshold', 'Reconstruction Loss']})
	
	result_fig = go.Figure(go.Bar(
            x=result.value,
            y=result.type,
            marker_color=result.value,
            orientation='h'))
	result_fig.update_layout(height=230, width=600, hovermode="y")


	st.plotly_chart(result_fig)

	message.markdown(text)


