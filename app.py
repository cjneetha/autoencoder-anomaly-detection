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


@st.cache(suppress_st_warning=True)
def get_data(distribution='normal', seed=42):
	np.random.seed(seed)
	if distribution == 'gaussian':
		x = np.random.normal(0, 1, 100)
		y = np.random.normal(0, 1, 100)
	elif distribution == 'beta (α=β=0.5)':
		x = np.random.beta(0.5, 0.5, 100)
		y = np.random.beta(0.5, 0.5, 100)
	elif distribution == 'uniform':
		x = np.random.uniform(-1.5, 1.5, 100)
		y = np.random.uniform(-1.5, 1.5, 100)

	data = pd.DataFrame(zip(x, y), columns=['x', 'y'])
	return data

@st.cache(suppress_st_warning=True)
def plot_data(data):
	data = data.copy()
	data['size'] = 12
	fig = px.scatter(data, x='x', y='y', size='size')
	return fig

@st.cache(suppress_st_warning=True)
def autoencoder(data, num_units_h1, num_units_h2, act_func, epochs):
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
	model.compile(loss='mae', optimizer='adam')
	model.fit(train_x, train_x, batch_size=20, epochs=epochs, verbose=0)
	pred_x = pd.DataFrame(model.predict(train_x), columns=train_x.columns)
	train_x['mae'] = (pred_x - train_x).abs().mean(axis=1).values
	return model, train_x


st.title('Anomaly Detector')
#st.markdown('This app uses an Autoencoder Network to classify a data point as an Anomaly.')
#st.markdown('1) On the left sidebar, configure the Network Architecture.')
#st.markdown('2) Create a new data point.')
#st.markdown('3) Hit \'Go\' to test if it\'s an Anomaly based on a threshold.')
instructions = st.empty()
instructions.markdown('This app uses an Autoencoder Network to classify a data point as an Anomaly.<br>\
	1) On the left sidebar, configure the Network Architecture. <br>\
	2) Create a new data point. <br> \
	3) Hit \'Go\' to test if it\'s an Anomaly based on a threshold.',
	 unsafe_allow_html=True)


# Sidebar data inputs
st.sidebar.header('Configure Data')
data_dist = st.sidebar.selectbox('Select Data Distribution', 
	['gaussian', 'uniform', 'beta (α=β=0.5)'])
seed = st.sidebar.slider('Change Random Seed to Resample Data', 0, 100)
data = get_data(data_dist, seed)

# Sidebar network inputs
st.sidebar.header('Configure Network Architecture')
num_units_h1 = st.sidebar.slider('Neurons in the 1st Hidden Layer', 
	1, 5, 5)
num_units_h2 = st.sidebar.slider('Neurons in the 2nd Hidden Layer', 
	1, 5, 1)
act_function = st.sidebar.selectbox('Activation Function', 
	['elu', 'relu', 'tanh', 'sigmoid'], index=0)
epochs = st.sidebar.slider('Epochs', 100, 1000, 200)

# Sidebar new point inputs
st.sidebar.header('Create New Point')
x = float(st.sidebar.text_input('Enter X:', 2))
y = float(st.sidebar.text_input('Enter Y:', 2))


# Plot data and get reconstruction mae for training data
fig = plot_data(data)
fig_placeholder = st.empty()
fig_placeholder.plotly_chart(fig)

st.write('	')
message = st.empty()
message.warning('Training Network...')
model, train_x = autoencoder(data, num_units_h1, num_units_h2, act_function, epochs)
message.success('Network Trained... select a threshold on the left & click \'Go\'!')

# Sidebar threshold inputs
st.sidebar.header('Select Anomaly Threshold')
# Display training mae distribution
sns.distplot(train_x.mae, bins=50)
plt.title('Reconstruction Loss Distribution for Training Points')
st.sidebar.pyplot()

# Select threshold
threshold = float(st.sidebar.text_input('Anomaly Threshold:', train_x.mae.quantile(0.95).round(2)))
go_button = st.sidebar.button('Go')


# If Go button pressed
if go_button:

	# Remove the initial instructions to clear some space
	instructions.empty()

	message.warning('Calculating Reconstruction Loss...')
	# Create dataframe with entered data
	test_x = pd.DataFrame({'x': [x], 'y': [y]})
	# Get the prediction
	pred = pd.DataFrame(model.predict(test_x.values), 
		columns=test_x.columns)
	# Get the reconstruction mae
	test_x['mae'] = [(test_x - pred).abs().mean(axis=1).iloc[0].round(4)]
	to_plot = pd.concat([train_x, test_x, pred], ignore_index=True, axis=0)
	to_plot['size'] = 12
	to_plot['type'] = 'Inliers'
	to_plot.at[to_plot.index[-2], 'type'] = 'New Point'
	to_plot.at[to_plot.index[-1], 'type'] = 'Reconstructed'

		
	rc_loss = test_x.mae.iloc[0]
	mean_rc_loss = train_x['mae'].mean().round(4)
	if rc_loss > threshold:
		text = 'The new point is an **Anomaly**, because its reconstruction loss is greater than the threshold'
		explanation = '**Explanation:** <br> The average loss for reconstructing the training data was **\
		' + str(mean_rc_loss) +'**. The loss to reconstruct the the new point was **' \
		+ str(rc_loss) + '**, which is much higher than that. Therfore, it is unlikely to belong to the\
		training data distribution. <br><br>If you think the prediction is wrong, you can try:<br> \
		- tweaking the network by **decreasing** the number of neurons. <br>\
		- training for **fewer** epochs, as the network could have **overfit**. <br>\
		- **increasing** the outlier threshold. <br><br>\
		Also, keep in mind that in an autoencoder network, the 2nd hidden layer is where the compression happens, \
		and since there are only 2 features, the number of neurons in the 2nd hidden layer should be 1 or at most 2.'
	else:
		text = 'The new point is **not an Anomaly** since its reconstruction loss is less than the threshold.'
		explanation = '**Explanation:** <br> The average loss for reconstructing the training data was **\
		' + str(mean_rc_loss) +'**. The loss to reconstruct the new point was **' \
		+ str(rc_loss) + '**, which is not too far away. Therfore, it is likely to belong to the\
		training data distribution. <br><br>If you think the prediction is wrong, you can try:<br> \
		- tweaking the network by **increasing** the number of neurons. <br>\
		- training for **more** epochs, as the network could have **underfit** <br>\
		- **decreasing** the outlier threshold. <br><br>\
		Also, keep in mind that in an autoencoder network, the 2nd hidden layer is where the compression happens, \
		and since there are only 2 features, the number of neurons in the 2nd hidden layer should be 1 or at most 2.'

	result = pd.DataFrame({'value': [threshold, rc_loss],
		'type': ['Threshold', 'Reconstruction Loss']})
	result_fig = go.Figure(go.Bar(
            x=result.value,
            y=result.type,
            marker_color=result.value,
            orientation='h'))
	result_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=100, width=600, hovermode="y")


	scatter_fig = px.scatter(to_plot, x='x', y='y', color='type', size='size')
	fig_placeholder.plotly_chart(scatter_fig)
	st.plotly_chart(result_fig)
	message.markdown(text)

	st.markdown(explanation, unsafe_allow_html=True)


