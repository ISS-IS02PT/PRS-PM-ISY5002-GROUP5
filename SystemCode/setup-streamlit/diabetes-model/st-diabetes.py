import streamlit as st
import pickle
import numpy as np

st.title('Testing Diabetes model with KNN')

# Load the scaler 
with open('./diabetes-scaler.pkl', 'rb') as scaler_pkl:
  scaler_load = pickle.load(scaler_pkl)
  st.text('Loading scaler ... DONE')

# Load the model
with open('./diabetes-knn-model.pkl', 'rb') as model_pkl:
  knn_load = pickle.load(model_pkl)
  st.text('Loading model ... DONE')

st.text('Loading test data')
X_unseen = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50], [1, 85, 66, 29, 0, 26.6, 0.351, 31], [8, 183, 64, 0, 0, 23.3, 0.672, 32]])
X_unseen
# X_unseen = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Apply the scale
st.text('Scaling the test data')
X_unseen_scale = scaler_load.transform(X_unseen)
X_unseen_scale

# Get the result
st.text('Prediction result')
result = knn_load.predict(X_unseen_scale)
result