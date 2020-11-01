import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

st.title('Use Case #1: Prediction of Write-Off Cases')
# st.title('Use Case #2: Prediction of Inpatient Revenue based on Patient Country of Residence')
# st.title('Use Case #3: Hospital Bill Estimation Upon Admission')


st.write(time.time())

##### Text_input
st.header("=== Text Input ===")
text_input = st.text_input("Text_input", "Default value")

##### Text_area
st.header("=== Text Area ===")
text_area = st.text_area("Text_area", "Default value")

##### Select box
st.header("=== Select box ===")
display = ("Male", "Female")
options = list(range(len(display)))
option_value = st.selectbox("Please select your gender:", display)
# option_value = st.selectbox("Please select your gender:", options, format_func=lambda x: display[x])


##### Radio
st.header("=== Radio ===")
genre = st.radio(
  "What's your favorite movie genre",
  ('Comedy', 'Drama', 'Documentary'))


##### Date picker
st.header("=== Date Picker ===")
start_date = st.date_input('Start date')
end_date = st.date_input('End date')


##### Date Slider
st.header("=== Date Slider ===")
d3 = st.date_input("Date range", [])

##### Button
st.header("=== Submit Button to collect all the inputs ===")

submit = st.button("Submit")
if not submit:
  st.stop()

st.write(text_input)
st.write(text_area)
st.write(option_value)
st.write(genre)
st.write(start_date)
st.write(end_date)
st.write(d3)

##### Progress Bar - 100%
st.header("=== Progress Bar ===")

import time

# Progress time = 5 seconds
progress_time = 5

# Add a text_placeholder
text_placeholder = st.empty()
bar = st.progress(0)

for i in range(progress_time):
  # Update the progress bar with each iteration.
  time.sleep(1)
  current_percentage = (i+1) * 1 / progress_time
  # temp_text = 
  text_placeholder.text(f'Percentage = {current_percentage*100:0.0f}%')
  bar.progress(current_percentage)
'...and now we\'re done!'