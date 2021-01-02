
import streamlit as st
import numpy as np
from pickle import load

st.title("California Housing Prediction")
 
col1, col2, col3, col4 = st.beta_columns(4)

with col1:
    result1 = st.number_input('F1')
    st.text(f'{result1}')

with col2:
    result2 = st.number_input('F2')
    st.text(f'{result2}')

with col3:
    result3 = st.number_input('F3')
    st.text(f'{result3}')

with col4:
    result4 = st.number_input('F4')
    st.text(f'{result4}')

col5, col6, col7, col8 = st.beta_columns(4)

with col5:
    result5 = st.number_input('F5')
    st.text(f'{result1}')

with col6:
    result6 = st.number_input('F6')
    st.text(f'{result2}')

with col7:
    result7 = st.number_input('F7')
    st.text(f'{result3}')

with col8:
    result8 = st.number_input('F8')
    st.text(f'{result4}')

st.cache() # chache to store the model 
model = load(open('testmodel.pkl', 'rb'))

input = np.array([[float(result1), float(result2), float(result3), float(result4), float(result5), float(result6), float(result7),
                   float(result8)]])

st.text(model.predict(input)[0])
