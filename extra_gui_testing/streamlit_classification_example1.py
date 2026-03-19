# pip install streamlit
import streamlit as st
import numpy as np
import pandas as pd

# NOTE!
# using the model outside your training code:
# It's recommended to use Pipelines etc.
# in order to avoid scaling problems
# if not using pipelines, you have TO ALSO SAVE THE SCALER
# instance so we can scale the new values correctly here

# load the model, see the KNN-classification example in the bottom
# for how to save a model in joblib -format
from joblib import load
model = load("social_model1.joblib")

# category names for predictions
labels = ["No", "Yes"]

tester_row = {
    'Gender': 0, 
    'Age': 55, 
    'EstimatedSalary': 67000 
}

# create a pandas DataFrame and scale the values
# this is just a sanity check to see our model 
# loaded correctly, and the we got an output that makes sense
# NOTE! use a Pipeline in order to avoid any potential 
# scaling issues!
tester_row = pd.DataFrame([tester_row])

print("All probabilities by category:")
print(model.predict_proba(tester_row))
print()

print("Did this customer buy the service (Yes/No):")
result = labels[model.predict(tester_row)[0]]
print(result)
print("-------------------")

# STREAMLIT APP START
 
# title of the streamlit app
st.title("Social network classifier 2026")

# title of the sidebar for inputs
st.sidebar.title("Input features")

# 'Gender': 0, 
# 'Age': 55, 
# 'EstimatedSalary': 67000 

continuous_var1 = st.sidebar.slider("Gender", min_value=0.0, max_value=1.0, value=0.0, step=1.0)
continuous_var2 = st.sidebar.slider("Age", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
continuous_var3 = st.sidebar.slider("EstimatedSalary", min_value=0.0, max_value=500000.0, value=30000.0, step=1000.0)

# wrap up the variables in the original order
# in NumPy -array format

# FOLLOW THE SAME ORDER AS IN THE ORIGINAL TESTER ROW
input_data = np.array([[
    continuous_var1,
    continuous_var2,
    continuous_var3,
]], dtype=float)

st.image("socialmedia.png", caption="Social Media", use_container_width=True)

# the button that predicts through the model
if st.button("Predict"):
    st.subheader("Prediction (did the custom buy service):")

    # MODIFY THE PREDICTION FUNCTION AS NEEDED BASED ON THE MODEL YOU HAVE
    # this is using the classification predict_proba
    # in regression it's usually different (compare to tester_row -codes)
    result = model.predict_proba(input_data)
    result_text = labels[np.argmax(result)]
    st.write(result_text)


# first change directory to where the streamlit -script is (streamlit_classification_example1.py)
# in this case
# for example in terminal => cd extra_gui_testing

# run the streamlit app in terminal:
# streamlit run streamlit_classification_example1.py
