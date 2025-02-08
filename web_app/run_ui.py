import os
import sys

import streamlit as st
sys.path.append(os.getcwd())

from classifiers.BestClassifier import TextClassifier

from parameter_finder.ParameterFinder import ParameterFinder

@st.cache_resource
def get_parameter_finder():
    return ParameterFinder()

@st.cache_resource
def get_classifier():
    return TextClassifier()
# Page Configuration
st.set_page_config(page_title="Action Validation System", page_icon="🎮", layout="centered")

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
    background-color: #e5ff00;
    }
    </style>
    """, unsafe_allow_html=True)

# Logo Placeholder (Replace with actual logo path if available)
st.image(os.path.abspath(os.path.join(os.getcwd(), "web_app/wsc.png")))  # Replace "logo.png" with your actual logo file

# Title of the Web App
st.title("Action Validation System")
st.subheader("Identify the validity of action-phrases in basketball game commentary")

# Input Field for User to Enter the Transcript
user_input = st.text_input("Enter the transcript here:", "")

parameter_finder = get_parameter_finder()
classifier = get_classifier()

# Output Section
if user_input:
    # Simulating the output with a mock action and label for now
    # Ideally, this would run a model to predict the action and validity
    found_params = parameter_finder.get_parameters(user_input)
    label, prob = classifier.predict_with_probabilities(user_input)
    valid = True if label == 1 else False
    print(found_params)
    if len(found_params) >= 1 and label:
        st.markdown(f"**Found Parameters:** {', '.join(found_params)}")
    else:
        st.markdown("**No parameters found.**")

st.markdown("<div class='footer'>Made by Idan Arbiv</div>", unsafe_allow_html=True)
