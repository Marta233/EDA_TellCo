import streamlit as st
import pandas as pd
from Pagelist import Home, customer_overview, experience, user_engagement

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('notebook/raw_data.csv')
        return data
    except FileNotFoundError:
        st.error("The file 'notebook/raw_data.csv' was not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading the data: {e}")
        return pd.DataFrame()

data = load_data()

# Clear sidebar
st.sidebar.empty()
st.sidebar.header("Navigation")

# Custom CSS for sidebar styling
css = """
<style>
    .sidebar .sidebar-content {
        background-color: #f4f4f4; /* Light background */
    }
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 5px 0;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #45a049; /* Darker green */
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Define sidebar icons and labels
sidebar_icons = {
    "Home": "🏠",
    "User Experience": "👥",
    "Customer Overview": "📋",
    "User Engagement Analysis": "🔍"
}

# Initialize selected page variable
selected_page = "Home"

# Create styled buttons in the sidebar
if st.sidebar.button(f"{sidebar_icons['Home']} Home"):
    selected_page = "Home"

if st.sidebar.button(f"{sidebar_icons['Customer Overview']} Customer Overview"):
    selected_page = "Customer Overview"

if st.sidebar.button(f"{sidebar_icons['User Experience']} User Experience"):
    selected_page = "User Experience"

if st.sidebar.button(f"{sidebar_icons['User Engagement Analysis']} User Engagement"):
    selected_page = "User Engagement Analysis"

# Render the selected page
if selected_page == "Home":
    Home.show_default_page()
elif selected_page == "Customer Overview":
    customer_overview.show_customer_overview(data)
elif selected_page == "User Experience":
    experience.show_experience(data)
elif selected_page == "User Engagement Analysis":
    user_engagement.show_user_engagement_page(data)
