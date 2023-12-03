import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import load

# Load the best model
best_model = load('random_forest_model_reduced.joblib')
# Load your original DataFrame (replace this with your actual data loading code)
df = pd.read_csv('combined.csv')

# Function to preprocess user input
def preprocess_input(user_input):
    # Your preprocessing logic here
    return user_input

# Function to predict using the best model
def predict_price(model, input_data):
    # Your prediction logic here
    prediction = model.predict(input_data)
    return prediction

# Streamlit App
def main():
    st.set_page_config(
            page_title="Housing Price Prediction App",
            layout="wide",
        )
    
    
    # Define CSS styles for the title
    title_style = """
        color: white;
        text-align: center;
        padding: 10px;
        background-color: Grey;
        border-radius: 15px; /* Adjust the value to control the curvature */
    """
    
    st.markdown(f'<h1 style="{title_style}">Housing Price Prediction App</h1>', unsafe_allow_html=True)
    # User input form
    st.header("User Input")
    town = st.selectbox("Town", df['town'].unique())
    month = st.selectbox("Month", df['month'].unique())
    flat_type = st.selectbox("Flat Type", df['flat_type'].unique())


    cbd_dist = st.slider("CBD Distance", min_value=0, max_value=15000, value=5000)
    min_dist_mrt = st.slider("Min Distance to MRT", min_value=0, max_value=2000, value=500)
    floor_area_sqm = st.slider("Floor Area (sqm)", min_value=40, max_value=200, value=80)
    lease_remain_years = st.slider("Lease Remaining (years)", min_value=30, max_value=99, value=60)
    storey_median = st.slider("Storey Range Median", min_value=0, max_value=50, value=25)

    
# Dropdowns for categorical features
   

    user_input = {
        'cbd_dist': cbd_dist,
        'min_dist_mrt': min_dist_mrt,
        'floor_area_sqm': floor_area_sqm,
        'lease_remain_years': lease_remain_years,
        'storey_median': storey_median
    }

    # Preprocess user input
    input_data = preprocess_input(pd.DataFrame(user_input, index=[0]))

    # Standardize input data
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)
    
    col1, col2, col3 = st.columns(3)
    
    if col2.button("Predict"):
        # Get prediction
        prediction = predict_price(best_model, input_data)

        # Display prediction
        st.success(f"The predicted housing price is: {prediction[0]:,.2f} SGD")

if __name__ == "__main__":
    main()
