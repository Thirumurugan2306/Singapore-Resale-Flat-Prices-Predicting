# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load  # Used for saving and loading models
import streamlit as st

# Data Collection and Preprocessing
url = "https://beta.data.gov.sg/collections/189/download"
data = pd.read_csv(url)
# Preprocess the data as needed (cleaning, structuring)

# Feature Engineering
# Extract relevant features and create additional features
features = ['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'lease_commence_date']
target = 'resale_price'
X = data[features]
y = data[target]

# Model Selection and Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()  # You can choose a different model based on your analysis
model.fit(X_train, y_train)

# Save the trained model
dump(model, 'resale_price_model.joblib')

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Streamlit Web Application
def predict_price(town, flat_type, storey_range, floor_area, flat_model, lease_commence_date):
    input_data = pd.DataFrame([[town, flat_type, storey_range, floor_area, flat_model, lease_commence_date]], columns=features)
    prediction = model.predict(input_data)
    return prediction[0]

st.title("Singapore HDB Resale Price Prediction")
# Create input elements for user input
town = st.selectbox("Town", data['town'].unique())
flat_type = st.selectbox("Flat Type", data['flat_type'].unique())
storey_range = st.selectbox("Storey Range", data['storey_range'].unique())
floor_area = st.number_input("Floor Area (sqm)", min_value=1)
flat_model = st.selectbox("Flat Model", data['flat_model'].unique())
lease_commence_date = st.number_input("Lease Commence Date", min_value=1960, max_value=2023)

# Get prediction
if st.button("Predict Resale Price"):
    prediction = predict_price(town, flat_type, storey_range, floor_area, flat_model, lease_commence_date)
    st.success(f"Predicted Resale Price: ${prediction:,.2f}")

# Streamlit API for Model
@st.cache(allow_output_mutation=True)
def load_model():
    return load('resale_price_model.joblib')

# Load the model
loaded_model = load_model()

# Streamlit API endpoint for prediction
@st.experimental_singleton
def predict_api(input_data):
    return loaded_model.predict(input_data)

# Streamlit API endpoint for user input
@st.experimental_singleton
def input_api(town, flat_type, storey_range, floor_area, flat_model, lease_commence_date):
    return pd.DataFrame([[town, flat_type, storey_range, floor_area, flat_model, lease_commence_date]], columns=features)


