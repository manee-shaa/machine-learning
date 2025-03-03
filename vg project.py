import os
import joblib
import numpy as np
import streamlit as st
import pandas as pd

# Load dataset
DATASET_PATH = r"C:\Users\manis\Downloads\cleaned_global_video_game_sales.csv"
if os.path.exists(DATASET_PATH):
    try:
        data = pd.read_csv(DATASET_PATH)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()
else:
    st.error("Dataset file not found.")
    st.stop()

# Function to load the saved model and label encoders
def load_model():
    try:
        model_path = r"C:\Users\manis\Desktop\python 2\module\ML\project\project 2\Rf_model.joblib"
        label_paths = {
            "Name": r"C:\Users\manis\Desktop\python 2\module\ML\project\project 2\L1.pkl",
            "Platform": r"C:\Users\manis\Desktop\python 2\module\ML\project\project 2\L2.pkl",
            "Genre": r"C:\Users\manis\Desktop\python 2\module\ML\project\project 2\L3.pkl",
            "Publisher": r"C:\Users\manis\Desktop\python 2\module\ML\project\project 2\L4.pkl",
        }
        model = joblib.load(model_path)
        label_encoders = {key: joblib.load(path) for key, path in label_paths.items()}
        return model, label_encoders
    except Exception as e:
        st.error(f"Error loading model or label encoders: {e}")
        st.stop()

# Load the model and label encoders
model, label_encoders = load_model()

# Streamlit app
st.title("Video Game Global Sales Prediction")
st.write("Predict global sales based on game details.")

# Input fields for user data
try:
    required_columns = ["Rank", "Name", "Platform", "Year", "Genre", "Publisher", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"Missing columns in dataset: {missing_columns}")
        st.stop()

    # Set up the "Name" field with pre-selected value
    names = sorted(data["Name"].dropna().unique())
    default_name = names[0]  # Default to the first game in the list
    input_data = {
        "Rank": st.number_input("Rank", min_value=1, step=1),
        "Name": st.selectbox("Game Name", names, index=names.index(default_name)),  # Selectbox with pre-selected name
        "Platform": st.selectbox("Platform", sorted(data["Platform"].dropna().unique())),
        "Year": st.slider("Year of Release", int(data['Year'].min()), int(data['Year'].max()), 2010),
        "Genre": st.selectbox("Genre", sorted(data["Genre"].dropna().unique())),
        "Publisher": st.selectbox("Publisher", sorted(data["Publisher"].dropna().unique())),
        "NA_Sales": st.number_input("North America Sales (in millions)", min_value=0.0, step=0.01),
        "EU_Sales": st.number_input("Europe Sales (in millions)", min_value=0.0, step=0.01),
        "JP_Sales": st.number_input("Japan Sales (in millions)", min_value=0.0, step=0.01),
        "Other_Sales": st.number_input("Other Regions Sales (in millions)", min_value=0.0, step=0.01),
    }
except KeyError as e:
    st.error(f"Error accessing dataset columns: {e}")
    st.stop()

# Function to preprocess input data
def preprocess_input(input_data, label_encoders, expected_features):
    processed_data = []
    try:
        for feature in expected_features:
            if feature in label_encoders:
                processed_data.append(label_encoders[feature].transform([input_data[feature]])[0])
            else:
                processed_data.append(input_data[feature])
    except Exception as e:
        st.error(f"Error during input preprocessing: {e}")
        st.stop()
    return np.array(processed_data).reshape(1, -1)

# Prediction logic
if st.button("Predict Global Sales"):
    try:
        # Expected features based on model training
        expected_features = ["Rank", "Name", "Platform", "Year", "Genre", "Publisher", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]

        # Preprocess the input data
        processed_data = preprocess_input(input_data, label_encoders, expected_features)

        # Ensure processed data matches model input shape
        if processed_data.shape[1] != model.n_features_in_:
            st.error("Mismatch between processed data and model input features.")
            st.stop()

        # Adjust prediction by summing regional sales if necessary
        regional_sales_sum = input_data["NA_Sales"] + input_data["EU_Sales"] + input_data["JP_Sales"] + input_data["Other_Sales"]
        prediction = model.predict(processed_data)
        adjusted_prediction = regional_sales_sum

        # Display the result
        st.success(f"The predicted global sales for the game is: {adjusted_prediction:.2f} million units")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
