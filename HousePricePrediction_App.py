import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

st.set_page_config(
    page_title="Prapti's House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("Prapti's House Price Prediction Dashboard")
st.write("California Housing | Bagging vs Gradient Boosting")

# Step 1: Check result and model files
required_files = [
    "bagging_results.csv",
    "boosting_results.csv",
    "bagging_predictions.csv",
    "boosting_predictions.csv",
    "bagging_model.pkl",
    "boosting_model.pkl"
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:
    st.warning("Please run both ML Python files first.")
    st.write("Missing files:", missing_files)
    st.stop()

# Step 2: Load saved results and models
bagging_results = pd.read_csv("bagging_results.csv")
boosting_results = pd.read_csv("boosting_results.csv")

bagging_predictions = pd.read_csv("bagging_predictions.csv")
boosting_predictions = pd.read_csv("boosting_predictions.csv")

bagging_model = joblib.load("bagging_model.pkl")
boosting_model = joblib.load("boosting_model.pkl")

results = pd.concat(
    [bagging_results, boosting_results],
    ignore_index=True
)

# Step 3: Model performance
st.subheader("1. Model Performance")

col1, col2 = st.columns(2)

col1.metric(
    "Bagging R² Score",
    f"{bagging_results['R2'].iloc[0]:.4f}"
)

col2.metric(
    "Boosting R² Score",
    f"{boosting_results['R2'].iloc[0]:.4f}"
)

st.dataframe(results, use_container_width=True)

# Step 4: Compare R² scores
st.subheader("2. Model Comparison")

fig, ax = plt.subplots(figsize=(8, 4))

sns.barplot(data=results, x="Model", y="R2", ax=ax)

ax.set_title("R² Score Comparison")
ax.set_ylabel("R² Score")
ax.tick_params(axis="x", rotation=10)

st.pyplot(fig)
plt.close(fig)

# Step 5: Compare MSE
fig, ax = plt.subplots(figsize=(8, 4))

sns.barplot(data=results, x="Model", y="MSE", ax=ax)

ax.set_title("Mean Squared Error Comparison")
ax.set_ylabel("MSE")
ax.tick_params(axis="x", rotation=10)

st.pyplot(fig)
plt.close(fig)

# Step 6: Select predictions to view
st.subheader("3. Actual vs Predicted Values")

selected_model = st.selectbox(
    "Select Model",
    ["Bagging Regressor", "Gradient Boosting Regressor"]
)

if selected_model == "Bagging Regressor":
    predictions = bagging_predictions
else:
    predictions = boosting_predictions

st.dataframe(predictions.head(20), use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    predictions["Actual"],
    predictions["Predicted"],
    alpha=0.5
)

lower = min(
    predictions["Actual"].min(),
    predictions["Predicted"].min()
)

upper = max(
    predictions["Actual"].max(),
    predictions["Predicted"].max()
)

ax.plot([lower, upper], [lower, upper], linestyle="--")

ax.set_xlabel("Actual House Value")
ax.set_ylabel("Predicted House Value")
ax.set_title(f"Actual vs Predicted: {selected_model}")

st.pyplot(fig)
plt.close(fig)

# Step 7: Predict a new house price
st.subheader("4. Predict a New House Price")

st.write("Enter the house details below.")

with st.form("house_prediction_form"):

    prediction_model = st.selectbox(
        "Select Prediction Model",
        ["Bagging Regressor", "Gradient Boosting Regressor"]
    )

    col1, col2 = st.columns(2)

    with col1:
        MedInc = st.number_input(
            "Median Income (MedInc)",
            min_value=0.0,
            value=3.5,
            step=0.1
        )

        HouseAge = st.number_input(
            "House Age (HouseAge)",
            min_value=0.0,
            value=25.0,
            step=1.0
        )

        AveRooms = st.number_input(
            "Average Rooms (AveRooms)",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

        AveBedrms = st.number_input(
            "Average Bedrooms (AveBedrms)",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

    with col2:
        Population = st.number_input(
            "Population",
            min_value=0.0,
            value=1000.0,
            step=10.0
        )

        AveOccup = st.number_input(
            "Average Occupancy",
            min_value=0.0,
            value=3.0,
            step=0.1
        )

        Latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=34.0,
            step=0.1
        )

        Longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-118.0,
            step=0.1
        )

    predict_button = st.form_submit_button("Predict House Price")

if predict_button:

    house_data = pd.DataFrame([{
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }])

    if prediction_model == "Bagging Regressor":
        prediction = bagging_model.predict(house_data)[0]
    else:
        prediction = boosting_model.predict(house_data)[0]

    # California Housing target is measured in units of $100,000
    price = prediction * 100000

    st.success(f"Predicted House Value: ${price:,.2f}")

    st.info(
        "This is an estimated value based on the California Housing "
        "dataset, not a live real-estate market price."
    )

# Step 8: Download comparison
st.subheader("5. Download Results")

st.download_button(
    "Download Model Comparison",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="house_price_model_comparison.csv",
    mime="text/csv"
)

st.success("Dashboard loaded successfully!")
