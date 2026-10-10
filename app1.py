import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("PriceModel.joblib")

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠"
)

st.title("🏠 Airbnb Rental Price Predictor")
st.caption(
    "Educational demonstration using the Airbnb NYC rental dataset."
)

# Numerical inputs
minimum_nights = st.number_input(
    "Minimum Nights", 1, 365, 1, 1
)

latitude = st.number_input(
    "Latitude", value=40.75, format="%.6f"
)

longitude = st.number_input(
    "Longitude", value=-73.98, format="%.6f"
)

number_of_reviews = st.number_input(
    "Number of Reviews", 0, 10000, 10, 1
)

reviews_per_month = st.number_input(
    "Reviews Per Month", 0.0, 100.0, 1.0, 0.1
)

availability_365 = st.number_input(
    "Availability Per Year (Days)", 0, 365, 180, 1
)

last_review_days = st.number_input(
    "Days Since Last Review", 0, 5000, 30, 1
)

calculated_host_listings_count = st.number_input(
    "Host's Total Listings", 1, 1000, 1, 1
)

# Categorical inputs
neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
)

neighbourhood = st.text_input(
    "Neighbourhood",
    "Midtown"
)

room_type = st.selectbox(
    "Room Type",
    [
        "Entire home/apt",
        "Private room",
        "Shared room"
    ]
)

# Prediction
if st.button("Predict Rental Price"):

    row = pd.DataFrame([{
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "latitude": latitude,
        "longitude": longitude,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "availability_365": availability_365,
        "last_review_days": last_review_days,
        "calculated_host_listings_count":
            calculated_host_listings_count
    }])

    try:
        prediction = model.predict(row)[0]

        st.success(
            f"Predicted Nightly Price: ${prediction:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.info(
    "Educational demonstration only. "
    "Predictions are estimates, not guaranteed market prices."
)