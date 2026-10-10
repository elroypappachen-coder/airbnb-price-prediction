import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("airbnb_best_model.pkl")

st.set_page_config(
    page_title="Airbnb Nightly Price Predictor",
    page_icon="🏡",
    layout="centered"
)

st.title("🏡 Airbnb Nightly Price Predictor")
st.caption("Estimate a listing's nightly price using your trained regression model.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place airbnb_best_model.pkl in the same folder as this app."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Property details")

with st.form("airbnb_price_form"):
    left, right = st.columns(2)

    with left:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=1, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
        minimum_nights = st.number_input("Minimum nights", min_value=1, max_value=365, value=1, step=1)
        neighbourhood_group = st.selectbox(
            "Neighbourhood group",
            ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        )
        room_type = st.selectbox(
            "Room type",
            ["Entire home/apt", "Private room", "Shared room"]
        )
        latitude = st.number_input("Latitude", value=40.750000, format="%.6f")
        longitude = st.number_input("Longitude", value=-73.980000, format="%.6f")
        availability_365 = st.number_input("Availability (days per year)", min_value=0, max_value=365, value=180)

    with right:
        number_of_reviews = st.number_input("Number of reviews", min_value=0, max_value=10000, value=10)
        reviews_per_month = st.number_input("Reviews per month", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
        calculated_host_listings_count = st.number_input(
            "Host's calculated listing count", min_value=1, max_value=1000, value=1
        )
        days_since_last_review = st.number_input(
            "Days since last review (use 0 if unknown)", min_value=0, max_value=5000, value=0
        )
        feature_count = st.number_input("Number of listed features", min_value=0, max_value=100, value=4)
        photo_count = st.number_input("Number of photos", min_value=0, max_value=100, value=6)
        listing_hour = st.number_input("Listing hour (0–23)", min_value=0, max_value=23, value=12)
        listing_dayofweek = st.selectbox(
            "Listing day of week",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda d: [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"
            ][d],
            index=2
        )

    submitted = st.form_submit_button("Predict Nightly Price")

if submitted:
    # Column names must match the features used when the model was trained.
    new_listing = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "minimum_nights": minimum_nights,
        "neighbourhood_group": neighbourhood_group,
        "room_type": room_type,
        "latitude": latitude,
        "longitude": longitude,
        "availability_365": availability_365,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "days_since_last_review": days_since_last_review,
        "feature_count": feature_count,
        "photo_count": photo_count,
        "listing_hour": listing_hour,
        "listing_dayofweek": listing_dayofweek
    }])

    try:
        predicted_price = model.predict(new_listing)[0]
        st.success(f"Estimated nightly price: ${predicted_price:,.2f}")
        st.caption("This is an estimate from the model, not a guaranteed market price.")
    except Exception as error:
        st.error(
            "Prediction failed. The app now includes the columns shown as missing "
            "in your screenshot. If it still fails, check the exact training column "
            "names and categorical values used by your saved model."
        )
        st.exception(error)
