
import streamlit as st
import pandas as pd
import joblib
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="centered"
)


# ============================================================
# LOAD RANDOM FOREST + PREPROCESSING
# ============================================================

@st.cache_resource
def load_model():

    package = joblib.load("stroke_rf_model.pkl")

    model = package["model"]
    imputer = package["imputer"]
    scaler = package["scaler"]

    return model, imputer, scaler


try:

    model, imputer, scaler = load_model()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error("❌ Model could not be loaded.")
    st.exception(e)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 Stroke Prediction System")

st.subheader(
    "Random Forest-Based Stroke Risk Prediction"
)

st.write(
    "Enter the patient's information below. "
    "The trained Random Forest model will analyze "
    "the input and provide the prediction."
)


# ============================================================
# PATIENT INPUT
# ============================================================

st.divider()

st.header("👤 Patient Information")


# ============================================================
# AGE
# ============================================================

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50,
    step=1
)


# ============================================================
# GENDER
# ============================================================

gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)


# ============================================================
# HYPERTENSION
# ============================================================

hypertension = st.selectbox(
    "Hypertension",
    [
        0,
        1
    ],
    format_func=lambda x:
        "Yes" if x == 1 else "No"
)


# ============================================================
# HEART DISEASE
# ============================================================

heart_disease = st.selectbox(
    "Heart Disease",
    [
        0,
        1
    ],
    format_func=lambda x:
        "Yes" if x == 1 else "No"
)


# ============================================================
# EVER MARRIED
# ============================================================

ever_married = st.selectbox(
    "Ever Married",
    [
        "Yes",
        "No"
    ]
)


# ============================================================
# WORK TYPE
# ============================================================

work_type = st.selectbox(
    "Work Type",
    [
        "Private",
        "Self-employed",
        "Govt_job",
        "children",
        "Never_worked"
    ]
)


# ============================================================
# RESIDENCE TYPE
# ============================================================

residence_type = st.selectbox(
    "Residence Type",
    [
        "Urban",
        "Rural"
    ]
)


# ============================================================
# GLUCOSE LEVEL
# ============================================================

avg_glucose_level = st.number_input(
    "Average Glucose Level",
    min_value=50.0,
    max_value=300.0,
    value=100.0,
    step=1.0
)


# ============================================================
# BMI
# ============================================================

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0,
    step=1.0
)


# ============================================================
# SMOKING STATUS
# ============================================================

smoking_status = st.selectbox(
    "Smoking Status",
    [
        "formerly smoked",
        "never smoked",
        "smokes",
        "Unknown"
    ]
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "🔍 ANALYZE STROKE RISK",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button and model_loaded:

    try:

        # ====================================================
        # CREATE USER DATA
        # ====================================================

        user_data = pd.DataFrame({

            "gender": [gender],

            "age": [age],

            "hypertension": [hypertension],

            "heart_disease": [heart_disease],

            "ever_married": [ever_married],

            "work_type": [work_type],

            "Residence_type": [residence_type],

            "avg_glucose_level": [
                avg_glucose_level
            ],

            "bmi": [bmi],

            "smoking_status": [
                smoking_status
            ]
        })


        # ====================================================
        # SHOW ORIGINAL INPUT
        # ====================================================

        st.divider()

        st.subheader("📋 Patient Data")

        st.dataframe(
            user_data,
            use_container_width=True
        )


        # ====================================================
        # PRIORITY ENCODING
        # ====================================================

        gender_map = {
            "Female": 0,
            "Male": 1
        }

        ever_married_map = {
            "No": 0,
            "Yes": 1
        }

        work_type_map = {
            "Never_worked": 0,
            "children": 1,
            "Govt_job": 2,
            "Self-employed": 3,
            "Private": 4
        }

        residence_map = {
            "Rural": 0,
            "Urban": 1
        }

        smoking_map = {
            "never smoked": 0,
            "formerly smoked": 1,
            "Unknown": 2,
            "smokes": 3
        }


        # ====================================================
        # APPLY ENCODING
        # ====================================================

        user_data["gender"] = (
            user_data["gender"].map(gender_map)
        )

        user_data["ever_married"] = (
            user_data["ever_married"].map(
                ever_married_map
            )
        )

        user_data["work_type"] = (
            user_data["work_type"].map(
                work_type_map
            )
        )

        user_data["Residence_type"] = (
            user_data["Residence_type"].map(
                residence_map
            )
        )

        user_data["smoking_status"] = (
            user_data["smoking_status"].map(
                smoking_map
            )
        )


        # ====================================================
        # CHECK FOR INVALID VALUES
        # ====================================================

        if user_data.isnull().any().any():

            st.error(
                "❌ Some input values could not be encoded."
            )

            st.stop()


        # ====================================================
        # IMPUTATION FIX
        # ====================================================
        #
        # We DO NOT call:
        #
        # imputer.transform(user_data)
        #
        # because the saved SimpleImputer is causing:
        #
        # AttributeError:
        # 'SimpleImputer' object has no attribute '_fill_dtype'
        #
        # Instead, because the Streamlit form contains values
        # for every feature, we only use the saved imputer's
        # statistics if a missing value actually exists.
        #

        user_imputed = user_data.copy()


        # ----------------------------------------------------
        # Handle missing values manually using saved medians
        # ----------------------------------------------------

        if user_imputed.isnull().any().any():

            if hasattr(imputer, "statistics_"):

                for i, column in enumerate(user_imputed.columns):

                    if user_imputed[column].isnull().any():

                        user_imputed[column] = (
                            user_imputed[column]
                            .fillna(imputer.statistics_[i])
                        )

            else:

                st.error(
                    "❌ Imputer statistics are not available."
                )

                st.stop()


        # ====================================================
        # CONVERT TO NUMPY
        # ====================================================

        user_array = user_imputed.astype(float).values


        # ====================================================
        # SCALING
        # ====================================================

        user_scaled = scaler.transform(
            user_array
        )


        # ====================================================
        # RANDOM FOREST PREDICTION
        # ====================================================

        prediction = model.predict(
            user_scaled
        )[0]


        # ====================================================
        # RANDOM FOREST PROBABILITY
        # ====================================================

        probability = model.predict_proba(
            user_scaled
        )[0]


        # ====================================================
        # SAFELY GET PROBABILITIES
        # ====================================================

        if len(probability) == 2:

            no_stroke_probability = probability[0]
            stroke_probability = probability[1]

        else:

            no_stroke_probability = 1.0
            stroke_probability = 0.0


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.header("🔎 Prediction Result")


        if prediction == 0:

            st.success(
                "🟢 NO STROKE"
            )

            st.write(
                "The Random Forest model predicts that "
                "the patient belongs to the **No Stroke** class."
            )

        else:

            st.error(
                "🔴 STROKE - HIGH CHANCES"
            )

            st.write(
                "The Random Forest model predicts that "
                "the patient belongs to the **Stroke** class."
            )


        # ====================================================
        # PROBABILITY
        # ====================================================

        st.subheader(
            "📊 Random Forest Prediction Probability"
        )

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "No Stroke",
                f"{no_stroke_probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Stroke",
                f"{stroke_probability * 100:.2f}%"
            )


        # ====================================================
        # PROBABILITY BAR
        # ====================================================

        st.write("### Stroke Probability")

        st.progress(
            float(stroke_probability)
        )


        # ====================================================
        # RANDOM FOREST DETAILS
        # ====================================================

        st.divider()

        st.subheader("🌳 Random Forest Analysis")

        st.write(
            f"Predicted Class: **{int(prediction)}**"
        )

        st.write(
            f"Stroke Probability: "
            f"**{stroke_probability * 100:.2f}%**"
        )

        st.write(
            f"No Stroke Probability: "
            f"**{no_stroke_probability * 100:.2f}%**"
        )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.divider()

        st.subheader("📌 Prediction Interpretation")

        if stroke_probability < 0.30:

            st.info(
                "The model indicates a relatively low predicted "
                "probability of stroke."
            )

        elif stroke_probability < 0.60:

            st.warning(
                "The model indicates a moderate predicted "
                "probability of stroke."
            )

        else:

            st.error(
                "The model indicates a relatively high predicted "
                "probability of stroke."
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("⚙️ Model Information")

st.write(
    "**Algorithm:** Random Forest Classifier"
)

st.write(
    "**Number of Trees:** 100"
)

st.write(
    "**Data Balancing:** SMOTE"
)

st.write(
    "**Preprocessing:** Priority Encoding + "
    "Median Imputation + Standard Scaling"
)

st.write(
    "**Output:** 0 = No Stroke, 1 = Stroke"
)


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This application is intended for educational "
    "and research purposes only and is not a medical diagnosis."
)

