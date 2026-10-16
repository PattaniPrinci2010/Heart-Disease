import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Heart Disease Portal",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# PAGE 1: HOME
# ==========================================
def home_page():
    st.title("❤️ Welcome to Heart Disease Prediction System")
    st.write("This application uses a machine learning model to predict the risk of heart disease[cite: 3].")

    st.info("👈 Please select a page from the left-side navigation panel to get started!")

    st.markdown("""
    ### 🌟 Key Features:
    - **Accurate Prediction:** Utilizes a trained Logistic Regression model[cite: 3].
    - **Interactive UI:** Simple and clean form for dynamic user inputs.
    - **Dataset Analysis:** Built-in data visualizer and exploratory charts.
    """)

# ==========================================
# PAGE 2: PREDICTION
# ==========================================
def prediction_page():
    st.title("🩺 Heart Disease Prediction Form")
    st.write("Please fill in your medical details below:")

    # Load artifacts safely
    @st.cache_resource
    def load_artifacts():
        model = joblib.load('logistic_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('columns.pkl')
        return model, scaler, model_columns

    try:
        model, scaler, model_columns = load_artifacts()
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return

    # Input Form Layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
        resting_bp = st.number_input("Resting BP", min_value=50, max_value=250, value=120)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
        fasting_bs = st.selectbox("Fasting BS > 120 mg/dl", [0, 1])

    with col2:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Max HR", min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    if st.button("🚀 Predict Heart Disease", use_container_width=True):
        # Prepare input matching trained columns
        input_data = {col: 0 for col in model_columns}
        
        input_data['Age'] = age
        input_data['RestingBP'] = resting_bp
        input_data['Cholesterol'] = cholesterol
        input_data['FastingBS'] = fasting_bs
        input_data['MaxHR'] = max_hr
        input_data['Oldpeak'] = oldpeak
        
        if sex == 'M': input_data['Sex_M'] = 1
        if chest_pain == 'ATA': input_data['ChestPainType_ATA'] = 1
        elif chest_pain == 'NAP': input_data['ChestPainType_NAP'] = 1
        elif chest_pain == 'TA': input_data['ChestPainType_TA'] = 1
            
        if resting_ecg == 'Normal': input_data['RestingECG_Normal'] = 1
        elif resting_ecg == 'ST': input_data['RestingECG_ST'] = 1
            
        if exercise_angina == 'Y': input_data['ExerciseAngina_Y'] = 1
            
        if st_slope == 'Flat': input_data['ST_Slope_Flat'] = 1
        elif st_slope == 'Up': input_data['ST_Slope_Up'] = 1
            
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        st.divider()
        st.subheader("Result Summary:")
        if prediction[0] == 1:
            st.error(f"⚠️ Warning: Heart Disease detected! (Probability: {probability[0][1]*100:.2f}%)")
        else:
            st.success(f"✅ Normal: No Heart Disease detected. (Probability: {probability[0][0]*100:.2f}%)")

# ==========================================
# PAGE 3: DATASET ANALYSIS
# ==========================================
def analysis_page():
    st.title("📊 Dataset & Insights")
    
    @st.cache_data
    def load_data():
        return pd.read_csv('heart.csv')

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading heart.csv: {e}")
        return

    if st.checkbox("Show Raw Dataset"):
        st.dataframe(df)

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Heart Disease Outcome Distribution")
    if 'HeartDisease' in df.columns:
        counts = df['HeartDisease'].value_counts()
        st.bar_chart(counts)

# ==========================================
# NAVIGATION SETUP
# ==========================================
p1 = st.Page(home_page, title="Home", icon="🏠")
p2 = st.Page(prediction_page, title="Heart Predictor", icon="🩺")
p3 = st.Page(analysis_page, title="Dataset Insights", icon="📊")

# Setup Navigation Menu
pg = st.navigation([p1, p2, p3])

# Run active page
pg.run()