import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="Ovarian Cancer Prediction",
    page_icon="🩺",
    layout="centered"
)

# --------------------------
# Load Model Safely (Auto-detecting Path Structure)
# --------------------------
@st.cache_resource
def load_my_model():
    folder_path = os.path.join("models", "ovarian_cancer_model.pkl")
    root_path = "ovarian_cancer_model.pkl"
    
    if os.path.exists(folder_path):
        target_path = folder_path
    elif os.path.exists(root_path):
        target_path = root_path
    else:
        all_files = os.listdir(".")
        pkl_files = [f for f in all_files if f.endswith(".pkl")]
        
        if pkl_files:
            target_path = pkl_files[0]
        elif os.path.exists("models") and os.path.isdir("models"):
            sub_files = [os.path.join("models", f) for f in os.listdir("models") if f.endswith(".pkl")]
            target_path = sub_files[0] if sub_files else None
        else:
            target_path = None

    if not target_path:
        return None

    try:
        return joblib.load(target_path)
    except Exception as e:
        return e

# Initialize the model
model = load_my_model()

if model is None:
    st.error("❌ Critical Error: The model file `ovarian_cancer_model.pkl` could not be found anywhere in your repository.")
    st.info("💡 **Fix:** Please make sure the trained model file is uploaded to GitHub.")
    st.stop()
elif isinstance(model, Exception):
    st.error("❌ Critical Error: Failed to deserialize or read the machine learning model.")
    st.exception(model)
    st.info("💡 **Fix:** Re-run your local training pipeline to output a clean binary model file, then re-upload.")
    st.stop()

# --------------------------
# Sidebar Navigation
# --------------------------
st.sidebar.title("🩺 Healthcare Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "ℹ️ About Ovarian Cancer",
        "🧪 Biomarker Information",
        "📋 How to Use",
        "⚠️ Risk Factors",
        "📅 Health Monitoring Tips",
        "🥗 Diet & Lifestyle Tips",
        "🩺 Symptoms Checker",
        "⚖️ BMI Calculator",
        "📊 Dataset Insights",
        "🌿 Wellness Tips",
        "📈 Feature Importance",
        "🏥 Nearby Specialists"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("🔍 Machine Learning Based Risk Assessment\n\nThis system assists in analyzing ovarian cancer risk using patient biomarker values.")

# ======================================================
# HOME PAGE
# ======================================================
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center; color:#E63946;'>🩺 Ovarian Cancer Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px;'>Enter patient biomarker values to predict ovarian cancer risk.</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Patient Information")
    HE4 = st.number_input("HE4", min_value=0.0, value=50.0)
    CA125 = st.number_input("CA125", min_value=0.0, value=20.0)
    Age = st.number_input("Age", min_value=1, max_value=120, value=40)
    LYM_percent = st.number_input("LYM (%)", min_value=0.0, value=30.0)
    LYM_count = st.number_input("LYM (#)", min_value=0.0, value=2.0)
    CEA = st.number_input("CEA", min_value=0.0, value=3.0)
    NEU = st.number_input("NEU", min_value=0.0, value=60.0)
    Mg = st.number_input("Magnesium (Mg)", min_value=0.0, value=1.8)
    PDW = st.number_input("PDW", min_value=0.0, value=12.0)
    Menopause = st.selectbox("Menopause Status", ["No", "Yes"])
    menopause_value = 0 if Menopause == "No" else 1

    st.markdown("---")

    if st.button("Predict Risk"):
        feature_names = [
            'MPV','BASO#','PHOS','GLU.','K','AST','BASO%','Mg',
            'Menopause','CL','CEA','EO#','CA19-9','ALB','IBIL',
            'GGT','MCH','GLO','ALT','DBIL','Age','RDW','PDW',
            'CREA','AFP','HGB','Na','HE4','LYM#','CA125','BUN',
            'LYM%','Ca','AG','MONO#','PLT','NEU','EO%','TP',
            'UA','RBC','PCT','CO2CP','TBIL','HCT','MONO%',
            'MCV','ALP'
        ]

        data = {feature: 0 for feature in feature_names}
        data['HE4'] = HE4
        data['CA125'] = CA125
        data['Age'] = Age
        data['LYM%'] = LYM_percent
        data['LYM#'] = LYM_count
        data['CEA'] = CEA
        data['NEU'] = NEU
        data['Mg'] = Mg
        data['PDW'] = PDW
        data['Menopause'] = menopause_value

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df).max() * 100

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("⚠️ Higher Risk Category Detected")
        else:
            st.success("✅ Lower Risk Category Detected")

        st.write(f"Confidence Score: {probability:.2f}%")
        st.progress(int(probability))
        
        if probability < 40:
            st.success("🟢 Low Risk")
        elif probability < 70:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")
            
        st.markdown("---")
        st.subheader("📊 Key Feature Contributions")
        features_summary = ["HE4", "CA125", "Age", "LYM%", "LYM#"]
        importance_summary = [0.18, 0.08, 0.05, 0.05, 0.04]

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(features_summary, importance_summary, color="#62214E")
        ax.set_xlabel('Relative Importance Weight')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        
        report = f"OVARIAN CANCER PREDICTION REPORT\n\nPrediction Result:\n{'Higher Risk Category' if prediction == 1 else 'Lower Risk Category'}\n\nRisk Status Assessment:\n{'Low Risk' if probability < 40 else 'Moderate Risk' if probability < 70 else 'High Risk'}\n\nConfidence Score:\n{probability:.2f}%\n\nPatient Details Metrics:\nAge = {Age}\nHE4 = {HE4}\nCA125 = {CA125}\nCEA = {CEA}\nNEU = {NEU}\n"
        st.download_button(label="📄 Download Report", data=report, file_name="Prediction_Report.txt", mime="text/plain")

    st.markdown("---")
    st.warning("This application is intended for educational and research purposes only.\n\nThe prediction generated by this model should not be considered a medical diagnosis.\nPlease consult a qualified healthcare professional for medical advice.")

elif page == "ℹ️ About Ovarian Cancer":
    st.title("ℹ️ About Ovarian Cancer")
    st.write("Ovarian cancer occurs when abnormal cells in the ovary grow uncontrollably.\n\nEarly detection is important because treatment is more effective when the disease is identified at an early stage.\n\nCommon symptoms include:\n- Pelvic pain\n- Abdominal bloating\n- Loss of appetite\n- Frequent urination")

elif page == "🧪 Biomarker Information":
    st.title("🧪 Biomarker Information")
    st.write("**HE4** – Human Epididymis Protein 4\n\n**CA125** – Cancer Antigen 125\n\n**CEA** – Carcinoembryonic Antigen\n\n**LYM%** – Lymphocyte Percentage\n\n**LYM#** – Lymphocyte Count\n\n**NEU** – Neutrophils\n\n**Mg** – Magnesium\n\n**PDW** – Platelet Distribution Width\n\n**Menopause** – Menopausal Status")

elif page == "📋 How to Use":
    st.title("📋 How to Use")
    st.write("1. Open the Home page.\n2. Enter patient biomarker values.\n3. Select menopause status.\n4. Click Predict Risk.\n5. View the prediction and confidence score.")

elif page == "⚠️ Risk Factors":
    st.title("⚠️ Risk Factors")
    st.write("Common ovarian cancer risk factors:\n\n• Increasing age\n\n• Family history of ovarian cancer\n\n• Genetic mutations (BRCA1/BRCA2)\n\n• Post-menopausal status\n\n• Obesity\n\n• Hormonal factors")

elif page == "📅 Health Monitoring Tips":
    st.title("📅 Health Monitoring Tips")
    st.success("Regular health monitoring can help in early detection and better management of ovarian cancer risk.")
    st.subheader("✅ Preventive Care Tips")
    st.write("🔹 Regular gynecological checkups\n\n🔹 Family history awareness\n\n🔹 Monitor unusual symptoms\n\n🔹 Follow prescribed treatments\n\n🔹 Maintain a healthy lifestyle\n\n🔹 Exercise regularly\n\n🔹 Eat a balanced diet\n\n🔹 Stay hydrated")
    st.subheader("⚠️ Symptoms to Watch")
    st.write("• Abdominal bloating\n\n• Pelvic pain\n\n• Feeling full quickly\n\n• Frequent urination\n\n• Unexplained weight changes\n\n• Fatigue")

elif page == "🥗 Diet & Lifestyle Tips":
    st.title("🥗 Diet & Lifestyle Tips")
    st.success("A healthy lifestyle may help support overall health and reduce the risk of many diseases.")
    st.subheader("🥦 Recommended Foods")
    st.write("✅ Green leafy vegetables\n- Spinach\n- Fenugreek leaves\n- Kale\n\n✅ Fruits\n- Apples\n- Oranges\n- Berries\n- Pomegranate\n\n✅ Whole Grains\n- Brown rice\n- Oats\n- Whole wheat\n\n✅ Protein Sources\n- Lentils\n- Beans\n- Chickpeas\n- Eggs\n\n✅ Healthy Fats\n- Almonds\n- Walnuts\n- Flax seeds\n\n✅ Hydration\n- Drink 2–3 liters of water daily")
    st.subheader("🚫 Foods to Limit")
    st.write("❌ Excessive processed foods\n\n❌ Sugary drinks\n\n❌ Excessive fried foods\n\n❌ Highly processed meats\n\n❌ Excessive junk food")

elif page == "🩺 Symptoms Checker":
    st.title("🩺 Symptoms Checker")
    st.write("Select any current symptoms below to assess common observations.")
    symptoms = [st.checkbox("Abdominal Bloating"), st.checkbox("Pelvic Pain"), st.checkbox("Fatigue"), st.checkbox("Frequent Urination"), st.checkbox("Loss of Appetite"), st.checkbox("Back Pain")]
    if sum(symptoms) >= 3:
        st.warning("⚠️ Multiple symptoms detected. Please consult a healthcare professional.")
    else:
        st.success("✅ Few symptoms reported.")

elif page == "⚖️ BMI Calculator":
    st.title("⚖️ BMI Calculator")
    weight = st.number_input("Weight (kg)", min_value=1.0, value=60.0)
    height = st.number_input("Height (cm)", min_value=10.0, value=165.0)
    if st.button("Calculate BMI"):
        bmi = weight / ((height / 100) ** 2)
        st.metric("Your Calculated BMI", f"{bmi:.2f}")
        if bmi < 18.5: st.info("Category: Underweight")
        elif bmi < 25: st.success("Category: Normal")
        elif bmi < 30: st.warning("Category: Overweight")
        else: st.error("Category: Obese")

elif page == "📊 Dataset Insights":
    st.title("📊 Dataset Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Patients", 235)
        st.metric("Cancer Cases", 89)
    with col2:
        st.metric("Non-Cancer Cases", 146)
        st.metric("Validation Accuracy", "85.11%")

elif page == "🌿 Wellness Tips":
    st.title("🌿 Wellness Tips")
    st.write("💧 **Hydration:** Drink 2–3 liters water daily\n\n🥗 **Nutrition:** Eat clean vegetables daily\n\n🚶 **Activity:** Walk 30 minutes daily\n\n😴 **Recovery:** Sleep 7–8 hours nightly\n\n🧘 **Mindfulness:** Actively manage stress daily")

elif page == "📈 Feature Importance":
    st.title("📈 Feature Importance")
    features = ["HE4", "CA125", "Age", "LYM%", "LYM#", "CEA", "NEU", "Mg", "PDW", "Menopause"]
    importance = [0.18, 0.08, 0.05, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(features, importance, color="#741246")
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance Score")
    st.pyplot(fig)

elif page == "🏥 Nearby Specialists":
    st.title("🏥 Specialist Consultation Guide")
    st.subheader("👩‍⚕️ Gynecologist")
    st.write("• Regular reproductive health checkups\n• Ovarian cyst evaluation\n• Menopause-related concerns\n• Preventive screening")
    st.subheader("🎗️ Oncologist")
    st.write("• Cancer diagnosis and treatment\n• Tumor evaluation\n• Cancer risk assessment\n• Follow-up monitoring")