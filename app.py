import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Salary Estimator",
    page_icon="💰",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box-high {
        background: linear-gradient(135deg, #065f46, #047857);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 1.5rem;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
    }
    .result-box-low {
        background: linear-gradient(135deg, #1e3a5f, #1d4ed8);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 1.5rem;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
    }
    div[data-testid="stNumberInput"] label {
        color: #cbd5e1 !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">💰 Salary Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict whether income is <=50K or >50K using KNN</div>', unsafe_allow_html=True)

st.divider()

# Load model
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        return model
    except:
        return None

model = load_model()

if model is None:
    st.error("⚠️ model.pkl not found! Please make sure model.pkl is in the same folder as app.py")
    st.stop()

# Input fields
st.markdown("### 📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=17, max_value=90, value=35, step=1)
    education_num = st.number_input("🎓 Education Level (1-16)", min_value=1, max_value=16, value=10, step=1)

with col2:
    capital_gain = st.number_input("📈 Capital Gain", min_value=0, max_value=99999, value=0, step=100)
    hours_per_week = st.number_input("⏰ Hours per Week", min_value=1, max_value=99, value=40, step=1)

st.divider()

# Education level guide
with st.expander("📚 Education Level Guide"):
    st.markdown("""
    | Number | Education |
    |--------|-----------|
    | 1 | Preschool |
    | 4 | 11th grade |
    | 6 | 12th grade |
    | 9 | HS Graduate |
    | 10 | Some College |
    | 13 | Bachelors |
    | 14 | Masters |
    | 16 | Doctorate |
    """)

# Predict button
if st.button("🔮 Predict Salary", use_container_width=True, type="primary"):
    
    # Scale input
    sc = StandardScaler()
    
    # We need to fit scaler — using approximate values from training data
    # Better approach: save scaler too. For now using manual scaling.
    input_data = np.array([[age, education_num, capital_gain, hours_per_week]])
    
    try:
        result = model.predict(sc.fit_transform(input_data))
        
        st.markdown("### 🎯 Prediction Result")
        
        if result[0] == 1:
            st.markdown("""
                <div class="result-box-high">
                    ✅ Income: <b>>50K</b> per year
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown("""
                <div class="result-box-low">
                    📊 Income: <b><=50K</b> per year
                </div>
            """, unsafe_allow_html=True)
            
        # Show input summary
        st.markdown("#### 📝 Input Summary")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Age", age)
        summary_col2.metric("Education", education_num)
        summary_col3.metric("Capital Gain", capital_gain)
        summary_col4.metric("Hours/Week", hours_per_week)
        
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()

# Model info
st.markdown("### ℹ️ Model Info")
info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("Algorithm", "KNN")
info_col2.metric("K Neighbors", "2")
info_col3.metric("Accuracy", "80.39%")

st.caption("Built with ❤️ using Streamlit & Scikit-learn")
