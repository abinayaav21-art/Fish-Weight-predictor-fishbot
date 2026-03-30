import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Page Config & Luxury UI ---
st.set_page_config(page_title="Fish Weight AI", layout="centered", page_icon="🐟")

st.markdown("""
    <style>
    .stApp { 
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), 
                    url("https://www.pixelstalk.net/wp-content/uploads/2025/05/A-close-up-of-a-betta-fish-detailed-shimmering-fins-with-its-body-reflecting-light-swimming-in-a-tranquil-aquatic-environment.webp"); 
        background-size: cover; 
    }
    .main .block-container { 
        background: rgba(255, 255, 255, 0.03); 
        backdrop-filter: blur(20px); 
        border-radius: 25px; 
        padding: 50px; 
        border: 1px solid rgba(255,255,255,0.1); 
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    h1, label, p, .stSelectbox label { color: #f0f2f6 !important; }
    .stButton>button { 
        background: linear-gradient(to right, #3a7bd5, #00d2ff) !important;
        color: white !important; font-weight: bold; width: 100%; border: none; 
        height: 3.5em; border-radius: 12px; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); }
    </style>
""", unsafe_allow_html=True)

# --- Asset Loader ---
@st.cache_resource
def load_assets():
    file_path = 'fish_model_elite.pkl'
    if not os.path.exists(file_path):
        st.error(f"Error: '{file_path}' not found in the repository.")
        return None
    with open(file_path, 'rb') as f:
        return pickle.load(f)

assets = load_assets()

if assets:
    st.title("🐟 Fish Weight Predictor")
    st.write("Advanced Morphological Analysis & Growth Estimation")

    # --- Input Sections ---
    col1, col2 = st.columns(2)
    
    with col1:
        species = st.selectbox("Fish Species", assets['species_list'])
        # We use Length3 (Total Length) as established in our refined roadmap
        length = st.number_input("Total Length (cm)", min_value=1.0, max_value=200.0, value=30.0)
        height = st.number_input("Body Height (cm)", min_value=1.0, max_value=100.0, value=12.0)
    
    with col2:
        width = st.number_input("Body Width (cm)", min_value=1.0, max_value=100.0, value=6.0)
        girth = st.number_input("Girth Measurement (cm)", min_value=1.0, max_value=150.0, value=15.0)
        age = st.slider("Estimated Age (Years)", 0.1, 20.0, 3.0)

    temp = st.slider("Water Temperature (°C)", 0.0, 40.0, 18.0)

    # --- Prediction Logic ---
    if st.button("Analyze & Predict Weight"):
        try:
            # 1. Create Input DataFrame matching the model's exact training columns
            input_df = pd.DataFrame(0, index=[0], columns=assets['columns'])
            
            # 2. Assign numerical features
            input_df['length3_cm'] = length
            input_df['height_cm'] = height
            input_df['width_cm'] = width
            input_df['girth_cm'] = girth
            input_df['age_years'] = age
            input_df['water_temp_c'] = temp
            
            # 3. Assign Categorical (Species) Feature
            species_col = f"species_{species}"
            if species_col in input_df.columns:
                input_df[species_col] = 1

            # 4. Standardize/Scale the input using saved training parameters
            scaled_input = (input_df - assets['mean']) / assets['std']

            # 5. Predict log(weight) and convert back to grams using exp()
            log_prediction = assets['model'].predict(scaled_input)[0]
            final_weight = np.exp(log_prediction)

            # 6. High-End Result Display
            st.markdown(f"""
                <div style="background: rgba(0, 210, 255, 0.1); padding: 25px; border-radius: 15px; text-align: center; border: 1px solid #00d2ff; margin-top: 20px;">
                    <h3 style="margin:0; color: #00d2ff; font-size: 1.2em;">Estimated Mass</h3>
                    <h1 style="margin:0; color: white; font-size: 3em;">{final_weight:,.2f} g</h1>
                    <p style="margin-top:10px; opacity:0.8;">Equivalent to {(final_weight/1000):,.2f} kg</p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

else:
    st.info("👋 Welcome! Please ensure your 'fish_model_elite.pkl' file is uploaded to GitHub to begin.")