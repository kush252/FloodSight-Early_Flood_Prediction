"""
FloodSight - Early Flood Prediction System
A Streamlit app for water level prediction using LSTM model.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="FloodSight | Water Level Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    /* Main theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Hero section */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out;
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        border-color: rgba(0, 210, 255, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d2ff;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status indicators */
    .status-safe {
        color: #48bb78;
        font-weight: 600;
    }
    
    .status-warning {
        color: #ed8936;
        font-weight: 600;
    }
    
    .status-danger {
        color: #f56565;
        font-weight: 600;
    }
    
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.1) 0%, rgba(58, 123, 213, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(0, 210, 255, 0.3);
        border-radius: 25px;
        padding: 2rem;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 210, 255, 0.3); }
        50% { box-shadow: 0 0 40px rgba(0, 210, 255, 0.5); }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 12, 41, 0.9);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0, 210, 255, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 210, 255, 0.1);
        border-left: 4px solid #00d2ff;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0, 210, 255, 0.5), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    """Load the trained model and scaler."""
    try:
        import tensorflow as tf
        
        # Load metadata and scaler
        with open(os.path.join(SCRIPT_DIR, "model.pkl"), 'rb') as f:
            model_data = pickle.load(f)
        
        window_size = model_data['window_size']
        
        # Rebuild model architecture (must match save_model.py)
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                64,
                activation='tanh',
                return_sequences=False,
                input_shape=(window_size, 1)
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Load weights
        model.load_weights(os.path.join(SCRIPT_DIR, "lstm_weights.weights.h5"))
        
        return model, model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def get_flood_status(level, warning_threshold=2.0, danger_threshold=4.0):
    """Determine flood status based on water level."""
    if level >= danger_threshold:
        return "🚨 DANGER", "status-danger", "Immediate evacuation may be required!"
    elif level >= warning_threshold:
        return "⚠️ WARNING", "status-warning", "Monitor closely. Prepare for possible evacuation."
    else:
        return "✅ SAFE", "status-safe", "Water levels are within normal range."

def predict_future(model, scaler, last_sequence, window_size, horizon):
    """Predict future water levels."""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(horizon):
        # Scale the input
        scaled_input = scaler.transform(current_sequence.reshape(-1, 1)).reshape(1, window_size, 1)
        
        # Predict
        next_pred_scaled = model.predict(scaled_input, verbose=0)
        next_pred = scaler.inverse_transform(next_pred_scaled)[0][0]
        predictions.append(next_pred)
        
        # Update sequence
        current_sequence = np.append(current_sequence[1:], next_pred)
    
    return predictions

def main():
    # Hero Section
    st.markdown('<h1 class="hero-title">🌊 FloodSight</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Advanced AI-Powered Water Level Prediction System</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Load model
    model, model_data = load_model()
    
    if model is None or model_data is None:
        st.error("⚠️ Model not found. Please run `python save_model.py` first to train and save the model.")
        
        st.markdown("""
        ### Quick Setup Guide
        
        1. Open terminal in the project directory
        2. Run: `python save_model.py`
        3. Wait for training to complete
        4. Refresh this page
        """)
        return
    
    scaler = model_data['scaler']
    window_size = model_data['window_size']
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        # Alert thresholds
        st.markdown("#### 🚨 Alert Thresholds")
        warning_threshold = st.slider("Warning Level (m)", 0.5, 5.0, 2.0, 0.1)
        danger_threshold = st.slider("Danger Level (m)", 1.0, 10.0, 4.0, 0.1)
        
        st.markdown("---")
        st.markdown("#### 📊 Model Info")
        st.markdown(f"""
        - **R² Score**: {model_data.get('r2', 0.95):.3f}
        - **RMSE**: {model_data.get('rmse', 0.126):.3f} m
        - **MAE**: {model_data.get('mae', 0.041):.3f} m
        - **Window Size**: {window_size} readings
        """)
        
        st.markdown("---")
        st.markdown("#### 📍 Location")
        st.markdown("**Godavari River**")
        st.markdown("Anantwadi, Maharashtra")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Enter Recent Water Levels")
        st.markdown('<div class="info-box">Enter the last 10 water level readings (in meters) to predict future levels.</div>', unsafe_allow_html=True)
        
        # Input for water levels
        cols = st.columns(5)
        water_levels = []
        
        for i in range(10):
            with cols[i % 5]:
                val = st.number_input(
                    f"Reading {i+1}",
                    min_value=0.0,
                    max_value=20.0,
                    value=0.5,
                    step=0.01,
                    key=f"level_{i}"
                )
                water_levels.append(val)
        
        st.markdown("---")
        
        # Prediction horizon
        horizon = st.slider("🔮 Prediction Horizon (future readings)", 1, 10, 3)
        
        predict_btn = st.button("🚀 Generate Prediction", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        
        current_level = water_levels[-1] if water_levels else 0
        avg_level = np.mean(water_levels) if water_levels else 0
        max_level = max(water_levels) if water_levels else 0
        
        status, status_class, status_msg = get_flood_status(max_level, warning_threshold, danger_threshold)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Level</div>
            <div class="metric-value">{current_level:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Level</div>
            <div class="metric-value">{avg_level:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Status</div>
            <div class="{status_class}" style="font-size: 1.5rem;">{status}</div>
            <p style="color: #a0aec0; font-size: 0.85rem; margin-top: 0.5rem;">{status_msg}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction Results
    if predict_btn:
        with st.spinner("🔄 Generating predictions..."):
            try:
                # Convert to numpy array
                input_sequence = np.array(water_levels)
                
                # Get predictions
                predictions = predict_future(model, scaler, input_sequence, window_size, horizon)
                
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("## 🔮 Prediction Results")
                
                # Create prediction display
                pred_cols = st.columns(min(horizon, 5))
                for i, pred in enumerate(predictions):
                    with pred_cols[i % 5]:
                        status, status_class, _ = get_flood_status(pred, warning_threshold, danger_threshold)
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div style="color: #a0aec0; font-size: 0.8rem;">+{(i+1)*9}h</div>
                            <div class="metric-value">{pred:.2f} m</div>
                            <div class="{status_class}">{status.split()[0]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Create visualization
                st.markdown("### 📊 Water Level Trend")
                
                # Prepare data for chart
                all_levels = water_levels + predictions
                time_labels = [f"-{(10-i)*9}h" for i in range(10)] + [f"+{(i+1)*9}h" for i in range(horizon)]
                
                # Create figure
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=time_labels[:10],
                    y=water_levels,
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='#00d2ff', width=3),
                    marker=dict(size=8, color='#00d2ff'),
                    fill='tozeroy',
                    fillcolor='rgba(0, 210, 255, 0.1)'
                ))
                
                # Predictions
                fig.add_trace(go.Scatter(
                    x=time_labels[9:],
                    y=[water_levels[-1]] + predictions,
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#ff6b6b', width=3, dash='dash'),
                    marker=dict(size=10, color='#ff6b6b', symbol='diamond'),
                ))
                
                # Warning threshold line
                fig.add_hline(
                    y=warning_threshold,
                    line_dash="dash",
                    line_color="#ed8936",
                    annotation_text="Warning Level",
                    annotation_position="right"
                )
                
                # Danger threshold line
                fig.add_hline(
                    y=danger_threshold,
                    line_dash="dash",
                    line_color="#f56565",
                    annotation_text="Danger Level",
                    annotation_position="right"
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(
                        title="Time Offset",
                        gridcolor='rgba(255,255,255,0.1)',
                        showgrid=True
                    ),
                    yaxis=dict(
                        title="Water Level (m)",
                        gridcolor='rgba(255,255,255,0.1)',
                        showgrid=True
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=40, r=40, t=40, b=40),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Alert message
                max_pred = max(predictions)
                pred_status, pred_class, pred_msg = get_flood_status(max_pred, warning_threshold, danger_threshold)
                
                if max_pred >= warning_threshold:
                    st.warning(f"⚠️ **Alert**: Predicted water level may reach {max_pred:.2f}m. {pred_msg}")
                elif max_pred >= danger_threshold:
                    st.error(f"🚨 **Critical Alert**: Predicted water level may reach {max_pred:.2f}m! {pred_msg}")
                else:
                    st.success(f"✅ Water levels are predicted to remain safe (max: {max_pred:.2f}m)")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.exception(e)
    
    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #a0aec0; padding: 2rem;">
        <p>FloodSight | Early Flood Prediction System</p>
        <p style="font-size: 0.8rem;">Powered by LSTM Neural Network | Godavari River Basin</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
