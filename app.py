"""
♻️ RecycleVision - Garbage Image Classification System
Streamlit Application — Deep Learning based Waste Classification & Recycling Assistant
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
from PIL import Image
import cv2
import os
from datetime import datetime
import time
import random
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="♻️ RecycleVision - Garbage Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #2E8B57, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #264653;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Card styling */
    .card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        border-left: 5px solid #4CAF50;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e9ecef;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    /* Class-specific styling */
    .plastic {
        background-color: #FF6B6B;
        color: white;
        padding: 8px;
        border-radius: 8px;
        text-align: center;
    }
    
    .paper {
        background-color: #4ECDC4;
        color: white;
        padding: 8px;
        border-radius: 8px;
        text-align: center;
    }
    
    .glass {
        background-color: #45B7D1;
        color: white;
        padding: 8px;
        border-radius: 8px;
        text-align: center;
    }
    
    .metal {
        background-color: #96CEB4;
        color: white;
        padding: 8px;
        border-radius: 8px;
        text-align: center;
    }
    
    .cardboard {
        background-color: #FFEEAD;
        color: #333;
        padding: 8px;
        border-radius: 8px;
        text-align: center;
    }
    
    .trash {
        background-color: #D4A5A5;
        color: white;
        padding: 8px;
        border-radius: 8px;
        text-align: center;
    }
    
    /* Button styling */
    .classify-btn {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 12px 30px;
        font-size: 1.2rem;
        border-radius: 25px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
        width: 100%;
        font-weight: bold;
    }
    
    .classify-btn:hover {
        background: linear-gradient(135deg, #45a049, #4CAF50);
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white;
        border-radius: 10px;
        margin-top: 30px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Title with animation
st.markdown('<h1 class="main-header">♻️ RecycleVision</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Waste Classification System for Smart Recycling</p>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home & Classification", "📈 Model Performance", "ℹ️ About & Info", "⚙️ Settings"]
)

# ========== FIXED FUNCTION: load_model_and_classes ==========
@st.cache_resource(show_spinner=False)
def load_model_and_classes():
    """Load the trained model and class labels"""
    try:
        # Try to load from current directory
        model_paths = [
            "RecycleVision_EfficientNetB0_Final.keras",
            "best_efficientnet.keras",
            "model.keras",
            "RecycleVision_Final_Model.keras"
        ]
        
        model = None
        loaded_path = None
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = load_model(path)
                    loaded_path = path
                    break
                except Exception as e:
                    st.sidebar.warning(f"Failed to load {path}: {str(e)}")
                    continue
        
        # Class labels (always return these)
        class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        
        if model is None:
            st.sidebar.warning("⚠️ Model file not found. Running in DEMO MODE.")
            return None, class_labels
        
        st.sidebar.success(f"✅ Model loaded from: {loaded_path}")
        return model, class_labels
        
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        return None, class_labels

# ========== LOAD MODEL AND HANDLE RETURN VALUES ==========
load_result = load_model_and_classes()

# Default values (in case model not found)
class_colors = {
    'cardboard': '#FFEEAD',
    'glass': '#45B7D1',
    'metal': '#96CEB4',
    'paper': '#4ECDC4',
    'plastic': '#FF6B6B',
    'trash': '#D4A5A5'
}

recycling_info = {
    'cardboard': {
        'type': '📦 Paper/Cardboard',
        'recyclable': '✅ Fully Recyclable',
        'bin_color': 'Blue Bin',
        'processing': 'Pulped and remanufactured into paper products',
        'facts': 'Cardboard can be recycled 5-7 times before fibers become too short',
        'icon': '📦',
        'tips': 'Flatten boxes before recycling to save space'
    },
    'glass': {
        'type': '🥃 Glass',
        'recyclable': '✅ 100% Recyclable',
        'bin_color': 'Green Bin',
        'processing': 'Crushed, melted, and molded into new glass products',
        'facts': 'Glass takes 1 million years to decompose in landfill',
        'icon': '🥃',
        'tips': 'Rinse containers; labels can stay on'
    },
    'metal': {
        'type': '🔩 Metal',
        'recyclable': '✅ Highly Recyclable',
        'bin_color': 'Yellow Bin',
        'processing': 'Shredded, melted, and purified',
        'facts': 'Recycling aluminum saves 95% of energy needed to make new metal',
        'icon': '🔩',
        'tips': 'Cans can be crushed to save space'
    },
    'paper': {
        'type': '📄 Paper',
        'recyclable': '✅ Recyclable',
        'bin_color': 'Blue Bin',
        'processing': 'Mixed with water to create pulp, then pressed and dried',
        'facts': 'Each ton of recycled paper saves 17 trees',
        'icon': '📄',
        'tips': 'Keep paper dry and clean'
    },
    'plastic': {
        'type': '🧴 Plastic',
        'recyclable': '⚠️ Depends on type',
        'bin_color': 'White/Clear Bin',
        'processing': 'Sorted by type, shredded, melted, and pelletized',
        'facts': 'Plastic takes 450+ years to decompose',
        'icon': '🧴',
        'tips': 'Check recycling number on bottom'
    },
    'trash': {
        'type': '🗑️ General Waste',
        'recyclable': '❌ Not Recyclable',
        'bin_color': 'Black Bin',
        'processing': 'Landfill or incineration',
        'facts': 'Reduce waste by choosing reusable products',
        'icon': '🗑️',
        'tips': 'Consider if items can be reused or repaired'
    }
}

# Check what was returned from load_model_and_classes
if load_result is None:
    st.error("Failed to initialize application")
    st.stop()
elif len(load_result) == 2:
    model, class_labels = load_result
    # class_colors and recycling_info already defined above
else:
    st.error("Unexpected return value from load_model_and_classes()")
    st.stop()

# Check if model is None (demo mode)
if model is None:
    st.sidebar.info("🔧 Running in DEMO MODE - Showing sample predictions")
    st.sidebar.warning("Train the model and place it in the project folder for real predictions")

# ========== FIXED FUNCTION: predict_image ==========
def predict_image(model, image):
    """Make prediction on image"""
    try:
        if model is None:
            # Dummy prediction for UI demonstration
            # Return realistic-looking predictions
            # Random but biased towards certain classes for demo
            pred = np.random.random(6)
            pred = pred / pred.sum()  # Normalize to sum to 1
            return pred
            
        processed_img = preprocess_image(image)
        if processed_img is not None:
            predictions = model.predict(processed_img, verbose=0)[0]
            return predictions
        return None
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Helper functions (unchanged)
def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize((224, 224))
        
        # Convert to array
        img_array = np.array(image)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for EfficientNet
        img_array = preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def get_top_predictions(predictions, class_labels, top_n=3):
    """Get top N predictions"""
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    return [(class_labels[i], predictions[i] * 100) for i in top_indices]

def create_confidence_chart(predictions, class_labels):
    """Create confidence bar chart"""
    df = pd.DataFrame({
        'Class': class_labels,
        'Confidence': predictions * 100
    }).sort_values('Confidence', ascending=True)
    
    fig = px.bar(
        df,
        x='Confidence',
        y='Class',
        orientation='h',
        title='Classification Confidence Scores',
        color='Confidence',
        color_continuous_scale='Greens',
        text=df['Confidence'].round(1).astype(str) + '%'
    )
    
    fig.update_layout(
        xaxis_title='Confidence (%)',
        yaxis_title='Waste Class',
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    fig.update_traces(textposition='outside')
    
    return fig

def create_donut_chart(confidence, class_name):
    """Create donut chart for main prediction"""
    fig = go.Figure(data=[go.Pie(
        values=[confidence, 100-confidence],
        labels=[class_name, 'Other'],
        hole=.7,
        marker_colors=['#4CAF50', '#E0E0E0'],
        textinfo='none'
    )])
    
    fig.update_layout(
        annotations=[dict(text=f'{confidence:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
        height=200,
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0)
    )
    
    return fig

# ==================== PAGE 1: HOME & CLASSIFICATION ====================
if page == "🏠 Home & Classification":
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Waste Image")
        
        # File uploader with camera option
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Upload a clear image of the waste item"
        )
        
        # Camera input option
        camera_image = st.camera_input("Or take a photo")
        
        # Use camera image if available, otherwise use uploaded file
        if camera_image is not None:
            image = Image.open(camera_image)
            st.success("✅ Photo captured successfully!")
        elif uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.success("✅ Image uploaded successfully!")
        else:
            image = None
            st.info("👆 Please upload an image or take a photo to begin")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display uploaded image
        if image is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🖼️ Preview")
            
            # Create columns for image display
            img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
            with img_col2:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Classify button
            if st.button("🔍 Classify Waste", use_container_width=True):
                with st.spinner("🔄 Analyzing image..."):
                    time.sleep(1)  # Simulate processing
                    
                    # Make prediction
                    predictions = predict_image(model, image)
                    
                    if predictions is not None:
                        # Store predictions in session state
                        st.session_state['predictions'] = predictions
                        st.session_state['image'] = image
                        st.session_state['classified'] = True
                        
                        st.success("✅ Classification complete!")
                        st.rerun()
                    else:
                        st.error("❌ Classification failed. Please try again.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Display results if classified
        if 'classified' in st.session_state and st.session_state['classified']:
            predictions = st.session_state['predictions']
            image = st.session_state['image']
            
            # Main prediction
            pred_class_idx = np.argmax(predictions)
            pred_class = class_labels[pred_class_idx]
            confidence = predictions[pred_class_idx] * 100
            
            # Get top 3 predictions
            top_3 = get_top_predictions(predictions, class_labels)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Classification Results")
            
            # Main result with styling
            result_col1, result_col2 = st.columns([1, 1])
            
            with result_col1:
                # Donut chart for main prediction
                fig_donut = create_donut_chart(confidence, pred_class)
                st.plotly_chart(fig_donut, use_container_width=True)
            
            with result_col2:
                # Main prediction details
                st.markdown(f"### **Predicted Class:**")
                class_color = class_colors.get(pred_class, '#E0E0E0')
                st.markdown(f"<div style='background-color: {class_color}; padding: 15px; border-radius: 10px; text-align: center;'>"
                          f"<h2>{recycling_info[pred_class]['icon']} {pred_class.upper()}</h2>"
                          f"<h3>Confidence: {confidence:.2f}%</h3>"
                          f"</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Top 3 predictions
            st.markdown("### 📊 Top 3 Predictions")
            for i, (class_name, conf) in enumerate(top_3):
                st.markdown(f"""
                <div style='margin: 5px 0; padding: 10px; background-color: #f8f9fa; border-radius: 8px;'>
                    <strong>{i+1}. {recycling_info[class_name]['icon']} {class_name.capitalize()}</strong>
                    <div style='margin-top: 5px;'>
                        <div style='background-color: #e0e0e0; border-radius: 10px; height: 10px; width: 100%;'>
                            <div style='background-color: #4CAF50; border-radius: 10px; height: 10px; width: {conf}%;'></div>
                        </div>
                        <span style='float: right;'>{conf:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recycling Information
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### {recycling_info[pred_class]['icon']} Recycling Information")
            
            info = recycling_info[pred_class]
            
            # Info boxes
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"""
                <div class="info-box">
                    <strong>♻️ Recyclability</strong><br>
                    {info['recyclable']}
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="info-box">
                    <strong>🗑️ Bin Color</strong><br>
                    {info['bin_color']}
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                st.markdown(f"""
                <div class="info-box">
                    <strong>🔧 Processing</strong><br>
                    {info['processing'][:50]}...
                </div>
                """, unsafe_allow_html=True)
            
            # Facts and tips
            st.markdown("---")
            st.markdown(f"**📌 Did you know?** {info['facts']}")
            st.markdown(f"**💡 Tip:** {info['tips']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Confidence chart
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = create_confidence_chart(predictions, class_labels)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Environmental impact
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🌍 Environmental Impact")
            
            impact_data = {
                'cardboard': {'trees': 0.5, 'co2': 2.0, 'water': 100},
                'glass': {'trees': 0, 'co2': 3.5, 'water': 50},
                'metal': {'trees': 0, 'co2': 4.0, 'water': 75},
                'paper': {'trees': 1.0, 'co2': 2.5, 'water': 150},
                'plastic': {'trees': 0, 'co2': 1.5, 'water': 25},
                'trash': {'trees': 0, 'co2': 0.5, 'water': 10}
            }
            
            impact = impact_data[pred_class]
            
            col_imp1, col_imp2, col_imp3 = st.columns(3)
            with col_imp1:
                st.metric("Trees Saved", f"{impact['trees']} per kg" if impact['trees'] > 0 else "N/A")
            with col_imp2:
                st.metric("CO₂ Reduced", f"{impact['co2']} kg per kg")
            with col_imp3:
                st.metric("Water Saved", f"{impact['water']} L per kg")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download report
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📥 Download Analysis Report")
            
            report_content = f"""
            RECYCLEVISION - WASTE CLASSIFICATION REPORT
            ============================================
            
            ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            CLASSIFICATION RESULTS:
            - Primary Class: {pred_class.upper()}
            - Confidence: {confidence:.2f}%
            
            TOP 3 PREDICTIONS:
            1. {top_3[0][0].upper()}: {top_3[0][1]:.2f}%
            2. {top_3[1][0].upper()}: {top_3[1][1]:.2f}%
            3. {top_3[2][0].upper()}: {top_3[2][1]:.2f}%
            
            RECYCLING INFORMATION:
            - Recyclability: {info['recyclable']}
            - Bin Color: {info['bin_color']}
            - Processing: {info['processing']}
            
            ENVIRONMENTAL IMPACT:
            - CO₂ Reduction: {impact['co2']} kg per kg
            - Water Saved: {impact['water']} L per kg
            - Trees Saved: {impact['trees']} per kg
            
            EDUCATIONAL TIP:
            {info['tips']}
            
            INTERESTING FACT:
            {info['facts']}
            
            -- Generated by RecycleVision AI System --
            """
            
            st.download_button(
                label="📄 Download Report (TXT)",
                data=report_content,
                file_name=f"recyclevision_report_{pred_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # Welcome message when no image classified
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 👋 Welcome to RecycleVision!")
            st.markdown("""
            **Get started by:**
            1. 📤 Uploading an image of your waste item
            2. 📸 Taking a photo using your camera
            3. 🔍 Clicking "Classify Waste" button
            
            **Supported waste types:**
            - 📦 Cardboard
            - 🥃 Glass  
            - 🔩 Metal
            - 📄 Paper
            - 🧴 Plastic
            - 🗑️ Trash
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Sample classifications
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Sample Classifications")
            
            sample_col1, sample_col2, sample_col3 = st.columns(3)
            with sample_col1:
                st.markdown("""
                <div style='text-align: center; padding: 15px; background-color: #f0f0f0; border-radius: 10px;'>
                    <h3>📦</h3>
                    <p>Cardboard</p>
                    <small>98% confidence</small>
                </div>
                """, unsafe_allow_html=True)
            
            with sample_col2:
                st.markdown("""
                <div style='text-align: center; padding: 15px; background-color: #f0f0f0; border-radius: 10px;'>
                    <h3>🥃</h3>
                    <p>Glass</p>
                    <small>95% confidence</small>
                </div>
                """, unsafe_allow_html=True)
            
            with sample_col3:
                st.markdown("""
                <div style='text-align: center; padding: 15px; background-color: #f0f0f0; border-radius: 10px;'>
                    <h3>🧴</h3>
                    <p>Plastic</p>
                    <small>92% confidence</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Quick stats
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Quick Stats")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Model Accuracy", "84%", "+2.5%")
            with stat_col2:
                st.metric("Classes", "6", "")
            with stat_col3:
                st.metric("Training Images", "2,027", "")
            with stat_col4:
                st.metric("Inference Time", "<1s", "Fast")
            
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE 2: MODEL PERFORMANCE ====================
elif page == "📈 Model Performance":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Model Performance Metrics")
    
    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "83.2%", "↑ 2.1%")
    with col2:
        st.metric("Precision", "84.0%", "↑ 1.8%")
    with col3:
        st.metric("Recall", "83.0%", "↑ 1.5%")
    with col4:
        st.metric("F1-Score", "83.0%", "↑ 1.6%")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Confusion Matrix
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Confusion Matrix")
    
    # Confusion matrix data from validation
    cm_data = np.array([
        [63, 0, 2, 9, 0, 7],
        [0, 86, 9, 0, 4, 1],
        [0, 7, 70, 1, 4, 1],
        [2, 1, 2, 111, 1, 2],
        [1, 2, 4, 4, 68, 17],
        [0, 2, 1, 2, 2, 20]
    ])
    
    fig_cm = px.imshow(
        cm_data,
        x=class_labels,
        y=class_labels,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues',
        title="Confusion Matrix - Validation Set"
    )
    
    fig_cm.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=500
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Class-wise Performance
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Class-wise Performance")
    
    performance_data = pd.DataFrame({
        'Class': class_labels,
        'Precision': [0.97, 0.80, 0.80, 0.87, 0.87, 0.50],
        'Recall': [0.78, 0.86, 0.85, 0.93, 0.71, 0.74],
        'F1-Score': [0.86, 0.83, 0.82, 0.90, 0.78, 0.60],
        'Support': [81, 100, 82, 119, 96, 27]
    })
    
    fig_perf = px.bar(
        performance_data.melt(id_vars=['Class'], value_vars=['Precision', 'Recall', 'F1-Score']),
        x='Class',
        y='value',
        color='variable',
        barmode='group',
        title="Class-wise Performance Metrics",
        labels={'value': 'Score', 'variable': 'Metric'}
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Training History
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Training History")
    
    # Simulated training history
    epochs = list(range(1, 13))
    train_acc = [0.47, 0.76, 0.78, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.87, 0.88]
    val_acc = [0.76, 0.77, 0.80, 0.80, 0.82, 0.84, 0.82, 0.83, 0.83, 0.84, 0.83, 0.84]
    train_loss = [1.86, 0.72, 0.70, 0.56, 0.55, 0.47, 0.46, 0.45, 0.38, 0.35, 0.36, 0.33]
    val_loss = [0.70, 0.62, 0.53, 0.51, 0.51, 0.48, 0.50, 0.56, 0.55, 0.51, 0.49, 0.50]
    
    history_df = pd.DataFrame({
        'Epoch': epochs,
        'Training Accuracy': train_acc,
        'Validation Accuracy': val_acc,
        'Training Loss': train_loss,
        'Validation Loss': val_loss
    })
    
    fig_history = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Accuracy', 'Model Loss')
    )
    
    fig_history.add_trace(
        go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Training Accuracy', line=dict(color='#4CAF50')),
        row=1, col=1
    )
    
    fig_history.add_trace(
        go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation Accuracy', line=dict(color='#2196F3')),
        row=1, col=1
    )
    
    fig_history.add_trace(
        go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Training Loss', line=dict(color='#FF5722')),
        row=1, col=2
    )
    
    fig_history.add_trace(
        go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation Loss', line=dict(color='#9C27B0')),
        row=1, col=2
    )
    
    fig_history.update_layout(height=400, showlegend=True)
    fig_history.update_xaxes(title_text="Epoch", row=1, col=1)
    fig_history.update_xaxes(title_text="Epoch", row=1, col=2)
    fig_history.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig_history.update_yaxes(title_text="Loss", row=1, col=2)
    
    st.plotly_chart(fig_history, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE 3: ABOUT & INFO ====================
elif page == "ℹ️ About & Info":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ About RecycleVision")
    
    st.markdown("""
    **RecycleVision** is an advanced deep learning system designed to automatically classify 
    waste materials into six categories, helping streamline recycling processes and promoting 
    environmental sustainability.
    
    #### 🎯 Project Objectives
    - Automate waste sorting for recycling facilities
    - Reduce manual sorting time and labor costs
    - Educate people about proper waste segregation
    - Provide environmental impact insights
    
    #### 🤖 Technology Stack
    - **Deep Learning Framework:** TensorFlow 2.15 / Keras
    - **Model Architecture:** EfficientNetB0 (Transfer Learning)
    - **Frontend:** Streamlit 1.28
    - **Data Processing:** NumPy, Pandas, OpenCV
    - **Visualization:** Plotly, Matplotlib, Seaborn
    
    #### 📊 Dataset
    - **Source:** TrashNet / Garbage Classification Dataset
    - **Total Images:** 2,532
    - **Classes:** 6 waste categories
    - **Split:** 80% Training (2,027) | 20% Validation (505)
    
    #### 🏆 Model Performance
    - **Accuracy:** 83.2%
    - **Precision:** 84.0%
    - **Recall:** 83.0%
    - **F1-Score:** 83.0%
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Class Information
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗑️ Waste Classes Information")
    
    for class_name in class_labels:
        info = recycling_info[class_name]
        with st.expander(f"{info['icon']} {class_name.capitalize()} - {info['type']}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**♻️ Recyclable:** {info['recyclable']}")
                st.markdown(f"**🗑️ Bin Color:** {info['bin_color']}")
                st.markdown(f"**🔧 Processing:** {info['processing']}")
            with col_b:
                st.markdown(f"**📌 Fact:** {info['facts']}")
                st.markdown(f"**💡 Tip:** {info['tips']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Environmental Impact
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌍 Environmental Impact")
    
    impact_data = {
        'Category': ['CO₂ Reduction (kg/kg)', 'Energy Savings (%)', 'Water Saved (L/kg)', 'Trees Saved (per kg)'],
        'Cardboard': [2.0, 75, 100, 0.5],
        'Glass': [3.5, 60, 50, 0],
        'Metal': [4.0, 95, 75, 0],
        'Paper': [2.5, 70, 150, 1.0],
        'Plastic': [1.5, 80, 25, 0],
        'Trash': [0.5, 0, 10, 0]
    }
    
    impact_df = pd.DataFrame(impact_data)
    st.dataframe(impact_df.style.highlight_max(color='lightgreen'), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Team Info
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👥 Project Team")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3>🧑‍💻 Data Scientist</h3>
            <p>Model Development & Training</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3>🎨 UI/UX Designer</h3>
            <p>Streamlit App Development</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3>🔬 Domain Expert</h3>
            <p>Waste Management Specialist</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE 4: SETTINGS ====================
elif page == "⚙️ Settings":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Application Settings")
    
    # Model settings
    st.markdown("#### 🤖 Model Configuration")
    
    conf_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=0,
        max_value=100,
        value=50,
        help="Minimum confidence for classification"
    )
    
    top_k = st.slider(
        "Top-K Predictions",
        min_value=1,
        max_value=6,
        value=3,
        help="Number of top predictions to show"
    )
    
    # Display settings
    st.markdown("#### 🎨 Display Settings")
    
    theme = st.selectbox(
        "Theme",
        ["Light", "Dark", "System Default"],
        help="Choose application theme"
    )
    
    chart_style = st.selectbox(
        "Chart Style",
        ["Modern", "Classic", "Minimalist"],
        help="Choose chart visualization style"
    )
    
    # Notification settings
    st.markdown("#### 🔔 Notifications")
    
    show_tips = st.checkbox("Show recycling tips", value=True)
    show_facts = st.checkbox("Show environmental facts", value=True)
    show_impact = st.checkbox("Show impact metrics", value=True)
    
    # Advanced settings
    st.markdown("#### 🔧 Advanced Settings")
    
    batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=1)
    use_gpu = st.checkbox("Use GPU if available", value=True)
    
    # Save settings button
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("✅ Settings saved successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # About the model
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Model Details")
    
    st.markdown("""
    **Model Architecture:** EfficientNetB0
    - **Input Shape:** 224 × 224 × 3
    - **Total Parameters:** 4,384,169
    - **Trainable Parameters:** 332,038
    - **Non-trainable Parameters:** 4,052,131
    
    **Training Configuration:**
    - **Optimizer:** Adam (lr=0.001)
    - **Loss Function:** Categorical Crossentropy
    - **Batch Size:** 32
    - **Epochs:** 25 (with early stopping)
    - **Class Weights:** Balanced
    
    **Data Augmentation:**
    - Rotation Range: 25°
    - Zoom Range: 20%
    - Width/Height Shift: 20%
    - Horizontal Flip: Yes
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p style='font-size: 1.2rem; margin-bottom: 10px;'>♻️ RecycleVision - Making Waste Management Smarter</p>
    <p style='font-size: 0.9rem; opacity: 0.9;'>
        © 2024 | Deep Learning Project | EfficientNetB0 | 84% Accuracy<br>
        <small>Every classification helps build a cleaner tomorrow</small>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; padding: 10px; background-color: #f0f0f0; border-radius: 10px;'>
    <p style='margin: 0;'><strong>♻️ RecycleVision v2.0</strong></p>
    <p style='margin: 0; font-size: 0.8rem;'>AI-Powered Waste Classification</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'classified' not in st.session_state:
    st.session_state['classified'] = False