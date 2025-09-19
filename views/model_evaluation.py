# pages/model_evaluation.py  
import streamlit as st
from config.settings import PAGE_CONFIG

def show():
    """Display the Model Evaluation page"""
    st.markdown(f"# {PAGE_CONFIG['model_evaluation']['title']}")
    st.markdown(f"*{PAGE_CONFIG['model_evaluation']['description']}*")
    st.markdown("---")
    
    st.info("🚧 Model building implementation will be added here")
    st.markdown("This page will include:")
    st.markdown("- RFM score calculation")
    st.markdown("- Rule-based segmentation")
    st.markdown("- K-Means clustering analysis")
    st.markdown("- Model evaluation metrics")

