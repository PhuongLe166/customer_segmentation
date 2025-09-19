# pages/eda.py
import streamlit as st
from config.settings import PAGE_CONFIG

def show():
    """Display the EDA page"""
    st.markdown(f"# {PAGE_CONFIG['eda']['title']}")
    st.markdown(f"*{PAGE_CONFIG['eda']['description']}*")
    st.markdown("---")
    
    st.info("🚧 EDA implementation will be added here")
    st.markdown("This page will include:")
    st.markdown("- Data overview and statistics")
    st.markdown("- Sales trends and patterns")
    st.markdown("- Customer behavior analysis")
    st.markdown("- Product performance insights")

