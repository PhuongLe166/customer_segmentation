import streamlit as st
from config.settings import PAGE_CONFIG

def show():
    """Display the Introduction page"""
    
    # Page header
    st.markdown(f"# {PAGE_CONFIG['introduction']['title']}")
    st.markdown("---")
    
    # Images with subtle shadow and hover interaction (via CSS on st.image)
    st.markdown("""
    <style>
    .stImage img {
        border-radius: 16px !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.15) !important;
        transition: transform .25s ease, box-shadow .25s ease, filter .25s ease !important;
    }
    .stImage img:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 12px 28px rgba(0,0,0,0.22) !important;
        filter: saturate(1.02) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    try:
        st.image("assets/images/customer_segmentation.jpg", width='stretch')
        st.image("assets/images/glocery.jpg", width='stretch')
        st.markdown("---")
    except Exception:
        pass

