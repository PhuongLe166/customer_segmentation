import streamlit as st

def setup_navigation():
    """Setup the main navigation system"""
    
    # Initialize session state for navigation if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Introduction"
    
    # Navigation handled by sidebar only (no top tabs or buttons)
    st.markdown("---")
    # Top area reserved; no controls here now


def get_current_page():
    """Get the currently selected page"""
    return st.session_state.get('current_page', "Introduction")


def navigate_to(page_name):
    """Navigate programmatically to a specific page"""
    if page_name in ["Introduction", "About", "EDA", "Model Evaluation", "BI Dashboard"]:
        st.session_state.current_page = page_name
        st.rerun()
    else:
        st.error(f"Page '{page_name}' not found!")


def get_page_icon(page_name):
    """Get icon for specific page"""
    icons = {
        "Introduction": "🏠",
        "About": "ℹ️", 
        "EDA": "📊",
        "Model Evaluation": "🤖",
        "BI Dashboard": "📈",
        "Deep Insight": "🔍"
    }
    return icons.get(page_name, "📄")