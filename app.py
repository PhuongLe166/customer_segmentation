import streamlit as st
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "components"))
sys.path.append(str(Path(__file__).parent / "views"))

# Import page components
from views import introduction, about, eda, model_evaluation, bi_dashboard
from components.navigation import setup_navigation, get_current_page
from components.sidebar import setup_sidebar
from config.settings import APP_CONFIG

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title=APP_CONFIG["app_title"],
        page_icon=APP_CONFIG["page_icon"],
        layout=APP_CONFIG["layout"],
        initial_sidebar_state=APP_CONFIG["sidebar_state"]
    )
    
    # Custom CSS
    st.markdown(APP_CONFIG["custom_css"], unsafe_allow_html=True)
    
    # Sync current page from URL query params (e.g., ?page=Introduction)
    try:
        params = st.query_params if hasattr(st, "query_params") else {}
        # st.query_params can be a Mapping[str, Union[str, List[str]]]
        page_value = params.get("page") if isinstance(params, dict) else None
        if isinstance(page_value, list):
            page_param = page_value[0] if page_value else None
        else:
            page_param = page_value
        allowed_pages = [
            "Introduction",
            "About",
            "EDA",
            "Model Evaluation",
            "BI Dashboard",
        ]
        if page_param in allowed_pages:
            st.session_state.current_page = page_param
    except Exception:
        pass

    # Setup navigation
    setup_navigation()
    
    # Setup sidebar
    setup_sidebar()
    
    # Get current page from session state
    current_page = get_current_page()
    
    # Route to appropriate page
    if current_page == "Introduction":
        introduction.show()
    elif current_page == "About":
        about.show()
    elif current_page == "EDA":
        eda.show()
    elif current_page == "Model Evaluation":
        model_evaluation.show()
    elif current_page == "BI Dashboard":
        bi_dashboard.show()
    else:
        # Default to introduction
        introduction.show()

if __name__ == "__main__":
    main()