import streamlit as st
from components.navigation import get_current_page

def setup_sidebar():
    """Setup the sidebar with navigation and additional info"""
    
    with st.sidebar:
        # Title (remove "RFM Analysis" per request)
        st.markdown("### Customer Segmentation")
        st.markdown("---")
        
        # Current page indicator (no icons)
        current_page = get_current_page()
        st.markdown(f"**Current Page:** {current_page}")
        st.markdown("---")
        
        # Sidebar navigation (clickable, no icons)
        st.markdown("### Navigation")
        ordered_pages = [
            "Introduction",
            "About",
            "EDA",
            "Model Evaluation",
            "BI Dashboard",
            "Deep Insight",
        ]
        selected = st.selectbox(
            "",
            options=ordered_pages,
            index=ordered_pages.index(current_page) if current_page in ordered_pages else 0,
        )
        if selected != current_page:
            st.session_state.current_page = selected
            st.rerun()
        st.markdown("---")
        
        # Project info
        st.markdown("### 📋 Project Info")
        with st.expander("Dataset Info"):
            st.write("**Transactions:** 38,765 records")
            st.write("**Products:** 167 items")
            st.write("**Time Period:** Variable")
            st.write("**Customers:** Unique members")
        
        with st.expander("Analysis Methods"):
            st.write("• RFM Analysis")
            st.write("• Rule-based Segmentation") 
            st.write("• K-Means Clustering")
            st.write("• Customer Profiling")
        
        st.markdown("---")
        
        # Additional controls based on current page
        page_specific_sidebar(current_page)


def page_specific_sidebar(current_page):
    """Add page-specific sidebar controls"""
    
    if current_page == "EDA":
        st.markdown("### 📊 EDA Controls")
        
        # Date range selector
        if st.checkbox("Custom Date Range"):
            st.date_input("Start Date")
            st.date_input("End Date")
        
        # Chart options
        st.selectbox("Chart Type", ["Line", "Bar", "Area"])
        
    elif current_page == "Model Evaluation":
        st.markdown("### 🤖 Model Settings")
        
        # Clustering options
        st.slider("Number of Clusters", 2, 10, 5)
        st.selectbox("Clustering Method", ["K-Means", "Hierarchical"])
        
        # RFM options
        st.checkbox("Use Log Transform")
        st.checkbox("Standardize Features")
        
    elif current_page == "BI Dashboard":
        st.markdown("### 📈 Dashboard Filters")
        
        # Filters
        st.multiselect("Customer Segments", 
                      ["Champions", "Loyal Customers", "Potential Loyalists", 
                       "At Risk", "Can't Lose Them", "Hibernating", "Need Attention"])
        
        st.selectbox("Time Aggregation", ["Daily", "Weekly", "Monthly"])
        st.checkbox("Show Trends")
        
    elif current_page == "Deep Insight":
        st.markdown("### 🔍 Advanced Analytics")
        
        # Advanced filters
        st.selectbox("Analysis Type", ["Predictive", "Cohort", "Churn", "CLV"])
        st.slider("Prediction Horizon (days)", 30, 365, 90)
        st.checkbox("Include Seasonality")


def add_sidebar_metric(label, value, delta=None):
    """Add a metric to the sidebar"""
    with st.sidebar:
        st.metric(label, value, delta)


def add_sidebar_info(title, content):
    """Add information box to sidebar"""
    with st.sidebar:
        with st.expander(title):
            st.write(content)