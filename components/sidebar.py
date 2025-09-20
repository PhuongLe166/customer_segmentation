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
        
        # Uploads for all pages that need data
        if current_page in ["EDA", "Model Evaluation", "BI Dashboard"]:
            st.markdown("### Upload data files")
            
            # Check if files are already uploaded
            has_transactions = hasattr(st.session_state, 'upload_transactions') and st.session_state.upload_transactions is not None
            has_products = hasattr(st.session_state, 'upload_products') and st.session_state.upload_products is not None
            
            if has_transactions and has_products:
                # Show current uploaded files
                st.success("✅ Files loaded successfully!")
                st.info(f"📄 {st.session_state.upload_transactions['name']}")
                st.info(f"📄 {st.session_state.upload_products['name']}")
                
                # Option to reload files
                if st.button("🔄 Reload Files", key="reload_files"):
                    # Clear existing files
                    if 'upload_transactions' in st.session_state:
                        del st.session_state.upload_transactions
                    if 'upload_products' in st.session_state:
                        del st.session_state.upload_products
                    st.success("Files cleared. Please upload new files.")
                    st.rerun()
            else:
                # File uploaders (only show when no files are loaded)
                uploaded_transactions = st.file_uploader(
                    "Transactions.csv", type=["csv"], key="sidebar_upload_transactions"
                )
                uploaded_products = st.file_uploader(
                    "Products_with_Categories.csv", type=["csv"], key="sidebar_upload_products"
                )
                
                # Apply button
                if st.button("Apply", key="apply_uploads", type="primary"):
                    if uploaded_transactions is not None:
                        # Store file content and metadata
                        st.session_state.upload_transactions = {
                            'name': uploaded_transactions.name,
                            'content': uploaded_transactions.getvalue(),
                            'type': uploaded_transactions.type
                        }
                    if uploaded_products is not None:
                        # Store file content and metadata
                        st.session_state.upload_products = {
                            'name': uploaded_products.name,
                            'content': uploaded_products.getvalue(),
                            'type': uploaded_products.type
                        }
                    st.success("Files applied successfully!")
                    st.rerun()
            
            st.markdown("---")

        # Remove Project Info section per request
        st.markdown("---")
        
        # Additional controls based on current page
        page_specific_sidebar(current_page)


def page_specific_sidebar(current_page):
    """Add page-specific sidebar controls"""
    
    if current_page == "EDA":
        # EDA Controls removed per request
        pass
        
    elif current_page == "Model Evaluation":
        # Per request, remove Model Settings from sidebar for Model Evaluation page
        pass
        
    elif current_page == "BI Dashboard":
        # Dashboard filters removed per request
        pass
        
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