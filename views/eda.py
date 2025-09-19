# pages/eda.py
import streamlit as st
from config.settings import PAGE_CONFIG
import pandas as pd
from pathlib import Path

def show():
    """Display the EDA page"""
    st.markdown(f"# {PAGE_CONFIG['eda']['title']}")
    st.markdown(f"*{PAGE_CONFIG['eda']['description']}*")
    st.markdown("---")
    
    # Default file paths
    default_transactions = Path("data/raw/Transactions.csv")
    default_products = Path("data/raw/Products_with_Categories.csv")
    
    # Uploaders are shown in top navigation; read from session_state if present
    transactions_file = getattr(st.session_state, "upload_transactions", None)
    products_file = getattr(st.session_state, "upload_products", None)
    
    # Load data
    def read_csv(file_or_path: Path | bytes):
        try:
            if hasattr(file_or_path, "read"):
                return pd.read_csv(file_or_path)
            return pd.read_csv(file_or_path)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return None
    
    df_transactions = read_csv(transactions_file if transactions_file else default_transactions)
    df_products = read_csv(products_file if products_file else default_products)
    
    # Preview
    st.markdown("---")
    st.markdown("### Preview")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Transactions**")
        if df_transactions is not None:
            st.dataframe(df_transactions.head(10), use_container_width=True)
    with p2:
        st.markdown("**Products**")
        if df_products is not None:
            st.dataframe(df_products.head(10), use_container_width=True)

