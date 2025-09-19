# pages/eda.py
import streamlit as st
from config.settings import PAGE_CONFIG
import pandas as pd
from pathlib import Path
from src.eda_utils import load_datasets, infer_join_keys, merge_datasets

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
    
    # Load data via src utilities
    try:
        df_transactions, df_products, src_tx, src_pd = load_datasets(transactions_file, products_file)
    except Exception as e:
        st.error(f"Failed to load datasets: {e}")
        return
    
    # Status messages
    st.success(f"Loaded Transactions ({src_tx}) with {len(df_transactions):,} rows • Products ({src_pd}) with {len(df_products):,} rows")
    # Light CSS for nicer tables/sections
    st.markdown("""
    <style>
      [data-testid="stDataFrame"] { border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,.04); }
      .stat-badges { display:flex; gap:8px; margin: 6px 0 8px; flex-wrap: wrap; }
      .stat { background:#f7fbff; border:1px solid #e3f0ff; color:#1f77b4; padding:6px 10px; border-radius:9999px; font-weight:600; font-size:12px; }
    </style>
    """, unsafe_allow_html=True)

    # Tabs for EDA workflow
    tab_overview, tab_merge = st.tabs(["Overview Data", "Merge Datasets"])
    
    with tab_overview:
        # Two cards side-by-side for a cleaner overview
        col_tx, col_pd = st.columns(2, gap="large")
        with col_tx:
            st.markdown("#### Transactions")
            st.markdown(f"<div class='stat-badges'><span class='stat'>Rows: {len(df_transactions):,}</span><span class='stat'>Columns: {df_transactions.shape[1]}</span></div>", unsafe_allow_html=True)
            st.dataframe(df_transactions.head(10), use_container_width=True)
            

        with col_pd:
            st.markdown("#### Products")
            st.markdown(f"<div class='stat-badges'><span class='stat'>Rows: {len(df_products):,}</span><span class='stat'>Columns: {df_products.shape[1]}</span></div>", unsafe_allow_html=True)
            st.dataframe(df_products.head(10), use_container_width=True)
            
    
    with tab_merge:
        st.markdown("#### Merge Transactions + Products")
        # Infer join column using utility
        left_key, right_key = infer_join_keys(df_transactions, df_products)
        c5, c6, c7 = st.columns(3)
        with c5:
            left_on = st.selectbox("Left key (Transactions)", options=list(df_transactions.columns), index=list(df_transactions.columns).index(left_key))
        with c6:
            right_on = st.selectbox("Right key (Products)", options=list(df_products.columns), index=list(df_products.columns).index(right_key))
        with c7:
            how = st.selectbox("Join type", ["inner", "left", "right", "outer"], index=0)
        try:
            merged = merge_datasets(df_transactions, df_products, left_on, right_on, how)
            st.info(f"Merged shape: {merged.shape[0]:,} rows × {merged.shape[1]:,} columns")
            st.dataframe(merged.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Merge failed: {e}")
        

