# pages/bi_dashboard.py
import streamlit as st
import pandas as pd
import altair as alt
from config.settings import PAGE_CONFIG
from src.customer_segmentation_service import CustomerSegmentationService
from components import KPICards, ChartComponents, TableComponents, FormComponents

def show():
    """Display the BI Dashboard page"""
    st.markdown(f"# {PAGE_CONFIG['bi_dashboard']['title']}")
    st.markdown(f"*{PAGE_CONFIG['bi_dashboard']['description']}*")
    st.markdown("---")
    
    # Check if files are uploaded, use default if not
    transactions_file = getattr(st.session_state, "upload_transactions", None)
    products_file = getattr(st.session_state, "upload_products", None)
    
    # Initialize service and load data
    try:
        service = CustomerSegmentationService()
        
        # Load and prepare data
        data_prep = service.load_and_prepare_data(transactions_file, products_file)
        if data_prep['status'] != 'success':
            st.error(f"Data preparation failed: {data_prep.get('error', 'Unknown error')}")
            return
        
        merged = data_prep['merged_df']
        merged_rfm = data_prep['merged_rfm_df']
        
        # Perform RFM analysis
        rfm_analysis = service.perform_rfm_analysis(merged_rfm)
        if rfm_analysis['status'] != 'success':
            st.error(f"RFM analysis failed: {rfm_analysis.get('error', 'Unknown error')}")
            return
        
        rfm = rfm_analysis['rfm_df']
        
    except Exception as e:
        st.error(f"Service initialization failed: {e}")
        return

    # Status messages with file source info
    if transactions_file is None and products_file is None:
        st.info("📁 Using default files from data/raw/")
    else:
        # Show specific file names
        tx_name = "Default" if transactions_file is None else (
            transactions_file['name'] if isinstance(transactions_file, dict) else 
            getattr(transactions_file, 'name', 'Unknown')
        )
        pd_name = "Default" if products_file is None else (
            products_file['name'] if isinstance(products_file, dict) else 
            getattr(products_file, 'name', 'Unknown')
        )
        st.info(f"📤 Using uploaded files: {tx_name} • {pd_name}")
    
    # Key Performance Indicators Section (including RFM)
    st.markdown("### Key Performance Indicators")
    
    # Calculate KPIs using service
    kpis = service.calculate_kpis(merged, rfm)
    if kpis['status'] != 'success':
        st.error(f"KPI calculation failed: {kpis.get('error', 'Unknown error')}")
        return
    
    kpi_data = kpis['kpi_data']
    
    # Render advanced KPI cards using component
    KPICards.render_advanced_kpi_cards(kpi_data)
    
    # Render RFM cards using component
    st.markdown("---")
    KPICards.render_rfm_cards(kpi_data)
    st.markdown("---")
    
    # =========================
    # CLUSTERING ANALYSIS
    # =========================
    
    # Default k = 4 (can be adjusted by user)
    k = FormComponents.render_cluster_selector(min_clusters=2, max_clusters=10, default=4)
    
    # Perform K-Means clustering using service
    clustering = service.perform_kmeans_clustering(rfm.copy(), n_clusters=k)
    if clustering['status'] != 'success':
        st.error(f"K-Means clustering failed: {clustering.get('error', 'Unknown error')}")
        return
    
    rfm_km = clustering['rfm_clustered_df']
    clustering_metrics = clustering['clustering_metrics']
    
    
    # Create visualizations for K-Means clustering
    clustering_visualizations = service.create_visualizations({
        'merged_df': merged,
        'rfm_df': rfm,
        'rfm_clustered_df': rfm_km,
        'kpi_data': kpi_data
    })
    
    if clustering_visualizations['status'] != 'success':
        st.error(f"Clustering visualization creation failed: {clustering_visualizations.get('error', 'Unknown error')}")
        return
    
    # =========================
    # TWO CHARTS ON SAME LINE - ALIGNED LAYOUT
    # =========================
    
    # Create two columns for side-by-side charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Recency and Monetary of each Cluster")
        ChartComponents.render_cluster_bubble_chart(rfm_km)
    
    with col2:
        st.markdown("### Customer Segmentation: RFM Analysis")
        ChartComponents.render_cluster_scatter_chart(rfm_km)
    
    # =========================
    # KPI TABLE
    # =========================
    # Calculate segment KPIs using K-Means clustering
    segment_kpis = service.preprocess_core.calculate_segment_kpis(rfm_km)
    display_seg = service.evaluate_core.create_segment_table(segment_kpis)
    
    # Render segment table using component
    TableComponents.render_segment_table(display_seg, "Recency, Frequency and Monetary KPIs per Segment")
    