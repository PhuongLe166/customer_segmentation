# pages/bi_dashboard.py
import streamlit as st
import pandas as pd
import altair as alt
from config.settings import PAGE_CONFIG
from src.customer_segmentation_service import CustomerSegmentationService
from components import KPICards, ChartComponents, TableComponents, FormComponents, Footer

def show():
    """Display the BI Dashboard page"""
    # Hero header
    st.markdown(
        f"""
        <style>
        .bi-hero {{
            /* Higher-contrast vibrant gradient */
            background: linear-gradient(120deg, #0a3d62 0%, #1f77b4 45%, #56ccf2 100%);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 14px;
            padding: 22px 24px;
            margin-bottom: 16px;
            box-shadow: 0 10px 28px rgba(16, 81, 126, 0.28);
        }}
        .bi-hero h1 {{
            margin: 0 0 6px 0;
            font-size: 26px;
            line-height: 1.25;
            color: #ffffff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }}
        .bi-hero p {{
            margin: 0;
            color: #e9f3fb;
            font-size: 14px;
        }}
        .section-title {{
            font-weight: 700;
            font-size: 18px;
            margin: 12px 0 8px 0;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #0f2942;
        }}
        .section-title:before {{
            content: "";
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            background: #1f77b4;
            box-shadow: 0 0 0 4px rgba(31,119,180,0.12);
        }}
        </style>
        <div class="bi-hero">
            <h1>{PAGE_CONFIG['bi_dashboard']['title']}</h1>
            <p>{PAGE_CONFIG['bi_dashboard']['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
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
    
    # Default k = 4 (fixed)
    k = 4
    
    # Perform K-Means clustering using service
    clustering = service.perform_kmeans_clustering(rfm.copy(), n_clusters=k)
    if clustering['status'] != 'success':
        st.error(f"K-Means clustering failed: {clustering.get('error', 'Unknown error')}")
        return
    
    rfm_km = clustering['rfm_clustered_df']
    clustering_metrics = clustering['clustering_metrics']
    
    # =========================
    # TWO CHARTS ON SAME LINE - ALIGNED LAYOUT
    # =========================
    
    # Custom CSS: style Altair chart containers directly (no extra wrapper divs)
    st.markdown("""
    <style>
    /* Style the native Altair chart container */
    [data-testid="stAltairChart"] {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin: 10px 0;
        transition: box-shadow 0.3s ease, border-color 0.3s ease;
        width: 100% !important;
        max-width: 100% !important;
        display: block !important;
        overflow: hidden;
    }
    [data-testid="stAltairChart"]:hover {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        border-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create two columns for side-by-side charts
    col1, col2 = st.columns(2)
    
    with col1:
        ChartComponents.render_cluster_bubble_chart(rfm_km)
    
    with col2:
        ChartComponents.render_cluster_scatter_chart(rfm_km)
    
    # =========================
    # KPI TABLE
    # =========================
    # Calculate segment KPIs using K-Means clustering
    segment_kpis = service.preprocess_core.calculate_segment_kpis(rfm_km)
    display_seg = service.evaluate_core.create_segment_table(segment_kpis)
    
    # Render segment table using component
    TableComponents.render_segment_table(display_seg, "Recency, Frequency and Monetary KPIs per Segment")
    
    # Footer
    Footer.render()
    