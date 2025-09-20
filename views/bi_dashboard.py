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
    
    # =========================
    # K-MEANS CLUSTERING (needed for segment slicer)
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
    # CLUSTERING ANALYSIS (K-Means already performed above)
    # =========================
    
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
    
    # =========================
    # NEW CHARTS SECTION
    # =========================
    st.markdown("---")
    st.markdown("### Advanced Analytics")
    
    # Create tabs for different chart types
    tab_cohort, tab_revenue_orders, tab_customer, tab_strategies = st.tabs(["Cohort Analysis", "Revenue & Orders", "Customer Explorer", "Strategies by Segment"])
    
    with tab_cohort:
        st.markdown("#### Cohort Analysis")
        st.markdown("Customer retention analysis showing how customer groups behave over time.")
        ChartComponents.render_cohort_analysis_chart(merged, "Date", "Member_number")
    
    with tab_revenue_orders:
        st.markdown("#### Revenue & Orders Over Time")
        st.markdown("Track revenue and order trends with different time granularities.")
        
        # Granularity selector
        granularity = FormComponents.render_granularity_selector()
        ChartComponents.render_revenue_orders_chart(merged, "Date", granularity)
    
    with tab_customer:
        st.markdown("#### Customer Explorer")
        st.markdown("Select an existing customer or input a new customer's RFM to see their profile and cluster.")
        # Scoped styles for nice blocks
        st.markdown(
            """
            <style>
            .ce-card {border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px; background:#ffffff; box-shadow: 0 1px 2px rgba(16,24,40,.04); margin: 6px 0 14px}
            .ce-card--scores {background:#f0f7ff; border-color:#cfe3ff}
            .ce-card--cat {background:#f3fafc; border-color:#cfe8f3}
            .ce-card--prod {background:#fffaf2; border-color:#f3e2c1}
            .ce-title {font-weight:800; color:#0f172a; margin:0 0 12px; font-size:16px}
            .ce-section-title {display:inline-block; font-weight:800; padding:6px 10px; border-radius:10px; margin:0 0 10px;}
            /* Make Revenue over time title visually consistent with Preferred Categories */
            .ce-section-title.hist {background:#e6f2fb; color:#0a3d62; border:1px solid #cfe3ff}
            .ce-section-title.cat {background:#e6f2fb; color:#0a3d62; border:1px solid #cfe3ff}
            .ce-section-title.prod {background:#fff2df; color:#8a5200; border:1px solid #f3e2c1}
            .ce-grid {display:grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap:12px;}
            .ce-tile {background:#ffffff; border:1px solid #e6eaf0; border-radius:12px; padding:14px; box-shadow: 0 1px 2px rgba(16,24,40,.04)}
            .ce-tile--accent {background:#edf7ff; border-color:#cbe6ff}
            .ce-tile .value {font-size:24px; font-weight:800; color:#0f172a}
            .ce-tile .label {margin-top:6px; font-size:12px; color:#475569; font-weight:700; letter-spacing:.2px}
            /* Match height with side cards (reduced) */
            .ce-equal {height: 420px; display:flex; flex-direction:column}
            /* Native Streamlit Altair container styled to look like a card */
            /* Style the native Streamlit Altair container directly (no extra wrappers) */
            [data-testid="stAltairChart"]{ 
              border:1px solid #e5e7eb; border-radius:12px; background:#ffffff; 
              box-shadow:0 1px 2px rgba(16,24,40,.04); height:420px; overflow:hidden; margin-top:0;
            }
            [data-testid="stAltairChart"] svg{height:320px !important}
            .ce-list {margin:0; padding-left:18px; line-height:1.7; overflow:auto; flex:1}
            .ce-card .vega-embed, .ce-card .vega-embed > div { width: 100% !important; }
            .ce-card svg { width: 100% !important; }
            [data-testid="stAltairChart"] { background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:12px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
            </style>
            """,
            unsafe_allow_html=True,
        )
        import numpy as np
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Base RFM data with clusters
        rfm_base = rfm_km.copy()
        rfm_base = rfm_base.reset_index().rename(columns={rfm_base.index.name or 'index': 'member_number'})
        
        # UI mode
        mode = st.radio("Mode", ["Existing customer", "New customer"], horizontal=True)
        
        # Fit KMeans model on existing RFM (for predicting new customer and consistent cluster naming)
        X = rfm_base[["Recency", "Frequency", "Monetary"]].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        # Derive cluster names by Monetary mean like other charts
        centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=["Recency", "Frequency", "Monetary"])  # inverse to get original units
        centroid_order = centroids.sort_values("Monetary", ascending=False).index.tolist()
        cluster_names_map = {centroid_order[0]: 'VIPs', centroid_order[1]: 'Regulars', centroid_order[2]: 'Potential Loyalists', centroid_order[3]: 'At-Risk'}
        
        def compute_rfm_scores_single(recency: float, frequency: float, monetary: float) -> dict:
            q_rec = rfm_base['Recency'].quantile([0.2,0.4,0.6,0.8]).values
            q_freq = rfm_base['Frequency'].quantile([0.2,0.4,0.6,0.8]).values
            q_mon = rfm_base['Monetary'].quantile([0.2,0.4,0.6,0.8]).values
            # Recency lower is better
            R = 5 - np.searchsorted(q_rec, recency, side='right')
            F = 1 + np.searchsorted(q_freq, frequency, side='right')
            M = 1 + np.searchsorted(q_mon, monetary, side='right')
            R = int(np.clip(R, 1, 5)); F = int(np.clip(F, 1, 5)); M = int(np.clip(M, 1, 5))
            return {"R": R, "F": F, "M": M, "RFM_Score": f"{R}{F}{M}"}
        
        if mode == "Existing customer":
            cust_ids = rfm_base['member_number'].astype(str).tolist()
            selected_id = st.selectbox("Customer ID", cust_ids)
            row = rfm_base[rfm_base['member_number'].astype(str) == selected_id].iloc[0]
            rec, freq, mon = float(row['Recency']), float(row['Frequency']), float(row['Monetary'])
            # Predict cluster via model for consistent naming
            pred = kmeans.predict(scaler.transform([[rec, freq, mon]]))[0]
            cluster_name = cluster_names_map.get(pred, str(pred))
            scores = compute_rfm_scores_single(rec, freq, mon)
        else:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                rec = st.number_input("Recency (days)", min_value=0.0, value=float(rfm_base['Recency'].median()))
            with col_b:
                freq = st.number_input("Frequency (orders)", min_value=0.0, value=float(rfm_base['Frequency'].median()))
            with col_c:
                mon = st.number_input("Monetary ($)", min_value=0.0, value=float(rfm_base['Monetary'].median()))
            pred = kmeans.predict(scaler.transform([[rec, freq, mon]]))[0]
            cluster_name = cluster_names_map.get(pred, str(pred))
            scores = compute_rfm_scores_single(rec, freq, mon)
        
        # Scores + values wrapped in a pretty card
        st.markdown(
            f"""
            <div class="ce-card ce-card--scores">
              <div class="ce-title">Customer Profile</div>
              <div class="ce-grid">
                <div class="ce-tile"><div class="value">{scores['R']}</div><div class="label">R score</div></div>
                <div class="ce-tile"><div class="value">{scores['F']}</div><div class="label">F score</div></div>
                <div class="ce-tile"><div class="value">{scores['M']}</div><div class="label">M score</div></div>
                <div class="ce-tile"><div class="value">{int(rec):,} days</div><div class="label">Recency</div></div>
                <div class="ce-tile"><div class="value">{int(freq):,}</div><div class="label">Frequency</div></div>
                <div class="ce-tile"><div class="value">${int(mon):,}</div><div class="label">Monetary</div></div>
                <div class="ce-tile ce-tile--accent" style="grid-column: span 3; text-align:center"><div class="value">{cluster_name}</div><div class="label">Cluster</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Purchase history and preferences (only for existing customers)
        if mode == "Existing customer":
            # Try to detect columns
            def ensure_amount_column(df):
                if 'amount' not in df.columns:
                    if 'items' in df.columns and 'price' in df.columns:
                        df['amount'] = df['items'] * df['price']
                return df

            cust_col = 'Member_number' if 'Member_number' in merged.columns else ('member_number' if 'member_number' in merged.columns else None)
            date_col = 'Date' if 'Date' in merged.columns else ('date' if 'date' in merged.columns else None)
            cust_df = merged.copy()
            if date_col:
                cust_df[date_col] = pd.to_datetime(cust_df[date_col], errors='coerce')
                cust_df = cust_df.dropna(subset=[date_col])
            if cust_col:
                cust_df = cust_df[cust_df[cust_col].astype(str) == str(selected_id)]
            cust_df = ensure_amount_column(cust_df)

            # Left panel: purchase history; Right panels: favorite categories and common products
            left, right1, right2 = st.columns([2,1,1])
            with left:
                if not cust_df.empty and date_col and 'amount' in cust_df.columns:
                    tmp = cust_df[[date_col, 'amount']].copy()
                    tmp['month'] = tmp[date_col].dt.to_period('M').dt.start_time
                    history = tmp.groupby('month', as_index=False)['amount'].sum().rename(columns={'amount':'revenue'})
                    history['label'] = history['month'].dt.strftime('%Y %b')
                    line = (
                        alt.Chart(history)
                        .mark_line(point=True, strokeWidth=3, color='#4C78A8')
                        .encode(
                            x=alt.X('label:N', sort=list(history['label'].tolist()), title='Month'),
                            y=alt.Y('revenue:Q', title='Revenue', axis=alt.Axis(format='$,.0f')),
                            tooltip=['label', alt.Tooltip('revenue:Q', format='$,.0f')]
                        )
                        .properties(width=540, height=340, title='Revenue over time')
                    )
                    chart_html = line.to_html()
                    # Inline styles so the title matches Preferred Categories and the chart stays inside the card
                    # Render only chart; container is styled via CSS selector
                    st.altair_chart(line, use_container_width=True)
                else:
                    st.info('No purchase history available for this customer.')
            
            # Detect category and product columns similar to EDA
            def pick_column(df, candidates):
                for c in candidates:
                    if c in df.columns:
                        return c
                return None
            cat_col = pick_column(merged, ["category", "Category", "product_category", "ProductCategory"]) 
            prod_col = pick_column(merged, [
                "productName", "ProductName", "product", "Product", "product_name", "item_name", "description", "Description", "sku"
            ])
            
            with right1:
                if not cust_df.empty and cat_col and 'amount' in cust_df.columns:
                    top_cats = cust_df.groupby(cat_col)['amount'].sum().sort_values(ascending=False).head(8)
                    items = "".join([f"<li>{name}</li>" for name, _ in top_cats.items()])
                    html_block = f"""
                    <div class='ce-card ce-card--cat ce-equal'>
                      <div class='ce-section-title cat'>Preferred Categories</div>
                      <ul class='ce-list'>{items}</ul>
                    </div>
                    """
                    st.markdown(html_block, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='ce-card ce-card--cat ce-equal'>
                      <div class='ce-section-title cat'>Preferred Categories</div>
                      <div>No category data.</div>
                    </div>
                    """, unsafe_allow_html=True)
            with right2:
                if not cust_df.empty and prod_col:
                    top_prods = cust_df[prod_col].value_counts().head(10)
                    items = "".join([f"<li>{name}</li>" for name, _ in top_prods.items()])
                    html_block = f"""
                    <div class='ce-card ce-card--prod ce-equal'>
                      <div class='ce-section-title prod'>Frequently Bought Products</div>
                      <ul class='ce-list'>{items}</ul>
                    </div>
                    """
                    st.markdown(html_block, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='ce-card ce-card--prod ce-equal'>
                      <div class='ce-section-title prod'>Frequently Bought Products</div>
                      <div>No product data.</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Removed scatter chart "Customer vs Population (Recency vs Monetary)" per request
    
    with tab_strategies:
        st.markdown("#### Strategies for Each Segment")
        st.markdown(
            """
            <style>
              .strat-grid { display:grid; grid-template-columns: repeat(2, minmax(240px, 1fr)); gap:12px; }
              .strat-card { border:1px solid #e5e7eb; border-radius:14px; padding:12px 14px; box-shadow:0 1px 2px rgba(16,24,40,.06); min-height: 180px; display:flex; flex-direction:column; overflow:auto; }
              .strat-title { font-weight:800; margin:0 0 8px; font-size:16px; color:#0f172a; }
              .strat-card ul { margin:8px 0 0 18px; padding:0; }
              .strat-card li { margin:4px 0; }
              .strat-card.vip { background:#fff7e6; border-color:#ffd591; }
              .strat-card.reg { background:#e6f4ff; border-color:#91caff; }
              .strat-card.pot { background:#ecfdf5; border-color:#86efac; }
              .strat-card.risk { background:#fff1f2; border-color:#fda4af; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        # Row 1: VIPs and Regulars
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                <div class="strat-card vip">
                  <div class="strat-title">VIPs & Loyal</div>
                  <ul>
                    <li>VIP/Premium membership with exclusive privileges.</li>
                    <li>Personalized incentives: special discounts, birthday vouchers, early access to new products.</li>
                    <li>Upsell/Cross‑sell and high‑value bundles.</li>
                    <li>Track: retention rate, ARPU, VIP program engagement.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="strat-card reg">
                  <div class="strat-title">Regulars</div>
                  <ul>
                    <li>Maintain purchase frequency with value bundles/combos.</li>
                    <li>Referral program plus loyalty points.</li>
                    <li>Emails suggesting substitutes and accessories.</li>
                    <li>Track: purchase frequency, AOV, retention rate.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        # Add vertical space between two rows
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        # Row 2: Potential Loyalists and At-Risk
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(
                """
                <div class="strat-card pot">
                  <div class="strat-title">Potential Loyalists</div>
                  <ul>
                    <li>Discount on the next order; small return gift.</li>
                    <li>Start the loyalty program from the first purchase to build habit.</li>
                    <li>Email/Remarketing: cart reminders and related product suggestions.</li>
                    <li>Track: conversion to Loyal and average return interval.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                """
                <div class="strat-card risk">
                  <div class="strat-title">At‑Risk</div>
                  <ul>
                    <li>Selective win‑back campaigns with strong vouchers.</li>
                    <li>Personalized offers based on history; direct outreach (SMS/Call/Zalo).</li>
                    <li>Survey churn reasons to improve product and service.</li>
                    <li>Track: return rate and recovered revenue.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Footer
    Footer.render()
    