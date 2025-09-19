# pages/bi_dashboard.py
import streamlit as st
import pandas as pd
import altair as alt
from config.settings import PAGE_CONFIG
from src.eda_utils import load_datasets, infer_join_keys, merge_datasets
from src.data_processing_eda import compute_rfm

def show():
    """Display the BI Dashboard page"""
    st.markdown(f"# {PAGE_CONFIG['bi_dashboard']['title']}")
    st.markdown(f"*{PAGE_CONFIG['bi_dashboard']['description']}*")
    st.markdown("---")
    
    # Load data
    try:
        transactions_file = getattr(st.session_state, "upload_transactions", None)
        products_file = getattr(st.session_state, "upload_products", None)
        df_tx, df_pd, _, _ = load_datasets(transactions_file, products_file)
    except Exception as e:
        st.error(f"Failed to load datasets: {e}")
        return

    # Build merged data for RFM
    left_key, right_key = infer_join_keys(df_tx, df_pd)
    try:
        merged = merge_datasets(df_tx, df_pd, left_key, right_key, "inner")
    except Exception as e:
        st.error(f"Merge failed: {e}")
        return

    # Validate columns
    date_col = "Date" if "Date" in merged.columns else ("date" if "date" in merged.columns else None)
    customer_col = "Member_number" if "Member_number" in merged.columns else ("member_number" if "member_number" in merged.columns else None)
    if not date_col or not customer_col or "amount" not in merged.columns:
        st.warning("Required columns for dashboard not found.")
        return

    merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce")
    merged = merged.dropna(subset=[date_col, customer_col])

    # Build RFM data
    merged_rfm = merged.copy()
    merged_rfm = merged_rfm.rename(columns={
        customer_col: 'member_number', 
        date_col: 'date'
    })
    
    if 'items' not in merged_rfm.columns:
        items_col = None
        for col in merged_rfm.columns:
            if 'item' in col.lower():
                items_col = col
                break
        if items_col:
            merged_rfm['items'] = merged_rfm[items_col]
        else:
            merged_rfm['items'] = 1
    
    rfm = compute_rfm(
        merged_rfm,
        customer_col='member_number',
        date_col='date',
        amount_col='amount'
    )
    rfm = rfm.set_index('member_number')

    # RFM scoring and segmentation
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
    rfm[['R_Score','F_Score','M_Score']] = rfm[['R_Score','F_Score','M_Score']].astype(int)

    # Customer segmentation
    def segment_customers(row):
        if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
            return 'A.CHAMPIONS'
        elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
            return 'B.LOYAL'
        elif row['R_Score'] >= 4 and row['F_Score'] <= 2:
            return 'C.POTENTIAL_LOYALIST'
        elif row['R_Score'] >= 3 and row['F_Score'] >= 2 and row['M_Score'] >= 2:
            return 'D.RECENT_CUSTOMERS'
        elif row['R_Score'] >= 2 and row['F_Score'] >= 2 and row['M_Score'] >= 2:
            return 'F.NEED_ATTENTION'
        elif row['R_Score'] <= 2 and row['F_Score'] >= 2 and row['M_Score'] >= 2:
            return 'H.AT_RISK'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] >= 2:
            return 'I.CANNOT_LOSE'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
            return 'K.LOST'
        else:
            return 'J.HIBERNATING'

    rfm['Segment'] = rfm.apply(segment_customers, axis=1)
    
    # Key Performance Indicators Section (including RFM)
    st.markdown("### Key Performance Indicators")
    
    # Calculate KPIs
    total_users = len(rfm)
    total_transactions = len(merged)
    total_revenue = merged['amount'].sum()
    
    # Calculate user types using RFM-based logic
    active_users = len(rfm[(rfm['Recency'] <= 30) & (rfm['Frequency'] >= rfm['Frequency'].quantile(0.5)) & (rfm['Monetary'] >= rfm['Monetary'].quantile(0.5))])
    at_risk_users = len(rfm[(rfm['Recency'] > 30) & (rfm['Recency'] <= 90) & (rfm['Frequency'] >= rfm['Frequency'].quantile(0.3)) & (rfm['Monetary'] >= rfm['Monetary'].quantile(0.3))])
    churned_users = len(rfm[(rfm['Recency'] > 90) | (rfm['Frequency'] < rfm['Frequency'].quantile(0.3)) | (rfm['Monetary'] < rfm['Monetary'].quantile(0.3))])
    
    pct_active = (active_users / total_users) * 100
    pct_at_risk = (at_risk_users / total_users) * 100
    pct_churned = (churned_users / total_users) * 100
    
    # Calculate RFM averages
    avg_recency = rfm['Recency'].mean()
    avg_frequency = rfm['Frequency'].mean()
    avg_monetary = rfm['Monetary'].mean()
    
    # Combined KPI Cards styling with tooltips
    st.markdown("""
    <style>
    .kpi-card {
        background-color: #ffffff;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        margin: 8px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        position: relative;
        cursor: help;
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        border-color: #1f77b4;
    }
    .kpi-title {
        font-size: 14px;
        color: #6c757d;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 28px;
        color: #212529;
        font-weight: 700;
        margin: 0;
    }
    .kpi-tooltip {
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #333;
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 12px;
        white-space: nowrap;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
        max-width: 300px;
        white-space: normal;
        text-align: center;
    }
    .kpi-card:hover .kpi-tooltip {
        opacity: 1;
        visibility: visible;
    }
    .rfm-card {
        background-color: #ffffff;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        margin: 8px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        position: relative;
        cursor: help;
        transition: all 0.3s ease;
    }
    .rfm-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        border-color: #1f77b4;
    }
    .rfm-letter {
        font-size: 32px;
        font-weight: 900;
        color: #1f77b4;
        margin-bottom: 8px;
    }
    .rfm-title {
        font-size: 12px;
        color: #6c757d;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .rfm-value {
        font-size: 24px;
        color: #212529;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .rfm-link {
        color: #1f77b4;
        text-decoration: none;
        font-weight: 600;
        font-size: 11px;
    }
    .rfm-link:hover {
        text-decoration: underline;
    }
    .rfm-tooltip {
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #333;
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 12px;
        white-space: nowrap;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
        max-width: 300px;
        white-space: normal;
        text-align: center;
    }
    .rfm-card:hover .rfm-tooltip {
        opacity: 1;
        visibility: visible;
    }
    .chart-container {
        background-color: white;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        padding: 15px;
        margin: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # First row: Total metrics (3 columns)
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-tooltip">Total number of unique customers in the dataset</div>
            <div class="kpi-title">Total Users</div>
            <div class="kpi-value">{total_users:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-tooltip">Total number of transactions across all customers</div>
            <div class="kpi-title">Total Transactions</div>
            <div class="kpi-value">{total_transactions:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-tooltip">Sum of all transaction amounts (items × price)</div>
            <div class="kpi-title">Total Net Revenue</div>
            <div class="kpi-value">${total_revenue:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row: Percentage metrics (3 columns)
    kpi4, kpi5, kpi6 = st.columns(3)
    
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-tooltip">Active Users: Recent (≤30 days) + High Frequency (≥50th percentile) + High Monetary (≥50th percentile)</div>
            <div class="kpi-title">% Users Active</div>
            <div class="kpi-value">{pct_active:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-tooltip">At Risk Users: Medium Recency (31-90 days) + Medium Frequency (≥30th percentile) + Medium Monetary (≥30th percentile)</div>
            <div class="kpi-title">% Users At Risk</div>
            <div class="kpi-value">{pct_at_risk:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-tooltip">Churned Users: Old (>90 days) OR Low Frequency (<30th percentile) OR Low Monetary (<30th percentile)</div>
            <div class="kpi-title">% Users Churned</div>
            <div class="kpi-value">{pct_churned:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Third row: RFM KPIs (3 columns on one line)
    st.markdown("---")
    rfm1, rfm2, rfm3 = st.columns(3)
    
    with rfm1:
        st.markdown(f"""
        <div class="rfm-card">
            <div class="rfm-tooltip">Recency: Average number of days since each customer's last transaction</div>
            <div class="rfm-letter">R</div>
            <div class="rfm-title">AVG Days Since Last Transaction</div>
            <div class="rfm-value">{avg_recency:.0f}</div>
            <div class="rfm-link">EXPLORE RECENCY</div>
        </div>
        """, unsafe_allow_html=True)
    
    with rfm2:
        st.markdown(f"""
        <div class="rfm-card">
            <div class="rfm-tooltip">Frequency: Average number of transactions per customer</div>
            <div class="rfm-letter">F</div>
            <div class="rfm-title">AVG Transactions per User</div>
            <div class="rfm-value">{avg_frequency:.1f}</div>
            <div class="rfm-link">EXPLORE FREQUENCY</div>
        </div>
        """, unsafe_allow_html=True)
    
    with rfm3:
        st.markdown(f"""
        <div class="rfm-card">
            <div class="rfm-tooltip">Monetary: Average total revenue per customer (sum of all their transactions)</div>
            <div class="rfm-letter">M</div>
            <div class="rfm-title">AVG Net Revenue Per User</div>
            <div class="rfm-value">${avg_monetary:.0f}</div>
            <div class="rfm-link">EXPLORE MONETARY</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Apply K-Means clustering with 4 clusters
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Prepare RFM data for clustering
    rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Standardize features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    
    # Apply K-Means with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Define colors for clusters
    cluster_colors = {
        0: '#1f77b4',  # Blue
        1: '#ff7f0e',  # Orange
        2: '#2ca02c',  # Green
        3: '#d62728'   # Red
    }
    
    # Create cluster labels based on characteristics
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean', 
        'Monetary': 'mean'
    }).round(2)
    
    # Add count of customers per cluster
    cluster_summary['Customer_Count'] = rfm.groupby('Cluster').size()
    
    # Assign meaningful names to clusters based on RFM characteristics
    cluster_summary_sorted = cluster_summary.sort_values('Monetary', ascending=False)
    
    cluster_names = {}
    cluster_names[cluster_summary_sorted.index[0]] = 'VIPs'
    cluster_names[cluster_summary_sorted.index[1]] = 'Regulars'
    cluster_names[cluster_summary_sorted.index[2]] = 'Potential Loyalists'
    cluster_names[cluster_summary_sorted.index[3]] = 'At-Risk'
    
    # Map cluster names
    rfm['Cluster_Name'] = rfm['Cluster'].map(cluster_names)
    
    # =========================
    # TWO CHARTS ON SAME LINE - ALIGNED LAYOUT
    # =========================
    
    if 'Cluster_Name' not in rfm.columns:
        if 'Cluster' not in rfm.columns:
            st.warning("No clustering found. Please run KMeans to create 'Cluster' first.")
        else:
            rfm = rfm.copy()
            rfm['Cluster_Name'] = rfm['Cluster'].apply(lambda x: f'Cluster {x}')

    # Define 4 main clusters (common for both charts)
    main_clusters = ['VIPs', 'Regulars', 'Potential Loyalists', 'At-Risk']
    palette = ['#4C78A8', '#F58518', '#54A24B', '#E45756']

    # Data for bubble chart
    bubble_data = (
        rfm.groupby(['Cluster', 'Cluster_Name'])
           .agg(Recency=('Recency', 'mean'),
                Total_Revenue=('Monetary', 'sum'),
                Avg_Revenue=('Monetary', 'mean'),
                User_Count=('Recency', 'size'),
                Avg_Frequency=('Frequency', 'mean'),
                Total_Transactions=('Frequency', 'sum'))
           .reset_index()
    )
    bubble_data = bubble_data[bubble_data['Cluster_Name'].isin(main_clusters)]
    
    # Create two columns for side-by-side charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Recency and Monetary of each Cluster")
        
        # Bubble Chart - Show aggregated cluster data as bubbles
        bubble_chart = (
            alt.Chart(bubble_data)
            .mark_circle(opacity=0.8, stroke='#ffffff', strokeWidth=2)
            .encode(
                x=alt.X('Recency:Q',
                        title='Days Since Last Transaction',
                        scale=alt.Scale(domain=[0, float(bubble_data['Recency'].max()*1.1)])),
                y=alt.Y('Total_Revenue:Q',
                        title='Total Revenue',
                        scale=alt.Scale(domain=[0, float(bubble_data['Total_Revenue'].max()*1.1)])),
                size=alt.Size('Total_Revenue:Q', 
                             legend=alt.Legend(
                                 title='Total Revenue ($)',
                                 titleFontSize=12,
                                 labelFontSize=10,
                                 symbolSize=100,
                                 orient='right'
                             ), 
                             scale=alt.Scale(range=[200, 800])),
                color=alt.Color('Cluster_Name:N',
                                title='Cluster',
                                scale=alt.Scale(domain=main_clusters, range=palette),
                                legend=alt.Legend(
                                    titleFontSize=12, 
                                    labelFontSize=11, 
                                    symbolSize=140,
                                    orient='right'
                                )),
                tooltip=[
                    alt.Tooltip('Cluster_Name:N', title='Cluster'),
                    alt.Tooltip('Cluster:Q', title='Cluster ID'),
                    alt.Tooltip('User_Count:Q', title='Total Users', format=','),
                    alt.Tooltip('Recency:Q', title='Avg Recency (days)', format=',.0f'),
                    alt.Tooltip('Total_Revenue:Q', title='Total Revenue ($)', format='$,.0f'),
                    alt.Tooltip('Avg_Revenue:Q', title='Avg Revenue per User ($)', format='$,.0f'),
                    alt.Tooltip('Avg_Frequency:Q', title='Avg Frequency (orders)', format=',.1f'),
                    alt.Tooltip('Total_Transactions:Q', title='Total Transactions', format=','),
                ],
            )
            .properties(
                width=380,  # Adjusted for better fit
                height=350,
                title=alt.TitleParams(
                    text='Clusters: Recency vs Total Revenue',
                    anchor='start', 
                    fontSize=14, 
                    fontWeight='bold'
                )
            )
            .configure_view(
                strokeWidth=0,
                fill='#ffffff'
            )
            .configure_axis(
                grid=True,
                gridOpacity=0.3,
                domainColor='#1f1f1f',
                tickColor='#1f1f1f',
                labelColor='#2b2b2b',
                titleColor='#2b2b2b'
            )
        )
        
        st.altair_chart(bubble_chart, use_container_width=True)
    
    with col2:
        st.markdown("### Customer Segmentation: RFM Analysis")
        
        # Scatter Plot - Show individual customer data points
        sample_rfm = rfm.sample(min(1000, len(rfm))).reset_index()
        sample_rfm = sample_rfm[sample_rfm['Cluster_Name'].isin(main_clusters)]
        
        # Add jitter to prevent overlapping points
        import numpy as np
        sample_rfm['Recency_Jitter'] = sample_rfm['Recency'] + np.random.normal(0, 0.1, len(sample_rfm))
        sample_rfm['Monetary_Jitter'] = sample_rfm['Monetary'] + np.random.normal(0, 0.1, len(sample_rfm))
        
        scatter_chart = (
            alt.Chart(sample_rfm)
            .mark_circle(opacity=0.6, stroke='#ffffff', strokeWidth=0.5)
            .encode(
                x=alt.X('Recency_Jitter:Q',
                        title='Recency (Days)',
                        scale=alt.Scale(type='linear', domain=[0, sample_rfm['Recency'].max() * 1.1])),
                y=alt.Y('Monetary_Jitter:Q',
                        title='Monetary Value ($)',
                        scale=alt.Scale(type='linear', domain=[0, sample_rfm['Monetary'].max() * 1.1])),
                color=alt.Color('Cluster_Name:N',
                                title='Cluster',
                                scale=alt.Scale(domain=main_clusters, range=palette),
                                legend=alt.Legend(
                                    titleFontSize=12, 
                                    labelFontSize=11, 
                                    symbolSize=140,
                                    orient='right'
                                )),
                size=alt.value(40),  # Smaller size for individual points
                tooltip=[
                    alt.Tooltip('member_number:N', title='Customer ID'),
                    alt.Tooltip('Cluster_Name:N', title='Cluster'),
                    alt.Tooltip('Recency:Q', title='Recency (days)', format=',.0f'),
                    alt.Tooltip('Frequency:Q', title='Frequency', format=',.0f'),
                    alt.Tooltip('Monetary:Q', title='Monetary ($)', format='$,.2f')
                ]
            )
            .properties(
                width=380,  # Same width as bubble chart
                height=350,  # Same height as bubble chart
                title=alt.TitleParams(
                    text='Recency vs Monetary by Cluster',
                    anchor='start',
                    fontSize=14,
                    fontWeight='bold'
                )
            )
            .configure_view(
                strokeWidth=0,
                fill='#ffffff'
            )
            .configure_axis(
                grid=True,
                gridOpacity=0.3,
                domainColor='#1f1f1f',
                tickColor='#1f1f1f',
                labelColor='#2b2b2b',
                titleColor='#2b2b2b'
            )
        )
        
        st.altair_chart(scatter_chart, use_container_width=True)
    
    # =========================
    # KPI TABLE
    # =========================
    st.markdown("#### Recency, Frequency and Monetary KPIs per Segment")

    # Calculate KPIs by Segment
    seg = (
        rfm.groupby('Cluster_Name')
           .agg(Total_Users=('Recency', 'size'),
                Avg_Recency=('Recency', 'mean'),
                Avg_Frequency=('Frequency', 'mean'),
                Avg_Monetary=('Monetary', 'mean'))
           .reset_index()
    )

    tot_users = seg['Total_Users'].sum()
    tot_revenue = (seg['Total_Users'] * seg['Avg_Monetary']).sum()
    tot_tx = (seg['Total_Users'] * seg['Avg_Frequency']).sum()

    seg['Pct_Users'] = seg['Total_Users'] / tot_users * 100
    seg['Pct_Revenue'] = (seg['Total_Users']*seg['Avg_Monetary']) / tot_revenue * 100
    seg['Pct_Transactions'] = (seg['Total_Users']*seg['Avg_Frequency']) / tot_tx * 100

    # Sort by Cluster_Name
    seg = seg.sort_values('Cluster_Name')

    # Create table with emoji bars as visual representation
    def create_table_with_emoji_bars(df):
        """Create table with emoji bars for visual representation"""
        df_display = df.copy()
        
        # Add emoji bars for each metric
        for col in ['Pct_Users', 'Pct_Revenue', 'Pct_Transactions', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']:
            if col in df.columns:
                max_val = df[col].max()
                df_display[f'{col}_bar'] = df[col].apply(
                    lambda x: '█' * int((x / max_val) * 10) + '░' * (10 - int((x / max_val) * 10))
                )
        
        return df_display
    
    # Format percentage columns to 2 decimal places BEFORE creating emoji bars
    seg_formatted = seg.copy()
    seg_formatted['Pct_Users'] = seg_formatted['Pct_Users'].round(2)
    seg_formatted['Pct_Revenue'] = seg_formatted['Pct_Revenue'].round(2)
    seg_formatted['Pct_Transactions'] = seg_formatted['Pct_Transactions'].round(2)
    
    # Create table with emoji bars
    display_seg = create_table_with_emoji_bars(
        seg_formatted[['Cluster_Name', 'Total_Users', 'Pct_Users', 'Pct_Revenue', 'Pct_Transactions', 
                      'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']]
    )
    
    # Add CSS for table header styling - place AFTER dataframe creation
    st.markdown("""
    <style>
    .stDataFrame table thead tr th {
        background-color: #1f77b4 !important;
        color: white !important;
        font-weight: bold !important;
    }
    .stDataFrame table thead tr th:first-child {
        background-color: #1f77b4 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display with custom column names
    st.dataframe(
        display_seg[['Cluster_Name', 'Total_Users', 'Pct_Users', 'Pct_Users_bar', 
                    'Pct_Revenue', 'Pct_Revenue_bar', 'Pct_Transactions', 'Pct_Transactions_bar',
                    'Avg_Recency', 'Avg_Recency_bar', 'Avg_Frequency', 'Avg_Frequency_bar',
                    'Avg_Monetary', 'Avg_Monetary_bar']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'Cluster_Name': 'Cluster Name',
            'Total_Users': 'Total Users',
            'Pct_Users': '% Users',
            'Pct_Users_bar': 'Users Bar',
            'Pct_Revenue': '% Revenue', 
            'Pct_Revenue_bar': 'Revenue Bar',
            'Pct_Transactions': '% Transactions',
            'Pct_Transactions_bar': 'Transactions Bar',
            'Avg_Recency': 'Avg Recency',
            'Avg_Recency_bar': 'Recency Bar',
            'Avg_Frequency': 'Avg Frequency',
            'Avg_Frequency_bar': 'Frequency Bar',
            'Avg_Monetary': 'Avg Monetary',
            'Avg_Monetary_bar': 'Monetary Bar'
        }
    )