# pages/eda.py
import streamlit as st
import altair as alt
from config.settings import PAGE_CONFIG
import pandas as pd
from pathlib import Path
from src.customer_segmentation_service import CustomerSegmentationService
from components import KPICards, ChartComponents, TableComponents, FormComponents, Footer

def ensure_amount_column(df):
    """Ensure amount column exists by calculating from items * price if needed."""
    if 'amount' not in df.columns:
        if 'items' in df.columns and 'price' in df.columns:
            df['amount'] = df['items'] * df['price']
        else:
            raise ValueError("Cannot calculate amount: missing 'amount' or 'items'/'price' columns")
    return df

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
    
    # Initialize service and load data
    try:
        service = CustomerSegmentationService()
        data_prep = service.load_and_prepare_data(transactions_file, products_file)
        if data_prep['status'] != 'success':
            st.error(f"Data preparation failed: {data_prep.get('error', 'Unknown error')}")
            return
        
        df_transactions = data_prep['transactions_df']
        df_products = data_prep['products_df']
        src_tx = data_prep['sources']['transactions']
        src_pd = data_prep['sources']['products']
        
    except Exception as e:
        st.error(f"Service initialization failed: {e}")
        return
    
    # Status messages with file source info
    if transactions_file is None and products_file is None:
        st.info("📁 Using default files from data/raw/")
        st.success(f"✅ Data loaded: {len(df_transactions):,} transactions • {len(df_products):,} products")
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
        st.success(f"✅ Data loaded: {len(df_transactions):,} transactions • {len(df_products):,} products")
    # Light CSS for nicer tables/sections
    st.markdown("""
    <style>
      [data-testid="stDataFrame"] { border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,.04); }
      .stat-badges { display:flex; gap:8px; margin: 6px 0 8px; flex-wrap: wrap; }
      .stat { background:#f7fbff; border:1px solid #e3f0ff; color:#1f77b4; padding:6px 10px; border-radius:9999px; font-weight:600; font-size:12px; }
      /* KPI cards */
      .kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 10px; }
      .kpi { background: linear-gradient(180deg, #ffffff, #f9fbff); border: 1px solid #e6effa; border-radius: 14px; padding: 14px 16px; box-shadow: 0 8px 22px rgba(31,119,180,.08); }
      .kpi .label { font-size: 12px; color: #356ea8; font-weight: 700; letter-spacing: .2px; }
      .kpi .value { margin-top: 6px; font-size: 26px; font-weight: 800; color: #123b63; }
      @media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
      @media (max-width: 600px) { .kpi-grid { grid-template-columns: 1fr; } }
    </style>
    """, unsafe_allow_html=True)

    # Tabs for EDA workflow
    tab_overview, tab_merge, tab_revenue, tab_category, tab_rfm = st.tabs(["Overview Data", "Merge Datasets", "Revenue Trend", "Category/Product Analysis", "RFM Analysis"])
    
    with tab_overview:
        # Transactions block
        TableComponents.render_data_preview(df_transactions, "Transactions", 10)
        st.markdown("---")

        # Products block
        TableComponents.render_data_preview(df_products, "Products", 10)
            
    
    with tab_merge:
        st.markdown("#### Merge Transactions + Products")
        # Use service for merge and analysis
        try:
            merged = data_prep['merged_df']
            merged_rfm = data_prep['merged_rfm_df']
            
            # Perform RFM analysis
            rfm_analysis = service.perform_rfm_analysis(merged_rfm)
            if rfm_analysis['status'] != 'success':
                st.error(f"RFM analysis failed: {rfm_analysis.get('error', 'Unknown error')}")
                return
            
            rfm = rfm_analysis['rfm_df']
            
            # Calculate KPIs using service
            kpis = service.calculate_kpis(merged, rfm)
            if kpis['status'] != 'success':
                st.error(f"KPI calculation failed: {kpis.get('error', 'Unknown error')}")
                return
            
            kpi_data = kpis['kpi_data']
            
            # Create KPI display using component
            KPICards.render_basic_kpi_cards(kpi_data, columns=3)
            
            # Then meta info and preview
            st.info(f"Merged shape: {merged.shape[0]:,} rows × {merged.shape[1]:,} columns")
            st.dataframe(merged.head(20), width='stretch')
        except Exception as e:
            st.error(f"Merge failed: {e}")
        
    with tab_revenue:
        st.markdown("#### Revenue Trend")
        # Use core service for merge
        try:
            merged_tr = data_prep['merged_df']
        except Exception as e:
            st.error(f"Data access failed: {e}")
            return

        # Date range slicer
        date_col = "Date" if "Date" in merged_tr.columns else "date"
        merged_tr[date_col] = pd.to_datetime(merged_tr[date_col], errors="coerce")
        merged_tr = merged_tr.dropna(subset=[date_col])
        
        # Use form components for date range and granularity
        start_date, end_date = FormComponents.render_date_range_selector(merged_tr, date_col)
        if start_date and end_date:
            # Filter data by date range
            merged_tr = merged_tr[
                (merged_tr[date_col].dt.date >= start_date.date()) & 
                (merged_tr[date_col].dt.date <= end_date.date())
            ]

        granularity = FormComponents.render_granularity_selector()
        
        # Calculate revenue trends using service
        trend = service.eda_core.calculate_revenue_trends(merged_tr, granularity)
        
        # Create chart using component
        ChartComponents.render_revenue_trend_chart(trend, "line")

        st.markdown("---")

        # Monthly total vs top 10 categories (always shown)
        cat_col = None
        for c in ["category", "Category", "product_category", "ProductCategory"]:
            if c in merged_tr.columns:
                cat_col = c
                break
        if cat_col is not None:
            # Ensure amount column exists
            merged_tr = ensure_amount_column(merged_tr)
            monthly = merged_tr[[date_col, cat_col, "amount"]].copy()
            monthly[date_col] = pd.to_datetime(monthly[date_col], errors="coerce")
            monthly = monthly.dropna(subset=[date_col])
            monthly["month"] = monthly[date_col].dt.to_period("M").dt.start_time

            # Total revenue per month
            total_rev = monthly.groupby("month", as_index=False)["amount"].sum()
            total_rev[cat_col] = "Total"

            # Top N categories across whole period
            top_cats = (
                monthly.groupby(cat_col)["amount"].sum().sort_values(ascending=False).head(10).index
            )
            top_df = monthly[monthly[cat_col].isin(top_cats)]
            top_df = top_df.groupby(["month", cat_col], as_index=False)["amount"].sum()

            combined = pd.concat([
                top_df.rename(columns={"amount": "revenue"}),
                total_rev.rename(columns={"amount": "revenue"}),
            ], ignore_index=True)

            combined["label"] = combined["month"].dt.strftime("%Y %b")

            chart2 = (
                alt.Chart(combined)
                .mark_line(point=False)
                .encode(
                    x=alt.X("label:N", sort=list(combined["label"].unique()), title="Month"),
                    y=alt.Y("revenue:Q", title="Revenue"),
                    color=alt.Color(f"{cat_col}:N", title="Category"),
                    tooltip=["label", alt.Tooltip("revenue:Q", format=",.0f"), alt.Tooltip(f"{cat_col}:N")],
                )
                .properties(title="Monthly Revenue: Total vs Top 10 Categories")
            )
            st.altair_chart(chart2, use_container_width=True)

        # Divider before growth charts
        st.markdown("---")
        st.markdown("#### MoM and YoY Analysis")

        # Prepare monthly total revenue for growth calculations
        merged_tr = ensure_amount_column(merged_tr)
        monthly_total = (
            merged_tr[[date_col, "amount"]]
            .assign(month=lambda df: pd.to_datetime(df[date_col]).dt.to_period("M").dt.start_time)
            .groupby("month", as_index=False)["amount"].sum()
            .rename(columns={"amount": "revenue"})
            .sort_values("month")
        )

        if len(monthly_total) < 2:
            st.info("Not enough monthly data to compute growth metrics.")
        else:
            monthly_total["MoM"] = monthly_total["revenue"].pct_change() * 100.0
            monthly_total["YoY"] = monthly_total["revenue"].pct_change(12) * 100.0
            monthly_total["label"] = monthly_total["month"].dt.strftime("%Y %b")

            # MoM Chart: Bar (Revenue) + Line (Growth %)
            mom_data = monthly_total.dropna(subset=["MoM"])
            if not mom_data.empty:
                # Revenue bars
                mom_bars = (
                    alt.Chart(mom_data)
                    .mark_bar(opacity=0.7)
                    .encode(
                        x=alt.X("label:N", sort=list(mom_data["label"].tolist()), title="Month"),
                        y=alt.Y("revenue:Q", title="Revenue", axis=alt.Axis(format=",.0f")),
                        tooltip=["label", alt.Tooltip("revenue:Q", format=",.0f")],
                        color=alt.value("#1f77b4")
                    )
                )
                
                # Growth line
                mom_line = (
                    alt.Chart(mom_data)
                    .mark_line(point=True, strokeWidth=3)
                    .encode(
                        x=alt.X("label:N", sort=list(mom_data["label"].tolist()), title="Month"),
                        y=alt.Y("MoM:Q", title="MoM Growth (%)", axis=alt.Axis(format=".1f")),
                        tooltip=["label", alt.Tooltip("MoM:Q", format=".1f")],
                        color=alt.value("#ff7f0e")
                    )
                )
                
                # Combine charts with dual y-axis
                mom_chart = alt.layer(mom_bars, mom_line).resolve_scale(
                    y='independent'
                ).properties(title="MoM: Revenue (Bar) vs Growth % (Line)")
                
                st.altair_chart(mom_chart, use_container_width=True)

            # YoY Chart: Bar (Revenue) + Line (Growth %)
            yoy_data = monthly_total.dropna(subset=["YoY"])
            if not yoy_data.empty:
                # Revenue bars
                yoy_bars = (
                    alt.Chart(yoy_data)
                    .mark_bar(opacity=0.7)
                    .encode(
                        x=alt.X("label:N", sort=list(yoy_data["label"].tolist()), title="Month"),
                        y=alt.Y("revenue:Q", title="Revenue", axis=alt.Axis(format=",.0f")),
                        tooltip=["label", alt.Tooltip("revenue:Q", format=",.0f")],
                        color=alt.value("#2ca02c")
                    )
                )
                
                # Growth line
                yoy_line = (
                    alt.Chart(yoy_data)
                    .mark_line(point=True, strokeWidth=3)
                    .encode(
                        x=alt.X("label:N", sort=list(yoy_data["label"].tolist()), title="Month"),
                        y=alt.Y("YoY:Q", title="YoY Growth (%)", axis=alt.Axis(format=".1f")),
                        tooltip=["label", alt.Tooltip("YoY:Q", format=".1f")],
                        color=alt.value("#d62728")
                    )
                )
                
                # Combine charts with dual y-axis
                yoy_chart = alt.layer(yoy_bars, yoy_line).resolve_scale(
                    y='independent'
                ).properties(title="YoY: Revenue (Bar) vs Growth % (Line)")
                
                st.altair_chart(yoy_chart, use_container_width=True)

        # Category Contribution Chart
        st.markdown("---")
        st.markdown("#### Category Contribution (Revenue Share %)")
        
        # Prepare category revenue data
        cat_col = None
        for c in ["category", "Category", "product_category", "ProductCategory"]:
            if c in merged_tr.columns:
                cat_col = c
                break
        
        if cat_col is not None:
            # Get monthly category revenue
            merged_tr = ensure_amount_column(merged_tr)
            monthly_cat = merged_tr[[date_col, cat_col, "amount"]].copy()
            monthly_cat[date_col] = pd.to_datetime(monthly_cat[date_col], errors="coerce")
            monthly_cat = monthly_cat.dropna(subset=[date_col])
            monthly_cat["month"] = monthly_cat[date_col].dt.to_period("M").dt.start_time
            
            # Calculate monthly revenue by category
            cat_revenue = monthly_cat.groupby(["month", cat_col], as_index=False)["amount"].sum()
            cat_revenue["label"] = cat_revenue["month"].dt.strftime("%Y %b")
            
            # Calculate total revenue per month for percentage calculation
            monthly_totals = cat_revenue.groupby("month", as_index=False)["amount"].sum().rename(columns={"amount": "total_revenue"})
            cat_revenue = cat_revenue.merge(monthly_totals, on="month")
            cat_revenue["revenue_share"] = (cat_revenue["amount"] / cat_revenue["total_revenue"]) * 100
            
            # Get top 10 categories by total revenue
            top_cats = (
                cat_revenue.groupby(cat_col)["amount"].sum()
                .sort_values(ascending=False).head(10).index
            )
            
            # Filter to top categories and "Others"
            cat_revenue_top = cat_revenue[cat_revenue[cat_col].isin(top_cats)].copy()
            
            # Calculate "Others" category
            others_data = []
            for month in cat_revenue["month"].unique():
                month_data = cat_revenue[cat_revenue["month"] == month]
                top_cats_month = month_data[month_data[cat_col].isin(top_cats)]
                others_revenue = month_data["amount"].sum() - top_cats_month["amount"].sum()
                others_share = (others_revenue / month_data["total_revenue"].iloc[0]) * 100
                
                others_data.append({
                    "month": month,
                    cat_col: "Others",
                    "amount": others_revenue,
                    "label": month_data["label"].iloc[0],
                    "total_revenue": month_data["total_revenue"].iloc[0],
                    "revenue_share": others_share
                })
            
            if others_data:
                others_df = pd.DataFrame(others_data)
                cat_revenue_final = pd.concat([cat_revenue_top, others_df], ignore_index=True)
            else:
                cat_revenue_final = cat_revenue_top
            
            # Create stacked area chart
            chart_type = st.selectbox("Chart Type", ["100% Stacked Bar Chart", "Stacked Area Chart"], index=0)
            
            if chart_type == "Stacked Area Chart":
                # Stacked Area Chart
                area_chart = (
                    alt.Chart(cat_revenue_final)
                    .mark_area()
                    .encode(
                        x=alt.X("label:N", sort=list(cat_revenue_final["label"].unique()), title="Month"),
                        y=alt.Y("amount:Q", title="Revenue", axis=alt.Axis(format=",.0f")),
                        color=alt.Color(f"{cat_col}:N", title="Category"),
                        tooltip=["label", alt.Tooltip("amount:Q", format=",.0f"), f"{cat_col}:N", alt.Tooltip("revenue_share:Q", format=".1f", title="Share %")],
                    )
                    .properties(title="Category Revenue Contribution (Stacked Area)")
                )
                st.altair_chart(area_chart, use_container_width=True)
            
            else:
                # 100% Stacked Bar Chart
                bar_chart = (
                    alt.Chart(cat_revenue_final)
                    .mark_bar()
                    .encode(
                        x=alt.X("label:N", sort=list(cat_revenue_final["label"].unique()), title="Month"),
                        y=alt.Y("revenue_share:Q", title="Revenue Share (%)", axis=alt.Axis(format=".1f")),
                        color=alt.Color(f"{cat_col}:N", title="Category"),
                        tooltip=["label", alt.Tooltip("revenue_share:Q", format=".1f", title="Share %"), f"{cat_col}:N", alt.Tooltip("amount:Q", format=",.0f", title="Revenue")],
                    )
                    .properties(title="Category Revenue Share % (100% Stacked Bar)")
                )
                st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.warning("No category column found to analyze category contribution.")

    with tab_category:
        st.markdown("#### Category/Product Analysis")
        
        # Use merged data for analysis
        try:
            merged_cat = data_prep['merged_df']
        except Exception as e:
            st.error(f"Data access failed: {e}")
            return

        # Ensure datetime and amount
        date_col = "Date" if "Date" in merged_cat.columns else ("date" if "date" in merged_cat.columns else None)
        if date_col is None:
            st.warning("No date column found for analysis.")
            return
        merged_cat[date_col] = pd.to_datetime(merged_cat[date_col], errors="coerce")
        merged_cat = merged_cat.dropna(subset=[date_col])
        try:
            merged_cat = ensure_amount_column(merged_cat)
        except ValueError as e:
            st.warning(f"Cannot analyze categories: {e}")
            return

        # Apply date filter if available
        if 'start_date' in locals() and 'end_date' in locals():
            merged_cat = merged_cat[
                (merged_cat[date_col].dt.date >= start_date.date()) & 
                (merged_cat[date_col].dt.date <= end_date.date())
            ]

        # Find category and product columns (robust auto-detect)
        import re
        def normalize(col: str) -> str:
            return re.sub(r"(_x|_y)$", "", col.lower())

        columns_norm = {col: normalize(col) for col in merged_cat.columns}

        def pick_column(prioritized_bases):
            # Exact base match by priority, considering _x/_y suffixes and case
            for base in prioritized_bases:
                base_l = base.lower()
                for original, norm in columns_norm.items():
                    if norm == base_l:
                        return original
            return None

        cat_col = pick_column([
            "Category", "category", "product_category", "ProductCategory", "cat"
        ])

        product_col = pick_column([
            "productName", "ProductName", "product_name", "product", "Product",
            "item_name", "item", "Item", "description", "Description", "productdesc", "ProductDesc", "sku", "code"
        ])

        if cat_col is None and product_col is None:
            st.warning("No category or product column found for analysis.")
            return

        # Analysis type selector
        analysis_type = FormComponents.render_analysis_type_selector()

        if analysis_type == "Category Analysis" and cat_col is not None:
            # Top categories by total revenue
            cat_revenue = merged_cat.groupby(cat_col, as_index=False)["amount"].sum().sort_values("amount", ascending=False)
            
            # Slider for number of top categories
            max_cats = min(len(cat_revenue), 20)  # Limit to available categories or 20
            num_cats = FormComponents.render_top_n_selector(max_cats, 10, "Number of Top Categories to Show")
            top_cats = cat_revenue.head(num_cats)

            # Category statistics FIRST
            st.markdown("##### Category Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Categories", len(cat_revenue))
            with col2:
                st.metric("Top Category Revenue", f"${cat_revenue.iloc[0]['amount']:,.0f}")
            with col3:
                st.metric("Avg Category Revenue", f"${cat_revenue['amount'].mean():,.0f}")

            st.markdown("##### Top Categories by Revenue")
            # Bar chart for top categories using component
            # Rename columns to what the component expects
            top_cats_named = top_cats.rename(columns={cat_col: "category", "amount": "total_revenue"})
            ChartComponents.render_category_analysis_chart(top_cats_named, "bar", num_cats)

        elif analysis_type == "Product Analysis" and product_col is not None:
            # Top products by total revenue
            product_revenue = merged_cat.groupby(product_col, as_index=False)["amount"].sum().sort_values("amount", ascending=False)
            
            # Slider for number of top products
            max_products = min(len(product_revenue), 50)  # Limit to available products or 50
            num_products = st.slider("Number of Top Products to Show", 1, max_products, 15)
            top_products = product_revenue.head(num_products)

            # Product statistics FIRST
            st.markdown("##### Product Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Products", len(product_revenue))
            with col2:
                st.metric("Top Product Revenue", f"${product_revenue.iloc[0]['amount']:,.0f}")
            with col3:
                st.metric("Avg Product Revenue", f"${product_revenue['amount'].mean():,.0f}")

            st.markdown("##### Top Products by Revenue")
            # Bar chart for top products
            product_chart = (
                alt.Chart(top_products)
                .mark_bar()
                .encode(
                    x=alt.X("amount:Q", title="Total Revenue"),
                    y=alt.Y(f"{product_col}:N", sort="-x", title="Product"),
                    tooltip=[f"{product_col}:N", alt.Tooltip("amount:Q", format=",.0f")],
                    color=alt.value("#2ca02c")
                )
                .properties(title=f"Top {num_products} Products by Revenue", height=500)
            )
            st.altair_chart(product_chart, use_container_width=True)

        elif analysis_type == "Category vs Product Comparison" and cat_col is not None and product_col is not None:
            st.markdown("##### Category vs Product Performance")
            
            # Category performance
            cat_perf = merged_cat.groupby(cat_col, as_index=False).agg({
                "amount": "sum",
                product_col: "nunique"
            }).rename(columns={product_col: "product_count"})
            cat_perf["avg_product_revenue"] = cat_perf["amount"] / cat_perf["product_count"]
            
            # Scatter plot: Category revenue vs Number of products
            scatter_chart = (
                alt.Chart(cat_perf)
                .mark_circle(size=100)
                .encode(
                    x=alt.X("product_count:Q", title="Number of Products"),
                    y=alt.Y("amount:Q", title="Total Revenue"),
                    color=alt.Color("avg_product_revenue:Q", title="Avg Product Revenue", scale=alt.Scale(scheme="viridis")),
                    tooltip=[
                        f"{cat_col}:N",
                        alt.Tooltip("amount:Q", format=",.0f", title="Total Revenue"),
                        alt.Tooltip("product_count:Q", title="Product Count"),
                        alt.Tooltip("avg_product_revenue:Q", format=",.0f", title="Avg Product Revenue")
                    ]
                )
                .properties(title="Category Performance: Revenue vs Product Count")
            )
            st.altair_chart(scatter_chart, use_container_width=True)
            
            # Top performing categories table
            st.markdown("##### Top Performing Categories")
            cat_perf_sorted = cat_perf.sort_values("amount", ascending=False).head(10)
            st.dataframe(
                cat_perf_sorted[[cat_col, "amount", "product_count", "avg_product_revenue"]].round(2),
                width='stretch'
            )

        else:
            st.warning(f"Required columns not found for {analysis_type}. Please check your data.")

    with tab_rfm:
        st.markdown("#### RFM Analysis")
        
        # Use merged data for RFM analysis
        try:
            merged_rfm = data_prep['merged_rfm_df']
        except Exception as e:
            st.error(f"Data access failed: {e}")
            return

        # Ensure datetime and amount
        date_col = "Date" if "Date" in merged_rfm.columns else ("date" if "date" in merged_rfm.columns else None)
        customer_col = "Member_number" if "Member_number" in merged_rfm.columns else ("member_number" if "member_number" in merged_rfm.columns else None)
        
        if date_col is None:
            st.warning("No date column found for RFM analysis.")
            return
        if customer_col is None:
            st.warning("No customer column found for RFM analysis.")
            return
        try:
            merged_rfm = ensure_amount_column(merged_rfm)
        except ValueError as e:
            st.warning(f"Cannot perform RFM analysis: {e}")
            return

        # Ensure datetime first
        merged_rfm[date_col] = pd.to_datetime(merged_rfm[date_col], errors="coerce")
        merged_rfm = merged_rfm.dropna(subset=[date_col, customer_col])

        # Apply date filter if available (after coercion to datetime)
        if 'start_date' in locals() and 'end_date' in locals():
            merged_rfm = merged_rfm[
                (merged_rfm[date_col].dt.date >= start_date.date()) & 
                (merged_rfm[date_col].dt.date <= end_date.date())
            ]

        # Perform RFM analysis using service
        rfm_analysis = service.perform_rfm_analysis(merged_rfm)
        if rfm_analysis['status'] != 'success':
            st.error(f"RFM analysis failed: {rfm_analysis.get('error', 'Unknown error')}")
            return
        
        rfm = rfm_analysis['rfm_df']

        # RFM analysis already includes segmentation

        # RFM Statistics using component
        st.markdown("##### RFM Statistics")
        rfm_metrics = {
            "Total Customers": len(rfm),
            "Avg Recency": f"{rfm['Recency'].mean():.1f} days",
            "Avg Frequency": f"{rfm['Frequency'].mean():.1f}",
            "Avg Monetary": f"${rfm['Monetary'].mean():,.0f}"
        }
        FormComponents.render_metric_cards(rfm_metrics, 4)

        # RFM Distributions (EDA only)
        st.markdown("##### RFM Distributions (EDA)")
        bins = FormComponents.render_bins_selector(20)
        rfm_reset = rfm.reset_index()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            rec_chart = (
                alt.Chart(rfm_reset)
                .mark_bar()
                .encode(
                    x=alt.X('Recency:Q', bin=alt.Bin(maxbins=bins), title='Recency'),
                    y=alt.Y('count()', title='Count')
                )
                .properties(title='Recency Distribution', height=300)
            )
            st.altair_chart(rec_chart, use_container_width=True)
        with col_b:
            freq_chart = (
                alt.Chart(rfm_reset)
                .mark_bar()
                .encode(
                    x=alt.X('Frequency:Q', bin=alt.Bin(maxbins=bins), title='Frequency'),
                    y=alt.Y('count()', title='Count')
                )
                .properties(title='Frequency Distribution', height=300)
            )
            st.altair_chart(freq_chart, use_container_width=True)
        with col_c:
            mon_chart = (
                alt.Chart(rfm_reset)
                .mark_bar()
                .encode(
                    x=alt.X('Monetary:Q', bin=alt.Bin(maxbins=bins), title='Monetary'),
                    y=alt.Y('count()', title='Count')
                )
                .properties(title='Monetary Distribution', height=300)
            )
            st.altair_chart(mon_chart, use_container_width=True)

        # Note: Modeling & Evaluation moved to Model Evaluation page

        # Note: Removed Customer Segments Distribution, RFM Score Distribution, and Top Customers by Segment as requested

    # Footer
    Footer.render()


