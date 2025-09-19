# pages/eda.py
import streamlit as st
import altair as alt
from config.settings import PAGE_CONFIG
import pandas as pd
from pathlib import Path
from src.eda_utils import load_datasets, infer_join_keys, merge_datasets, compute_recency_frequency_metrics

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
      /* KPI cards */
      .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-top: 10px; }
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
        st.markdown("#### Transactions")
        st.markdown(f"<div class='stat-badges'><span class='stat'>Rows: {len(df_transactions):,}</span><span class='stat'>Columns: {df_transactions.shape[1]}</span></div>", unsafe_allow_html=True)
        st.dataframe(df_transactions.head(10), use_container_width=True)
        st.markdown("---")

        # Products block
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
            # Summary cards FIRST
            total_tx = int(merged.shape[0])
            total_revenue = float(merged.get("amount", pd.Series(dtype=float)).sum()) if "amount" in merged.columns else 0.0
            avg_order_value = float(merged.get("amount", pd.Series(dtype=float)).mean()) if "amount" in merged.columns else 0.0
            avg_recency, avg_frequency = compute_recency_frequency_metrics(
                merged, customer_col="Member_number" if "Member_number" in merged.columns else "member_number",
                date_col="Date" if "Date" in merged.columns else "date",
            )
            kpi_html = f"""
            <div class='kpi-grid'>
              <div class='kpi'><div class='label'>Total Transactions</div><div class='value'>{total_tx:,}</div></div>
              <div class='kpi'><div class='label'>Total Revenue</div><div class='value'>${total_revenue:,.0f}</div></div>
              <div class='kpi'><div class='label'>Avg Price per Order</div><div class='value'>${avg_order_value:,.2f}</div></div>
              <div class='kpi'><div class='label'>Avg Recency</div><div class='value'>{avg_recency:,.1f} days</div></div>
              <div class='kpi'><div class='label'>Avg Frequency</div><div class='value'>{avg_frequency:,.1f}</div></div>
            </div>
            """
            st.markdown(kpi_html, unsafe_allow_html=True)
            # Then meta info and preview
            st.info(f"Merged shape: {merged.shape[0]:,} rows × {merged.shape[1]:,} columns")
            st.dataframe(merged.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Merge failed: {e}")
        
    with tab_revenue:
        st.markdown("#### Revenue Trend")
        # Use sensible defaults for merging (no key controls)
        left_key, right_key = infer_join_keys(df_transactions, df_products)
        try:
            merged_tr = merge_datasets(df_transactions, df_products, left_key, right_key, "inner")
        except Exception as e:
            st.error(f"Merge failed: {e}")
            return

        # Ensure datetime and amount
        date_col = "Date" if "Date" in merged_tr.columns else ("date" if "date" in merged_tr.columns else None)
        if date_col is None:
            st.warning("No date column found for trend.")
            return
        merged_tr[date_col] = pd.to_datetime(merged_tr[date_col], errors="coerce")
        merged_tr = merged_tr.dropna(subset=[date_col])
        if "amount" not in merged_tr.columns:
            st.warning("No amount column found after merge.")
            return

        # Date range slicer
        min_date = merged_tr[date_col].min().date()
        max_date = merged_tr[date_col].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        
        # Filter data by date range
        merged_tr = merged_tr[
            (merged_tr[date_col].dt.date >= start_date) & 
            (merged_tr[date_col].dt.date <= end_date)
        ]

        granularity = st.selectbox("Aggregate by", ["Date", "Week", "Month"], index=2)
        base = merged_tr[[date_col, "amount"]].copy()
        base[date_col] = pd.to_datetime(base[date_col], errors="coerce")
        base = base.dropna(subset=[date_col])

        if granularity == "Date":
            periods = base[date_col].dt.floor("D")
        elif granularity == "Week":
            periods = base[date_col].dt.to_period("W").dt.start_time
        else:
            periods = base[date_col].dt.to_period("M").dt.start_time

        trend = base.groupby(periods)["amount"].sum().reset_index()
        trend.columns = ["period", "revenue"]
        trend = trend.dropna(subset=["period"]).sort_values("period")

        # Friendly x-axis labels by granularity
        if granularity == "Month":
            trend["label"] = trend["period"].dt.strftime("%Y %b")
        elif granularity == "Week":
            iso = trend["period"].dt.isocalendar()
            trend["label"] = iso["year"].astype(str) + " W" + iso["week"].astype(str)
        else:  # Date
            trend["label"] = trend["period"].dt.strftime("%Y %b %d")

        # Ensure numeric and ordered labels, then draw with Altair
        trend["revenue"] = pd.to_numeric(trend["revenue"], errors="coerce").fillna(0)
        order = trend["label"].tolist()
        chart = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("label:N", sort=order, title="Period"),
                y=alt.Y("revenue:Q", title="Revenue"),
                tooltip=["label", alt.Tooltip("revenue:Q", format=",.0f")],
            )
            .properties(title="Revenue Over Time")
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("---")

        # Monthly total vs top 10 categories (always shown)
        cat_col = None
        for c in ["category", "Category", "product_category", "ProductCategory"]:
            if c in merged_tr.columns:
                cat_col = c
                break
        if cat_col is not None:
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
        left_key, right_key = infer_join_keys(df_transactions, df_products)
        try:
            merged_cat = merge_datasets(df_transactions, df_products, left_key, right_key, "inner")
        except Exception as e:
            st.error(f"Merge failed: {e}")
            return

        # Ensure datetime and amount
        date_col = "Date" if "Date" in merged_cat.columns else ("date" if "date" in merged_cat.columns else None)
        if date_col is None:
            st.warning("No date column found for analysis.")
            return
        merged_cat[date_col] = pd.to_datetime(merged_cat[date_col], errors="coerce")
        merged_cat = merged_cat.dropna(subset=[date_col])
        if "amount" not in merged_cat.columns:
            st.warning("No amount column found after merge.")
            return

        # Apply date filter if available
        if 'start_date' in locals() and 'end_date' in locals():
            merged_cat = merged_cat[
                (merged_cat[date_col].dt.date >= start_date) & 
                (merged_cat[date_col].dt.date <= end_date)
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
        analysis_type = st.selectbox("Analysis Type", ["Category Analysis", "Product Analysis", "Category vs Product Comparison"], index=0)

        if analysis_type == "Category Analysis" and cat_col is not None:
            # Top categories by total revenue
            cat_revenue = merged_cat.groupby(cat_col, as_index=False)["amount"].sum().sort_values("amount", ascending=False)
            
            # Slider for number of top categories
            max_cats = min(len(cat_revenue), 20)  # Limit to available categories or 20
            num_cats = st.slider("Number of Top Categories to Show", 1, max_cats, 10)
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
            # Bar chart for top categories
            cat_chart = (
                alt.Chart(top_cats)
                .mark_bar()
                .encode(
                    x=alt.X("amount:Q", title="Total Revenue"),
                    y=alt.Y(f"{cat_col}:N", sort="-x", title="Category"),
                    tooltip=[f"{cat_col}:N", alt.Tooltip("amount:Q", format=",.0f")],
                    color=alt.value("#1f77b4")
                )
                .properties(title=f"Top {num_cats} Categories by Revenue", height=400)
            )
            st.altair_chart(cat_chart, use_container_width=True)

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
                use_container_width=True
            )

        else:
            st.warning(f"Required columns not found for {analysis_type}. Please check your data.")

    with tab_rfm:
        st.markdown("#### RFM Analysis")
        
        # Use merged data for RFM analysis
        left_key, right_key = infer_join_keys(df_transactions, df_products)
        try:
            merged_rfm = merge_datasets(df_transactions, df_products, left_key, right_key, "inner")
        except Exception as e:
            st.error(f"Merge failed: {e}")
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
        if "amount" not in merged_rfm.columns:
            st.warning("No amount column found for RFM analysis.")
            return

        # Ensure datetime first
        merged_rfm[date_col] = pd.to_datetime(merged_rfm[date_col], errors="coerce")
        merged_rfm = merged_rfm.dropna(subset=[date_col, customer_col])

        # Apply date filter if available (after coercion to datetime)
        if 'start_date' in locals() and 'end_date' in locals():
            merged_rfm = merged_rfm[
                (merged_rfm[date_col].dt.date >= start_date) & 
                (merged_rfm[date_col].dt.date <= end_date)
            ]

        # Calculate RFM metrics
        snapshot_date = merged_rfm[date_col].max() + pd.Timedelta(days=1)
        
        rfm = merged_rfm.groupby(customer_col).agg({
            date_col: lambda x: (snapshot_date - x.max()).days,  # Recency
            customer_col: 'count',  # Frequency
            'amount': 'sum'  # Monetary
        }).rename(columns={
            date_col: 'Recency',
            customer_col: 'Frequency',
            'amount': 'Monetary'
        })

        # RFM Scoring (1-5 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')

        # Convert to numeric
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)

        # RFM Score combination
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

        # Customer Segmentation
        def segment_customers(rfm):
            if rfm['R_Score'] >= 4 and rfm['F_Score'] >= 4 and rfm['M_Score'] >= 4:
                return 'Champions'
            elif rfm['R_Score'] >= 3 and rfm['F_Score'] >= 3 and rfm['M_Score'] >= 3:
                return 'Loyal Customers'
            elif rfm['R_Score'] >= 4 and rfm['F_Score'] <= 2:
                return 'New Customers'
            elif rfm['R_Score'] >= 3 and rfm['F_Score'] >= 2 and rfm['M_Score'] >= 2:
                return 'Potential Loyalists'
            elif rfm['R_Score'] >= 2 and rfm['F_Score'] >= 2 and rfm['M_Score'] >= 2:
                return 'Need Attention'
            elif rfm['R_Score'] <= 2 and rfm['F_Score'] >= 2 and rfm['M_Score'] >= 2:
                return 'At Risk'
            elif rfm['R_Score'] <= 2 and rfm['F_Score'] <= 2 and rfm['M_Score'] >= 2:
                return 'Cannot Lose Them'
            else:
                return 'Hibernating'

        rfm['Segment'] = rfm.apply(segment_customers, axis=1)

        # RFM Statistics
        st.markdown("##### RFM Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", len(rfm))
        with col2:
            st.metric("Avg Recency", f"{rfm['Recency'].mean():.1f} days")
        with col3:
            st.metric("Avg Frequency", f"{rfm['Frequency'].mean():.1f}")
        with col4:
            st.metric("Avg Monetary", f"${rfm['Monetary'].mean():,.0f}")

        # RFM Distributions (EDA only)
        st.markdown("##### RFM Distributions (EDA)")
        bins = st.slider("Number of Bins", 5, 60, 20)
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

        # Note: Removed Customer Segments Distribution, RFM Score Distribution, and Top Customers by Segment as requested


