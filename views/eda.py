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
    tab_overview, tab_merge, tab_revenue = st.tabs(["Overview Data", "Merge Datasets", "Revenue Trend"])
    
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
            st.info(f"Merged shape: {merged.shape[0]:,} rows × {merged.shape[1]:,} columns")
            st.dataframe(merged.head(20), use_container_width=True)
            # Summary cards
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
        )
        st.altair_chart(chart, use_container_width=True)


