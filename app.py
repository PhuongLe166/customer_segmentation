import streamlit as st
import pandas as pd
import src.data_processing_eda as dpe   # Data preprocessing + EDA utils
import src.rfm_segmentation as rfm_seg  # RFM + clustering utils

# -----------------------------------------
# Streamlit page config
# -----------------------------------------
st.set_page_config(page_title = "Customer Clustering EDA", layout = "wide")

st.title("📊 Customer Clustering - Data Exploration App")

# -----------------------------------------
# Sidebar: File upload
# -----------------------------------------
st.sidebar.header("📥 Upload Datasets")
products_file = st.sidebar.file_uploader("Upload Products file (CSV/XLSX)", type = ["csv", "xlsx"])
transactions_file = st.sidebar.file_uploader("Upload Transactions file (CSV/XLSX)", type = ["csv", "xlsx"])

# -----------------------------------------
# Main app logic
# -----------------------------------------
if products_file and transactions_file :
    # Load datasets
    products_df = dpe.load_dataset(products_file)
    transactions_df = dpe.load_dataset(transactions_file)
    st.success("✅ Datasets loaded successfully!")

    # Dataset overview
    with st.expander("📋 Products Dataset Overview") :
        st.write(dpe.dataset_overview(products_df, "Products"))

    with st.expander("📋 Transactions Dataset Overview") :
        st.write(dpe.dataset_overview(transactions_df, "Transactions"))

    # -----------------------------------------
    # Section 2: Overview
    # -----------------------------------------
    st.header("🏠 Overview")

    # Preprocess + Merge
    merged_df = dpe.preprocess_transactions(transactions_df, products_df)
    st.subheader("Merged Dataset Preview")
    st.dataframe(merged_df.head())

    # Aggregate transactions
    transactions_agg = (
        merged_df.groupby(["date", "member_number"])
        .agg(total_items = ("items", "sum"), total_amount = ("amount", "sum")).reset_index()
    )

    # Compute RFM
    rfm_df = dpe.compute_rfm(merged_df)

    # KPI metrics
    total_revenue = merged_df["amount"].sum()
    total_customers = merged_df["member_number"].nunique()
    avg_recency = rfm_df["Recency"].mean()
    avg_frequency = rfm_df["Frequency"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("👥 Customers", f"{total_customers:,}")
    col3.metric("📅 Avg Recency", f"{avg_recency:.1f} days")
    col4.metric("🛍 Avg Frequency", f"{avg_frequency:.1f}")

    # -----------------------------------------
    # Section 3: Data Analysis
    # -----------------------------------------
    st.header("📈 Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Revenue Trends", "Category/Product Revenue", "RFM Analysis"])

    # Revenue Trends
    with tab1 :
        st.subheader("Revenue Trends")
        st.markdown("We explore how sales evolve over time to understand seasonality and growth.")
        if st.checkbox("Show Daily Sales Trend") :
            st.pyplot(dpe.plot_daily_sales(transactions_agg))
        if st.checkbox("Show Monthly Sales Trend") :
            st.pyplot(dpe.plot_monthly_sales(transactions_agg))
        if st.checkbox("Show Monthly Total vs Top Categories") :
            st.pyplot(dpe.plot_monthly_total_with_breakdown(merged_df))

    # Category/Product
    with tab2 :
        st.subheader("Revenue by Category & Product")
        st.markdown("We look at which categories and products generate the most revenue.")
        if st.checkbox("Show Revenue by Category") :
            st.pyplot(dpe.plot_category_revenue(merged_df))
        if st.checkbox("Show Revenue by Product") :
            top_n = st.slider("Select Top N Products", 5, 30, 15)
            st.pyplot(dpe.plot_product_revenue(merged_df, top_n = top_n))

    # RFM
    with tab3 :
        st.subheader("RFM Distributions")
        st.markdown("RFM (Recency, Frequency, Monetary) summarizes customer activity. "
                    "We explore the distribution of each dimension to understand behavior patterns.")
        bins = st.slider("Number of Bins", 10, 50, 30)
        st.pyplot(dpe.plot_rfm_histograms(rfm_df, bins = bins))

    # -----------------------------------------
    # Section 4: Customer Segmentation
    # -----------------------------------------
    st.header("🧩 Customer Segmentation")

    st.markdown("""
    Customer segmentation helps us group customers based on their behavior.
    
    We use **two approaches**:
    1. **Rule-based segmentation (RFM scores):** Customers are grouped using thresholds on recency, frequency, and monetary values.
    2. **KMeans clustering:** A machine learning algorithm that automatically finds groups of similar customers.
    """)

    seg_tab1, seg_tab2 = st.tabs(["RFM Rule-Based Segmentation", "KMeans Clustering"])

    # --- Rule-based segmentation ---
    with seg_tab1 :
        st.subheader("📊 RFM Rule-Based Segmentation")
        st.markdown("""
        **How it works:**  
        - Each customer is scored on **Recency**, **Frequency**, and **Monetary** using quartiles (1–4).  
        - These scores are combined into an overall **RFM Index**.  
        - Customers are then assigned to intuitive segments.
        """)
        
        with st.expander("📖 Explanation of Segments") :
            st.markdown("""
                - **🏆 Champions** – The very best customers. They buy often, spend the most, and made a purchase recently. These are your VIPs.  
                - **💎 Loyal Customers** – Purchase frequently and spend a lot, but not always the most recent. They form the backbone of recurring revenue.  
                - **🌱 Potential Loyalists** – New or growing customers with high Recency and good Frequency. With nurturing, they can become Loyal or Champions.  
                - **⚠️ At Risk** – Used to be active buyers (high Frequency before), but it’s been a long time since their last purchase. Need re-engagement campaigns.  
                - **🚨 Can’t Lose Them** – Previously high-spending customers (high Monetary), but haven’t purchased recently. High-value but disengaged.  
                - **😴 Hibernating** – Least engaged group. Rarely purchase, spend little, and haven’t been active in a long time.  
                - **👀 Need Attention** – Customers who don’t fit neatly into other categories but show some activity. With promotions, they might become more engaged.  
            """)

        rfm_scored = rfm_seg.calculate_rfm_scores(rfm_df.copy())
        rfm_segmented = rfm_seg.apply_segmentation(rfm_scored)

        st.write("Example segmentation results:")
        st.dataframe(rfm_segmented[['Recency', 'Frequency', 'Monetary', 'Segment']].head())

        st.subheader("Segment Distribution")
        fig = rfm_seg.plot_segment_distribution(rfm_segmented)
        st.pyplot(fig)
        
        st.subheader("Segment Visualizations")

        st.markdown("**1. Boxplots:** Compare Recency, Frequency, and Monetary across rule-based segments.")
        fig_box = rfm_seg.plot_cluster_boxplots(rfm_segmented, "Segment")
        st.pyplot(fig_box)

        st.markdown("**2. Pairplot:** Visualize relationships between RFM variables, colored by segment.")
        fig_pair = rfm_seg.plot_pairplot(rfm_segmented, "Segment")
        st.pyplot(fig_pair)

        st.markdown("**3. Treemap:** Show size and average RFM values per segment.")
        fig_tree = rfm_seg.plot_cluster_treemap(rfm_segmented, "Segment")
        st.pyplot(fig_tree)


    # --- KMeans clustering ---
    with seg_tab2 :
        st.subheader("🤖 KMeans Clustering")
        st.markdown("""
        **How it works:**  
        - KMeans is an **unsupervised machine learning algorithm** that groups customers 
        into *k* clusters based on RFM similarity.  
        - Unlike rule-based segmentation, KMeans does not rely on thresholds; instead, it 
        finds natural groupings in the data.  
        - To decide the optimal **k**, we use:  
        1. **Elbow Method** – shows where adding more clusters gives diminishing returns.  
        2. **Silhouette Score** – measures how well-separated the clusters are (closer to 1 is better).  
        """)

        rfm_scaled = rfm_seg.scale_rfm(rfm_df)
        
        # Interactive selection of k
        st.markdown("**Elbow & Silhouette Diagnostic Plots:** Help decide the optimal number of clusters.")
        st.pyplot(rfm_seg.plot_elbow_and_silhouette(rfm_scaled, k_range = range(2, 11)))
        k_value = st.slider("Select number of clusters (k)", min_value = 2, max_value = 10, value = 3, step = 1)

        clustered_df = rfm_seg.run_kmeans(rfm_df.copy(), rfm_scaled, n_clusters = k_value)

        st.write(f"Cluster assignments (normalized labels) for **k = {k_value}**:")
        st.dataframe(clustered_df[['Recency', 'Frequency', 'Monetary', f'Cluster_{k_value}_Normalized']].head())

        # 👉 Silhouette score
        sil_metrics = rfm_seg.compute_silhouette_scores(clustered_df, rfm_scaled, f"Cluster_{k_value}_Normalized")
        st.info(f"📏 Silhouette Score (k = {k_value}): {sil_metrics['overall']:.3f}")

        # 👉 Cluster Quality Evaluation
        st.subheader("📏 Cluster Quality Evaluation")

        # Overall silhouette score
        st.markdown(f"**Overall Silhouette Score (k = {k_value}):** `{sil_metrics['overall']:.3f}`")
        st.markdown("""
        A higher score (closer to **1**) means customers are well separated into distinct groups.  
        A score close to **0** means clusters overlap, and negative values suggest misclassification.
        """)

        # Per-cluster silhouette scores
        st.markdown("**Per-cluster Silhouette Scores:**")
        fig_silbar = rfm_seg.plot_silhouette_bar(sil_metrics["per_cluster"])
        st.pyplot(fig_silbar)
        st.markdown("""
        📊 This bar chart shows the **average silhouette score per cluster**.  
        - It helps identify which clusters are well-defined and which ones may overlap with others.  
        - Higher bars = better separation for that cluster.
        """)

        # Silhouette diagnostic plot
        st.markdown("**Silhouette Plot:**")
        fig_sil = rfm_seg.plot_silhouette(clustered_df, rfm_scaled, f"Cluster_{k_value}_Normalized")
        st.pyplot(fig_sil)
        st.markdown("""
        📈 The silhouette plot shows the **distribution of silhouette values** for every customer within each cluster.  
        - Wider clusters (horizontally) mean more customers are in that group.  
        - Taller values mean those customers fit well inside their cluster.  
        - The red dashed line shows the **average silhouette score** across all clusters.
        """)

        st.subheader("Cluster Visualizations")
        st.markdown("We now explore the clusters visually:")
        
        st.markdown("**1. Boxplots:** Compare Recency, Frequency, and Monetary across clusters.")
        fig_box = rfm_seg.plot_cluster_boxplots(clustered_df, f"Cluster_{k_value}_Normalized")
        st.pyplot(fig_box)

        st.markdown("**2. Pairplot:** Visualize relationships between RFM variables.")
        fig_pair = rfm_seg.plot_pairplot(clustered_df, f"Cluster_{k_value}_Normalized")
        st.pyplot(fig_pair)

        st.markdown("**3. Treemap:** A big-picture view of cluster sizes and averages.")
        fig_tree = rfm_seg.plot_cluster_treemap(clustered_df, f"Cluster_{k_value}_Normalized")
        st.pyplot(fig_tree)
        
    # -----------------------------------------
    # Section 5: Define Fixed 4-Cluster Model
    # -----------------------------------------
    st.header("🏷️ Interpreting the 4-Cluster Model")

    st.markdown("""
    While you can experiment with any number of clusters above, we also define a **4-cluster model** 
    that provides meaningful business insights:

    - **Cluster 0 (💎 Top Customers)** – Very recent buyers, purchase frequently, and spend the most.  
    → They are the *VIP group*, crucial for growth.  
    - **Cluster 1 (📉 Regular Customers)** – Moderate/low spenders, purchase occasionally.  
    → Stable revenue source, but less loyalty. Maintenance costs must be considered.  
    - **Cluster 2 (🌱 High Potential)** – Good frequency and value, but not yet consistent.  
    → With nurturing campaigns, they can evolve into Loyal Customers or Champions.  
    - **Cluster 3 (❌ Lost Customers)** – Long time since last purchase, low frequency, low spend.  
    → They bring almost no current value and require reactivation efforts.  
    """)

    # Train fixed 4-cluster model
    rfm_scaled_fixed = rfm_seg.scale_rfm(rfm_df)
    clustered_4 = rfm_seg.run_kmeans(rfm_df.copy(), rfm_scaled_fixed, n_clusters=4)

    st.subheader("📊 Cluster Distribution (k=4)")
    fig_box_4 = rfm_seg.plot_cluster_boxplots(clustered_4, "Cluster_4_Normalized")
    st.pyplot(fig_box_4)

    st.subheader("Cluster Treemap (k=4)")
    fig_tree_4 = rfm_seg.plot_cluster_treemap(clustered_4, "Cluster_4_Normalized")
    st.pyplot(fig_tree_4)

    # Input prediction with fixed 4-cluster model
    st.subheader("🔮 Predict with 4-Cluster Model")
    col1, col2, col3 = st.columns(3)
    r4 = col1.number_input("Recency (days)", min_value=0, max_value=1000, value=50, key="r4")
    f4 = col2.number_input("Frequency", min_value=1, max_value=100, value=5, key="f4")
    m4 = col3.number_input("Monetary ($)", min_value=1, max_value=10000, value=500, key="m4")

    if st.button("Predict with 4-Cluster Model"):
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        scaler4 = StandardScaler().fit(rfm_df[["Recency","Frequency","Monetary"]])
        kmeans4 = KMeans(n_clusters=4, random_state=42).fit(
            scaler4.transform(rfm_df[["Recency","Frequency","Monetary"]])
        )
        cluster4 = rfm_seg.predict_kmeans_cluster(kmeans4, scaler4, r4, f4, m4)
        
        st.success(f"📌 This customer belongs to **Cluster {cluster4}**")
        
        explanations = {
            0: "💎 **Top Customers** – High spend, frequent purchases, and recent activity.",
            1: "📉 **Regular Customers** – Moderate spenders with occasional purchases.",
            2: "🌱 **High Potential** – Promising frequency and spend, but not consistent yet.",
            3: "❌ **Lost Customers** – Low spend, rare purchases, disengaged.",
        }
        
        st.markdown(explanations[cluster4])

else :
    st.warning("⚠️ Please upload both Products and Transactions datasets to continue.")
