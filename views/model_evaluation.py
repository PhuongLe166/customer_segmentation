# pages/model_evaluation.py  
import streamlit as st
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
from config.settings import PAGE_CONFIG
from src.customer_segmentation_service import CustomerSegmentationService
from components import Footer
from src.evaluate_core import EvaluateCore

def show():
    """Display the Model Evaluation page"""
    # Page title and intro
    st.markdown(f"# {PAGE_CONFIG['model_evaluation']['title']}")
    # Description removed per request
    st.markdown("---")

    # Objective (collapsible)
    with st.expander("Objective"):
        st.markdown("Customer segmentation helps us group customers based on their behavior.")
        st.markdown("\n**We use two approaches:**")
        st.markdown("- **Rule-based segmentation (RFM scores)**: Customers are grouped using thresholds on recency, frequency, and monetary values.")
        st.markdown("- **KMeans clustering**: A machine learning algorithm that automatically finds groups of similar customers.")

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
        
        # Calculate KPIs using service
        kpis = service.calculate_kpis(merged, rfm)
        if kpis['status'] != 'success':
            st.error(f"KPI calculation failed: {kpis.get('error', 'Unknown error')}")
            return
        
        kpi_data = kpis['kpi_data']
        
    except Exception as e:
        st.error(f"Service initialization failed: {e}")
        return

    # Tabs: Rules vs KMeans
    tab_rules, tab_kmeans = st.tabs(["Rule-Based Segmentation", "K-Means Clustering"]) 

    with tab_rules:
        st.markdown("#### Rule-based Segmentation")

        # KPI Cards with custom styling
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
            transition: transform 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .kpi-icon {
            font-size: 24px;
            margin-bottom: 8px;
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
        </style>
        """, unsafe_allow_html=True)
        
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">💰</div>
                <div class="kpi-title">Total Revenue</div>
                <div class="kpi-value">${kpi_data['total_revenue']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">👥</div>
                <div class="kpi-title">Customers</div>
                <div class="kpi-value">{kpi_data['total_users']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🗓️</div>
                <div class="kpi-title">Avg Recency</div>
                <div class="kpi-value">{kpi_data['avg_recency']:,.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📈</div>
                <div class="kpi-title">Avg Frequency</div>
                <div class="kpi-value">{kpi_data['avg_frequency']:,.1f}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("How it works"):
            st.markdown("- Each customer is scored on **Recency**, **Frequency**, and **Monetary** using quartiles (1–4).")
            st.markdown("- These scores are combined into an overall **RFM Index**.")
            st.markdown("- Customers are then assigned to intuitive **segments**.")

        with st.expander("Explanation of Segments"):
            st.markdown("- 🏆 **Champions** – Best customers: buy often, spend the most, and purchased recently.")
            st.markdown("- 💎 **Loyal Customers** – Purchase frequently and spend a lot; backbone of recurring revenue.")
            st.markdown("- 🌱 **Potential Loyalists** – New or growing customers with high Recency and good Frequency.")
            st.markdown("- ⚠️ **At Risk** – Used to be active buyers (high Frequency) but it’s been a long time since last purchase.")
            st.markdown("- 🚫 **Can’t Lose Them** – High Monetary before, but haven’t purchased recently; high value but disengaged.")
            st.markdown("- 😴 **Hibernating** – Least engaged group; rarely purchase and spend little.")
            st.markdown("- 👀 **Need Attention** – Do not fit other categories but show some activity; can be nurtured.")
        # Apply rule-based segmentation using service
        rfm_rules = rfm.copy()  # RFM analysis already includes segmentation

        # Table first
        st.markdown("Example segmentation results:")
        st.dataframe(rfm_rules[['Recency','Frequency','Monetary','Segment']].head(5), width='stretch')
        
        # Chart second
        fig = EvaluateCore.plot_segment_distribution(rfm_rules.reset_index())
        st.pyplot(fig, width='stretch')

        # Treemap for rule-based (use Segment as groups)
        treemap_fig_rules = EvaluateCore.plot_cluster_treemap(rfm_rules.reset_index().rename(columns={'Segment':'Cluster'}), 'Cluster')
        st.pyplot(treemap_fig_rules, width='stretch')

        # Optional visualizations dropdown
        viz_option_rules = st.selectbox(
            "Additional Visualization (Rules Based)",
            ["None", "Boxplots by Segment", "Pairplot by Segment"],
            index=0,
        )
        if viz_option_rules == "Boxplots by Segment":
            fig_box = EvaluateCore.plot_cluster_boxplots(
                rfm_rules.reset_index().rename(columns={"Segment": "Cluster"}),
                "Cluster",
            )
            st.pyplot(fig_box, width='stretch')
        elif viz_option_rules == "Pairplot by Segment":
            fig_pair = EvaluateCore.plot_pairplot(
                rfm_rules.reset_index().rename(columns={"Segment": "Cluster"}),
                "Cluster",
            )
            st.pyplot(fig_pair, width='stretch')

    with tab_kmeans:
        st.markdown("#### K-Means Clustering on RFM")
        with st.expander("How it works"):
            st.markdown("- KMeans is an **unsupervised** algorithm that groups customers into k clusters based on RFM similarity.")
            st.markdown("- Unlike rule-based segmentation, KMeans doesn't rely on thresholds; it finds natural groupings.")
            st.markdown("- To decide the optimal k, we use:")
            st.markdown("  1. **Elbow Method** – diminishing returns in inertia.")
            st.markdown("  2. **Silhouette Score** – separation quality (closer to 1 is better).")
        try:
            # Evaluate clustering quality using service
            evaluation = service.evaluate_clustering_quality(rfm)
            if evaluation['status'] != 'success':
                st.error(f"Clustering evaluation failed: {evaluation.get('error', 'Unknown error')}")
                return
            
            evaluation_results = evaluation['evaluation_results']
            
            # Display elbow and silhouette plots
            c1p, c2p = st.columns(2)
            with c1p:
                fig1, ax1 = plt.subplots(figsize=(6,4))
                ax1.plot(evaluation_results['k_range'], evaluation_results['inertias'], marker='o')
                ax1.set_title('Elbow Method (Inertia vs. k)')
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('Inertia')
                ax1.grid(True, linestyle='--', alpha=0.4)
                st.pyplot(fig1, width='stretch')
            with c2p:
                fig2, ax2 = plt.subplots(figsize=(6,4))
                ax2.plot(evaluation_results['k_range'], evaluation_results['silhouettes'], marker='o', color='#2ca02c')
                ax2.set_title('Silhouette Score vs. k')
                ax2.set_xlabel('Number of Clusters (k)')
                ax2.set_ylabel('Silhouette Score')
                ax2.grid(True, linestyle='--', alpha=0.4)
                st.pyplot(fig2, width='stretch')

            # Default k = 4 (can be adjusted by user)
            k = st.slider("Select number of clusters (k)", 2, 10, 4)

            # Perform K-Means clustering using service
            clustering = service.perform_kmeans_clustering(rfm.copy(), n_clusters=k)
            if clustering['status'] != 'success':
                st.error(f"K-Means clustering failed: {clustering.get('error', 'Unknown error')}")
                return
            
            rfm_km = clustering['rfm_clustered_df']
            clustering_metrics = clustering['clustering_metrics']
            
            sil = clustering_metrics['silhouette_score']
            dbi = clustering_metrics['davies_bouldin_score']
            
            # Custom CSS for metric cards
            st.markdown("""
            <style>
            .metric-card {
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 20px;
                margin: 10px 0;
                text-align: left;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-title {
                font-size: 14px;
                color: #6c757d;
                font-weight: 500;
                margin-bottom: 8px;
            }
            .metric-value {
                font-size: 24px;
                color: #212529;
                font-weight: 600;
                margin: 0;
            }
            </style>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Silhouette Score</div>
                    <div class="metric-value">{sil:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Davies-Bouldin Index</div>
                    <div class="metric-value">{dbi:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("##### Cluster Visualization")

            cluster_avg = rfm_km.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().reset_index()
            recency_median = rfm_km['Recency'].median()
            monetary_median = rfm_km['Monetary'].median()

            # Modern color palette (Tableau10)
            cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # Main bubble chart
            chart = alt.Chart(cluster_avg).mark_circle(
                opacity=0.85,
                stroke='black',
                strokeWidth=1.5
            ).encode(
                x=alt.X('Recency:Q',
                        title='Recency (avg days – lower = more recent)',
                        scale=alt.Scale(domain=[0, 400])),
                y=alt.Y('Monetary:Q',
                        title='Monetary (average $)',
                        scale=alt.Scale(domain=[0, 220])),
                size=alt.Size('Frequency:Q',
                            legend=None,
                            scale=alt.Scale(range=[100, 1000])),
                color=alt.Color('Cluster:N',
                                title='Cluster',
                                scale=alt.Scale(domain=list(range(k)), range=cluster_colors[:k]),
                                legend=alt.Legend(titleFontSize=13,
                                                labelFontSize=12,
                                                symbolSize=120)),
                tooltip=[
                    alt.Tooltip('Cluster:N', title='Cluster'),
                    alt.Tooltip('Recency:Q', title='Avg Recency (days)', format=',.0f'),
                    alt.Tooltip('Frequency:Q', title='Avg Frequency', format=',.1f'),
                    alt.Tooltip('Monetary:Q', title='Avg Monetary ($)', format='$,.0f')
                ]
            ).properties(
                width=700,
                height=400,
                title=alt.TitleParams(
                    text='Clusters: Recency vs Monetary (Bubble size = Frequency)',
                    fontSize=16,
                    fontWeight='bold',
                    anchor='start'
                )
            )

            # Median lines with label
            vline = alt.Chart(pd.DataFrame({'x': [recency_median]})).mark_rule(
                strokeDash=[6, 4], color='#555', strokeWidth=2
            ).encode(x='x:Q')

            hline = alt.Chart(pd.DataFrame({'y': [monetary_median]})).mark_rule(
                strokeDash=[6, 4], color='#555', strokeWidth=2
            ).encode(y='y:Q')

            # Label boxes
            vlabel = alt.Chart(pd.DataFrame({'x': [recency_median], 'y':[210]})).mark_text(
                text='Recency Median',
                align='center', fontSize=12, fontWeight='bold',
                color='#444', dy=-10
            ).encode(x='x:Q', y='y:Q')

            hlabel = alt.Chart(pd.DataFrame({'x':[390], 'y':[monetary_median]})).mark_text(
                text='Monetary Median',
                align='left', fontSize=12, fontWeight='bold',
                color='#444', dx=10
            ).encode(x='x:Q', y='y:Q')

            # Combine
            final_chart = (chart + vline + hline + vlabel + hlabel).configure_view(
                strokeWidth=0
            ).configure_axis(
                grid=True, gridColor='#eaeaea',
                domainColor='#333', tickColor='#333',
                labelColor='#333', titleColor='#333'
            )

            st.altair_chart(final_chart, use_container_width=True)

            
            # Treemap - Second line (header removed per request)
            treemap_fig = EvaluateCore.plot_cluster_treemap(rfm_km.reset_index(), 'Cluster')
            st.pyplot(treemap_fig, width='stretch')

            # Optional visualizations dropdown
            viz_option_km = st.selectbox(
                "Additional Visualization (K-Means)",
                ["None", "Silhouette by Cluster", "Boxplots by Cluster", "Pairplot by Cluster"],
                index=0,
            )
            if viz_option_km == "Boxplots by Cluster":
                fig_box_km = EvaluateCore.plot_cluster_boxplots(rfm_km.reset_index(), "Cluster")
                st.pyplot(fig_box_km, width='stretch')
            elif viz_option_km == "Pairplot by Cluster":
                fig_pair_km = EvaluateCore.plot_pairplot(rfm_km.reset_index(), "Cluster")
                st.pyplot(fig_pair_km, width='stretch')
            elif viz_option_km == "Silhouette by Cluster":
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.metrics import silhouette_samples
                    # Rebuild the feature matrix consistently (standard scaling)
                    features = rfm[['Recency','Frequency','Monetary']].copy()
                    scaler_tmp = StandardScaler()
                    X_tmp = scaler_tmp.fit_transform(features)
                    labels_tmp = rfm_km['Cluster'].to_numpy()

                    sil_samples = silhouette_samples(X_tmp, labels_tmp)
                    rfm_km['_silhouette'] = sil_samples
                    per_cluster_sil = (
                        rfm_km.groupby('Cluster')['_silhouette'].mean().reset_index()
                        .rename(columns={'_silhouette': 'Avg Silhouette'})
                    )
                    fig_sil, ax_sil = plt.subplots(figsize=(6,4))
                    ax_sil.bar(per_cluster_sil['Cluster'].astype(str), per_cluster_sil['Avg Silhouette'], color=['#69b3a2','#e78ac3','#8da0cb','#a6d854','#ffd92f'][:k])
                    ax_sil.set_ylim(0, max(0.01, per_cluster_sil['Avg Silhouette'].max()*1.15))
                    ax_sil.set_xlabel('Cluster')
                    ax_sil.set_ylabel('Average Silhouette Score')
                    ax_sil.grid(True, axis='y', linestyle='--', alpha=0.3)
                    st.pyplot(fig_sil, width='stretch')
                except Exception as e:
                    st.warning(f"Could not compute silhouette chart: {e}")
        except Exception as e:
            st.error(f"K-Means clustering failed: {e}")
    
    # Footer
    Footer.render()
