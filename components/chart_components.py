"""
Chart Components - Reusable chart display components
"""

import streamlit as st
import altair as alt
import pandas as pd
from typing import Dict, Any, Optional, List

class ChartComponents:
    """Chart components for displaying various types of charts."""
    
    @staticmethod
    def render_revenue_trend_chart(trend_df: pd.DataFrame, chart_type: str = "line") -> None:
        """
        Render revenue trend chart.
        
        Args:
            trend_df: DataFrame with trend data
            chart_type: Type of chart ('line', 'bar', 'area')
        """
        if trend_df.empty:
            st.warning("No trend data available to display.")
            return
        
        if chart_type == "line":
            chart = alt.Chart(trend_df).mark_line(
                point=True,
                strokeWidth=3,
                color='#1f77b4'
            ).encode(
                x=alt.X('label:N', sort=list(trend_df['label'].unique()), title='Period'),
                y=alt.Y('revenue:Q', title='Revenue', axis=alt.Axis(format=",.0f")),
                tooltip=['label', alt.Tooltip('revenue:Q', format=",.0f")]
            ).properties(
                title='Revenue Trend Over Time',
                width=700,
                height=400
            )
        elif chart_type == "bar":
            chart = alt.Chart(trend_df).mark_bar(
                color='#2ca02c'
            ).encode(
                x=alt.X('label:N', sort=list(trend_df['label'].unique()), title='Period'),
                y=alt.Y('revenue:Q', title='Revenue', axis=alt.Axis(format=",.0f")),
                tooltip=['label', alt.Tooltip('revenue:Q', format=",.0f")]
            ).properties(
                title='Revenue Trend Over Time',
                width=700,
                height=400
            )
        elif chart_type == "area":
            chart = alt.Chart(trend_df).mark_area(
                color='#ff7f0e',
                opacity=0.7
            ).encode(
                x=alt.X('label:N', sort=list(trend_df['label'].unique()), title='Period'),
                y=alt.Y('revenue:Q', title='Revenue', axis=alt.Axis(format=",.0f")),
                tooltip=['label', alt.Tooltip('revenue:Q', format=",.0f")]
            ).properties(
                title='Revenue Trend Over Time',
                width=700,
                height=400
            )
        else:
            st.error(f"Unsupported chart type: {chart_type}")
            return
        
        st.altair_chart(chart, use_container_width=True)
    
    @staticmethod
    def render_category_analysis_chart(category_df: pd.DataFrame, chart_type: str = "bar", top_n: int = 10) -> None:
        """
        Render category analysis chart.
        
        Args:
            category_df: DataFrame with category data
            chart_type: Type of chart ('bar', 'pie', 'treemap')
            top_n: Number of top categories to show
        """
        if category_df.empty:
            st.warning("No category data available to display.")
            return
        
        # Get top N categories
        top_categories = category_df.head(top_n)
        
        if chart_type == "bar":
            chart = alt.Chart(top_categories).mark_bar(
                color='#1f77b4'
            ).encode(
                x=alt.X('total_revenue:Q', title='Total Revenue'),
                y=alt.Y('category:N', sort='-x', title='Category'),
                tooltip=['category:N', alt.Tooltip('total_revenue:Q', format=",.0f")]
            ).properties(
                title=f'Top {top_n} Categories by Revenue',
                height=400
            )
        elif chart_type == "pie":
            chart = alt.Chart(top_categories).mark_arc(
                innerRadius=50,
                outerRadius=150
            ).encode(
                theta=alt.Theta('total_revenue:Q'),
                color=alt.Color('category:N', scale=alt.Scale(scheme='category20')),
                tooltip=['category:N', alt.Tooltip('total_revenue:Q', format=",.0f")]
            ).properties(
                title=f'Top {top_n} Categories by Revenue',
                width=400,
                height=400
            )
        else:
            st.error(f"Unsupported chart type: {chart_type}")
            return
        
        st.altair_chart(chart, use_container_width=True)
    
    @staticmethod
    def render_cluster_scatter_chart(rfm_df: pd.DataFrame, x_col: str = "Recency", y_col: str = "Monetary", 
                                   color_col: str = "Cluster_Name", size_col: str = "Frequency") -> None:
        """
        Render cluster scatter chart for RFM analysis with K-Means clustering results.
        
        Args:
            rfm_df: DataFrame with RFM and cluster data
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_col: Column for color encoding
            size_col: Column for size encoding
        """
        if rfm_df.empty:
            st.warning("No cluster data available to display.")
            return
        
        # Apply K-Means clustering if not already done
        if 'Cluster' not in rfm_df.columns:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare RFM data for clustering
            rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
            
            # Standardize features
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_features)
            
            # Apply K-Means with 4 clusters
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            rfm_df = rfm_df.copy()
            rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Create cluster labels based on characteristics
        cluster_summary = rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean', 
            'Monetary': 'mean'
        }).round(2)
        
        # Assign meaningful names to clusters based on RFM characteristics
        cluster_summary_sorted = cluster_summary.sort_values('Monetary', ascending=False)
        
        cluster_names = {}
        cluster_names[cluster_summary_sorted.index[0]] = 'VIPs'
        cluster_names[cluster_summary_sorted.index[1]] = 'Regulars'
        cluster_names[cluster_summary_sorted.index[2]] = 'Potential Loyalists'
        cluster_names[cluster_summary_sorted.index[3]] = 'At-Risk'
        
        # Map cluster names
        rfm_df['Cluster_Name'] = rfm_df['Cluster'].map(cluster_names)
        
        # Define 4 main clusters
        main_clusters = ['VIPs', 'Regulars', 'Potential Loyalists', 'At-Risk']
        palette = ['#4C78A8', '#F58518', '#54A24B', '#E45756']
        
        # Sample data for scatter plot (to avoid overcrowding)
        sample_rfm = rfm_df.sample(min(1000, len(rfm_df))).reset_index()
        sample_rfm = sample_rfm[sample_rfm['Cluster_Name'].isin(main_clusters)]
        
        # Add jitter to prevent overlapping points
        import numpy as np
        sample_rfm['Recency_Jitter'] = sample_rfm['Recency'] + np.random.normal(0, 0.1, len(sample_rfm))
        sample_rfm['Monetary_Jitter'] = sample_rfm['Monetary'] + np.random.normal(0, 0.1, len(sample_rfm))
        
        # Scatter Plot - Show individual customer data points
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
    
    @staticmethod
    def render_cluster_bubble_chart(rfm_df: pd.DataFrame) -> None:
        """
        Render cluster bubble chart for RFM analysis with K-Means clustering results.
        
        Args:
            rfm_df: DataFrame with RFM and cluster data
        """
        if rfm_df.empty:
            st.warning("No cluster data available to display.")
            return
        
        # Apply K-Means clustering if not already done
        if 'Cluster' not in rfm_df.columns:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare RFM data for clustering
            rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
            
            # Standardize features
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_features)
            
            # Apply K-Means with 4 clusters
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            rfm_df = rfm_df.copy()
            rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Create cluster labels based on characteristics
        cluster_summary = rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean', 
            'Monetary': 'mean'
        }).round(2)
        
        # Add count of customers per cluster
        cluster_summary['Customer_Count'] = rfm_df.groupby('Cluster').size()
        
        # Assign meaningful names to clusters based on RFM characteristics
        cluster_summary_sorted = cluster_summary.sort_values('Monetary', ascending=False)
        
        cluster_names = {}
        cluster_names[cluster_summary_sorted.index[0]] = 'VIPs'
        cluster_names[cluster_summary_sorted.index[1]] = 'Regulars'
        cluster_names[cluster_summary_sorted.index[2]] = 'Potential Loyalists'
        cluster_names[cluster_summary_sorted.index[3]] = 'At-Risk'
        
        # Map cluster names
        rfm_df['Cluster_Name'] = rfm_df['Cluster'].map(cluster_names)
        
        # Define 4 main clusters
        main_clusters = ['VIPs', 'Regulars', 'Potential Loyalists', 'At-Risk']
        palette = ['#4C78A8', '#F58518', '#54A24B', '#E45756']
        
        # Data for bubble chart
        bubble_data = (
            rfm_df.groupby(['Cluster', 'Cluster_Name'])
               .agg(Recency=('Recency', 'mean'),
                    Total_Revenue=('Monetary', 'sum'),
                    Avg_Revenue=('Monetary', 'mean'),
                    User_Count=('Recency', 'size'),
                    Avg_Frequency=('Frequency', 'mean'),
                    Total_Transactions=('Frequency', 'sum'))
               .reset_index()
        )
        bubble_data = bubble_data[bubble_data['Cluster_Name'].isin(main_clusters)]
        
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
    
    @staticmethod
    def render_growth_chart(growth_df: pd.DataFrame, metric: str = "MoM") -> None:
        """
        Render growth chart (MoM or YoY).
        
        Args:
            growth_df: DataFrame with growth data
            metric: Growth metric ('MoM' or 'YoY')
        """
        if growth_df.empty:
            st.warning("No growth data available to display.")
            return
        
        # Filter data for the specific metric
        metric_data = growth_df.dropna(subset=[metric])
        
        if metric_data.empty:
            st.warning(f"No {metric} data available to display.")
            return
        
        # Revenue bars
        bars = alt.Chart(metric_data).mark_bar(
            opacity=0.7
        ).encode(
            x=alt.X('label:N', sort=list(metric_data['label'].tolist()), title='Month'),
            y=alt.Y('revenue:Q', title='Revenue', axis=alt.Axis(format=",.0f")),
            tooltip=['label', alt.Tooltip('revenue:Q', format=",.0f")],
            color=alt.value("#1f77b4" if metric == "MoM" else "#2ca02c")
        )
        
        # Growth line
        line = alt.Chart(metric_data).mark_line(
            point=True,
            strokeWidth=3
        ).encode(
            x=alt.X('label:N', sort=list(metric_data['label'].tolist()), title='Month'),
            y=alt.Y(f'{metric}:Q', title=f'{metric} Growth (%)', axis=alt.Axis(format=".1f")),
            tooltip=['label', alt.Tooltip(f'{metric}:Q', format=".1f")],
            color=alt.value("#ff7f0e" if metric == "MoM" else "#d62728")
        )
        
        # Combine charts with dual y-axis
        chart = alt.layer(bars, line).resolve_scale(
            y='independent'
        ).properties(
            title=f'{metric}: Revenue (Bar) vs Growth % (Line)',
            width=700,
            height=400
        )
        
        st.altair_chart(chart, use_container_width=True)
    
    @staticmethod
    def render_cluster_treemap(rfm_df: pd.DataFrame, cluster_col: str = "Cluster") -> None:
        """
        Render cluster treemap with detailed information (cluster name, days, orders, $, total customer).
        Based on the original plot_cluster_treemap function from rfm_segmentation.py
        
        Args:
            rfm_df: DataFrame with RFM and cluster data
            cluster_col: Column name for cluster grouping
        """
        if rfm_df.empty:
            st.warning("No cluster data available to display.")
            return
        
        # Apply K-Means clustering if not already done
        if 'Cluster' not in rfm_df.columns:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare RFM data for clustering
            rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
            
            # Standardize features
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_features)
            
            # Apply K-Means with 4 clusters
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            rfm_df = rfm_df.copy()
            rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Calculate cluster summary exactly like in rfm_segmentation.py
        cluster_summary = rfm_df.groupby(cluster_col).agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
            Count=(cluster_col, "count")
        ).reset_index()
        
        total_customers = cluster_summary["Count"].sum()
        cluster_summary["Percent"] = 100 * cluster_summary["Count"] / total_customers
        
        # Create labels exactly like in rfm_segmentation.py
        labels = [
            (f"CLUSTER {row[cluster_col]}\n"
             f"{int(row.Recency)} days\n"
             f"{int(row.Frequency)} orders\n"
             f"{int(row.Monetary)} $\n"
             f"{row.Count} customers ({row.Percent:.2f}%)")
            for _, row in cluster_summary.iterrows()
        ]
        
        # Create treemap using matplotlib and squarify (exactly like rfm_segmentation.py)
        import matplotlib.pyplot as plt
        import squarify
        
        # Set matplotlib backend to avoid display issues
        plt.switch_backend('Agg')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        squarify.plot(
            sizes=cluster_summary["Count"],
            label=labels,
            alpha=0.8,
            color=plt.cm.Set3(range(len(cluster_summary))),
            text_kwargs={"fontsize": 10, "weight": "bold"},
            ax=ax
        )
        
        ax.axis("off")
        ax.set_title(f"{len(cluster_summary)} Clusters – Treemap", fontsize=16, weight="bold")
        fig.tight_layout()
        
        # Display the treemap
        st.pyplot(fig, width='stretch')
        
        # Close the figure to free memory
        plt.close(fig)
