"""
Chart Components - Reusable chart display components
"""

import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import pandas as pd
from typing import Dict, Any, Optional, List
from operator import attrgetter

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
    
    # @staticmethod
    # def render_cohort_analysis_chart(transactions_df: pd.DataFrame, date_col: str = "Date", 
    #                                customer_col: str = "Member_number") -> None:
    #     """
    #     Render cohort analysis as heatmap matrix with table structure.
        
    #     Args:
    #         transactions_df: DataFrame with transaction data
    #         date_col: Name of the date column
    #         customer_col: Name of the customer column
    #     """
    #     if transactions_df.empty:
    #         st.warning("No transaction data available for cohort analysis.")
    #         return
        
    #     try:
    #         # Prepare data for cohort analysis
    #         df = transactions_df.copy()
    #         df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    #         df = df.dropna(subset=[date_col, customer_col])
            
    #         if df.empty:
    #             st.warning("No valid date or customer data for cohort analysis.")
    #             return
            
    #         # Create cohort month (first purchase month for each customer)
    #         df['cohort_month'] = df.groupby(customer_col)[date_col].transform('min').dt.to_period('M')
    #         df['order_month'] = df[date_col].dt.to_period('M')
            
    #         # Calculate cohort data
    #         cohort_data = df.groupby(['cohort_month', 'order_month']).agg({
    #             customer_col: 'nunique'
    #         }).reset_index()
            
    #         # Calculate period number (months since first purchase)
    #         cohort_data['period_number'] = (cohort_data['order_month'] - cohort_data['cohort_month']).apply(attrgetter('n'))
            
    #         # Create cohort table
    #         cohort_table = cohort_data.pivot_table(
    #             index='cohort_month', 
    #             columns='period_number', 
    #             values=customer_col, 
    #             aggfunc='sum'
    #         )
            
    #         # Calculate retention rates
    #         cohort_sizes = cohort_table.iloc[:, 0]
    #         retention_table = cohort_table.divide(cohort_sizes, axis=0)
            
    #         # Fill NaN values with 0
    #         retention_table = retention_table.fillna(0)
            
    #         # Create heatmap data for Altair
    #         heatmap_data = []
            
    #         # Add "All" row data
    #         all_retention = retention_table.mean()
    #         all_start_value = cohort_sizes.sum()
            
    #         # Add All row
    #         for period in range(min(6, len(all_retention))):
    #             if period < len(all_retention):
    #                 retention_value = all_retention.iloc[period] * 100
    #                 heatmap_data.append({
    #                     'Cohort': 'All',
    #                     'Cohort_Display': 'All',
    #                     'Period': period,
    #                     'Period_Display': f'Month {period + 1}',
    #                     'Retention': retention_value,
    #                     'Start_Value': all_start_value,
    #                     'Display_Value': f"{retention_value:.0f}%" if retention_value > 0 else ""
    #                 })
            
    #         # Add individual cohort data
    #         for cohort in retention_table.index:
    #             cohort_date = cohort.to_timestamp()
    #             cohort_name = cohort_date.strftime('%b %Y')
    #             start_value = cohort_sizes[cohort]
                
    #             for period in range(min(6, len(retention_table.columns))):
    #                 if period < len(retention_table.columns):
    #                     retention_value = retention_table.loc[cohort, period] * 100
    #                     heatmap_data.append({
    #                         'Cohort': str(cohort),
    #                         'Cohort_Display': cohort_name,
    #                         'Period': period,
    #                         'Period_Display': f'Month {period + 1}',
    #                         'Retention': retention_value,
    #                         'Start_Value': start_value,
    #                         'Display_Value': f"{retention_value:.0f}%" if retention_value > 0 else ""
    #                     })
            
    #         heatmap_df = pd.DataFrame(heatmap_data)
            
    #         # Create the heatmap chart
    #         st.markdown("### Cohort Analysis")
    #         st.markdown("Customer retention analysis showing how customer groups behave over time.")
            
    #         # Create base chart
    #         base = alt.Chart(heatmap_df).encode(
    #             x=alt.X('Period_Display:O', 
    #                    title='Period',
    #                    axis=alt.Axis(labelAngle=0, labelFontSize=12)),
    #             y=alt.Y('Cohort_Display:O', 
    #                    title='Cohort',
    #                    sort=alt.SortField(field='Cohort', order='descending'),
    #                    axis=alt.Axis(labelFontSize=12))
    #         )
            
    #         # Create heatmap rectangles
    #         heatmap = base.mark_rect(
    #             stroke='white',
    #             strokeWidth=2,
    #             cornerRadius=4
    #         ).encode(
    #             color=alt.Color('Retention:Q',
    #                           title='Retention Rate (%)',
    #                           scale=alt.Scale(scheme='blues', domain=[0, 100]),
    #                           legend=alt.Legend(
    #                               format='.0f',
    #                               titleFontSize=14,
    #                               labelFontSize=12,
    #                               titleFontWeight='bold'
    #                           ))
    #         )
            
    #         # Add text labels
    #         text = base.mark_text(
    #             align='center',
    #             baseline='middle',
    #             fontSize=12,
    #             fontWeight='bold'
    #         ).encode(
    #             text=alt.Text('Display_Value:N'),
    #             color=alt.condition(
    #                 alt.datum.Retention > 50,
    #                 alt.value('white'),
    #                 alt.value('black')
    #             ),
    #             tooltip=[
    #                 alt.Tooltip('Cohort_Display:N', title='Cohort'),
    #                 alt.Tooltip('Period_Display:N', title='Period'),
    #                 alt.Tooltip('Start_Value:Q', title='Start Value'),
    #                 alt.Tooltip('Retention:Q', title='Retention Rate (%)', format='.1f')
    #             ]
    #         )
            
    #         # Combine heatmap and text
    #         chart = (heatmap + text).resolve_scale(
    #             color='independent'
    #         ).properties(
    #             title=alt.TitleParams(
    #                 text='Cohort Analysis - Retention Heatmap',
    #                 subtitle='Customer retention rates by cohort and period',
    #                 fontSize=18,
    #                 fontWeight='bold',
    #                 anchor='start',
    #                 subtitleFontSize=14,
    #                 subtitleColor='#666'
    #             ),
    #             width=800,
    #             height=600
    #         ).configure_view(
    #             strokeWidth=0,
    #             fill='#f8f9fa'
    #         ).configure_axis(
    #             grid=False,
    #             domainColor='#333',
    #             tickColor='#333',
    #             labelColor='#333',
    #             titleColor='#333',
    #             titleFontSize=14,
    #             labelFontSize=12,
    #             titleFontWeight='bold'
    #         ).configure_legend(
    #             titleFontSize=14,
    #             labelFontSize=12,
    #             titleFontWeight='bold'
    #         )
            
    #         st.altair_chart(chart, use_container_width=True)
            
    #         # Add summary statistics
    #         st.markdown("#### Tóm tắt Cohort")
    #         col1, col2, col3 = st.columns(3)
            
    #         with col1:
    #             avg_retention_1m = retention_table.iloc[:, 1].mean() * 100 if len(retention_table.columns) > 1 else 0
    #             st.metric("Tỷ lệ giữ chân TB tháng 1", f"{avg_retention_1m:.1f}%")
            
    #         with col2:
    #             avg_retention_3m = retention_table.iloc[:, 3].mean() * 100 if len(retention_table.columns) > 3 else 0
    #             st.metric("Tỷ lệ giữ chân TB tháng 3", f"{avg_retention_3m:.1f}%")
            
    #         with col3:
    #             total_cohorts = len(retention_table)
    #             st.metric("Tổng số Cohort", f"{total_cohorts}")
            
    #     except Exception as e:
    #         st.error(f"Error creating cohort analysis: {str(e)}")

    @staticmethod
    def render_cohort_analysis_chart(merged_df: pd.DataFrame, date_col: str = "Date", customer_col: str = "Member_number") -> None:
        """
        Render cohort analysis chart with retention percentages in a heatmap style.
        
        Args:
            merged_df: DataFrame with transaction data
            date_col: Name of date column
            customer_col: Name of customer ID column
        """
        if merged_df.empty:
            st.warning("No data available for cohort analysis.")
            return
        
        # Prepare cohort data
        df = merged_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Get first purchase date for each customer
        df['FirstPurchaseDate'] = df.groupby(customer_col)[date_col].transform('min')
        
        # Create cohort groups (monthly)
        df['CohortMonth'] = df['FirstPurchaseDate'].dt.to_period('M')
        df['TransactionMonth'] = df[date_col].dt.to_period('M')
        
        # Calculate cohort index (months since first purchase)
        df['CohortIndex'] = (df['TransactionMonth'] - df['CohortMonth']).apply(lambda x: x.n)
        
        # Build counts by cohort and month offsets (0..6) and render as styled table only
        counts_by_index = (
            df.groupby(['CohortMonth', 'CohortIndex'])[customer_col]
              .nunique()
              .reset_index(name='Customers')
        )
        counts_pivot = counts_by_index.pivot(index='CohortMonth', columns='CohortIndex', values='Customers').sort_index()
        desired_offsets = list(range(0, 7))
        for off in desired_offsets:
            if off not in counts_pivot.columns:
                counts_pivot[off] = pd.NA
        counts_pivot = counts_pivot[desired_offsets]
        
        # Cohort sizes and retention
        cohort_sizes = counts_pivot[0].fillna(0).astype('Int64')
        # Avoid division by zero by replacing 0 with NA for division
        denom = cohort_sizes.replace({0: pd.NA})
        retention_by_index = counts_pivot.divide(denom, axis=0)
        
        # Table in percentages with Start value
        table_pct = (retention_by_index * 100).round(0)
        table_pct.insert(0, 'Start value', cohort_sizes.astype('Int64'))
        
        # Add "All" row
        total_start_value = int(cohort_sizes.fillna(0).sum())
        all_counts = counts_pivot.sum(axis=0).reindex(desired_offsets)
        all_retention = (all_counts / total_start_value * 100).round(0)
        all_row = pd.DataFrame([[total_start_value] + list(all_retention.tolist())], index=['All'], columns=table_pct.columns)
        table_pct = pd.concat([all_row, table_pct])
        
        # Pretty row/column labels
        def _cohort_label(val):
            if val == 'All':
                return 'All'
            try:
                return pd.Period(val).to_timestamp().strftime('%b %Y')
            except Exception:
                return str(val)
        table_pct.index = [_cohort_label(i) for i in table_pct.index]
        table_pct.columns = ['Start value'] + [str(i) for i in desired_offsets]
        
        # Formatters
        def fmt_pct(val):
            if pd.isna(val):
                return ''
            try:
                return f"{int(val)}%"
            except Exception:
                return ''
        
        # Compact height: smaller row height and lower max height
        table_height = int(max(260, min(600, (len(table_pct) + 2) * 24)))
        styled = (table_pct.style
                  .background_gradient(cmap='Blues', axis=None, vmin=0, vmax=100, subset=[str(i) for i in desired_offsets])
                  .format({str(i): fmt_pct for i in desired_offsets} | {'Start value': '{:,}'})
                  .set_properties(**{'text-align': 'center'}))
        # Reliable horizontal scroll using iframe component with improved aesthetics
        num_cols = table_pct.shape[1]
        table_min_width = max(980, 220 + 120 + len(desired_offsets) * 120)
        table_html = styled.to_html()
        html = f"""
        <style>
        .cohort-table-wrapper {{
            overflow-x: auto; width: 100%; border: 1px solid #e9edf3; border-radius: 12px; padding: 8px; background: #ffffff;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }}
        .cohort-table-wrapper table {{
            min-width: {table_min_width}px; border-collapse: separate; border-spacing: 0; font: 14px/1.4 Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji'; color: #111827;
        }}
        .cohort-table-wrapper thead th {{
            position: sticky; top: 0; background: #f8fafc; z-index: 5; font-weight: 600; color: #0f172a;
        }}
        .cohort-table-wrapper th, .cohort-table-wrapper td {{
            border: 1px solid #e5e7eb; padding: 6px 10px; white-space: nowrap;
        }}
        /* Sticky first (index) column */
        .cohort-table-wrapper tr > :is(th,td):first-child {{
            position: sticky; left: 0; background: #ffffff; z-index: 4; min-width: 220px; width: 220px; font-weight: 700;
        }}
        /* Sticky second (Start value) column */
        .cohort-table-wrapper tr > :is(th,td):nth-child(2) {{
            position: sticky; left: 220px; background: #ffffff; z-index: 4; min-width: 120px; width: 120px; text-align: right; font-variant-numeric: tabular-nums;
        }}
        .cohort-table-wrapper tbody tr:nth-child(even) td {{ background: #fbfdff; }}
        .cohort-table-wrapper tbody tr:hover td {{ background: #f0f7ff; }}
        /* Round top corners */
        .cohort-table-wrapper table thead tr th:first-child {{ border-top-left-radius: 10px; }}
        .cohort-table-wrapper table thead tr th:last-child {{ border-top-right-radius: 10px; }}
        </style>
        <div class=\"cohort-table-wrapper\">{table_html}</div>
        """
        components.html(html, height=table_height + 120, scrolling=True)
    
    @staticmethod
    def render_revenue_orders_chart(transactions_df: pd.DataFrame, date_col: str = "Date", 
                                  granularity: str = "Month") -> None:
        """
        Render revenue and orders over time chart.
        
        Args:
            transactions_df: DataFrame with transaction data
            date_col: Name of the date column
            granularity: Time granularity ('Day', 'Week', 'Month', 'Quarter', 'Year')
        """
        if transactions_df.empty:
            st.warning("No transaction data available for revenue/orders analysis.")
            return
        
        try:
            # Prepare data
            df = transactions_df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            
            if df.empty:
                st.warning("No valid date data for revenue/orders analysis.")
                return
            
            # Ensure amount column exists
            if 'amount' not in df.columns:
                if 'items' in df.columns and 'price' in df.columns:
                    df['amount'] = df['items'] * df['price']
                else:
                    st.warning("Cannot calculate revenue: missing 'amount' or 'items'/'price' columns")
                    return
            
            # Create time period column and friendly label like in EDA revenue trend
            if granularity == "Date":
                df['period'] = df[date_col].dt.to_period('D').dt.start_time
                df['label'] = df['period'].dt.strftime('%Y-%m-%d')
            elif granularity == "Week":
                df['period'] = df[date_col].dt.to_period('W').dt.start_time
                df['label'] = df['period'].dt.strftime('W%W %Y-%m-%d')
            elif granularity == "Month":
                df['period'] = df[date_col].dt.to_period('M').dt.start_time
                df['label'] = df['period'].dt.strftime('%Y %b')
            elif granularity == "Year":
                df['period'] = df[date_col].dt.to_period('Y').dt.start_time
                df['label'] = df['period'].dt.strftime('%Y')
            else:
                df['period'] = df[date_col].dt.to_period('M').dt.start_time
                df['label'] = df['period'].dt.strftime('%Y %b')
            
            # Aggregate data
            period_data = df.groupby(['period', 'label']).agg(
                revenue=('amount', 'sum'),
                orders=(date_col, 'count')
            ).reset_index().sort_values('period')
            
            # Create dual-axis chart
            base = alt.Chart(period_data).encode(
                x=alt.X('label:N', sort=list(period_data['label'].tolist()), title=granularity)
            )
            
            # Revenue line
            revenue_line = base.mark_line(
                color='#1f77b4',
                strokeWidth=3
            ).encode(
                y=alt.Y('revenue:Q', 
                       title='Revenue ($)',
                       axis=alt.Axis(format='$,.0f')),
                tooltip=[
                    alt.Tooltip('label:N', title='Period'),
                    alt.Tooltip('revenue:Q', title='Revenue', format='$,.0f')
                ]
            )
            
            # Orders line (scaled)
            orders_line = base.mark_line(
                color='#ff7f0e',
                strokeWidth=2,
                strokeDash=[5, 5]
            ).encode(
                y=alt.Y('orders:Q', 
                       title='Orders (scaled)',
                       scale=alt.Scale(domain=[0, period_data['orders'].max() * 1.1]),
                       axis=alt.Axis(format=',.0f')),
                tooltip=[
                    alt.Tooltip('label:N', title='Period'),
                    alt.Tooltip('orders:Q', title='Orders', format=',.0f')
                ]
            )
            
            # Combine charts
            chart = alt.layer(revenue_line, orders_line).resolve_scale(
                y='independent'
            ).properties(
                title=f'Revenue and Orders Over Time ({granularity})',
                width=700,
                height=400
            )
            
            st.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating revenue/orders chart: {str(e)}")
    
    @staticmethod
    def render_funnel_chart(transactions_df: pd.DataFrame, date_col: str = "Date", 
                          customer_col: str = "Member_number") -> None:
        """
        Render funnel chart showing customer journey stages.
        
        Args:
            transactions_df: DataFrame with transaction data
            date_col: Name of the date column
            customer_col: Name of the customer column
        """
        if transactions_df.empty:
            st.warning("No transaction data available for funnel analysis.")
            return
        
        try:
            # Prepare data
            df = transactions_df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col, customer_col])
            
            if df.empty:
                st.warning("No valid data for funnel analysis.")
                return
            
            # Calculate funnel stages
            total_customers = df[customer_col].nunique()
            customers_with_orders = df[customer_col].nunique()
            
            # Calculate different stages based on available data
            funnel_data = []
            
            # Stage 1: Total unique customers (viewers)
            funnel_data.append({
                'Stage': 'View',
                'Count': total_customers,
                'Percentage': 100.0
            })
            
            # Stage 2: Customers with transactions (ATC - Add to Cart equivalent)
            customers_with_transactions = df[customer_col].nunique()
            funnel_data.append({
                'Stage': 'ATC',
                'Count': customers_with_transactions,
                'Percentage': (customers_with_transactions / total_customers) * 100
            })
            
            # Stage 3: Customers with multiple transactions (Checkout equivalent)
            customers_multiple_transactions = df[df.groupby(customer_col)[date_col].transform('count') > 1][customer_col].nunique()
            funnel_data.append({
                'Stage': 'Checkout',
                'Count': customers_multiple_transactions,
                'Percentage': (customers_multiple_transactions / total_customers) * 100
            })
            
            # Stage 4: High-value customers (Purchase equivalent)
            if 'amount' in df.columns or ('items' in df.columns and 'price' in df.columns):
                if 'amount' not in df.columns:
                    df['amount'] = df['items'] * df['price']
                
                # Customers with above-average spending
                avg_spending = df.groupby(customer_col)['amount'].sum().mean()
                high_value_customers = df.groupby(customer_col)['amount'].sum()
                high_value_customers = high_value_customers[high_value_customers > avg_spending].index.nunique()
                
                funnel_data.append({
                    'Stage': 'Purchase',
                    'Count': high_value_customers,
                    'Percentage': (high_value_customers / total_customers) * 100
                })
            
            funnel_df = pd.DataFrame(funnel_data)
            
            # Create funnel chart
            funnel_chart = alt.Chart(funnel_df).mark_bar(
                cornerRadius=5
            ).encode(
                x=alt.X('Count:Q', title='Number of Customers'),
                y=alt.Y('Stage:O', 
                       title='Funnel Stage',
                       sort=['View', 'ATC', 'Checkout', 'Purchase']),
                color=alt.Color('Stage:N', 
                              scale=alt.Scale(domain=['View', 'ATC', 'Checkout', 'Purchase'],
                                            range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])),
                tooltip=[
                    alt.Tooltip('Stage:N', title='Stage'),
                    alt.Tooltip('Count:Q', title='Customers', format=',.0f'),
                    alt.Tooltip('Percentage:Q', title='Conversion Rate (%)', format='.1f')
                ]
            ).properties(
                title='Customer Journey Funnel',
                width=600,
                height=300
            )
            
            st.altair_chart(funnel_chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating funnel chart: {str(e)}")