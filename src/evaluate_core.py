# src/evaluate_core.py
"""
Evaluate Core module - Contains all model evaluation and visualization logic
for customer segmentation application.

This module provides comprehensive visualization and evaluation capabilities
with enhanced performance, interactivity, and customization options.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class EvaluateCore:
    """Core evaluation operations for model assessment and visualization with enhanced capabilities."""
    
    @staticmethod
    def create_kpi_cards(kpi_data: Dict[str, Any], style: str = "basic") -> str:
        """
        Create HTML for KPI cards display with enhanced styling options.
        
        Args:
            kpi_data: Dictionary with KPI values
            style: Style type ('basic', 'modern', 'minimal')
            
        Returns:
            HTML string for KPI cards
        """
        try:
            if style == "basic":
                return EvaluateCore._create_basic_kpi_cards(kpi_data)
            elif style == "modern":
                return EvaluateCore._create_modern_kpi_cards(kpi_data)
            elif style == "minimal":
                return EvaluateCore._create_minimal_kpi_cards(kpi_data)
            else:
                logger.warning(f"Unknown style '{style}', using basic style")
                return EvaluateCore._create_basic_kpi_cards(kpi_data)
                
        except Exception as e:
            logger.error(f"Failed to create KPI cards: {e}")
            return "<div>Error creating KPI cards</div>"
    
    @staticmethod
    def _create_basic_kpi_cards(kpi_data: Dict[str, Any]) -> str:
        """Create basic KPI cards."""
        kpi_html = f"""
        <div class='kpi-grid'>
          <div class='kpi'><div class='label'>Total Users</div><div class='value'>{kpi_data['total_users']:,}</div></div>
          <div class='kpi'><div class='label'>Total Transactions</div><div class='value'>{kpi_data['total_transactions']:,}</div></div>
          <div class='kpi'><div class='label'>Total Revenue</div><div class='value'>${kpi_data['total_revenue']:,.0f}</div></div>
          <div class='kpi'><div class='label'>Avg Recency</div><div class='value'>{kpi_data['avg_recency']:,.1f} days</div></div>
          <div class='kpi'><div class='label'>Avg Frequency</div><div class='value'>{kpi_data['avg_frequency']:,.1f}</div></div>
          <div class='kpi'><div class='label'>Avg Monetary</div><div class='value'>${kpi_data['avg_monetary']:,.0f}</div></div>
        </div>
        """
        return kpi_html
    
    @staticmethod
    def _create_modern_kpi_cards(kpi_data: Dict[str, Any]) -> str:
        """Create modern KPI cards with enhanced styling."""
        kpi_html = f"""
        <div class='modern-kpi-grid'>
          <div class='modern-kpi-card'>
            <div class='kpi-icon'>👥</div>
            <div class='kpi-content'>
              <div class='kpi-label'>Total Users</div>
              <div class='kpi-value'>{kpi_data['total_users']:,}</div>
            </div>
          </div>
          <div class='modern-kpi-card'>
            <div class='kpi-icon'>🛒</div>
            <div class='kpi-content'>
              <div class='kpi-label'>Total Transactions</div>
              <div class='kpi-value'>{kpi_data['total_transactions']:,}</div>
            </div>
          </div>
          <div class='modern-kpi-card'>
            <div class='kpi-icon'>💰</div>
            <div class='kpi-content'>
              <div class='kpi-label'>Total Revenue</div>
              <div class='kpi-value'>${kpi_data['total_revenue']:,.0f}</div>
            </div>
          </div>
          <div class='modern-kpi-card'>
            <div class='kpi-icon'>📅</div>
            <div class='kpi-content'>
              <div class='kpi-label'>Avg Recency</div>
              <div class='kpi-value'>{kpi_data['avg_recency']:,.1f} days</div>
            </div>
          </div>
          <div class='modern-kpi-card'>
            <div class='kpi-icon'>🔄</div>
            <div class='kpi-content'>
              <div class='kpi-label'>Avg Frequency</div>
              <div class='kpi-value'>{kpi_data['avg_frequency']:,.1f}</div>
            </div>
          </div>
          <div class='modern-kpi-card'>
            <div class='kpi-icon'>💎</div>
            <div class='kpi-content'>
              <div class='kpi-label'>Avg Monetary</div>
              <div class='kpi-value'>${kpi_data['avg_monetary']:,.0f}</div>
            </div>
          </div>
        </div>
        """
        return kpi_html
    
    @staticmethod
    def _create_minimal_kpi_cards(kpi_data: Dict[str, Any]) -> str:
        """Create minimal KPI cards."""
        kpi_html = f"""
        <div class='minimal-kpi-grid'>
          <div class='minimal-kpi'>Users: {kpi_data['total_users']:,}</div>
          <div class='minimal-kpi'>Transactions: {kpi_data['total_transactions']:,}</div>
          <div class='minimal-kpi'>Revenue: ${kpi_data['total_revenue']:,.0f}</div>
          <div class='minimal-kpi'>Recency: {kpi_data['avg_recency']:,.1f}d</div>
          <div class='minimal-kpi'>Frequency: {kpi_data['avg_frequency']:,.1f}</div>
          <div class='minimal-kpi'>Monetary: ${kpi_data['avg_monetary']:,.0f}</div>
        </div>
        """
        return kpi_html
    
    @staticmethod
    def create_advanced_kpi_cards(kpi_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Create HTML for advanced KPI cards with user type percentages and enhanced tooltips.
        
        Args:
            kpi_data: Dictionary with KPI values
            
        Returns:
            Dictionary with HTML strings for different KPI cards
        """
        try:
            # First row: Total metrics
            kpi1_html = f"""
            <div class="kpi-card">
                <div class="kpi-tooltip">Total number of unique customers in the dataset</div>
                <div class="kpi-title">Total Users</div>
                <div class="kpi-value">{kpi_data['total_users']:,}</div>
            </div>
            """
            
            kpi2_html = f"""
            <div class="kpi-card">
                <div class="kpi-tooltip">Total number of transactions across all customers</div>
                <div class="kpi-title">Total Transactions</div>
                <div class="kpi-value">{kpi_data['total_transactions']:,}</div>
            </div>
            """
            
            kpi3_html = f"""
            <div class="kpi-card">
                <div class="kpi-tooltip">Sum of all transaction amounts (items × price)</div>
                <div class="kpi-title">Total Net Revenue</div>
                <div class="kpi-value">${kpi_data['total_revenue']:,.0f}</div>
            </div>
            """
            
            # Second row: Percentage metrics
            kpi4_html = f"""
            <div class="kpi-card">
                <div class="kpi-tooltip">Active Users: Recent (≤30 days) + High Frequency (≥50th percentile) + High Monetary (≥50th percentile)</div>
                <div class="kpi-title">% Users Active</div>
                <div class="kpi-value">{kpi_data['pct_active']:.0f}%</div>
            </div>
            """
            
            kpi5_html = f"""
            <div class="kpi-card">
                <div class="kpi-tooltip">At Risk Users: Medium Recency (31-90 days) + Medium Frequency (≥30th percentile) + Medium Monetary (≥30th percentile)</div>
                <div class="kpi-title">% Users At Risk</div>
                <div class="kpi-value">{kpi_data['pct_at_risk']:.0f}%</div>
            </div>
            """
            
            kpi6_html = f"""
            <div class="kpi-card">
                <div class="kpi-tooltip">Churned Users: Old (>90 days) OR Low Frequency (<30th percentile) OR Low Monetary (<30th percentile)</div>
                <div class="kpi-title">% Users Churned</div>
                <div class="kpi-value">{kpi_data['pct_churned']:.0f}%</div>
            </div>
            """
            
            # Third row: RFM KPIs
            rfm1_html = f"""
            <div class="rfm-card">
                <div class="rfm-tooltip">Recency: Average number of days since each customer's last transaction</div>
                <div class="rfm-letter">R</div>
                <div class="rfm-title">AVG Days Since Last Transaction</div>
                <div class="rfm-value">{kpi_data['avg_recency']:.0f}</div>
            </div>
            """
            
            rfm2_html = f"""
            <div class="rfm-card">
                <div class="rfm-tooltip">Frequency: Average number of transactions per customer</div>
                <div class="rfm-letter">F</div>
                <div class="rfm-title">AVG Transactions per User</div>
                <div class="rfm-value">{kpi_data['avg_frequency']:.1f}</div>
            </div>
            """
            
            rfm3_html = f"""
            <div class="rfm-card">
                <div class="rfm-tooltip">Monetary: Average total revenue per customer (sum of all their transactions)</div>
                <div class="rfm-letter">M</div>
                <div class="rfm-title">AVG Net Revenue Per User</div>
                <div class="rfm-value">${kpi_data['avg_monetary']:.0f}</div>
            </div>
            """
            
            return {
                'kpi1': kpi1_html, 'kpi2': kpi2_html, 'kpi3': kpi3_html,
                'kpi4': kpi4_html, 'kpi5': kpi5_html, 'kpi6': kpi6_html,
                'rfm1': rfm1_html, 'rfm2': rfm2_html, 'rfm3': rfm3_html
            }
            
        except Exception as e:
            logger.error(f"Failed to create advanced KPI cards: {e}")
            return {}
    
    @staticmethod
    def create_revenue_trend_chart(
        trend_df: pd.DataFrame, 
        chart_type: str = "line",
        interactive: bool = True
    ) -> alt.Chart:
        """
        Create revenue trend chart using Altair with enhanced customization.
        
        Args:
            trend_df: DataFrame with trend data
            chart_type: Type of chart ('line', 'bar', 'area')
            interactive: Whether to make chart interactive
            
        Returns:
            Altair chart object
        """
        try:
            if trend_df.empty:
                logger.warning("Empty trend data provided")
                return alt.Chart(pd.DataFrame()).mark_text(text="No data available")
            
            order = trend_df["label"].tolist()
            
            # Base chart configuration
            base_chart = alt.Chart(trend_df)
            
            # Choose chart type
            if chart_type == "line":
                chart = base_chart.mark_line(point=True, strokeWidth=3)
            elif chart_type == "bar":
                chart = base_chart.mark_bar(opacity=0.8)
            elif chart_type == "area":
                chart = base_chart.mark_area(opacity=0.6)
            else:
                chart = base_chart.mark_line(point=True)
            
            # Configure encoding
            chart = chart.encode(
                x=alt.X("label:N", sort=order, title="Period"),
                y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format=",.0f")),
                tooltip=[
                    "label:N",
                    alt.Tooltip("revenue:Q", format=",.0f", title="Revenue ($)"),
                    alt.Tooltip("transaction_count:Q", format=",.0f", title="Transactions")
                ],
                color=alt.value("#1f77b4")
            )
            
            # Add interactive features
            if interactive:
                chart = chart.interactive()
            
            # Configure properties
            chart = chart.properties(
                title=alt.TitleParams(
                    text="Revenue Over Time",
                    fontSize=16,
                    fontWeight="bold"
                ),
                width=600,
                height=400
            )
            
            logger.info(f"Successfully created {chart_type} revenue trend chart")
            return chart
            
        except Exception as e:
            logger.error(f"Failed to create revenue trend chart: {e}")
            return alt.Chart(pd.DataFrame()).mark_text(text="Error creating chart")
    
    @staticmethod
    def create_growth_charts(trend_df: pd.DataFrame) -> Dict[str, Optional[alt.Chart]]:
        """
        Create MoM and YoY growth charts with enhanced styling.
        
        Args:
            trend_df: DataFrame with trend data including growth metrics
            
        Returns:
            Dictionary with MoM and YoY charts
        """
        try:
            charts = {'mom_chart': None, 'yoy_chart': None}
            
            # MoM Chart
            mom_data = trend_df.dropna(subset=["MoM"])
            if not mom_data.empty:
                charts['mom_chart'] = EvaluateCore._create_mom_chart(mom_data)
            
            # YoY Chart
            yoy_data = trend_df.dropna(subset=["YoY"])
            if not yoy_data.empty:
                charts['yoy_chart'] = EvaluateCore._create_yoy_chart(yoy_data)
            
            logger.info("Successfully created growth charts")
            return charts
            
        except Exception as e:
            logger.error(f"Failed to create growth charts: {e}")
            return {'mom_chart': None, 'yoy_chart': None}
    
    @staticmethod
    def _create_mom_chart(mom_data: pd.DataFrame) -> alt.Chart:
        """Create month-over-month growth chart."""
        # Revenue bars
        mom_bars = (
            alt.Chart(mom_data)
            .mark_bar(opacity=0.7, color="#1f77b4")
            .encode(
                x=alt.X("label:N", sort=list(mom_data["label"].tolist()), title="Month"),
                y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format=",.0f")),
                tooltip=["label", alt.Tooltip("revenue:Q", format=",.0f")]
            )
        )
        
        # Growth line
        mom_line = (
            alt.Chart(mom_data)
            .mark_line(point=True, strokeWidth=3, color="#ff7f0e")
            .encode(
                x=alt.X("label:N", sort=list(mom_data["label"].tolist()), title="Month"),
                y=alt.Y("MoM:Q", title="MoM Growth (%)", axis=alt.Axis(format=".1f")),
                tooltip=["label", alt.Tooltip("MoM:Q", format=".1f")]
            )
        )
        
        return alt.layer(mom_bars, mom_line).resolve_scale(y='independent').properties(
            title="Month-over-Month: Revenue (Bar) vs Growth % (Line)",
            width=600,
            height=400
        )
    
    @staticmethod
    def _create_yoy_chart(yoy_data: pd.DataFrame) -> alt.Chart:
        """Create year-over-year growth chart."""
        # Revenue bars
        yoy_bars = (
            alt.Chart(yoy_data)
            .mark_bar(opacity=0.7, color="#2ca02c")
            .encode(
                x=alt.X("label:N", sort=list(yoy_data["label"].tolist()), title="Month"),
                y=alt.Y("revenue:Q", title="Revenue ($)", axis=alt.Axis(format=",.0f")),
                tooltip=["label", alt.Tooltip("revenue:Q", format=",.0f")]
            )
        )
        
        # Growth line
        yoy_line = (
            alt.Chart(yoy_data)
            .mark_line(point=True, strokeWidth=3, color="#d62728")
            .encode(
                x=alt.X("label:N", sort=list(yoy_data["label"].tolist()), title="Month"),
                y=alt.Y("YoY:Q", title="YoY Growth (%)", axis=alt.Axis(format=".1f")),
                tooltip=["label", alt.Tooltip("YoY:Q", format=".1f")]
            )
        )
        
        return alt.layer(yoy_bars, yoy_line).resolve_scale(y='independent').properties(
            title="Year-over-Year: Revenue (Bar) vs Growth % (Line)",
            width=600,
            height=400
        )
    
    @staticmethod
    def create_cluster_visualization(
        rfm_df: pd.DataFrame, 
        chart_type: str = "bubble",
        interactive: bool = True
    ) -> alt.Chart:
        """
        Create cluster visualization with multiple chart types.
        
        Args:
            rfm_df: DataFrame with cluster data
            chart_type: Type of visualization ('bubble', 'scatter', 'bar')
            interactive: Whether to make chart interactive
            
        Returns:
            Altair chart object
        """
        try:
            if chart_type == "bubble":
                return EvaluateCore._create_bubble_chart(rfm_df, interactive)
            elif chart_type == "scatter":
                return EvaluateCore._create_scatter_chart(rfm_df, interactive)
            elif chart_type == "bar":
                return EvaluateCore._create_cluster_bar_chart(rfm_df, interactive)
            else:
                logger.warning(f"Unknown chart type '{chart_type}', using bubble chart")
                return EvaluateCore._create_bubble_chart(rfm_df, interactive)
                
        except Exception as e:
            logger.error(f"Failed to create cluster visualization: {e}")
            return alt.Chart(pd.DataFrame()).mark_text(text="Error creating chart")
    
    @staticmethod
    def _create_bubble_chart(rfm_df: pd.DataFrame, interactive: bool = True) -> alt.Chart:
        """Create bubble chart for cluster visualization."""
        # Define main clusters and colors
        main_clusters = ['VIPs', 'Regulars', 'Potential Loyalists', 'At-Risk']
        palette = ['#4C78A8', '#F58518', '#54A24B', '#E45756']
        
        # Data for bubble chart
        bubble_data = (
            rfm_df.groupby(['Cluster', 'Cluster_Name'])
               .agg(
                   Recency=('Recency', 'mean'),
                   Total_Revenue=('Monetary', 'sum'),
                   Avg_Revenue=('Monetary', 'mean'),
                   User_Count=('Recency', 'size'),
                   Avg_Frequency=('Frequency', 'mean'),
                   Total_Transactions=('Frequency', 'sum')
               )
               .reset_index()
        )
        bubble_data = bubble_data[bubble_data['Cluster_Name'].isin(main_clusters)]
        
        # Create bubble chart
        chart = (
            alt.Chart(bubble_data)
            .mark_circle(opacity=0.8, stroke='#ffffff', strokeWidth=2)
            .encode(
                x=alt.X('Recency:Q',
                        title='Days Since Last Transaction',
                        scale=alt.Scale(domain=[0, float(bubble_data['Recency'].max()*1.1)])),
                y=alt.Y('Total_Revenue:Q',
                        title='Total Revenue ($)',
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
        )
        
        # Add interactive features
        if interactive:
            chart = chart.interactive()
        
        # Configure properties
        chart = chart.properties(
            width=400,
            height=350,
            title=alt.TitleParams(
                text='Clusters: Recency vs Total Revenue',
                anchor='start', 
                fontSize=14, 
                fontWeight='bold'
            )
        ).configure_view(
            strokeWidth=0,
            fill='#ffffff'
        ).configure_axis(
            grid=True,
            gridOpacity=0.3,
            domainColor='#1f1f1f',
            tickColor='#1f1f1f',
            labelColor='#2b2b2b',
            titleColor='#2b2b2b'
        )
        
        return chart
    
    @staticmethod
    def _create_scatter_chart(rfm_df: pd.DataFrame, interactive: bool = True) -> alt.Chart:
        """Create scatter plot for individual customer data points."""
        # Sample data for performance
        sample_size = min(1000, len(rfm_df))
        sample_rfm = rfm_df.sample(sample_size).reset_index()
        
        main_clusters = ['VIPs', 'Regulars', 'Potential Loyalists', 'At-Risk']
        sample_rfm = sample_rfm[sample_rfm['Cluster_Name'].isin(main_clusters)]
        
        # Add jitter to prevent overlapping points
        np.random.seed(42)  # For reproducibility
        sample_rfm['Recency_Jitter'] = sample_rfm['Recency'] + np.random.normal(0, 0.1, len(sample_rfm))
        sample_rfm['Monetary_Jitter'] = sample_rfm['Monetary'] + np.random.normal(0, 0.1, len(sample_rfm))
        
        palette = ['#4C78A8', '#F58518', '#54A24B', '#E45756']
        
        chart = (
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
                size=alt.value(40),
                tooltip=[
                    alt.Tooltip('member_number:N', title='Customer ID'),
                    alt.Tooltip('Cluster_Name:N', title='Cluster'),
                    alt.Tooltip('Recency:Q', title='Recency (days)', format=',.0f'),
                    alt.Tooltip('Frequency:Q', title='Frequency', format=',.0f'),
                    alt.Tooltip('Monetary:Q', title='Monetary ($)', format='$,.2f')
                ]
            )
        )
        
        # Add interactive features
        if interactive:
            chart = chart.interactive()
        
        # Configure properties
        chart = chart.properties(
            width=400,
            height=350,
            title=alt.TitleParams(
                text='Recency vs Monetary by Cluster',
                anchor='start',
                fontSize=14,
                fontWeight='bold'
            )
        ).configure_view(
            strokeWidth=0,
            fill='#ffffff'
        ).configure_axis(
            grid=True,
            gridOpacity=0.3,
            domainColor='#1f1f1f',
            tickColor='#1f1f1f',
            labelColor='#2b2b2b',
            titleColor='#2b2b2b'
        )
        
        return chart
    
    @staticmethod
    def _create_cluster_bar_chart(rfm_df: pd.DataFrame, interactive: bool = True) -> alt.Chart:
        """Create bar chart for cluster comparison."""
        cluster_summary = (
            rfm_df.groupby('Cluster_Name')
            .agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'Cluster': 'count'
            })
            .reset_index()
        )
        cluster_summary = cluster_summary.rename(columns={'Cluster': 'Customer_Count'})
        
        # Melt for grouped bar chart
        cluster_melted = cluster_summary.melt(
            id_vars=['Cluster_Name', 'Customer_Count'],
            value_vars=['Recency', 'Frequency', 'Monetary'],
            var_name='Metric',
            value_name='Value'
        )
        
        chart = (
            alt.Chart(cluster_melted)
            .mark_bar(opacity=0.8)
            .encode(
                x=alt.X('Cluster_Name:N', title='Cluster'),
                y=alt.Y('Value:Q', title='Value'),
                color=alt.Color('Metric:N', title='RFM Metric'),
                tooltip=[
                    alt.Tooltip('Cluster_Name:N', title='Cluster'),
                    alt.Tooltip('Metric:N', title='Metric'),
                    alt.Tooltip('Value:Q', title='Value', format=',.2f'),
                    alt.Tooltip('Customer_Count:Q', title='Customers', format=',')
                ]
            )
        )
        
        # Add interactive features
        if interactive:
            chart = chart.interactive()
        
        # Configure properties
        chart = chart.properties(
            width=600,
            height=400,
            title=alt.TitleParams(
                text='RFM Metrics by Cluster',
                fontSize=16,
                fontWeight='bold'
            )
        )
        
        return chart
    
    @staticmethod
    def create_segment_table(segment_kpis: pd.DataFrame, style: str = "emoji") -> pd.DataFrame:
        """
        Create formatted segment table with enhanced styling options.
        
        Args:
            segment_kpis: DataFrame with segment KPIs
            style: Style type ('emoji', 'bars', 'colors', 'simple')
            
        Returns:
            Formatted DataFrame for display
        """
        try:
            if style == "emoji":
                return EvaluateCore._create_emoji_table(segment_kpis)
            elif style == "bars":
                return EvaluateCore._create_bar_table(segment_kpis)
            elif style == "colors":
                return EvaluateCore._create_color_table(segment_kpis)
            else:
                return EvaluateCore._create_simple_table(segment_kpis)
                
        except Exception as e:
            logger.error(f"Failed to create segment table: {e}")
            return segment_kpis
    
    @staticmethod
    def _create_emoji_table(df: pd.DataFrame) -> pd.DataFrame:
        """Create table with emoji bars for visual representation."""
        df_display = df.copy()
        
        # Add emoji bars for each metric
        for col in ['Pct_Users', 'Pct_Revenue', 'Pct_Transactions', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']:
            if col in df.columns:
                max_val = df[col].max()
                if max_val > 0:
                    df_display[f'{col}_bar'] = df[col].apply(
                        lambda x: '█' * int((x / max_val) * 10) + '░' * (10 - int((x / max_val) * 10))
                    )
        
        return df_display
    
    @staticmethod
    def _create_bar_table(df: pd.DataFrame) -> pd.DataFrame:
        """Create table with text bars."""
        df_display = df.copy()
        
        for col in ['Pct_Users', 'Pct_Revenue', 'Pct_Transactions']:
            if col in df.columns:
                max_val = df[col].max()
                if max_val > 0:
                    df_display[f'{col}_bar'] = df[col].apply(
                        lambda x: '|' * int((x / max_val) * 20) + '-' * (20 - int((x / max_val) * 20))
                    )
        
        return df_display
    
    @staticmethod
    def _create_color_table(df: pd.DataFrame) -> pd.DataFrame:
        """Create table with color coding."""
        df_display = df.copy()
        
        # Add color indicators
        for col in ['Pct_Users', 'Pct_Revenue', 'Pct_Transactions']:
            if col in df.columns:
                df_display[f'{col}_color'] = df[col].apply(
                    lambda x: '🟢' if x >= 20 else '🟡' if x >= 10 else '🔴'
                )
        
        return df_display
    
    @staticmethod
    def _create_simple_table(df: pd.DataFrame) -> pd.DataFrame:
        """Create simple formatted table."""
        df_display = df.copy()
        
        # Format percentage columns
        for col in ['Pct_Users', 'Pct_Revenue', 'Pct_Transactions']:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(2)
        
        return df_display
    
    @staticmethod
    def create_comparison_chart(
        rule_data: pd.DataFrame, 
        kmeans_data: pd.DataFrame,
        metric: str = "revenue"
    ) -> alt.Chart:
        """
        Create comparison chart between rule-based and K-means segmentation.
        
        Args:
            rule_data: Rule-based segmentation data
            kmeans_data: K-means clustering data
            metric: Metric to compare ('revenue', 'users', 'frequency')
            
        Returns:
            Altair comparison chart
        """
        try:
            # Prepare data for comparison
            rule_summary = rule_data.groupby('Segment')[metric].sum().reset_index()
            rule_summary['Method'] = 'Rule-based'
            rule_summary = rule_summary.rename(columns={'Segment': 'Segment_Name'})
            
            kmeans_summary = kmeans_data.groupby('Cluster_Name')[metric].sum().reset_index()
            kmeans_summary['Method'] = 'K-means'
            kmeans_summary = kmeans_summary.rename(columns={'Cluster_Name': 'Segment_Name'})
            
            # Combine data
            comparison_data = pd.concat([rule_summary, kmeans_summary], ignore_index=True)
            
            # Create chart
            chart = (
                alt.Chart(comparison_data)
                .mark_bar(opacity=0.8)
                .encode(
                    x=alt.X('Segment_Name:N', title='Segment'),
                    y=alt.Y(f'{metric}:Q', title=f'{metric.title()}'),
                    color=alt.Color('Method:N', title='Segmentation Method'),
                    tooltip=[
                        alt.Tooltip('Segment_Name:N', title='Segment'),
                        alt.Tooltip('Method:N', title='Method'),
                        alt.Tooltip(f'{metric}:Q', title=f'{metric.title()}', format=',.0f')
                    ]
                )
                .properties(
                    width=600,
                    height=400,
                    title=f'{metric.title()} Comparison: Rule-based vs K-means'
                )
            )
            
            logger.info(f"Successfully created comparison chart for {metric}")
            return chart
            
        except Exception as e:
            logger.error(f"Failed to create comparison chart: {e}")
            return alt.Chart(pd.DataFrame()).mark_text(text="Error creating comparison chart")
    
    @staticmethod
    def plot_segment_distribution(df_rfm: pd.DataFrame):
        """Plot distribution of rule-based customer segments and return fig."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(
            data=df_rfm,
            x="Segment",
            order=df_rfm["Segment"].value_counts().index,
            palette="Set2",
            ax=ax
        )
        ax.set_title("Customer Segment Distribution", fontsize=14, weight="bold")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=45)
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_cluster_boxplots(df_rfm: pd.DataFrame, cluster_col: str):
        """Boxplots for Recency, Frequency, Monetary by cluster."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        rfm_cols = ['Recency', 'Frequency', 'Monetary']
        
        for i, col in enumerate(rfm_cols):
            sns.boxplot(data=df_rfm, x=cluster_col, y=col, ax=axes[i])
            axes[i].set_title(f'{col} by {cluster_col}')
            axes[i].tick_params(axis='x', rotation=45)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_pairplot(df_rfm: pd.DataFrame, cluster_col: str):
        """Pairplot for RFM metrics colored by cluster."""
        import seaborn as sns
        
        fig = sns.pairplot(df_rfm, hue=cluster_col, vars=['Recency', 'Frequency', 'Monetary'])
        return fig
    
    @staticmethod
    def plot_cluster_treemap(df_rfm: pd.DataFrame, cluster_col: str):
        """Create treemap visualization for cluster distribution with detailed RFM information."""
        import matplotlib.pyplot as plt
        import squarify
        
        # Calculate cluster summary with RFM statistics
        cluster_summary = df_rfm.groupby(cluster_col).agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
            Count=(cluster_col, "count")
        ).reset_index()
        
        total_customers = cluster_summary["Count"].sum()
        cluster_summary["Percent"] = 100 * cluster_summary["Count"] / total_customers
        
        # Create detailed labels with all information
        labels = [
            (f"CLUSTER {row[cluster_col]}\n"
             f"{int(row.Recency)} days\n"
             f"{int(row.Frequency)} orders\n"
             f"{int(row.Monetary)} $\n"
             f"{row.Count} customers ({row.Percent:.2f}%)")
            for _, row in cluster_summary.iterrows()
        ]

        fig, ax = plt.subplots(figsize=(12, 6))
        squarify.plot(
            sizes=cluster_summary["Count"],
            label=labels,
            alpha=0.8,
            color=plt.cm.Set3(range(len(cluster_summary))),
            text_kwargs={"fontsize": 10, "weight": "bold"},
            ax=ax
        )
        ax.set_title(f"{len(cluster_summary)} Clusters – Treemap", fontsize=16, weight="bold")
        ax.axis('off')
        fig.tight_layout()
        return fig