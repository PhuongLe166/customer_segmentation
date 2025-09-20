"""
Table Components - Reusable table display components
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List

class TableComponents:
    """Table components for displaying various types of data tables."""
    
    @staticmethod
    def render_data_preview(df: pd.DataFrame, title: str = "Data Preview", max_rows: int = 10) -> None:
        """
        Render data preview table.
        
        Args:
            df: DataFrame to display
            title: Title for the table
            max_rows: Maximum number of rows to display
        """
        if df.empty:
            st.warning("No data available to display.")
            return
        
        st.markdown(f"#### {title}")
        st.markdown(f"<div class='stat-badges'><span class='stat'>Rows: {len(df):,}</span><span class='stat'>Columns: {df.shape[1]}</span></div>", unsafe_allow_html=True)
        st.dataframe(df.head(max_rows), width='stretch')
    
    @staticmethod
    def render_segment_table(segment_df: pd.DataFrame, title: str = "Segment Analysis") -> None:
        """
        Render segment analysis table.
        
        Args:
            segment_df: DataFrame with segment data
            title: Title for the table
        """
        if segment_df.empty:
            st.warning("No segment data available to display.")
            return
        
        st.markdown(f"#### {title}")
        
        # Determine the correct segment column name
        segment_col = 'Cluster_Name' if 'Cluster_Name' in segment_df.columns else 'Segment'
        
        # Add CSS for table header styling
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
        display_columns = [segment_col, 'Total_Users', 'Pct_Users', 'Pct_Users_bar', 
                          'Pct_Revenue', 'Pct_Revenue_bar', 'Pct_Transactions', 'Pct_Transactions_bar',
                          'Avg_Recency', 'Avg_Recency_bar', 'Avg_Frequency', 'Avg_Frequency_bar',
                          'Avg_Monetary', 'Avg_Monetary_bar']
        
        # Filter to only show columns that exist
        available_columns = [col for col in display_columns if col in segment_df.columns]
        
        if available_columns:
            st.dataframe(
                segment_df[available_columns],
                width='stretch',
                hide_index=True,
                column_config={
                    segment_col: 'Cluster Name',
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
        else:
            st.warning("No suitable columns found for segment table display.")
    
    @staticmethod
    def render_category_performance_table(category_df: pd.DataFrame, title: str = "Category Performance") -> None:
        """
        Render category performance table.
        
        Args:
            category_df: DataFrame with category performance data
            title: Title for the table
        """
        if category_df.empty:
            st.warning("No category performance data available to display.")
            return
        
        st.markdown(f"#### {title}")
        
        # Sort by total revenue descending
        sorted_df = category_df.sort_values('amount', ascending=False).head(10)
        
        st.dataframe(
            sorted_df[['category', 'amount', 'product_count', 'avg_product_revenue']].round(2),
            width='stretch',
            column_config={
                'category': 'Category',
                'amount': st.column_config.NumberColumn('Total Revenue', format="$%.2f"),
                'product_count': 'Product Count',
                'avg_product_revenue': st.column_config.NumberColumn('Avg Product Revenue', format="$%.2f")
            }
        )
    
    @staticmethod
    def render_rfm_sample_table(rfm_df: pd.DataFrame, title: str = "RFM Sample Data", max_rows: int = 5) -> None:
        """
        Render RFM sample data table.
        
        Args:
            rfm_df: DataFrame with RFM data
            title: Title for the table
            max_rows: Maximum number of rows to display
        """
        if rfm_df.empty:
            st.warning("No RFM data available to display.")
            return
        
        st.markdown(f"#### {title}")
        
        # Select relevant columns
        display_columns = ['Recency', 'Frequency', 'Monetary']
        if 'Segment' in rfm_df.columns:
            display_columns.append('Segment')
        if 'Cluster' in rfm_df.columns:
            display_columns.append('Cluster')
        if 'Cluster_Name' in rfm_df.columns:
            display_columns.append('Cluster_Name')
        
        # Filter to only show columns that exist
        available_columns = [col for col in display_columns if col in rfm_df.columns]
        
        if available_columns:
            st.dataframe(
                rfm_df[available_columns].head(max_rows),
                width='stretch',
                column_config={
                    'Recency': st.column_config.NumberColumn('Recency (days)', format="%.0f"),
                    'Frequency': st.column_config.NumberColumn('Frequency', format="%.1f"),
                    'Monetary': st.column_config.NumberColumn('Monetary ($)', format="$%.2f"),
                    'Segment': 'Segment',
                    'Cluster': 'Cluster',
                    'Cluster_Name': 'Cluster Name'
                }
            )
        else:
            st.warning("No suitable columns found for RFM table display.")
    
    @staticmethod
    def render_clustering_metrics_table(metrics: Dict[str, Any], title: str = "Clustering Metrics") -> None:
        """
        Render clustering metrics table.
        
        Args:
            metrics: Dictionary with clustering metrics
            title: Title for the table
        """
        if not metrics:
            st.warning("No clustering metrics available to display.")
            return
        
        st.markdown(f"#### {title}")
        
        # Create metrics DataFrame
        metrics_data = {
            'Metric': ['Silhouette Score', 'Davies-Bouldin Index', 'Number of Clusters'],
            'Value': [
                f"{metrics.get('silhouette_score', 0):.3f}",
                f"{metrics.get('davies_bouldin_score', 0):.3f}",
                f"{metrics.get('n_clusters', 0)}"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        st.dataframe(
            metrics_df,
            width='stretch',
            hide_index=True,
            column_config={
                'Metric': 'Clustering Metric',
                'Value': 'Value'
            }
        )
    
    @staticmethod
    def render_summary_statistics_table(df: pd.DataFrame, title: str = "Summary Statistics") -> None:
        """
        Render summary statistics table.
        
        Args:
            df: DataFrame to analyze
            title: Title for the table
        """
        if df.empty:
            st.warning("No data available for summary statistics.")
            return
        
        st.markdown(f"#### {title}")
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for summary statistics.")
            return
        
        # Calculate summary statistics
        summary_stats = df[numeric_cols].describe()
        
        st.dataframe(
            summary_stats,
            width='stretch',
            column_config={
                col: st.column_config.NumberColumn(col, format="%.2f") for col in numeric_cols
            }
        )
