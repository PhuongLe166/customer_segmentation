"""
Form Components - Reusable form and input components
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

class FormComponents:
    """Form components for user inputs and data uploads."""
    
    @staticmethod
    def render_file_uploaders() -> Tuple[Optional[str], Optional[str]]:
        """
        Render file uploaders for transactions and products data.
        
        Returns:
            Tuple of (transactions_file, products_file) paths
        """
        st.markdown("### Data Upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transactions_file = st.file_uploader(
                "Upload Transactions CSV",
                type=['csv'],
                help="Upload your transactions data file"
            )
        
        with col2:
            products_file = st.file_uploader(
                "Upload Products CSV", 
                type=['csv'],
                help="Upload your products data file"
            )
        
        # Store in session state
        if transactions_file is not None:
            st.session_state.upload_transactions = transactions_file
        if products_file is not None:
            st.session_state.upload_products = products_file
        
        return transactions_file, products_file
    
    @staticmethod
    def render_date_range_selector(df: pd.DataFrame, date_col: str = "Date") -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Render date range selector for data filtering.
        
        Args:
            df: DataFrame with date data
            date_col: Name of the date column
            
        Returns:
            Tuple of (start_date, end_date)
        """
        if df.empty or date_col not in df.columns:
            return None, None
        
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        
        if df.empty:
            return None, None
        
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        
        st.markdown("#### Date Range Filter")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", 
                value=min_date, 
                min_value=min_date, 
                max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=max_date, 
                min_value=min_date, 
                max_value=max_date
            )
        
        return pd.Timestamp(start_date), pd.Timestamp(end_date)
    
    @staticmethod
    def render_granularity_selector() -> str:
        """
        Render granularity selector for time-based analysis.
        
        Returns:
            Selected granularity
        """
        granularity = st.selectbox(
            "Aggregate by",
            ["Date", "Week", "Month", "Year"],
            index=2,  # Default to Month
            help="Select the time granularity for analysis"
        )
        return granularity
    
    @staticmethod
    def render_cluster_selector(min_clusters: int = 2, max_clusters: int = 10, default: int = 4) -> int:
        """
        Render cluster number selector for K-Means clustering.
        
        Args:
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
            default: Default number of clusters
            
        Returns:
            Selected number of clusters
        """
        k = st.slider(
            "Select number of clusters (k)",
            min_value=min_clusters,
            max_value=max_clusters,
            value=default,
            help="Choose the number of clusters for K-Means clustering"
        )
        return k
    
    @staticmethod
    def render_chart_type_selector(chart_types: List[str], default: str = "line") -> str:
        """
        Render chart type selector.
        
        Args:
            chart_types: List of available chart types
            default: Default chart type
            
        Returns:
            Selected chart type
        """
        chart_type = st.selectbox(
            "Chart Type",
            chart_types,
            index=chart_types.index(default) if default in chart_types else 0,
            help="Select the type of chart to display"
        )
        return chart_type
    
    @staticmethod
    def render_top_n_selector(max_items: int, default: int = 10, label: str = "Number of Top Items") -> int:
        """
        Render top N items selector.
        
        Args:
            max_items: Maximum number of items
            default: Default number of items
            label: Label for the selector
            
        Returns:
            Selected number of items
        """
        top_n = st.slider(
            label,
            min_value=1,
            max_value=min(max_items, 50),  # Cap at 50 for performance
            value=default,
            help=f"Select the number of top items to display (max: {max_items})"
        )
        return top_n
    
    @staticmethod
    def render_analysis_type_selector() -> str:
        """
        Render analysis type selector for category/product analysis.
        
        Returns:
            Selected analysis type
        """
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Category Analysis", "Product Analysis", "Category vs Product Comparison"],
            index=0,
            help="Select the type of analysis to perform"
        )
        return analysis_type
    
    @staticmethod
    def render_visualization_selector(visualizations: List[str], default: str = "None") -> str:
        """
        Render visualization selector for additional charts.
        
        Args:
            visualizations: List of available visualizations
            default: Default visualization
            
        Returns:
            Selected visualization
        """
        viz_option = st.selectbox(
            "Additional Visualization",
            visualizations,
            index=visualizations.index(default) if default in visualizations else 0,
            help="Select an additional visualization to display"
        )
        return viz_option
    
    @staticmethod
    def render_bins_selector(default: int = 20) -> int:
        """
        Render bins selector for histograms.
        
        Args:
            default: Default number of bins
            
        Returns:
            Selected number of bins
        """
        bins = st.slider(
            "Number of Bins",
            min_value=5,
            max_value=60,
            value=default,
            help="Select the number of bins for histogram display"
        )
        return bins
    
    @staticmethod
    def render_expander_section(title: str, content: str) -> None:
        """
        Render expandable section with content.
        
        Args:
            title: Title of the expander
            content: Content to display in the expander
        """
        with st.expander(title):
            st.markdown(content)
    
    @staticmethod
    def render_info_section(message: str, message_type: str = "info") -> None:
        """
        Render info section with different message types.
        
        Args:
            message: Message to display
            message_type: Type of message ('info', 'success', 'warning', 'error')
        """
        if message_type == "info":
            st.info(message)
        elif message_type == "success":
            st.success(message)
        elif message_type == "warning":
            st.warning(message)
        elif message_type == "error":
            st.error(message)
        else:
            st.info(message)  # Default to info
    
    @staticmethod
    def render_metric_cards(metrics: Dict[str, Any], columns: int = 3) -> None:
        """
        Render metric cards in a grid layout.
        
        Args:
            metrics: Dictionary with metric values
            columns: Number of columns for the grid
        """
        cols = st.columns(columns)
        
        for i, (title, value) in enumerate(metrics.items()):
            with cols[i % columns]:
                st.metric(
                    label=title,
                    value=value
                )
    
    @staticmethod
    def render_date_slicer(df: pd.DataFrame, date_col: str = "Date") -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Render date range slicer for filtering data.
        
        Args:
            df: DataFrame with date data
            date_col: Name of the date column
            
        Returns:
            Tuple of (start_date, end_date)
        """
        if df.empty or date_col not in df.columns:
            return None, None
        
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        
        if df.empty:
            return None, None
        
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        
        st.markdown("**Date:**")
        date_range = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        
        return pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    
    @staticmethod
    def render_compare_slicer() -> str:
        """
        Render compare period slicer.
        
        Returns:
            Selected compare option
        """
        st.markdown("**Compare:**")
        compare_option = st.selectbox(
            "Compare with",
            ["Previous period", "Previous year", "None"],
            index=0,
            label_visibility="collapsed"
        )
        return compare_option
    
    @staticmethod
    def render_segment_slicer(rfm_df: pd.DataFrame) -> str:
        """
        Render segment slicer for filtering by customer segments.
        
        Args:
            rfm_df: DataFrame with RFM and segment data
            
        Returns:
            Selected segment
        """
        st.markdown("**Segments:**")
        
        # Get available segments (prioritize K-Means cluster names)
        if 'Cluster_Name' in rfm_df.columns:
            available_segments = rfm_df['Cluster_Name'].unique().tolist()
        elif 'Segment' in rfm_df.columns:
            available_segments = rfm_df['Segment'].unique().tolist()
        else:
            # Create default K-Means cluster segments
            available_segments = ['VIPs', 'Regulars', 'Potential Loyalists', 'At-Risk']
        
        # Add "All Segments" option at the beginning
        all_segments = ['All Segments'] + available_segments
        
        # Create dropdown for segments
        selected_segment = st.selectbox(
            "Select Segment",
            options=all_segments,
            index=0,  # Default to "All Segments"
            label_visibility="collapsed"
        )
        
        return selected_segment
    
    @staticmethod
    def render_reset_button() -> bool:
        """
        Render reset button for clearing all filters.
        
        Returns:
            True if reset button was clicked
        """
        return st.button("Reset", type="secondary")
    
    @staticmethod
    def render_simulate_anomaly_button() -> bool:
        """
        Render simulate anomaly button.
        
        Returns:
            True if simulate anomaly button was clicked
        """
        return st.button("Simulate anomaly", type="primary")