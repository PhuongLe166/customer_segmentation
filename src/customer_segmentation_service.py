# src/customer_segmentation_service.py
"""
Customer Segmentation Service - Unified service for managing the entire
customer segmentation workflow.

This service provides a high-level interface that orchestrates all core modules
to deliver a complete customer segmentation solution.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import streamlit as st
import logging
from datetime import datetime

from .eda_core import EDACore
from .preprocess_core import PreprocessCore
from .build_model_core import BuildModelCore
from .evaluate_core import EvaluateCore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerSegmentationService:
    """
    Unified service for customer segmentation workflow management.
    
    This service provides a high-level interface that orchestrates all core modules
    to deliver a complete customer segmentation solution with enhanced error handling,
    caching, and performance optimization.
    """
    
    def __init__(self):
        """Initialize the service with core modules."""
        self.eda_core = EDACore()
        self.preprocess_core = PreprocessCore()
        self.build_model_core = BuildModelCore()
        self.evaluate_core = EvaluateCore()
        
        # Cache for storing intermediate results
        self._cache = {}
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Customer Segmentation Service initialized with session ID: {self._session_id}")
    
    def load_and_prepare_data(
        self, 
        transactions_file: Optional[str] = None, 
        products_file: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Load and prepare data for customer segmentation analysis.
        
        Args:
            transactions_file: Path to transactions file
            products_file: Path to products file
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with prepared data and metadata
        """
        cache_key = f"data_preparation_{hash(str(transactions_file) + str(products_file))}"
        
        if use_cache and cache_key in self._cache:
            logger.info("Using cached data preparation results")
            return self._cache[cache_key]
        
        try:
            # Load and validate data
            df_tx, df_pd, src_tx, src_pd = self.eda_core.load_and_validate_data(
                transactions_file, products_file
            )
            
            # Merge datasets
            left_key, right_key = self.eda_core.infer_join_keys(df_tx, df_pd)
            merged = self.eda_core.merge_datasets(df_tx, df_pd, left_key, right_key)
            
            # Prepare RFM data
            merged_rfm = self.eda_core.prepare_rfm_data(merged)
            
            # Get data summary
            data_summary = self.eda_core.get_data_summary(merged)
            
            result = {
                'transactions_df': df_tx,
                'products_df': df_pd,
                'merged_df': merged,
                'merged_rfm_df': merged_rfm,
                'data_summary': data_summary,
                'sources': {'transactions': src_tx, 'products': src_pd},
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache results
            if use_cache:
                self._cache[cache_key] = result
            
            logger.info("Data preparation completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Data preparation failed: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def perform_rfm_analysis(
        self, 
        merged_rfm_df: pd.DataFrame,
        scoring_method: str = "quantile",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Perform complete RFM analysis including metrics, scoring, and segmentation.
        
        Args:
            merged_rfm_df: Prepared RFM data
            scoring_method: RFM scoring method
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with RFM analysis results
        """
        cache_key = f"rfm_analysis_{hash(str(merged_rfm_df.shape) + scoring_method)}"
        
        if use_cache and cache_key in self._cache:
            logger.info("Using cached RFM analysis results")
            return self._cache[cache_key]
        
        try:
            # Compute RFM metrics
            rfm = self.preprocess_core.compute_rfm_metrics(merged_rfm_df)
            
            # Calculate RFM scores
            rfm = self.preprocess_core.calculate_rfm_scores(rfm, method=scoring_method)
            
            # Apply rule-based segmentation
            rfm = self.preprocess_core.segment_customers_rule_based(rfm)
            
            # Validate RFM data
            validation_report = self.preprocess_core.validate_rfm_data(rfm)
            
            result = {
                'rfm_df': rfm,
                'validation_report': validation_report,
                'scoring_method': scoring_method,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache results
            if use_cache:
                self._cache[cache_key] = result
            
            logger.info("RFM analysis completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"RFM analysis failed: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def perform_kmeans_clustering(
        self, 
        rfm_df: pd.DataFrame,
        n_clusters: int = 4,
        scaler_type: str = "standard",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Perform K-Means clustering analysis.
        
        Args:
            rfm_df: RFM DataFrame
            n_clusters: Number of clusters
            scaler_type: Type of scaler to use
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with clustering results
        """
        cache_key = f"kmeans_{hash(str(rfm_df.shape) + str(n_clusters) + scaler_type)}"
        
        if use_cache and cache_key in self._cache:
            logger.info("Using cached K-Means clustering results")
            return self._cache[cache_key]
        
        try:
            # Perform clustering
            rfm_clustered, clustering_metrics = self.build_model_core.perform_kmeans_clustering(
                rfm_df, n_clusters=n_clusters, scaler_type=scaler_type
            )
            
            # Create cluster profiles
            cluster_profiles = self.build_model_core.create_cluster_profiles(rfm_clustered)
            
            result = {
                'rfm_clustered_df': rfm_clustered,
                'clustering_metrics': clustering_metrics,
                'cluster_profiles': cluster_profiles,
                'n_clusters': n_clusters,
                'scaler_type': scaler_type,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache results
            if use_cache:
                self._cache[cache_key] = result
            
            logger.info(f"K-Means clustering completed successfully with {n_clusters} clusters")
            return result
            
        except Exception as e:
            error_msg = f"K-Means clustering failed: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def evaluate_clustering_quality(
        self, 
        rfm_df: pd.DataFrame,
        k_range: List[int] = None,
        scaler_type: str = "standard"
    ) -> Dict[str, Any]:
        """
        Evaluate clustering quality across different k values.
        
        Args:
            rfm_df: RFM DataFrame
            k_range: Range of k values to test
            scaler_type: Type of scaler to use
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            evaluation_results = self.build_model_core.evaluate_clustering_quality(
                rfm_df, k_range=k_range, scaler_type=scaler_type
            )
            
            # Get optimal k recommendation
            optimal_k = self.build_model_core.get_optimal_clusters(evaluation_results)
            
            result = {
                'evaluation_results': evaluation_results,
                'optimal_k': optimal_k,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Clustering quality evaluation completed. Optimal k: {optimal_k}")
            return result
            
        except Exception as e:
            error_msg = f"Clustering quality evaluation failed: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_kpis(
        self, 
        merged_df: pd.DataFrame, 
        rfm_df: pd.DataFrame,
        kpi_type: str = "basic"
    ) -> Dict[str, Any]:
        """
        Calculate Key Performance Indicators.
        
        Args:
            merged_df: Merged transaction data
            rfm_df: RFM analysis results
            kpi_type: Type of KPIs to calculate ('basic', 'advanced')
            
        Returns:
            Dictionary with KPI results
        """
        try:
            if kpi_type == "basic":
                kpi_data = self.preprocess_core.calculate_basic_kpis(merged_df, rfm_df)
            else:
                # For advanced KPIs, use basic as base and extend
                kpi_data = self.preprocess_core.calculate_basic_kpis(merged_df, rfm_df)
                # Add additional advanced metrics here if needed
            
            # Calculate segment KPIs
            segment_kpis = self.preprocess_core.calculate_segment_kpis(rfm_df)
            
            result = {
                'kpi_data': kpi_data,
                'segment_kpis': segment_kpis,
                'kpi_type': kpi_type,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("KPI calculation completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"KPI calculation failed: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def create_visualizations(
        self, 
        data: Dict[str, Any],
        visualization_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Create comprehensive visualizations for the analysis.
        
        Args:
            data: Dictionary containing all analysis data
            visualization_type: Type of visualizations to create
            
        Returns:
            Dictionary with visualization objects
        """
        try:
            visualizations = {}
            
            # KPI Cards
            if 'kpi_data' in data:
                kpi_cards = self.evaluate_core.create_advanced_kpi_cards(data['kpi_data'])
                visualizations['kpi_cards'] = kpi_cards
            
            # Revenue Trend Charts
            if 'merged_df' in data:
                trend_df = self.eda_core.calculate_revenue_trends(data['merged_df'])
                trend_with_growth = self.eda_core.calculate_growth_metrics(trend_df)
                
                revenue_chart = self.evaluate_core.create_revenue_trend_chart(trend_with_growth)
                growth_charts = self.evaluate_core.create_growth_charts(trend_with_growth)
                
                visualizations['revenue_trend_chart'] = revenue_chart
                visualizations['growth_charts'] = growth_charts
            
            # Cluster Visualizations
            if 'rfm_clustered_df' in data:
                cluster_viz = self.evaluate_core.create_cluster_visualization(
                    data['rfm_clustered_df'], chart_type="bubble"
                )
                scatter_viz = self.evaluate_core.create_cluster_visualization(
                    data['rfm_clustered_df'], chart_type="scatter"
                )
                
                visualizations['cluster_bubble_chart'] = cluster_viz
                visualizations['cluster_scatter_chart'] = scatter_viz
            
            # Segment Tables
            if 'segment_kpis' in data:
                segment_table = self.evaluate_core.create_segment_table(
                    data['segment_kpis'], style="emoji"
                )
                visualizations['segment_table'] = segment_table
            
            result = {
                'visualizations': visualizations,
                'visualization_type': visualization_type,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Visualization creation completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Visualization creation failed: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def compare_segmentation_methods(
        self, 
        rfm_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compare rule-based vs K-Means segmentation methods.
        
        Args:
            rfm_df: DataFrame with both segmentation methods
            
        Returns:
            Dictionary with comparison results
        """
        try:
            comparison_results = self.build_model_core.compare_segmentation_methods(rfm_df)
            
            # Create comparison visualizations
            comparison_chart = self.evaluate_core.create_comparison_chart(
                rfm_df, rfm_df, metric="Monetary"
            )
            
            result = {
                'comparison_results': comparison_results,
                'comparison_chart': comparison_chart,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Segmentation method comparison completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Segmentation method comparison failed: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_complete_analysis(
        self, 
        transactions_file: Optional[str] = None, 
        products_file: Optional[str] = None,
        n_clusters: int = 4,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete customer segmentation analysis workflow.
        
        Args:
            transactions_file: Path to transactions file
            products_file: Path to products file
            n_clusters: Number of clusters for K-Means
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with complete analysis results
        """
        try:
            logger.info("Starting complete customer segmentation analysis")
            
            # Step 1: Load and prepare data
            data_prep = self.load_and_prepare_data(transactions_file, products_file, use_cache)
            if data_prep['status'] != 'success':
                return data_prep
            
            # Step 2: Perform RFM analysis
            rfm_analysis = self.perform_rfm_analysis(
                data_prep['merged_rfm_df'], use_cache=use_cache
            )
            if rfm_analysis['status'] != 'success':
                return rfm_analysis
            
            # Step 3: Perform K-Means clustering
            clustering = self.perform_kmeans_clustering(
                rfm_analysis['rfm_df'], n_clusters=n_clusters, use_cache=use_cache
            )
            if clustering['status'] != 'success':
                return clustering
            
            # Step 4: Calculate KPIs
            kpis = self.calculate_kpis(
                data_prep['merged_df'], rfm_analysis['rfm_df']
            )
            if kpis['status'] != 'success':
                return kpis
            
            # Step 5: Create visualizations
            visualizations = self.create_visualizations({
                'merged_df': data_prep['merged_df'],
                'rfm_df': rfm_analysis['rfm_df'],
                'rfm_clustered_df': clustering['rfm_clustered_df'],
                'kpi_data': kpis['kpi_data'],
                'segment_kpis': kpis['segment_kpis']
            })
            
            # Step 6: Compare segmentation methods
            comparison = self.compare_segmentation_methods(clustering['rfm_clustered_df'])
            
            # Compile complete results
            complete_results = {
                'data_preparation': data_prep,
                'rfm_analysis': rfm_analysis,
                'clustering': clustering,
                'kpis': kpis,
                'visualizations': visualizations,
                'comparison': comparison,
                'session_id': self._session_id,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Complete customer segmentation analysis finished successfully")
            return complete_results
            
        except Exception as e:
            error_msg = f"Complete analysis failed: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'session_id': self._session_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached results."""
        return {
            'cache_size': len(self._cache),
            'cache_keys': list(self._cache.keys()),
            'session_id': self._session_id
        }
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def export_results(self, results: Dict[str, Any], format: str = "json") -> str:
        """
        Export analysis results in specified format.
        
        Args:
            results: Analysis results dictionary
            format: Export format ('json', 'csv', 'excel')
            
        Returns:
            Path to exported file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format == "json":
                import json
                filename = f"customer_segmentation_results_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                    
            elif format == "csv":
                # Export key DataFrames as CSV
                filename = f"customer_segmentation_results_{timestamp}.zip"
                import zipfile
                with zipfile.ZipFile(filename, 'w') as zf:
                    for key, value in results.items():
                        if isinstance(value, dict) and 'df' in key:
                            csv_name = f"{key}.csv"
                            value.to_csv(csv_name, index=False)
                            zf.write(csv_name)
                            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Results exported to {filename}")
            return filename
            
        except Exception as e:
            error_msg = f"Export failed: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
