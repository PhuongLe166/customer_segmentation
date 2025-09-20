# src/build_model_core.py
"""
Build Model Core module - Contains all machine learning model building logic
for customer segmentation application.

This module provides enhanced clustering capabilities with improved algorithms,
validation, and model selection techniques.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import streamlit as st
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class BuildModelCore:
    """Core model building operations for customer segmentation with enhanced algorithms."""
    
    @staticmethod
    def perform_kmeans_clustering(
        rfm_df: pd.DataFrame, 
        n_clusters: int = 4,
        scaler_type: str = "standard",
        random_state: int = 42,
        n_init: int = 10
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform K-Means clustering on RFM data with enhanced configuration.
        
        Args:
            rfm_df: DataFrame with RFM metrics
            n_clusters: Number of clusters
            scaler_type: Type of scaler ('standard', 'robust', 'minmax')
            random_state: Random state for reproducibility
            n_init: Number of initializations
            
        Returns:
            Tuple of (DataFrame with clusters, clustering metrics)
        """
        try:
            # Validate input
            required_cols = ['Recency', 'Frequency', 'Monetary']
            missing_cols = [col for col in required_cols if col not in rfm_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required RFM columns: {missing_cols}")
            
            if len(rfm_df) < n_clusters:
                raise ValueError(f"Not enough data points ({len(rfm_df)}) for {n_clusters} clusters")
            
            # Prepare features
            rfm_features = rfm_df[required_cols].copy()
            
            # Handle outliers before scaling
            rfm_features = BuildModelCore._handle_outliers(rfm_features)
            
            # Choose scaler
            scaler = BuildModelCore._get_scaler(scaler_type)
            rfm_scaled = scaler.fit_transform(rfm_features)
            
            # Apply K-Means with enhanced parameters
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=random_state, 
                n_init=n_init,
                max_iter=300,
                algorithm='lloyd'
            )
            rfm_df = rfm_df.copy()
            rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
            
            # Calculate comprehensive metrics
            metrics = BuildModelCore._calculate_clustering_metrics(rfm_scaled, rfm_df['Cluster'])
            
            # Assign meaningful cluster names
            cluster_names = BuildModelCore._assign_cluster_names(rfm_df, n_clusters)
            rfm_df['Cluster_Name'] = rfm_df['Cluster'].map(cluster_names)
            
            # Add cluster characteristics
            rfm_df = BuildModelCore._add_cluster_characteristics(rfm_df)
            
            # Store model artifacts
            metrics.update({
                'scaler': scaler,
                'kmeans_model': kmeans,
                'cluster_names': cluster_names,
                'rfm_scaled': rfm_scaled,
                'n_clusters': n_clusters
            })
            
            logger.info(f"Successfully performed K-Means clustering with {n_clusters} clusters")
            return rfm_df, metrics
            
        except Exception as e:
            error_msg = f"Failed to perform K-Means clustering: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def _handle_outliers(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """Handle outliers in RFM data."""
        df_clean = df.copy()
        
        if method == "iqr":
            for col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    
    @staticmethod
    def _get_scaler(scaler_type: str):
        """Get appropriate scaler based on type."""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        return scalers.get(scaler_type, StandardScaler())
    
    @staticmethod
    def _calculate_clustering_metrics(rfm_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive clustering metrics."""
        try:
            metrics = {
                'silhouette_score': silhouette_score(rfm_scaled, labels),
                'davies_bouldin_score': davies_bouldin_score(rfm_scaled, labels),
                'calinski_harabasz_score': calinski_harabasz_score(rfm_scaled, labels)
            }
            
            # Calculate additional metrics
            n_clusters = len(np.unique(labels))
            n_samples = len(labels)
            
            metrics.update({
                'n_clusters': n_clusters,
                'n_samples': n_samples,
                'avg_cluster_size': n_samples / n_clusters,
                'cluster_balance': 1 - (np.std(np.bincount(labels)) / np.mean(np.bincount(labels)))
            })
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to calculate some clustering metrics: {e}")
            return {'silhouette_score': 0, 'davies_bouldin_score': float('inf')}
    
    @staticmethod
    def _assign_cluster_names(rfm_df: pd.DataFrame, n_clusters: int) -> Dict[int, str]:
        """Assign meaningful names to clusters based on RFM characteristics."""
        cluster_summary = rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean', 
            'Monetary': 'mean'
        }).round(2)
        cluster_summary['Customer_Count'] = rfm_df.groupby('Cluster').size()
        
        # Sort by composite score (lower recency, higher frequency and monetary is better)
        cluster_summary['Composite_Score'] = (
            -cluster_summary['Recency'] / cluster_summary['Recency'].max() +
            cluster_summary['Frequency'] / cluster_summary['Frequency'].max() +
            cluster_summary['Monetary'] / cluster_summary['Monetary'].max()
        )
        
        cluster_summary_sorted = cluster_summary.sort_values('Composite_Score', ascending=False)
        
        # Assign names based on ranking
        cluster_names = {}
        name_templates = ['VIPs', 'Regulars', 'Potential Loyalists', 'At-Risk', 'New Customers', 'Hibernating']
        
        for i, (cluster_id, _) in enumerate(cluster_summary_sorted.iterrows()):
            if i < len(name_templates):
                cluster_names[cluster_id] = name_templates[i]
            else:
                cluster_names[cluster_id] = f'Cluster_{cluster_id}'
        
        return cluster_names
    
    @staticmethod
    def _add_cluster_characteristics(rfm_df: pd.DataFrame) -> pd.DataFrame:
        """Add cluster characteristics to the DataFrame."""
        rfm_df = rfm_df.copy()
        
        # Add cluster centroids
        cluster_centroids = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        
        for col in ['Recency', 'Frequency', 'Monetary']:
            rfm_df[f'{col}_Centroid'] = rfm_df['Cluster'].map(cluster_centroids[col])
            rfm_df[f'{col}_Distance'] = abs(rfm_df[col] - rfm_df[f'{col}_Centroid'])
        
        # Add cluster size
        cluster_sizes = rfm_df['Cluster'].value_counts()
        rfm_df['Cluster_Size'] = rfm_df['Cluster'].map(cluster_sizes)
        
        return rfm_df
    
    @staticmethod
    def evaluate_clustering_quality(
        rfm_df: pd.DataFrame, 
        k_range: List[int] = None,
        scaler_type: str = "standard"
    ) -> Dict[str, Any]:
        """
        Evaluate clustering quality across different k values with enhanced metrics.
        
        Args:
            rfm_df: DataFrame with RFM metrics
            k_range: Range of k values to test
            scaler_type: Type of scaler to use
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if k_range is None:
                k_range = list(range(2, min(11, len(rfm_df) // 2)))
            
            # Prepare features
            rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
            rfm_features = BuildModelCore._handle_outliers(rfm_features)
            
            scaler = BuildModelCore._get_scaler(scaler_type)
            rfm_scaled = scaler.fit_transform(rfm_features)
            
            # Evaluate different k values
            inertias = []
            silhouettes = []
            davies_bouldin = []
            calinski_harabasz = []
            
            for k in k_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(rfm_scaled)
                    
                    inertias.append(kmeans.inertia_)
                    silhouettes.append(silhouette_score(rfm_scaled, labels))
                    davies_bouldin.append(davies_bouldin_score(rfm_scaled, labels))
                    calinski_harabasz.append(calinski_harabasz_score(rfm_scaled, labels))
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate k={k}: {e}")
                    inertias.append(np.nan)
                    silhouettes.append(np.nan)
                    davies_bouldin.append(np.nan)
                    calinski_harabasz.append(np.nan)
            
            # Find optimal k using multiple criteria
            optimal_k_silhouette = k_range[np.nanargmax(silhouettes)] if not all(np.isnan(silhouettes)) else k_range[0]
            optimal_k_davies = k_range[np.nanargmin(davies_bouldin)] if not all(np.isnan(davies_bouldin)) else k_range[0]
            
            return {
                'k_range': k_range,
                'inertias': inertias,
                'silhouettes': silhouettes,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski_harabasz,
                'rfm_scaled': rfm_scaled,
                'optimal_k_silhouette': optimal_k_silhouette,
                'optimal_k_davies': optimal_k_davies,
                'recommended_k': optimal_k_silhouette  # Use silhouette as primary criterion
            }
            
        except Exception as e:
            error_msg = f"Failed to evaluate clustering quality: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def get_optimal_clusters(evaluation_results: Dict[str, Any], method: str = "silhouette") -> int:
        """
        Determine optimal number of clusters based on evaluation results.
        
        Args:
            evaluation_results: Results from evaluate_clustering_quality
            method: Method to use ('silhouette', 'davies', 'elbow', 'combined')
            
        Returns:
            Optimal number of clusters
        """
        try:
            k_range = evaluation_results['k_range']
            
            if method == "silhouette":
                silhouettes = evaluation_results['silhouettes']
                optimal_k = k_range[np.nanargmax(silhouettes)]
                
            elif method == "davies":
                davies_bouldin = evaluation_results['davies_bouldin']
                optimal_k = k_range[np.nanargmin(davies_bouldin)]
                
            elif method == "elbow":
                inertias = evaluation_results['inertias']
                # Find elbow point using second derivative
                if len(inertias) > 2:
                    second_deriv = np.diff(inertias, 2)
                    optimal_k = k_range[np.argmax(second_deriv) + 2]
                else:
                    optimal_k = k_range[0]
                    
            elif method == "combined":
                # Use multiple criteria
                silhouettes = evaluation_results['silhouettes']
                davies_bouldin = evaluation_results['davies_bouldin']
                
                # Normalize scores
                sil_norm = (silhouettes - np.nanmin(silhouettes)) / (np.nanmax(silhouettes) - np.nanmin(silhouettes))
                davies_norm = 1 - (davies_bouldin - np.nanmin(davies_bouldin)) / (np.nanmax(davies_bouldin) - np.nanmin(davies_bouldin))
                
                combined_scores = sil_norm + davies_norm
                optimal_k = k_range[np.nanargmax(combined_scores)]
                
            else:
                optimal_k = evaluation_results.get('recommended_k', k_range[0])
            
            logger.info(f"Optimal k determined using {method} method: {optimal_k}")
            return optimal_k
            
        except Exception as e:
            logger.warning(f"Failed to determine optimal k using {method}: {e}")
            return evaluation_results.get('recommended_k', 4)
    
    @staticmethod
    def create_cluster_profiles(rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create detailed cluster profiles with enhanced statistics.
        
        Args:
            rfm_df: DataFrame with cluster assignments
            
        Returns:
            DataFrame with cluster profiles
        """
        try:
            if 'Cluster_Name' not in rfm_df.columns:
                raise ValueError("No Cluster_Name column found. Run clustering first.")
            
            # Calculate comprehensive cluster statistics
            cluster_profiles = rfm_df.groupby('Cluster_Name').agg({
                'Recency': ['mean', 'std', 'min', 'max', 'median'],
                'Frequency': ['mean', 'std', 'min', 'max', 'median'],
                'Monetary': ['mean', 'std', 'min', 'max', 'median'],
                'Cluster': 'count'
            }).round(2)
            
            # Add quantile calculations separately
            q25_stats = rfm_df.groupby('Cluster_Name')[['Recency', 'Frequency', 'Monetary']].quantile(0.25)
            q75_stats = rfm_df.groupby('Cluster_Name')[['Recency', 'Frequency', 'Monetary']].quantile(0.75)
            
            # Flatten column names
            cluster_profiles.columns = ['_'.join(col).strip() for col in cluster_profiles.columns]
            cluster_profiles = cluster_profiles.rename(columns={'Cluster_count': 'Customer_Count'})
            
            # Add quantile columns
            for col in ['Recency', 'Frequency', 'Monetary']:
                cluster_profiles[f'{col}_q25'] = q25_stats[col]
                cluster_profiles[f'{col}_q75'] = q75_stats[col]
            
            # Add additional metrics
            cluster_profiles['Customer_Percentage'] = (
                cluster_profiles['Customer_Count'] / cluster_profiles['Customer_Count'].sum() * 100
            ).round(2)
            
            # Calculate cluster quality metrics
            cluster_profiles['Recency_CV'] = (
                cluster_profiles['Recency_std'] / cluster_profiles['Recency_mean']
            ).round(3)
            cluster_profiles['Frequency_CV'] = (
                cluster_profiles['Frequency_std'] / cluster_profiles['Frequency_mean']
            ).round(3)
            cluster_profiles['Monetary_CV'] = (
                cluster_profiles['Monetary_std'] / cluster_profiles['Monetary_mean']
            ).round(3)
            
            # Calculate total revenue per cluster
            cluster_profiles['Total_Revenue'] = (
                cluster_profiles['Customer_Count'] * cluster_profiles['Monetary_mean']
            ).round(2)
            
            result = cluster_profiles.reset_index()
            logger.info(f"Successfully created cluster profiles for {len(result)} clusters")
            return result
            
        except Exception as e:
            error_msg = f"Failed to create cluster profiles: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def predict_customer_segment(
        rfm_values: Dict[str, float], 
        model_type: str = 'kmeans', 
        model_data: Dict[str, Any] = None
    ) -> str:
        """
        Predict customer segment for new data with enhanced logic.
        
        Args:
            rfm_values: Dictionary with 'Recency', 'Frequency', 'Monetary' values
            model_type: 'kmeans' or 'rule_based'
            model_data: Model artifacts for prediction
            
        Returns:
            Predicted segment/cluster
        """
        try:
            if model_type == 'rule_based':
                return BuildModelCore._predict_rule_based(rfm_values)
                
            elif model_type == 'kmeans' and model_data:
                return BuildModelCore._predict_kmeans(rfm_values, model_data)
            
            else:
                logger.warning(f"Invalid model_type or missing model_data: {model_type}")
                return 'Unknown'
                
        except Exception as e:
            logger.error(f"Failed to predict customer segment: {e}")
            return 'Unknown'
    
    @staticmethod
    def _predict_rule_based(rfm_values: Dict[str, float]) -> str:
        """Predict using rule-based logic."""
        r, f, m = rfm_values['Recency'], rfm_values['Frequency'], rfm_values['Monetary']
        
        # Enhanced rule-based logic
        if r <= 30 and f >= 5 and m >= 1000:
            return 'A.CHAMPIONS'
        elif r <= 60 and f >= 3 and m >= 500:
            return 'B.LOYAL'
        elif r <= 30 and f <= 2 and m >= 200:
            return 'C.POTENTIAL_LOYALIST'
        elif r <= 60 and f >= 2 and m >= 200:
            return 'D.RECENT_CUSTOMERS'
        elif r > 90 and f >= 2 and m >= 500:
            return 'H.AT_RISK'
        elif r > 90 and f >= 1 and m >= 200:
            return 'I.CANNOT_LOSE'
        elif r > 90 and f <= 1 and m <= 200:
            return 'K.LOST'
        else:
            return 'F.NEED_ATTENTION'
    
    @staticmethod
    def _predict_kmeans(rfm_values: Dict[str, float], model_data: Dict[str, Any]) -> str:
        """Predict using K-Means model."""
        scaler = model_data.get('scaler')
        kmeans_model = model_data.get('kmeans_model')
        cluster_names = model_data.get('cluster_names', {})
        
        if scaler and kmeans_model:
            # Prepare input data
            input_data = np.array([[rfm_values['Recency'], rfm_values['Frequency'], rfm_values['Monetary']]])
            input_scaled = scaler.transform(input_data)
            cluster = kmeans_model.predict(input_scaled)[0]
            
            return cluster_names.get(cluster, f'Cluster_{cluster}')
        
        return 'Unknown'
    
    @staticmethod
    def compare_segmentation_methods(rfm_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare rule-based vs K-Means segmentation results with enhanced metrics.
        
        Args:
            rfm_df: DataFrame with both segmentation methods
            
        Returns:
            Dictionary with comparison metrics
        """
        try:
            # Ensure both segmentations exist
            if 'Segment' not in rfm_df.columns or 'Cluster_Name' not in rfm_df.columns:
                raise ValueError("Both rule-based segments and K-Means clusters must exist for comparison.")
            
            # Calculate segment distribution
            rule_segments = rfm_df['Segment'].value_counts()
            kmeans_clusters = rfm_df['Cluster_Name'].value_counts()
            
            # Calculate average RFM values by segment
            rule_rfm_avg = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
            kmeans_rfm_avg = rfm_df.groupby('Cluster_Name')[['Recency', 'Frequency', 'Monetary']].mean()
            
            # Calculate segment diversity
            rule_diversity = len(rule_segments)
            kmeans_diversity = len(kmeans_clusters)
            
            # Calculate segment balance
            rule_balance = 1 - (rule_segments.std() / rule_segments.mean())
            kmeans_balance = 1 - (kmeans_clusters.std() / kmeans_clusters.mean())
            
            return {
                'rule_segments': rule_segments,
                'kmeans_clusters': kmeans_clusters,
                'rule_rfm_avg': rule_rfm_avg,
                'kmeans_rfm_avg': kmeans_rfm_avg,
                'total_customers': len(rfm_df),
                'rule_diversity': rule_diversity,
                'kmeans_diversity': kmeans_diversity,
                'rule_balance': rule_balance,
                'kmeans_balance': kmeans_balance,
                'comparison_summary': {
                    'rule_method': f"{rule_diversity} segments, balance: {rule_balance:.3f}",
                    'kmeans_method': f"{kmeans_diversity} clusters, balance: {kmeans_balance:.3f}"
                }
            }
            
        except Exception as e:
            error_msg = f"Failed to compare segmentation methods: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)