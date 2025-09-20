# src/preprocess_core.py
"""
Preprocess Core module - Contains all data preprocessing logic
for customer segmentation application.

This module provides optimized data preprocessing capabilities with
enhanced performance, validation, and error handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import streamlit as st
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessCore:
    """Core preprocessing operations for data preparation with enhanced performance."""
    
    @staticmethod
    def compute_rfm_metrics(merged_rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RFM metrics from prepared data with enhanced validation.
        
        Args:
            merged_rfm: Prepared DataFrame with standardized columns
            
        Returns:
            DataFrame with RFM metrics
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            # Import hashlib for transaction key generation
            import hashlib
            
            # Validate required columns
            required_cols = ['member_number', 'date']
            missing_cols = [col for col in required_cols if col not in merged_rfm.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Validate data quality
            if merged_rfm.empty:
                raise ValueError("Input data is empty")
            
            # Ensure amount column exists
            if 'amount' not in merged_rfm.columns:
                if 'items' in merged_rfm.columns and 'price' in merged_rfm.columns:
                    merged_rfm['amount'] = merged_rfm['items'] * merged_rfm['price']
                else:
                    raise ValueError("Cannot calculate amount: missing 'amount' or 'items'/'price' columns")
            
            if merged_rfm['amount'].sum() <= 0:
                raise ValueError("Total amount must be greater than 0")
            
            # Compute RFM metrics
            rfm = PreprocessCore.compute_rfm(
                merged_rfm,
                customer_col='member_number',
                date_col='date',
                amount_col='amount'
            )
            
            if rfm.empty:
                raise ValueError("RFM computation resulted in empty dataset")
            
            # Set index and validate results
            rfm = rfm.set_index('member_number')
            
            # Validate RFM metrics
            if rfm['Recency'].min() < 0:
                logger.warning("Negative recency values found, setting to 0")
                rfm['Recency'] = rfm['Recency'].clip(lower=0)
            
            if rfm['Frequency'].min() <= 0:
                logger.warning("Non-positive frequency values found")
            
            if rfm['Monetary'].min() < 0:
                logger.warning("Negative monetary values found, setting to 0")
                rfm['Monetary'] = rfm['Monetary'].clip(lower=0)
            
            logger.info(f"Successfully computed RFM metrics for {len(rfm)} customers")
            return rfm
            
        except Exception as e:
            error_msg = f"Failed to compute RFM metrics: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def calculate_rfm_scores(rfm_df: pd.DataFrame, method: str = "quantile") -> pd.DataFrame:
        """
        Calculate R, F, M scores using different methods with enhanced validation.
        
        Args:
            rfm_df: DataFrame with RFM metrics
            method: Scoring method ('quantile', 'percentile', 'custom')
            
        Returns:
            DataFrame with RFM scores
        """
        try:
            rfm = rfm_df.copy()
            
            # Validate input
            required_cols = ['Recency', 'Frequency', 'Monetary']
            missing_cols = [col for col in required_cols if col not in rfm.columns]
            if missing_cols:
                raise ValueError(f"Missing required RFM columns: {missing_cols}")
            
            # Handle edge cases
            if len(rfm) < 5:
                logger.warning("Small dataset: using simple ranking instead of quantiles")
                method = "rank"
            
            if method == "quantile":
                # Standard quantile-based scoring
                rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
                rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
                rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
                
            elif method == "percentile":
                # Percentile-based scoring
                rfm['R_Score'] = pd.cut(rfm['Recency'], bins=5, labels=[5,4,3,2,1], duplicates='drop')
                rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=5, labels=[1,2,3,4,5], duplicates='drop')
                rfm['M_Score'] = pd.cut(rfm['Monetary'], bins=5, labels=[1,2,3,4,5], duplicates='drop')
                
            elif method == "rank":
                # Simple ranking for small datasets
                rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1], duplicates='drop')
                rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
                rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
            
            else:
                raise ValueError(f"Unsupported scoring method: {method}")
            
            # Convert to integer and handle NaN values
            rfm[['R_Score','F_Score','M_Score']] = rfm[['R_Score','F_Score','M_Score']].astype('Int64')
            
            # Fill NaN values with median scores
            for col in ['R_Score', 'F_Score', 'M_Score']:
                if rfm[col].isna().any():
                    median_score = rfm[col].median()
                    rfm[col] = rfm[col].fillna(median_score)
                    logger.warning(f"Filled NaN values in {col} with median score: {median_score}")
            
            # Create composite RFM score
            rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
            rfm['RFM_Index'] = rfm['R_Score'] * 100 + rfm['F_Score'] * 10 + rfm['M_Score']
            
            logger.info(f"Successfully calculated RFM scores using {method} method")
            return rfm
            
        except Exception as e:
            error_msg = f"Failed to calculate RFM scores: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def segment_customers_rule_based(rfm_df: pd.DataFrame, custom_rules: Optional[Dict] = None) -> pd.DataFrame:
        """
        Apply rule-based customer segmentation with enhanced flexibility.
        
        Args:
            rfm_df: DataFrame with RFM scores
            custom_rules: Optional custom segmentation rules
            
        Returns:
            DataFrame with customer segments
        """
        try:
            rfm = rfm_df.copy()
            
            # Validate required columns
            required_cols = ['R_Score', 'F_Score', 'M_Score']
            missing_cols = [col for col in required_cols if col not in rfm.columns]
            if missing_cols:
                raise ValueError(f"Missing required score columns: {missing_cols}")
            
            # Default segmentation rules
            default_rules = {
                'A.CHAMPIONS': {'R': (4, 5), 'F': (4, 5), 'M': (4, 5)},
                'B.LOYAL': {'R': (3, 5), 'F': (3, 5), 'M': (3, 5)},
                'C.POTENTIAL_LOYALIST': {'R': (4, 5), 'F': (1, 2), 'M': (1, 5)},
                'D.RECENT_CUSTOMERS': {'R': (3, 5), 'F': (2, 5), 'M': (2, 5)},
                'F.NEED_ATTENTION': {'R': (2, 5), 'F': (2, 5), 'M': (2, 5)},
                'H.AT_RISK': {'R': (1, 2), 'F': (2, 5), 'M': (2, 5)},
                'I.CANNOT_LOSE': {'R': (1, 2), 'F': (1, 2), 'M': (2, 5)},
                'K.LOST': {'R': (1, 2), 'F': (1, 2), 'M': (1, 2)},
                'J.HIBERNATING': {'R': (1, 5), 'F': (1, 5), 'M': (1, 5)}
            }
            
            rules = custom_rules if custom_rules else default_rules
            
            def segment_customers(row):
                """Apply segmentation rules to a single row."""
                r_score = row['R_Score']
                f_score = row['F_Score']
                m_score = row['M_Score']
                
                # Check each rule in order of priority
                for segment, criteria in rules.items():
                    r_range = criteria.get('R', (1, 5))
                    f_range = criteria.get('F', (1, 5))
                    m_range = criteria.get('M', (1, 5))
                    
                    if (r_range[0] <= r_score <= r_range[1] and
                        f_range[0] <= f_score <= f_range[1] and
                        m_range[0] <= m_score <= m_range[1]):
                        return segment
                
                # Default fallback
                return 'J.HIBERNATING'
            
            # Apply segmentation
            rfm['Segment'] = rfm.apply(segment_customers, axis=1)
            
            # Validate segmentation results
            segment_counts = rfm['Segment'].value_counts()
            logger.info(f"Segmentation completed. Distribution: {segment_counts.to_dict()}")
            
            return rfm
            
        except Exception as e:
            error_msg = f"Failed to segment customers: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def calculate_basic_kpis(merged_df: pd.DataFrame, rfm_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic KPIs from merged data and RFM analysis with enhanced metrics.
        
        Args:
            merged_df: Merged transaction data
            rfm_df: RFM analysis results
            
        Returns:
            Dictionary of KPI values
        """
        try:
            # Basic counts
            total_users = len(rfm_df)
            total_transactions = len(merged_df)
            
            # Calculate total revenue from items * price if amount column doesn't exist
            if 'amount' in merged_df.columns:
                total_revenue = float(merged_df['amount'].sum())
            elif 'items' in merged_df.columns and 'price' in merged_df.columns:
                total_revenue = float((merged_df['items'] * merged_df['price']).sum())
            else:
                raise ValueError("Cannot calculate revenue: missing 'amount' or 'items'/'price' columns")
            
            if total_users == 0:
                raise ValueError("No users found in RFM data")
            
            # Calculate user types using enhanced RFM-based logic
            active_users = len(rfm_df[
                (rfm_df['Recency'] <= 30) & 
                (rfm_df['Frequency'] >= rfm_df['Frequency'].quantile(0.5)) & 
                (rfm_df['Monetary'] >= rfm_df['Monetary'].quantile(0.5))
            ])
            
            at_risk_users = len(rfm_df[
                (rfm_df['Recency'] > 30) & 
                (rfm_df['Recency'] <= 90) & 
                (rfm_df['Frequency'] >= rfm_df['Frequency'].quantile(0.3)) & 
                (rfm_df['Monetary'] >= rfm_df['Monetary'].quantile(0.3))
            ])
            
            churned_users = len(rfm_df[
                (rfm_df['Recency'] > 90) | 
                (rfm_df['Frequency'] < rfm_df['Frequency'].quantile(0.3)) | 
                (rfm_df['Monetary'] < rfm_df['Monetary'].quantile(0.3))
            ])
            
            # Calculate percentages
            pct_active = (active_users / total_users) * 100 if total_users > 0 else 0
            pct_at_risk = (at_risk_users / total_users) * 100 if total_users > 0 else 0
            pct_churned = (churned_users / total_users) * 100 if total_users > 0 else 0
            
            # Calculate RFM averages
            avg_recency = float(rfm_df['Recency'].mean())
            avg_frequency = float(rfm_df['Frequency'].mean())
            avg_monetary = float(rfm_df['Monetary'].mean())
            
            # Calculate additional metrics
            median_recency = float(rfm_df['Recency'].median())
            median_frequency = float(rfm_df['Frequency'].median())
            median_monetary = float(rfm_df['Monetary'].median())
            
            # Calculate customer lifetime value (CLV) approximation
            clv_approx = avg_monetary * avg_frequency
            
            # Calculate retention metrics
            recent_customers = len(rfm_df[rfm_df['Recency'] <= 30])
            retention_rate = (recent_customers / total_users) * 100 if total_users > 0 else 0
            
            kpis = {
                'total_users': total_users,
                'total_transactions': total_transactions,
                'total_revenue': total_revenue,
                'active_users': active_users,
                'at_risk_users': at_risk_users,
                'churned_users': churned_users,
                'pct_active': pct_active,
                'pct_at_risk': pct_at_risk,
                'pct_churned': pct_churned,
                'avg_recency': avg_recency,
                'avg_frequency': avg_frequency,
                'avg_monetary': avg_monetary,
                'median_recency': median_recency,
                'median_frequency': median_frequency,
                'median_monetary': median_monetary,
                'clv_approx': clv_approx,
                'retention_rate': retention_rate,
                'avg_order_value': total_revenue / total_transactions if total_transactions > 0 else 0
            }
            
            logger.info("Successfully calculated basic KPIs")
            return kpis
            
        except Exception as e:
            error_msg = f"Failed to calculate basic KPIs: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def calculate_segment_kpis(rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate KPIs by customer segment with enhanced metrics.
        
        Args:
            rfm_df: DataFrame with customer segments
            
        Returns:
            DataFrame with segment KPIs
        """
        try:
            # Determine segment column - prioritize K-Means clustering over rule-based
            if 'Cluster_Name' in rfm_df.columns:
                segment_col = 'Cluster_Name'
            elif 'Segment' in rfm_df.columns:
                segment_col = 'Segment'
            else:
                raise ValueError("No segment column found in RFM data")
            
            # Calculate segment statistics
            seg = (
                rfm_df.groupby(segment_col)
                   .agg(
                       Total_Users=('Recency', 'size'),
                       Avg_Recency=('Recency', 'mean'),
                       Median_Recency=('Recency', 'median'),
                       Avg_Frequency=('Frequency', 'mean'),
                       Median_Frequency=('Frequency', 'median'),
                       Avg_Monetary=('Monetary', 'mean'),
                       Median_Monetary=('Monetary', 'median'),
                       Total_Revenue=('Monetary', 'sum'),
                       Min_Recency=('Recency', 'min'),
                       Max_Recency=('Recency', 'max'),
                       Min_Monetary=('Monetary', 'min'),
                       Max_Monetary=('Monetary', 'max')
                   )
                   .reset_index()
            )
            
            # Calculate totals for percentage calculations
            tot_users = seg['Total_Users'].sum()
            tot_revenue = seg['Total_Revenue'].sum()
            tot_tx = (seg['Total_Users'] * seg['Avg_Frequency']).sum()
            
            # Calculate percentages
            seg['Pct_Users'] = (seg['Total_Users'] / tot_users * 100).round(2)
            seg['Pct_Revenue'] = (seg['Total_Revenue'] / tot_revenue * 100).round(2)
            seg['Pct_Transactions'] = ((seg['Total_Users'] * seg['Avg_Frequency']) / tot_tx * 100).round(2)
            
            # Calculate additional metrics
            seg['Revenue_per_User'] = (seg['Total_Revenue'] / seg['Total_Users']).round(2)
            seg['Recency_Range'] = (seg['Max_Recency'] - seg['Min_Recency']).round(2)
            seg['Monetary_Range'] = (seg['Max_Monetary'] - seg['Min_Monetary']).round(2)
            
            # Sort by segment name for consistency
            result = seg.sort_values(segment_col).round(2)
            
            logger.info(f"Successfully calculated segment KPIs for {len(result)} segments")
            return result
            
        except Exception as e:
            error_msg = f"Failed to calculate segment KPIs: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def validate_rfm_data(rfm_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate RFM data quality and return validation report.
        
        Args:
            rfm_df: DataFrame with RFM metrics
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_report = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'statistics': {}
            }
            
            # Check required columns
            required_cols = ['Recency', 'Frequency', 'Monetary']
            missing_cols = [col for col in required_cols if col not in rfm_df.columns]
            if missing_cols:
                validation_report['errors'].append(f"Missing required columns: {missing_cols}")
                validation_report['is_valid'] = False
            
            if not validation_report['is_valid']:
                return validation_report
            
            # Check for negative values
            if (rfm_df['Recency'] < 0).any():
                validation_report['warnings'].append("Negative recency values found")
            
            if (rfm_df['Frequency'] <= 0).any():
                validation_report['warnings'].append("Non-positive frequency values found")
            
            if (rfm_df['Monetary'] < 0).any():
                validation_report['warnings'].append("Negative monetary values found")
            
            # Check for extreme outliers
            for col in required_cols:
                q99 = rfm_df[col].quantile(0.99)
                q01 = rfm_df[col].quantile(0.01)
                outliers = rfm_df[(rfm_df[col] > q99 * 3) | (rfm_df[col] < q01 * 0.1)]
                if len(outliers) > 0:
                    validation_report['warnings'].append(f"Potential outliers found in {col}: {len(outliers)} records")
            
            # Calculate statistics
            validation_report['statistics'] = {
                'total_customers': len(rfm_df),
                'recency_stats': rfm_df['Recency'].describe().to_dict(),
                'frequency_stats': rfm_df['Frequency'].describe().to_dict(),
                'monetary_stats': rfm_df['Monetary'].describe().to_dict()
            }
            
            logger.info("RFM data validation completed")
            return validation_report
            
        except Exception as e:
            error_msg = f"Failed to validate RFM data: {e}"
            logger.error(error_msg)
            return {'is_valid': False, 'errors': [error_msg], 'warnings': [], 'statistics': {}}
    
    @staticmethod
    def compute_rfm(df_merged, customer_col="member_number", date_col="date", invoice_col=None,
                    amount_col="amount", snapshot_date=None):
        """
        Compute RFM dataset from merged dataframe.
        - A transaction = unique (Member_number, Date)
        - Recency = days since last purchase
        - Frequency = number of purchases
        - Monetary = total amount spent
        """
        import hashlib
        
        df = df_merged.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        def generate_transaction_key(row):
            raw_key = f"{row[customer_col]}_{row[date_col].date()}"
            return hashlib.md5(raw_key.encode()).hexdigest()

        df["transaction_key"] = df.apply(generate_transaction_key, axis=1)

        # First aggregate by transaction_key to get unique transactions
        transactions_agg = (
            df.groupby(["transaction_key", customer_col, date_col])
            .agg(total_items=("items", "sum"), total_amount=(amount_col, "sum")).reset_index()
        )

        # Reference date = last date in transactions_agg (not original df)
        if snapshot_date is None:
            reference_date = transactions_agg[date_col].max()
        else:
            reference_date = snapshot_date

        # Now calculate RFM from transactions_agg
        rfm = (
            transactions_agg.groupby(customer_col).agg(
                Recency=(date_col, lambda x: (reference_date - x.max()).days),
                Frequency=("transaction_key", "nunique"),
                Monetary=("total_amount", "sum")
            ).reset_index()
        )
        return rfm