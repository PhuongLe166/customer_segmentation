# src/eda_core.py
"""
EDA Core module - Contains all exploratory data analysis logic
for customer segmentation application.

This module provides comprehensive data exploration and analysis capabilities
with improved error handling, validation, and performance optimization.
"""

import pandas as pd
import numpy as np
import re
import io
from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDACore:
    """Core EDA operations for data exploration and analysis with enhanced error handling."""
    
    @staticmethod
    @st.cache_data(show_spinner=False, ttl=3600)
    def _cached_read_csv_from_path(path: str) -> pd.DataFrame:
        """Cached CSV reader for file paths."""
        return pd.read_csv(path)

    @staticmethod
    @st.cache_data(show_spinner=False, ttl=3600)
    def _cached_read_csv_from_bytes(content: bytes) -> pd.DataFrame:
        """Cached CSV reader for uploaded file bytes."""
        return pd.read_csv(io.BytesIO(content))

    @staticmethod
    def load_and_validate_data(
        transactions_file: Optional[Union[str, Path]] = None, 
        products_file: Optional[Union[str, Path]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
        """
        Load and validate datasets from files or session state.
        
        Args:
            transactions_file: Path to transactions file or None for session state
            products_file: Path to products file or None for session state
            
        Returns:
            Tuple of (transactions_df, products_df, transactions_source, products_source)
            
        Raises:
            ValueError: If required data cannot be loaded
            FileNotFoundError: If files are specified but not found
        """
        try:
            # Use internal method
            
            df_tx, df_pd, src_tx, src_pd = EDACore.load_datasets(transactions_file, products_file)
            
            # Validate loaded data
            if df_tx.empty or df_pd.empty:
                raise ValueError("One or both datasets are empty")
            
            logger.info(f"Successfully loaded datasets: {src_tx} ({len(df_tx)} rows), {src_pd} ({len(df_pd)} rows)")
            return df_tx, df_pd, src_tx, src_pd
            
        except Exception as e:
            error_msg = f"Failed to load datasets: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    @staticmethod
    def merge_datasets(
        df_transactions: pd.DataFrame, 
        df_products: pd.DataFrame,
        join_type: str = "inner"
    ) -> pd.DataFrame:
        """
        Merge transactions and products datasets using inferred join keys.
        
        Args:
            df_transactions: Transactions DataFrame
            df_products: Products DataFrame
            join_type: Type of join to perform ('inner', 'left', 'right', 'outer')
            
        Returns:
            Merged DataFrame
            
        Raises:
            ValueError: If merge fails or results in empty dataset
        """
        try:
            # Use internal methods
            
            # Infer join keys
            left_key, right_key = EDACore.infer_join_keys(df_transactions, df_products)
            
            if not left_key or not right_key:
                raise ValueError("Could not infer join keys between datasets")
            
            # Perform merge
            merged = EDACore.merge_datasets(df_transactions, df_products, left_key, right_key, join_type)
            
            if merged.empty:
                raise ValueError(f"Merge resulted in empty dataset with {join_type} join")
            
            logger.info(f"Successfully merged datasets: {len(merged)} rows")
            return merged
            
        except Exception as e:
            error_msg = f"Merge failed: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    @staticmethod
    def prepare_rfm_data(merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare merged data for RFM analysis by standardizing column names.
        
        Args:
            merged_df: Merged DataFrame from transactions and products
            
        Returns:
            DataFrame ready for RFM analysis
            
        Raises:
            ValueError: If required columns are not found
        """
        try:
            # Find and standardize column names
            date_col = EDACore._find_column(merged_df, ["Date", "date", "transaction_date", "purchase_date"])
            customer_col = EDACore._find_column(merged_df, ["Member_number", "member_number", "customer_id", "user_id"])
            items_col = EDACore._find_column(merged_df, ["items", "quantity", "qty"])
            price_col = EDACore._find_column(merged_df, ["price", "unit_price", "cost"])
            
            if not all([date_col, customer_col, items_col, price_col]):
                missing_cols = [col for col, found in zip(
                    ["date", "customer", "items", "price"], [date_col, customer_col, items_col, price_col]
                ) if not found]
                raise ValueError(f"Required columns for RFM analysis not found: {missing_cols}")
            
            # Prepare RFM data
            merged_rfm = merged_df.copy()
            merged_rfm = merged_rfm.rename(columns={
                customer_col: 'member_number', 
                date_col: 'date',
                items_col: 'items',
                price_col: 'price'
            })
            
            # Calculate amount = items * price
            merged_rfm['amount'] = merged_rfm['items'] * merged_rfm['price']
            
            # Ensure datetime conversion
            merged_rfm['date'] = pd.to_datetime(merged_rfm['date'], errors="coerce")
            merged_rfm = merged_rfm.dropna(subset=['date', 'member_number'])
            
            # Add items column if not present
            merged_rfm = EDACore._ensure_items_column(merged_rfm)
            
            logger.info(f"Successfully prepared RFM data: {len(merged_rfm)} rows")
            return merged_rfm
            
        except Exception as e:
            error_msg = f"Failed to prepare RFM data: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def _find_column(df: pd.DataFrame, possible_names: list) -> Optional[str]:
        """Find column by possible names (case-insensitive)."""
        for name in possible_names:
            for col in df.columns:
                if col.lower() == name.lower():
                    return col
        return None
    
    @staticmethod
    def _ensure_items_column(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure items column exists in DataFrame."""
        if 'items' not in df.columns:
            items_col = EDACore._find_column(df, ["items", "quantity", "qty", "count"])
            if items_col:
                df['items'] = df[items_col]
            else:
                df['items'] = 1
        return df
    
    @staticmethod
    def find_category_columns(merged_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """
        Find category and product columns in merged data with improved logic.
        
        Args:
            merged_df: Merged DataFrame
            
        Returns:
            Tuple of (category_column, product_column)
        """
        def normalize_column_name(col: str) -> str:
            """Normalize column name for matching."""
            return re.sub(r"(_x|_y)$", "", col.lower().strip())
        
        # Create normalized column mapping
        columns_norm = {col: normalize_column_name(col) for col in merged_df.columns}
        
        def find_best_match(prioritized_names: list) -> Optional[str]:
            """Find best matching column from prioritized list."""
            for name in prioritized_names:
                name_lower = name.lower()
                for original, norm in columns_norm.items():
                    if norm == name_lower:
                        return original
            return None
        
        # Find category column
        cat_col = find_best_match([
            "category", "product_category", "productcategory", "cat", 
            "product_type", "producttype", "classification"
        ])
        
        # Find product column
        product_col = find_best_match([
            "productname", "product_name", "product", "item_name", "item", 
            "description", "productdesc", "productdesc", "sku", "code",
            "product_title", "producttitle"
        ])
        
        return cat_col, product_col
    
    @staticmethod
    def analyze_categories(
        merged_df: pd.DataFrame, 
        cat_col: str, 
        top_n: int = 10,
        min_revenue: float = 0
    ) -> pd.DataFrame:
        """
        Analyze category performance with enhanced filtering.
        
        Args:
            merged_df: Merged DataFrame
            cat_col: Category column name
            top_n: Number of top categories to return
            min_revenue: Minimum revenue threshold
            
        Returns:
            DataFrame with category analysis
        """
        if not cat_col or cat_col not in merged_df.columns:
            raise ValueError(f"Category column '{cat_col}' not found in data")
        
        # Ensure amount column exists
        if 'amount' not in merged_df.columns:
            if 'items' in merged_df.columns and 'price' in merged_df.columns:
                merged_df['amount'] = merged_df['items'] * merged_df['price']
            else:
                raise ValueError("Cannot analyze categories: missing 'amount' or 'items'/'price' columns")
        
        cat_revenue = (
            merged_df.groupby(cat_col, as_index=False)["amount"]
            .agg(['sum', 'count', 'mean'])
            .round(2)
        )
        cat_revenue.columns = ['total_revenue', 'transaction_count', 'avg_revenue']
        cat_revenue = cat_revenue.reset_index()
        
        # Filter by minimum revenue
        cat_revenue = cat_revenue[cat_revenue['total_revenue'] >= min_revenue]
        
        # Sort and return top N
        return cat_revenue.sort_values("total_revenue", ascending=False).head(top_n)
    
    @staticmethod
    def analyze_products(
        merged_df: pd.DataFrame, 
        product_col: str, 
        top_n: int = 15,
        min_revenue: float = 0
    ) -> pd.DataFrame:
        """
        Analyze product performance with enhanced filtering.
        
        Args:
            merged_df: Merged DataFrame
            product_col: Product column name
            top_n: Number of top products to return
            min_revenue: Minimum revenue threshold
            
        Returns:
            DataFrame with product analysis
        """
        if not product_col or product_col not in merged_df.columns:
            raise ValueError(f"Product column '{product_col}' not found in data")
        
        # Ensure amount column exists
        if 'amount' not in merged_df.columns:
            if 'items' in merged_df.columns and 'price' in merged_df.columns:
                merged_df['amount'] = merged_df['items'] * merged_df['price']
            else:
                raise ValueError("Cannot analyze products: missing 'amount' or 'items'/'price' columns")
        
        product_revenue = (
            merged_df.groupby(product_col, as_index=False)["amount"]
            .agg(['sum', 'count', 'mean'])
            .round(2)
        )
        product_revenue.columns = ['total_revenue', 'transaction_count', 'avg_revenue']
        product_revenue = product_revenue.reset_index()
        
        # Filter by minimum revenue
        product_revenue = product_revenue[product_revenue['total_revenue'] >= min_revenue]
        
        # Sort and return top N
        return product_revenue.sort_values("total_revenue", ascending=False).head(top_n)
    
    @staticmethod
    def calculate_revenue_trends(
        merged_df: pd.DataFrame, 
        granularity: str = "Month",
        date_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate revenue trends by different time granularities with enhanced error handling.
        
        Args:
            merged_df: Merged DataFrame
            granularity: Time granularity ('Date', 'Week', 'Month', 'Quarter', 'Year')
            date_col: Specific date column to use
            
        Returns:
            DataFrame with revenue trends
        """
        try:
            # Find date column if not specified
            if not date_col:
                date_col = EDACore._find_column(merged_df, ["Date", "date", "transaction_date", "purchase_date"])
            
            if not date_col:
                raise ValueError("No date column found in data")
            
            # Ensure datetime conversion
            merged_df = merged_df.copy()
            merged_df[date_col] = pd.to_datetime(merged_df[date_col], errors="coerce")
            merged_df = merged_df.dropna(subset=[date_col])
            
            if merged_df.empty:
                raise ValueError("No valid date data found after conversion")
            
            # Ensure amount column exists
            if 'amount' not in merged_df.columns:
                if 'items' in merged_df.columns and 'price' in merged_df.columns:
                    merged_df['amount'] = merged_df['items'] * merged_df['price']
                else:
                    raise ValueError("Cannot calculate revenue trends: missing 'amount' or 'items'/'price' columns")
            
            # Prepare base data
            base = merged_df[[date_col, "amount"]].copy()
            
            # Define time periods based on granularity
            period_mapping = {
                "Date": lambda x: x.dt.floor("D"),
                "Week": lambda x: x.dt.to_period("W").dt.start_time,
                "Month": lambda x: x.dt.to_period("M").dt.start_time,
                "Quarter": lambda x: x.dt.to_period("Q").dt.start_time,
                "Year": lambda x: x.dt.to_period("Y").dt.start_time
            }
            
            if granularity not in period_mapping:
                raise ValueError(f"Unsupported granularity: {granularity}")
            
            periods = period_mapping[granularity](base[date_col])
            
            # Calculate trends
            trend = base.groupby(periods)["amount"].agg(['sum', 'count', 'mean']).reset_index()
            trend.columns = ["period", "revenue", "transaction_count", "avg_revenue"]
            trend = trend.dropna(subset=["period"]).sort_values("period")
            
            # Add formatted labels
            trend = EDACore._add_period_labels(trend, granularity)
            
            # Ensure numeric values
            trend["revenue"] = pd.to_numeric(trend["revenue"], errors="coerce").fillna(0)
            trend["transaction_count"] = pd.to_numeric(trend["transaction_count"], errors="coerce").fillna(0)
            trend["avg_revenue"] = pd.to_numeric(trend["avg_revenue"], errors="coerce").fillna(0)
            
            logger.info(f"Successfully calculated {granularity} revenue trends: {len(trend)} periods")
            return trend
            
        except Exception as e:
            error_msg = f"Failed to calculate revenue trends: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def _add_period_labels(trend_df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Add formatted period labels based on granularity."""
        trend = trend_df.copy()
        
        if granularity == "Month":
            trend["label"] = trend["period"].dt.strftime("%Y %b")
        elif granularity == "Week":
            iso = trend["period"].dt.isocalendar()
            trend["label"] = iso["year"].astype(str) + " W" + iso["week"].astype(str)
        elif granularity == "Quarter":
            trend["label"] = trend["period"].dt.strftime("%Y Q%q")
        elif granularity == "Year":
            trend["label"] = trend["period"].dt.strftime("%Y")
        else:  # Date
            trend["label"] = trend["period"].dt.strftime("%Y %b %d")
        
        return trend
    
    @staticmethod
    def calculate_growth_metrics(trend_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate month-over-month and year-over-year growth metrics with enhanced validation.
        
        Args:
            trend_df: DataFrame with revenue trends
            
        Returns:
            DataFrame with growth metrics
        """
        try:
            trend = trend_df.copy()
            
            if len(trend) < 2:
                logger.warning("Insufficient data for growth calculations")
                trend["MoM"] = np.nan
                trend["YoY"] = np.nan
                return trend
            
            # Calculate MoM growth
            trend["MoM"] = trend["revenue"].pct_change() * 100.0
            
            # Calculate YoY growth (if enough data)
            if len(trend) >= 12:
                trend["YoY"] = trend["revenue"].pct_change(12) * 100.0
            else:
                trend["YoY"] = np.nan
                logger.info("Insufficient data for YoY calculations (need 12+ periods)")
            
            # Calculate additional metrics
            trend["revenue_change"] = trend["revenue"].diff()
            trend["cumulative_revenue"] = trend["revenue"].cumsum()
            
            logger.info("Successfully calculated growth metrics")
            return trend
            
        except Exception as e:
            error_msg = f"Failed to calculate growth metrics: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def get_data_summary(merged_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive data summary for validation and insights.
        
        Args:
            merged_df: Merged DataFrame
            
        Returns:
            Dictionary with data summary statistics
        """
        try:
            summary = {
                "total_rows": len(merged_df),
                "total_columns": len(merged_df.columns),
                "date_range": None,
                "revenue_stats": {},
                "customer_stats": {},
                "missing_data": {}
            }
            
            # Date range
            date_col = EDACore._find_column(merged_df, ["Date", "date", "transaction_date"])
            if date_col:
                dates = pd.to_datetime(merged_df[date_col], errors="coerce")
                summary["date_range"] = {
                    "start": dates.min(),
                    "end": dates.max(),
                    "span_days": (dates.max() - dates.min()).days
                }
            
            # Revenue stats
            if "amount" in merged_df.columns:
                summary["revenue_stats"] = {
                    "total": float(merged_df["amount"].sum()),
                    "mean": float(merged_df["amount"].mean()),
                    "median": float(merged_df["amount"].median()),
                    "std": float(merged_df["amount"].std()),
                    "min": float(merged_df["amount"].min()),
                    "max": float(merged_df["amount"].max())
                }
            
            # Customer stats
            customer_col = EDACore._find_column(merged_df, ["Member_number", "member_number", "customer_id"])
            if customer_col:
                summary["customer_stats"] = {
                    "unique_customers": int(merged_df[customer_col].nunique()),
                    "avg_transactions_per_customer": float(len(merged_df) / merged_df[customer_col].nunique())
                }
            
            # Missing data
            summary["missing_data"] = merged_df.isnull().sum().to_dict()
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate data summary: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def read_csv(file_or_path: Path | str | bytes) -> pd.DataFrame:
        """Read CSV file with proper type handling."""
        if isinstance(file_or_path, (str, Path)):
            return pd.read_csv(file_or_path)  # type: ignore[arg-type]
        return pd.read_csv(file_or_path)  # type: ignore[arg-type]
    
    @staticmethod
    def load_default_paths() -> Tuple[Path, Path]:
        """Load default file paths for transactions and products."""
        base_path = Path("data/raw")
        return base_path / "Transactions.csv", base_path / "Products_with_Categories.csv"
    
    @staticmethod
    def load_datasets(
        transactions_file: str | None = None, 
        products_file: str | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
        """
        Load transactions and products datasets.
        
        Args:
            transactions_file: Path to transactions file or file upload object
            products_file: Path to products file or file upload object
        
        Returns:
            Tuple of (transactions_df, products_df, transactions_source, products_source)
        """
        default_tx, default_pd = EDACore.load_default_paths()
        
        # Handle file upload objects from session state
        if transactions_file is None:
            tx_src = str(default_tx)
            df_tx = EDACore._cached_read_csv_from_path(tx_src)
        else:
            # Check if it's a file upload object (has .name attribute)
            if hasattr(transactions_file, 'name'):
                tx_src = f"Uploaded: {transactions_file.name}"
                # Use bytes cache if content available, else fallback to direct read
                try:
                    content = transactions_file.getvalue()  # type: ignore[attr-defined]
                    df_tx = EDACore._cached_read_csv_from_bytes(content)
                except Exception:
                    df_tx = EDACore.read_csv(transactions_file)
            # Check if it's a file content dict from session state
            elif isinstance(transactions_file, dict) and 'content' in transactions_file:
                tx_src = f"Uploaded: {transactions_file['name']}"
                df_tx = EDACore._cached_read_csv_from_bytes(transactions_file['content'])
            else:
                tx_src = str(transactions_file)
                df_tx = EDACore._cached_read_csv_from_path(tx_src)
        
        if products_file is None:
            pd_src = str(default_pd)
            df_pd = EDACore._cached_read_csv_from_path(pd_src)
        else:
            # Check if it's a file upload object (has .name attribute)
            if hasattr(products_file, 'name'):
                pd_src = f"Uploaded: {products_file.name}"
                try:
                    content = products_file.getvalue()  # type: ignore[attr-defined]
                    df_pd = EDACore._cached_read_csv_from_bytes(content)
                except Exception:
                    df_pd = EDACore.read_csv(products_file)
            # Check if it's a file content dict from session state
            elif isinstance(products_file, dict) and 'content' in products_file:
                pd_src = f"Uploaded: {products_file['name']}"
                df_pd = EDACore._cached_read_csv_from_bytes(products_file['content'])
            else:
                pd_src = str(products_file)
                df_pd = EDACore._cached_read_csv_from_path(pd_src)
        
        return df_tx, df_pd, tx_src, pd_src
    
    @staticmethod
    def infer_join_keys(df_tx: pd.DataFrame, df_pd: pd.DataFrame) -> Tuple[str, str]:
        """
        Infer join keys between transactions and products datasets.
        
        Returns:
            Tuple of (left_key, right_key)
        """
        # Common product ID columns
        tx_product_cols = [col for col in df_tx.columns if 'product' in col.lower()]
        pd_product_cols = [col for col in df_pd.columns if 'product' in col.lower()]
        
        if tx_product_cols and pd_product_cols:
            return tx_product_cols[0], pd_product_cols[0]
        
        # Fallback to first common column
        common_cols = set(df_tx.columns) & set(df_pd.columns)
        if common_cols:
            common_col = list(common_cols)[0]
            return common_col, common_col
        
        raise ValueError("No suitable join keys found between datasets")
    
    @staticmethod
    def merge_datasets(
        df_tx: pd.DataFrame, 
        df_pd: pd.DataFrame, 
        left_key: str, 
        right_key: str, 
        join_type: str = "left"
    ) -> pd.DataFrame:
        """
        Merge transactions and products datasets.
        
        Args:
            df_tx: Transactions DataFrame
            df_pd: Products DataFrame  
            left_key: Join key for transactions
            right_key: Join key for products
            join_type: Type of join ('left', 'inner', 'outer')
            
        Returns:
            Merged DataFrame
        """
        try:
            merged = df_tx.merge(df_pd, left_on=left_key, right_on=right_key, how=join_type)
            return merged
        except Exception as e:
            raise ValueError(f"Failed to merge datasets: {e}")