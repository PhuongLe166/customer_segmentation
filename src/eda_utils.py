from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime


def read_csv(file_or_path: Path | str | bytes) -> pd.DataFrame:
    """Read a CSV from a file-like object or path.

    Raises exceptions to the caller for explicit handling upstream.
    """
    if hasattr(file_or_path, "read"):
        return pd.read_csv(file_or_path)  # type: ignore[arg-type]
    return pd.read_csv(file_or_path)  # type: ignore[arg-type]


def load_default_paths() -> Tuple[Path, Path]:
    """Return default dataset paths for transactions and products."""
    return Path("data/raw/Transactions.csv"), Path("data/raw/Products_with_Categories.csv")


def load_datasets(
    transactions_src: Optional[Path | str | bytes],
    products_src: Optional[Path | str | bytes],
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Load transactions and products with fallbacks to defaults.

    Returns dataframes and source labels ("uploaded" or "default").
    """
    default_tx, default_pd = load_default_paths()

    tx_src = transactions_src if transactions_src is not None else default_tx
    pd_src = products_src if products_src is not None else default_pd

    df_tx = read_csv(tx_src)
    df_pd = read_csv(pd_src)

    tx_label = "uploaded" if transactions_src is not None else "default"
    pd_label = "uploaded" if products_src is not None else "default"
    return df_tx, df_pd, tx_label, pd_label


def infer_join_keys(df_tx: pd.DataFrame, df_pd: pd.DataFrame) -> Tuple[str, str]:
    """Infer likely join keys between transactions and products."""
    candidates = ("productid", "product_id", "product")
    def pick(df: pd.DataFrame) -> str:
        for c in df.columns:
            if c.lower() in candidates:
                return c
        return df.columns[0]
    return pick(df_tx), pick(df_pd)


def merge_datasets(
    df_tx: pd.DataFrame,
    df_pd: pd.DataFrame,
    left_on: str,
    right_on: str,
    how: str = "inner",
) -> pd.DataFrame:
    """Merge transactions and products with basic validation."""
    if left_on not in df_tx.columns:
        raise KeyError(f"Left key '{left_on}' not in transactions columns")
    if right_on not in df_pd.columns:
        raise KeyError(f"Right key '{right_on}' not in products columns")
    merged = df_tx.merge(df_pd, how=how, left_on=left_on, right_on=right_on)
    # Create amount if columns are present
    if "items" in merged.columns and "price" in merged.columns:
        try:
            merged["amount"] = merged["items"].astype(float) * merged["price"].astype(float)
        except Exception:
            # If casting fails, leave amount as NaN
            merged["amount"] = None
    return merged


def compute_recency_frequency_metrics(
    df: pd.DataFrame,
    customer_col: str = "member_number",
    date_col: str = "date",
) -> Tuple[float, float]:
    """Compute average recency (days) and average frequency per customer.

    Recency is measured as days since a customer's most recent transaction
    relative to the dataset snapshot date (max date in df).
    Frequency is the number of transactions per customer.
    """
    if date_col not in df.columns or customer_col not in df.columns:
        return float("nan"), float("nan")

    dates = df[date_col]
    if not is_datetime(dates):
        dates = pd.to_datetime(dates, errors="coerce")
    snapshot_date = dates.max()
    if pd.isna(snapshot_date):
        return float("nan"), float("nan")

    grouped = df.assign(__date__=dates).groupby(customer_col, as_index=False)
    last_dates = grouped["__date__"].max()
    recency_days = (snapshot_date - last_dates["__date__"]).dt.days
    avg_recency = float(np.nanmean(recency_days)) if len(recency_days) else float("nan")

    frequency = df.groupby(customer_col).size()
    avg_frequency = float(np.nanmean(frequency)) if len(frequency) else float("nan")

    return avg_recency, avg_frequency


def compute_transaction_based_metrics(
    df: pd.DataFrame,
    customer_col: str = "member_number",
    date_col: str = "date",
    amount_col: str = "amount",
) -> Tuple[int, float, float, float, float, float]:
    """Compute metrics based on unique transactions (customer + date combinations).
    
    All metrics are calculated using transaction key = customer + date
    
    Returns:
        - total_transactions: Count of unique transaction keys (customer + date)
        - total_revenue: Sum of all transaction amounts
        - avg_order_value: Average amount per transaction
        - avg_recency: Average days since last transaction per customer
        - avg_frequency: Average number of unique transactions per customer
        - avg_monetary: Average total revenue per customer
    """
    if date_col not in df.columns or customer_col not in df.columns:
        return 0, 0.0, 0.0, float("nan"), float("nan"), float("nan")
    
    # Convert date column to datetime
    dates = df[date_col]
    if not is_datetime(dates):
        dates = pd.to_datetime(dates, errors="coerce")
    
    # Create transaction key (customer + date)
    df_with_dates = df.copy()
    df_with_dates['__date__'] = dates
    
    # First, aggregate by transaction key (customer + date) to get unique transactions
    if amount_col in df.columns:
        transaction_summary = df_with_dates.groupby([customer_col, '__date__']).agg({
            amount_col: 'sum'  # Sum amount for each unique transaction
        }).reset_index()
    else:
        transaction_summary = df_with_dates.groupby([customer_col, '__date__']).size().reset_index()
        transaction_summary[amount_col] = 0
    
    # Count total unique transactions
    total_transactions = len(transaction_summary)
    
    # Calculate revenue metrics
    if amount_col in df.columns:
        total_revenue = float(transaction_summary[amount_col].sum())
        avg_order_value = float(transaction_summary[amount_col].mean())
    else:
        total_revenue = 0.0
        avg_order_value = 0.0
    
    # Calculate customer-level metrics based on unique transactions
    snapshot_date = dates.max()
    if pd.isna(snapshot_date):
        return total_transactions, total_revenue, avg_order_value, float("nan"), float("nan"), float("nan")
    
    # Group by customer to calculate RFM-like metrics from unique transactions
    customer_metrics = transaction_summary.groupby(customer_col).agg({
        '__date__': ['max', 'count'],  # last transaction date, count of unique transactions
        amount_col: 'sum'  # total revenue per customer from unique transactions
    }).reset_index()
    
    # Flatten column names
    customer_metrics.columns = [customer_col, 'last_date', 'frequency', 'monetary']
    
    # Calculate recency (days since last transaction)
    customer_metrics['recency'] = (snapshot_date - customer_metrics['last_date']).dt.days
    
    # Calculate averages
    avg_recency = float(np.nanmean(customer_metrics['recency'])) if len(customer_metrics) else float("nan")
    avg_frequency = float(np.nanmean(customer_metrics['frequency'])) if len(customer_metrics) else float("nan")
    avg_monetary = float(np.nanmean(customer_metrics['monetary'])) if len(customer_metrics) else float("nan")
    
    return total_transactions, total_revenue, avg_order_value, avg_recency, avg_frequency, avg_monetary


