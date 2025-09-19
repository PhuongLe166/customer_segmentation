from __future__ import annotations

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


