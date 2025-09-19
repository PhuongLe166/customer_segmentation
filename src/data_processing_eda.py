import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import hashlib


def load_dataset(path_or_buffer) :
    """Load dataset (CSV or Excel) and standardize column names."""

    if str(path_or_buffer).endswith((".xlsx", ".xls")) :
        df = pd.read_excel(path_or_buffer)
    else :
        df = pd.read_csv(path_or_buffer)

    df.columns = (
        df.columns.str.strip()           
                  .str.lower()           
                  .str.replace(r"[^0-9a-zA-Z]+", "_", regex = True)
                  .str.replace(r"__+", "_", regex = True)
                  .str.strip("_")
    )

    return df


def dataset_overview(df, name = "Dataset") :
    """
    Return dictionary with overview: shape, columns, dtypes,
    missing values, stats, unique counts (if applicable).
    """
    
    overview = {
        "name" : name,
        "shape" : df.shape,
        "columns" : list(df.columns),
        "dtypes" : df.dtypes.to_dict(),
        "missing" : df.isnull().sum().to_dict(),
        "describe" : df.describe(include = "all").transpose()
    }
    
    return overview


def preprocess_transactions(transactions_df, products_df) :
    """
    Preprocess transactions:
    - Convert 'Date' to datetime.
    - Merge transactions with products on productId.
    - Add 'amount' column = price * items.
    """
    
    transactions_df = transactions_df.copy()
    transactions_df['date'] = pd.to_datetime(transactions_df['date'], format = '%d-%m-%Y', errors = 'coerce')
    
    df_merged = transactions_df.merge(products_df, on = "productid", how = "left")
    df_merged = df_merged.copy()
    df_merged['amount'] = df_merged['price'] * df_merged['items']
    
    return df_merged


def compute_rfm(df_merged, customer_col = "member_number", date_col = "date", invoice_col =  None,
                amount_col = "amount", snapshot_date = None) :
    """
    Compute RFM dataset from merged dataframe.
    - A transaction = unique (Member_number, Date)
    - Recency = days since last purchase
    - Frequency = number of purchases
    - Monetary = total amount spent
    """
    
    df = df_merged.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    def generate_transaction_key(row) :
        raw_key = f"{row[customer_col]}_{row[date_col].date()}"
        return hashlib.md5(raw_key.encode()).hexdigest()

    df["transaction_key"] = df.apply(generate_transaction_key, axis = 1)

    # First aggregate by transaction_key to get unique transactions
    transactions_agg = (
        df.groupby(["transaction_key", customer_col, date_col])
        .agg(total_items = ("items", "sum"), total_amount = (amount_col, "sum")).reset_index()
    )

    # Reference date = last date in transactions_agg (not original df)
    if snapshot_date is None :
        reference_date = transactions_agg[date_col].max()
    else:
        reference_date = snapshot_date

    # Now calculate RFM from transactions_agg
    rfm = (
        transactions_agg.groupby(customer_col).agg(
            Recency = (date_col, lambda x : (reference_date - x.max()).days),
            Frequency = ("transaction_key", "nunique"),
            Monetary = ("total_amount", "sum")
        ).reset_index()
    )
    return rfm


def plot_daily_sales(transactions_agg, date_col : str = "date", amount_col : str = "total_amount") :
    """Plot overall daily sales trend and return figure."""

    daily_total = transactions_agg.groupby(date_col)[amount_col].sum()

    fig, ax = plt.subplots(figsize = (14, 7))
    sns.lineplot(
        x = daily_total.index, y = daily_total.values,
        marker = "o", color = "#2a9d8f", linewidth = 2,  ax = ax
    )
    ax.set_title("Daily Total Sales", fontsize = 16, fontweight = 'bold')
    ax.set_xlabel("Date", fontsize = 12)
    ax.set_ylabel("Revenue", fontsize = 12)
    plt.xticks(rotation = 45)
    ax.grid(True, linestyle = '--', alpha = 0.5)
    plt.tight_layout()

    return fig


def plot_monthly_sales(transactions_agg, date_col : str = "date", amount_col : str = "total_amount") :
    """Plot monthly total sales trend and return figure."""

    monthly_total = transactions_agg.set_index(date_col).resample("M")[amount_col].sum()

    fig, ax = plt.subplots(figsize = (14, 7))
    sns.lineplot(
        x = monthly_total.index, y = monthly_total.values,
        marker = "o", color = "#e76f51", linewidth = 2, ax = ax
    )
    ax.set_title("Monthly Total Sales", fontsize = 16, fontweight = 'bold')
    ax.set_xlabel("Month", fontsize = 12)
    ax.set_ylabel("Revenue", fontsize = 12)
    plt.xticks(rotation = 45)
    ax.grid(True, linestyle = '--', alpha = 0.5)
    plt.tight_layout()

    return fig



def plot_monthly_total_with_breakdown(df_merged, date_col : str = "date", category_col : str = "category", 
                                      amount_col : str = "amount", top_n : int = 10) :
    """Plot monthly total revenue vs monthly revenue of top-N products individually."""

    monthly_total = (df_merged.set_index(date_col).resample("M")[amount_col].sum().rename("Total"))
    top_products = (
        df_merged.groupby(category_col)[amount_col].sum().sort_values(ascending = False).head(top_n).index
    )
    
    monthly_by_product = (df_merged[df_merged[category_col].isin(top_products)].set_index(date_col)
                          .groupby(category_col)[amount_col].resample("M").sum().reset_index())

    fig, ax = plt.subplots(figsize = (16, 8))
    sns.lineplot(
        x = monthly_total.index, y = monthly_total.values, 
        label = "Total", color = "black", linewidth = 3, ax = ax)

    for prod in top_products :
        prod_data = monthly_by_product[monthly_by_product[category_col] == prod]
        sns.lineplot(x = prod_data[date_col], y = prod_data[amount_col], label = prod, ax = ax)

    ax.set_title(f"Monthly Revenue: Total vs Top {top_n} Categories", fontsize = 16, fontweight = "bold")
    ax.set_xlabel("Month", fontsize = 12)
    ax.set_ylabel("Revenue", fontsize = 12)
    plt.xticks(rotation = 45)
    ax.grid(True, linestyle = "--", alpha = 0.5)
    ax.legend()
    plt.tight_layout()

    return fig


def plot_category_revenue(df_merged, category_col : str = "category", amount_col : str = "amount") :
    """
    Plot total revenue by product category.
    """
    
    category_revenue = (df_merged.groupby(category_col)[amount_col].sum().sort_values(ascending = False))

    fig, ax = plt.subplots(figsize = (14, 7))
    sns.barplot(x = category_revenue.values, y = category_revenue.index, palette = "viridis", ax = ax)
    ax.set_title("Total Revenue by Product Category", fontsize = 16, fontweight = "bold")
    ax.set_xlabel("Revenue", fontsize = 12)
    ax.set_ylabel("Product Category", fontsize = 12)
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    plt.tight_layout()
    
    return fig


def plot_product_revenue(df_merged, product_col : str = "productid", name_col : str = "productname",
                         amount_col : str = "amount", top_n : int = 20) :
    """Plot top-N products by revenue."""
    
    product_revenue = (df_merged.groupby([product_col, name_col])[amount_col].sum().reset_index()
                       .sort_values(amount_col, ascending = False).head(top_n))
    product_revenue[name_col] = product_revenue[name_col].astype(str)
    order = product_revenue[name_col].tolist()    
    
    fig, ax = plt.subplots(figsize = (14, 7))
    sns.barplot(x = amount_col, y = name_col, data = product_revenue, order = order, palette = "magma", ax = ax)
    ax.set_title(f"Top {top_n} Products by Revenue", fontsize = 16, fontweight = "bold")
    ax.set_xlabel("Revenue", fontsize = 12)
    ax.set_ylabel("Product", fontsize = 12)
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    plt.tight_layout()

    return fig

def plot_rfm_histograms(rfm_df, r_cols : list = ["Recency", "Frequency", "Monetary"], bins : int = 30) :
    """Plot histograms for R, F, M distributions."""
    
    fig, axes = plt.subplots(1, len(r_cols), figsize = (18, 5))

    for i, col in enumerate(r_cols) :
        sns.histplot(rfm_df[col], bins = bins, kde = False, ax = axes[i])
        axes[i].set_title(f"{col} Distribution", fontsize = 14, fontweight = "bold")
        axes[i].set_xlabel(col, fontsize = 12)
        axes[i].set_ylabel("Count", fontsize = 12)
        axes[i].grid(True, linestyle = "--", alpha = 0.5)

    plt.tight_layout()

    return fig
