# src/utils.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import pickle
import os

@st.cache_data
def load_data_cached(file_path):
    """Cache data loading for better performance"""
    if file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    else:
        return pd.read_csv(file_path)

def format_currency(amount):
    """Format currency values"""
    if amount >= 1e6:
        return f"${amount/1e6:.1f}M"
    elif amount >= 1e3:
        return f"${amount/1e3:.1f}K"
    else:
        return f"${amount:.2f}"

def format_number(number):
    """Format large numbers"""
    if number >= 1e6:
        return f"{number/1e6:.1f}M"
    elif number >= 1e3:
        return f"{number/1e3:.1f}K"
    else:
        return f"{number:,.0f}"

def calculate_percentage_change(old_value, new_value):
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100

def create_date_range(start_date, end_date):
    """Create date range for filtering"""
    return pd.date_range(start=start_date, end=end_date, freq='D')

def get_time_period_options():
    """Get time period filtering options"""
    return {
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "Last 6 Months": 180,
        "Last 12 Months": 365,
        "All Time": None
    }

def filter_data_by_date_range(df, date_col, start_date, end_date):
    """Filter dataframe by date range"""
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    return df.loc[mask]

def generate_hash_id(text):
    """Generate hash ID for unique identification"""
    return hashlib.md5(text.encode()).hexdigest()[:8]

def save_model(model, filename):
    """Save model to pickle file"""
    os.makedirs('models', exist_ok=True)
    with open(f'models/{filename}', 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """Load model from pickle file"""
    try:
        with open(f'models/{filename}', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def validate_data_quality(df, required_columns):
    """Validate data quality and completeness"""
    issues = []
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for missing values
    missing_data = df.isnull().sum()
    if missing_data.any():
        issues.append(f"Missing values found in: {missing_data[missing_data > 0].to_dict()}")
    
    # Check data types
    if 'date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            issues.append("Date column is not in datetime format")
    
    return issues

def create_summary_stats(df, numeric_columns):
    """Create summary statistics for numeric columns"""
    summary = {}
    for col in numeric_columns:
        if col in df.columns:
            summary[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(), 
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count()
            }
    return summary

def export_results(data, filename, format='csv'):
    """Export results to file"""
    os.makedirs('exports', exist_ok=True)
    filepath = f'exports/{filename}.{format}'
    
    if format == 'csv':
        data.to_csv(filepath, index=False)
    elif format == 'excel':
        data.to_excel(filepath, index=False)
    elif format == 'json':
        data.to_json(filepath, orient='records')
    
    return filepath

