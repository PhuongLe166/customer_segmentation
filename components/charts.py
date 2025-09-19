# components/charts.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_metric_cards(metrics_data):
    """Create a row of metric cards"""
    cols = st.columns(len(metrics_data))
    
    for i, (col, metric) in enumerate(zip(cols, metrics_data)):
        with col:
            st.metric(
                label=metric.get('label', ''),
                value=metric.get('value', 0),
                delta=metric.get('delta', None),
                delta_color=metric.get('delta_color', 'normal')
            )

def plot_interactive_line_chart(df, x_col, y_col, title="Line Chart", color=None):
    """Create interactive line chart with Plotly"""
    fig = px.line(df, x=x_col, y=y_col, title=title, color=color)
    fig.update_layout(
        xaxis_title=x_col.title(),
        yaxis_title=y_col.title(),
        hovermode='x unified'
    )
    return fig

def plot_interactive_bar_chart(df, x_col, y_col, title="Bar Chart", orientation='v'):
    """Create interactive bar chart with Plotly"""
    if orientation == 'h':
        fig = px.bar(df, x=y_col, y=x_col, orientation='h', title=title)
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=title)
    
    fig.update_layout(
        xaxis_title=x_col.title() if orientation == 'v' else y_col.title(),
        yaxis_title=y_col.title() if orientation == 'v' else x_col.title()
    )
    return fig

def plot_correlation_heatmap(df, columns=None, title="Correlation Heatmap"):
    """Create correlation heatmap"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    corr_matrix = df[columns].corr()
    
    fig = px.imshow(
        corr_matrix,
        title=title,
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    
    return fig

def plot_distribution_chart(df, column, title="Distribution", chart_type="histogram"):
    """Create distribution charts"""
    if chart_type == "histogram":
        fig = px.histogram(df, x=column, title=title, nbins=30)
    elif chart_type == "box":
        fig = px.box(df, y=column, title=title)
    else:
        fig = px.violin(df, y=column, title=title)
    
    return fig

def plot_scatter_matrix(df, dimensions, color=None, title="Scatter Matrix"):
    """Create scatter plot matrix"""
    fig = px.scatter_matrix(
        df, 
        dimensions=dimensions, 
        color=color,
        title=title
    )
    return fig

def create_gauge_chart(value, title, min_val=0, max_val=100):
    """Create gauge chart for KPIs"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, max_val*0.5], 'color': "lightgray"},
                {'range': [max_val*0.5, max_val*0.8], 'color': "yellow"},
                {'range': [max_val*0.8, max_val], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val*0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_donut_chart(values, labels, title="Donut Chart"):
    """Create donut chart"""
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.3,
        title=title
    )])
    return fig

def style_matplotlib_chart():
    """Apply consistent styling to matplotlib charts"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
def create_comparison_chart(df, x_col, y_cols, title="Comparison Chart"):
    """Create multi-line comparison chart"""
    fig = go.Figure()
    
    for col in y_cols:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[col],
            mode='lines+markers',
            name=col,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col.title(),
        yaxis_title="Values",
        hovermode='x unified'
    )
    
    return fig