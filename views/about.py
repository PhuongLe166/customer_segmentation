import streamlit as st
from config.settings import PAGE_CONFIG

def show():
    """Display the About page"""
    
    # Page header
    st.markdown(f"# {PAGE_CONFIG['about']['title']}")
    st.markdown(f"*{PAGE_CONFIG['about']['description']}*")
    st.markdown("---")
    
    # Instructor
    st.markdown("## Instructor")
    st.markdown("**MSc. Khuat Thuy Phuong**")
    st.markdown("---")
    
    # Team
    st.markdown("## Team - Group I")
    st.markdown("- **Le Thi Ngoc Phuong** (Data Engineering)")
    st.markdown("- **Pham Hong Phat** (Student)")
    st.markdown("---")
    
    # Work allocation with styled table
    st.markdown("## Work Allocation")
    st.markdown("""
    <style>
      .about-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        box-shadow: 0 6px 18px rgba(0,0,0,.08);
        border-radius: 12px;
        overflow: hidden;
      }
      .about-table thead th {
        background: linear-gradient(90deg, #1f77b4, #5aa9e6);
        color: #fff;
        text-align: left;
        padding: 12px 16px;
        font-weight: 600;
        letter-spacing: .3px;
      }
      .about-table tbody td {
        padding: 12px 16px;
        background: #fff;
        border-top: 1px solid #eef2f7;
      }
      .about-table tbody tr:hover td { background: #f7fbff; }
      .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        background: #e6f2ff;
        color: #1f77b4;
        border: 1px solid #cfe6ff;
      }
    </style>
    <table class="about-table">
      <thead>
        <tr><th>Task</th><th>Owner(s)</th></tr>
      </thead>
      <tbody>
        <tr><td>EDA</td><td><span class="badge">Phuong</span>  <span class="badge">Phat</span></td></tr>
        <tr><td>Rules-based</td><td><span class="badge">Phuong</span></td></tr>
        <tr><td>KMeans + Hierarchical Clustering</td><td><span class="badge">Phat</span></td></tr>
        <tr><td>PySpark</td><td><span class="badge">Phuong</span></td></tr>
        <tr><td>GUI</td><td><span class="badge">Phuong</span>  <span class="badge">Phat</span></td></tr>
        <tr><td>Report</td><td><span class="badge">Phuong</span>  <span class="badge">Phat</span></td></tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Objectives
    st.markdown("## Objectives")
    st.markdown("- Build an end-to-end RFM customer segmentation pipeline")
    st.markdown("- Compare rule-based segmentation vs. clustering approaches")
    st.markdown("- Deliver actionable insights to support business decisions")
    st.markdown("---")
    
    # Pipeline (styled steps)
    st.markdown("## Pipeline")
    st.markdown("""
    <style>
      .pipeline {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin: 8px 0 12px;
      }
      .step {
        background: #ffffff;
        border: 1px solid #eef2f7;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,.06);
      }
      .step .num {
        width: 28px; height: 28px; line-height: 28px;
        margin: 0 auto 8px; border-radius: 9999px;
        background: #1f77b4; color: #fff; font-weight: 700;
      }
      @media (max-width: 900px) { .pipeline { grid-template-columns: repeat(2, 1fr);} }
      @media (max-width: 540px) { .pipeline { grid-template-columns: 1fr;} }
    </style>
    <div class="pipeline">
      <div class="step"><div class="num">1</div><div>Data Ingestion</div></div>
      <div class="step"><div class="num">2</div><div>Preprocessing</div></div>
      <div class="step"><div class="num">3</div><div>RFM Scoring</div></div>
      <div class="step"><div class="num">4</div><div>Segmentation</div></div>
      <div class="step"><div class="num">5</div><div>Dashboard & Report</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Tech Stack
    st.markdown("## Tech Stack")
    st.markdown("- Python, Pandas, NumPy")
    st.markdown("- Scikit-learn (KMeans, Hierarchical Clustering)")
    st.markdown("- PySpark (large-scale data processing)")
    st.markdown("- Streamlit (GUI)")
    st.markdown("- Matplotlib/Seaborn (EDA & Visualization)")

