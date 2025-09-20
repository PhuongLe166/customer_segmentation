"""
KPI Cards Components - Reusable KPI display components
"""

import streamlit as st
from typing import Dict, Any, Optional

class KPICards:
    """KPI Cards component for displaying key performance indicators."""
    
    @staticmethod
    def render_basic_kpi_cards(kpi_data: Dict[str, Any], columns: int = 3) -> None:
        """
        Render basic KPI cards in a grid layout.
        
        Args:
            kpi_data: Dictionary containing KPI values
            columns: Number of columns for the grid
        """
        # Create columns
        cols = st.columns(columns)
        
        # Basic KPIs
        kpis = [
            ("Total Users", kpi_data.get('total_users', 0), "👥"),
            ("Total Transactions", kpi_data.get('total_transactions', 0), "🛒"),
            ("Total Revenue", f"${kpi_data.get('total_revenue', 0):,.0f}", "💰"),
            ("Avg Recency", f"{kpi_data.get('avg_recency', 0):,.1f} days", "📅"),
            ("Avg Frequency", f"{kpi_data.get('avg_frequency', 0):,.1f}", "🔄"),
            ("Avg Monetary", f"${kpi_data.get('avg_monetary', 0):,.0f}", "💎")
        ]
        
        for i, (title, value, icon) in enumerate(kpis):
            with cols[i % columns]:
                st.metric(
                    label=title,
                    value=value,
                    help=f"{icon} {title}"
                )
    
    @staticmethod
    def render_rfm_cards(kpi_data: Dict[str, Any]) -> None:
        """
        Render RFM (Recency, Frequency, Monetary) cards.
        
        Args:
            kpi_data: Dictionary containing RFM values
        """
        # RFM Cards CSS
        st.markdown("""
        <style>
        .rfm-card {
            background-color: #ffffff;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 15px;
            margin: 8px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            position: relative;
            cursor: help;
            transition: all 0.3s ease;
        }
        .rfm-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            border-color: #1f77b4;
        }
        .rfm-letter {
            font-size: 28px;
            font-weight: 900;
            color: #1f77b4;
            margin-bottom: 6px;
        }
        .rfm-title {
            font-size: 11px;
            color: #6c757d;
            font-weight: 500;
            margin-bottom: 4px;
        }
        .rfm-value {
            font-size: 20px;
            color: #212529;
            font-weight: 700;
            margin-bottom: 0px;
        }
        .rfm-tooltip {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            z-index: 1000;
            max-width: 300px;
            white-space: normal;
            text-align: center;
        }
        .rfm-card:hover .rfm-tooltip {
            opacity: 1;
            visibility: visible;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create RFM cards
        rfm1, rfm2, rfm3 = st.columns(3)
        
        with rfm1:
            st.markdown(f"""
            <div class="rfm-card">
                <div class="rfm-tooltip">Recency: Average number of days since each customer's last transaction</div>
                <div class="rfm-letter">R</div>
                <div class="rfm-title">AVG Days Since Last Transaction</div>
                <div class="rfm-value">{kpi_data.get('avg_recency', 0):.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with rfm2:
            st.markdown(f"""
            <div class="rfm-card">
                <div class="rfm-tooltip">Frequency: Average number of transactions per customer</div>
                <div class="rfm-letter">F</div>
                <div class="rfm-title">AVG Transactions per User</div>
                <div class="rfm-value">{kpi_data.get('avg_frequency', 0):.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with rfm3:
            st.markdown(f"""
            <div class="rfm-card">
                <div class="rfm-tooltip">Monetary: Average total revenue per customer (sum of all their transactions)</div>
                <div class="rfm-letter">M</div>
                <div class="rfm-title">AVG Net Revenue Per User</div>
                <div class="rfm-value">${kpi_data.get('avg_monetary', 0):.0f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_advanced_kpi_cards(kpi_data: Dict[str, Any]) -> None:
        """
        Render advanced KPI cards with enhanced styling.
        
        Args:
            kpi_data: Dictionary containing KPI values
        """
        # Advanced KPI Cards CSS
        st.markdown("""
        <style>
        .kpi-card {
            background-color: #ffffff;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 20px;
            margin: 8px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            position: relative;
            cursor: help;
            transition: all 0.3s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            border-color: #1f77b4;
        }
        .kpi-title {
            font-size: 14px;
            color: #6c757d;
            font-weight: 500;
            margin-bottom: 8px;
        }
        .kpi-value {
            font-size: 28px;
            color: #212529;
            font-weight: 700;
            margin: 0;
        }
        .kpi-tooltip {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            z-index: 1000;
            max-width: 300px;
            white-space: normal;
            text-align: center;
        }
        .kpi-card:hover .kpi-tooltip {
            opacity: 1;
            visibility: visible;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # First row: Total metrics (3 columns)
        kpi1, kpi2, kpi3 = st.columns(3)
        
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-tooltip">Total number of unique customers in the dataset</div>
                <div class="kpi-title">Total Users</div>
                <div class="kpi-value">{kpi_data.get('total_users', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-tooltip">Total number of transactions across all customers</div>
                <div class="kpi-title">Total Transactions</div>
                <div class="kpi-value">{kpi_data.get('total_transactions', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-tooltip">Total revenue generated from all transactions</div>
                <div class="kpi-title">Total Revenue</div>
                <div class="kpi-value">${kpi_data.get('total_revenue', 0):,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Second row: Percentage metrics (3 columns)
        kpi4, kpi5, kpi6 = st.columns(3)
        
        with kpi4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-tooltip">Percentage of users who are considered active (recent and frequent buyers)</div>
                <div class="kpi-title">% Users Active</div>
                <div class="kpi-value">{kpi_data.get('pct_active', 0):.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-tooltip">Percentage of users who are at risk of churning</div>
                <div class="kpi-title">% Users At Risk</div>
                <div class="kpi-value">{kpi_data.get('pct_at_risk', 0):.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi6:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-tooltip">Percentage of users who have churned (inactive for long periods)</div>
                <div class="kpi-title">% Users Churned</div>
                <div class="kpi-value">{kpi_data.get('pct_churned', 0):.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
