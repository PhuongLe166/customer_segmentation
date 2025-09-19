import streamlit as st
from config.settings import PAGE_CONFIG

def show():
    """Display the About page"""
    
    # Page header
    st.markdown(f"# {PAGE_CONFIG['about']['title']}")
    st.markdown(f"*{PAGE_CONFIG['about']['description']}*")
    st.markdown("---")
    
    # RFM Overview
    st.markdown("## 🎯 What is RFM Analysis?")
    
    st.markdown("""
    **RFM Analysis** is a proven marketing technique used to quantitatively rank and group customers 
    based on the recency, frequency and monetary total of their recent transactions to identify the 
    best customers and perform targeted marketing campaigns.
    """)
    
    # RFM Components
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🕐 Recency (R)
        **How recently did the customer purchase?**
        
        - Days since last purchase
        - Lower recency = Better customer
        - Indicates customer engagement
        - Key for retention strategies
        """)
    
    with col2:
        st.markdown("""
        ### 🔄 Frequency (F)  
        **How often do they purchase?**
        
        - Total number of purchases
        - Higher frequency = Better customer
        - Shows customer loyalty
        - Predicts future behavior
        """)
    
    with col3:
        st.markdown("""
        ### 💰 Monetary (M)
        **How much do they spend?**
        
        - Total monetary value
        - Higher monetary = Better customer  
        - Indicates customer value
        - Revenue impact metric
        """)
    
    st.markdown("---")
    
    # Methodology
    st.markdown("## 🔬 Our Methodology")
    
    method_tabs = st.tabs(["📊 Data Processing", "🏷️ Segmentation", "🤖 Clustering", "📈 Evaluation"])
    
    with method_tabs[0]:
        st.markdown("""
        ### Data Processing Pipeline
        
        1. **Data Loading & Cleaning**
           - Load transaction and product data
           - Standardize column names  
           - Handle missing values
           
        2. **Feature Engineering**
           - Merge transactions with product data
           - Calculate transaction amounts
           - Create unique transaction keys
           
        3. **RFM Calculation**
           - Compute recency (days since last purchase)
           - Calculate frequency (number of transactions)
           - Sum monetary value (total spent)
        """)
        
        st.code("""
        # RFM Calculation Example
        rfm = transactions.groupby('customer').agg({
            'date': lambda x: (snapshot_date - x.max()).days,    # Recency
            'transaction': 'nunique',                            # Frequency  
            'amount': 'sum'                                      # Monetary
        })
        """, language="python")
    
    with method_tabs[1]:
        st.markdown("""
        ### Rule-Based Segmentation
        
        **Customer segments based on RFM scores (1-4 scale):**
        
        - **Champions** (444): Best customers - high value, frequent, recent
        - **Loyal Customers**: High frequency and monetary, may not be recent
        - **Potential Loyalists**: Recent customers with good frequency
        - **At Risk**: High-value customers who haven't purchased recently
        - **Can't Lose Them**: High monetary value but low recency
        - **Hibernating**: Lowest scores across all dimensions
        - **Need Attention**: All other customers requiring focus
        """)
        
        # Segment visualization
        st.markdown("**Segmentation Logic:**")
        st.code("""
        def segment_customer(row):
            if row['R_Score'] == 4 and row['F_Score'] == 4 and row['M_Score'] == 4:
                return 'Champions'
            elif row['F_Score'] >= 3 and row['M_Score'] >= 3:
                return 'Loyal Customers'  
            elif row['R_Score'] >= 3 and row['F_Score'] >= 2:
                return 'Potential Loyalists'
            # ... additional rules
        """, language="python")
    
    with method_tabs[2]:
        st.markdown("""
        ### K-Means Clustering
        
        **Unsupervised learning approach:**
        
        1. **Data Standardization**
           - Scale RFM features using StandardScaler
           - Ensure equal feature importance
           
        2. **Optimal Cluster Selection**  
           - Elbow method for inertia analysis
           - Silhouette score optimization
           - Business interpretability
           
        3. **Cluster Analysis**
           - Profile each cluster by RFM means
           - Assign business-friendly labels
           - Validate cluster quality
        """)
        
        st.code("""
        # K-Means Implementation
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(rfm_scaled)
        """, language="python")
    
    with method_tabs[3]:
        st.markdown("""
        ### Model Evaluation
        
        **Quality Metrics:**
        
        - **Silhouette Score**: Measures cluster cohesion and separation
        - **Elbow Method**: Identifies optimal number of clusters
        - **Business Validation**: Ensures segments make business sense
        - **Statistical Analysis**: Cluster size and distribution
        
        **Comparison Methods:**
        - Rule-based vs K-Means clustering
        - Cross-validation of segment stability  
        - Business outcome correlation
        """)
    
    st.markdown("---")
    
    # Project Structure
    st.markdown("## 🏗️ Project Architecture")
    
    architecture_cols = st.columns(2)
    
    with architecture_cols[0]:
        st.markdown("""
        ### 📁 Code Organization
        ```
        rfm_streamlit_app/
        ├── app.py                 # Main app
        ├── src/                   # Core logic
        │   ├── data_processing_eda.py
        │   └── rfm_segmentation.py
        ├── pages/                 # UI pages
        ├── components/            # UI components
        └── config/                # Settings
        ```
        """)
    
    with architecture_cols[1]:
        st.markdown("""
        ### 🛠️ Technology Stack
        - **Framework**: Streamlit
        - **Data Processing**: Pandas, NumPy  
        - **Visualization**: Matplotlib, Seaborn
        - **Machine Learning**: Scikit-learn
        - **Deployment**: Streamlit Cloud
        """)
    
    # Business Impact
    st.markdown("## 🎯 Business Impact")
    
    impact_cols = st.columns(2)
    
    with impact_cols[0]:
        st.success("""
        **Marketing Benefits**
        - Targeted campaign strategies
        - Personalized customer communication  
        - Improved conversion rates
        - Optimized marketing spend
        """)
    
    with impact_cols[1]:
        st.info("""
        **Business Outcomes**
        - Increased customer retention
        - Higher customer lifetime value
        - Better resource allocation
        - Data-driven decision making
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🚀 Ready to dive into the analysis? Navigate to EDA to start exploring the data!")

