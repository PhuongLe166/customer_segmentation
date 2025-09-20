# App configuration settings
APP_CONFIG = {
    "app_title": "RFM Customer Segmentation Analysis",
    "page_icon": "🎯",
    "layout": "wide",
    "sidebar_state": "expanded",
    "theme": "light",
    
    # Custom CSS
    "custom_css": """
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
        }
        
        .stButton > button {
            width: 100%;
            border-radius: 20px;
            height: 3em;
            background-color: transparent;
            border: 2px solid #1f77b4;
            color: #1f77b4;
        }
        
        .stButton > button:hover {
            background-color: #1f77b4;
            color: white;
            border: 2px solid #1f77b4;
        }
        
        .stButton > button[data-baseweb="button"][aria-pressed="true"] {
            background-color: #1f77b4;
            color: white;
        }
        
        .sidebar .sidebar-content {
            background-color: #fafbfc;
        }
        
        .stExpander > div:first-child {
            background-color: #e6f3ff;
        }
        
        h1, h2, h3 {
            color: #1f77b4;
        }
        
        .stMetric > div > div > div > div {
            color: #1f77b4;
        }
        </style>
    """
}

# Data configuration
DATA_CONFIG = {
    "transactions_file": "data/raw/Transactions.csv",
    "products_file": "data/raw/Products_with_Categories.csv",
    "processed_data_path": "data/processed/",
}

# Chart configuration
CHART_CONFIG = {
    "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "figure_size": (12, 6),
    "dpi": 100,
}

# Model configuration  
MODEL_CONFIG = {
    "default_clusters": 5,
    "random_state": 42,
    "rfm_quantiles": 4,
}

# Page content configuration
PAGE_CONFIG = {
    "introduction": {
        "title": "Welcome to RFM Customer Segmentation Analysis",
        "description": "Unlock the power of customer insights through RFM analysis"
    },
    "about": {
        "title": "About This Project",
        "description": "Overview, team, repository structure, and methodology"
    },
    "eda": {
        "title": "Exploratory Data Analysis", 
        "description": "Deep dive into the data patterns and trends"
    },
    "model_evaluation": {
        "title": "Model Building & Evaluation",
        "description": "RFM scoring, segmentation rules, and clustering analysis"
    },
    "bi_dashboard": {
        "title": "Business Intelligence Dashboard",
        "description": "Interactive dashboard with key business metrics"
    }
}