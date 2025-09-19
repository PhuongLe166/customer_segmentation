#!/usr/bin/env python3
"""
Development runner script for RFM Streamlit App
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'scikit-learn', 'squarify', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_data_files():
    """Check if required data files exist"""
    data_dir = Path("data/raw")
    required_files = ["Transactions.csv", "Products_with_Categories.csv"]
    
    missing_files = []
    
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing data files: {', '.join(missing_files)}")
        print(f"Please place them in the {data_dir} directory")
        return False
    
    print("✅ All required data files found")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw", "data/processed", "models", "exports", 
        "assets/images", "assets/styles"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")

def run_streamlit_app():
    """Run the Streamlit application"""
    print("🚀 Starting Streamlit application...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True

def main():
    """Main function"""
    print("🎯 RFM Customer Segmentation Analysis")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        print("\n💡 Tip: You can still run the app to see the interface,")
        print("   but data-dependent features won't work until you add the CSV files.")
        
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run the app
    run_streamlit_app()

if __name__ == "__main__":
    main()