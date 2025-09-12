# 🧩 Customer Clustering App

An interactive **Streamlit app** for customer segmentation using **RFM analysis** (rule-based) and **KMeans clustering**.  
It helps businesses understand customer behavior and identify groups like **Champions**, **Loyal Customers**, or **At Risk**.

---

## 📦 Features
- **RFM Rule-Based Segmentation** with intuitive customer groups
- **KMeans Clustering** with elbow method + silhouette score diagnostics
- Interactive **visualizations** (boxplots, pairplots, treemaps)
- **Cluster quality evaluation** with per-cluster silhouette scores

---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/your-username/customer_clustering_app.git
cd customer_clustering_app
```


Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate     # (Linux/Mac)
venv\Scripts\activate        # (Windows)

pip install -r requirements.txt
```

---

## ▶️ Run locally

```bash
streamlit run app.py
```

The app will start at [http://localhost:8501](http://localhost:8501).