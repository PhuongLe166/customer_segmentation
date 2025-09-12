# src/rfm_segmentation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


# ================================================================
# --- RULE-BASED RFM SCORING ---
# ================================================================

def calculate_rfm_scores(df_rfm : pd.DataFrame) -> pd.DataFrame :
    """Calculate R, F, M scores using quartiles and combine into RFM Score & Index."""

    # --- Recency Score ---
    df_rfm['R_Score'] = pd.qcut(df_rfm['Recency'], q = 4, labels = False, duplicates = 'drop')
    df_rfm['R_Score'] = 3 - df_rfm['R_Score']   
    df_rfm['R_Score'] = df_rfm['R_Score'] + 1

    # --- Frequency Score ---
    df_rfm['F_Score'] = pd.qcut(df_rfm['Frequency'], q = 4, labels = False, duplicates = 'drop')
    df_rfm['F_Score'] = df_rfm['F_Score'] + 1

    # --- Monetary Score ---
    df_rfm['M_Score'] = pd.qcut(df_rfm['Monetary'], q = 4, labels = False, duplicates = 'drop')
    df_rfm['M_Score'] = df_rfm['M_Score'] + 1

    # --- Combine RFM ---
    df_rfm['RFM_Score'] = (
        df_rfm['R_Score'].astype(str) +
        df_rfm['F_Score'].astype(str) +
        df_rfm['M_Score'].astype(str)
    )
    df_rfm['RFM_Index'] = (
        df_rfm['R_Score'] * 100 +
        df_rfm['F_Score'] * 10 +
        df_rfm['M_Score']
    )

    return df_rfm


def segment_customer(row) -> str :
    """Assign customer segment based on RFM scores."""

    if row['R_Score'] == 4 and row['F_Score'] == 4 and row['M_Score'] == 4 :
        return 'Champions'

    elif row['F_Score'] >= 3 and row['M_Score'] >= 3 :
        return 'Loyal Customers'

    elif row['R_Score'] >= 3 and row['F_Score'] >= 2 :
        return 'Potential Loyalists'

    elif row['R_Score'] == 1 and row['F_Score'] >= 3 :
        return 'At Risk'

    elif row['R_Score'] == 1 and row['M_Score'] >= 3 :
        return 'Can’t Lose Them'

    elif row['R_Score'] == 1 and row['F_Score'] == 1 and row['M_Score'] == 1 :
        return 'Hibernating'

    else :
        return 'Need Attention'


def apply_segmentation(df_rfm : pd.DataFrame) -> pd.DataFrame :
    """Apply segmentation rules to RFM-scored DataFrame."""
    df_rfm['Segment'] = df_rfm.apply(segment_customer, axis = 1)
    return df_rfm


# ================================================================
# --- KMEANS CLUSTERING ---
# ================================================================

def scale_rfm(df_rfm : pd.DataFrame) :
    """Standardize RFM features and return scaled array."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df_rfm[['Recency', 'Frequency', 'Monetary']])
    return rfm_scaled


def run_kmeans(df_rfm : pd.DataFrame, rfm_scaled, n_clusters : int = 3) -> pd.DataFrame :
    """Fit KMeans clustering, normalize cluster labels, and return updated df."""
    kmeans = KMeans(n_clusters = n_clusters, random_state = 5)
    df_rfm[f'KMeansCluster_{n_clusters}'] = kmeans.fit_predict(rfm_scaled)

    # Compute cluster profile
    cluster_profile = df_rfm.groupby(f'KMeansCluster_{n_clusters}')[
        ['Recency', 'Frequency', 'Monetary']
    ].mean()

    # Composite score for reordering clusters
    cluster_profile['ClusterScore'] = (
        -cluster_profile['Recency'] +
         cluster_profile['Frequency'] +
         cluster_profile['Monetary']
    )

    # Sort clusters and remap labels
    cluster_profile = cluster_profile.sort_values('ClusterScore', ascending = False)
    mapping = {old : new for new, old in enumerate(cluster_profile.index)}

    df_rfm[f'Cluster_{n_clusters}_Normalized'] = df_rfm[f'KMeansCluster_{n_clusters}'].map(mapping)

    return df_rfm


# ================================================================
# --- VISUALIZATIONS ---
# ================================================================

def plot_segment_distribution(df_rfm : pd.DataFrame) :
    """Plot distribution of rule-based customer segments and return fig."""
    fig, ax = plt.subplots(figsize = (10, 6))
    sns.countplot(
        data = df_rfm,
        x = "Segment",
        order = df_rfm["Segment"].value_counts().index,
        palette = "Set2",
        ax = ax
    )
    ax.set_title("Customer Segment Distribution", fontsize = 14, weight = "bold")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation = 45)
    fig.tight_layout()
    return fig


def plot_cluster_boxplots(df_rfm : pd.DataFrame, cluster_col : str) :
    """Boxplots for Recency, Frequency, Monetary by cluster."""
    fig, axes = plt.subplots(1, 3, figsize = (18, 5))

    for i, col in enumerate(["Recency", "Frequency", "Monetary"]) :
        sns.boxplot(
            data = df_rfm,
            x = cluster_col,
            y = col,
            hue = cluster_col,
            palette = "Set2",
            ax = axes[i]
        )
        axes[i].set_title(f"{col} by Cluster", fontsize = 12)
        axes[i].set_xlabel("")

        axes[i].tick_params(axis = "x", rotation = 30)
        if axes[i].get_legend() is not None :
            axes[i].get_legend().remove()

    fig.tight_layout()
    return fig


def plot_pairplot(df_rfm : pd.DataFrame, cluster_col : str) :
    """Pairplot of RFM variables colored by cluster."""
    g = sns.pairplot(
        df_rfm,
        hue = cluster_col,
        vars = ["Recency", "Frequency", "Monetary"],
        palette = "Set2"
    )
    return g.fig


def plot_cluster_treemap(df_rfm : pd.DataFrame, cluster_col : str) :
    """Treemap summarizing clusters by size and mean RFM values."""
    cluster_summary = df_rfm.groupby(cluster_col).agg(
        Recency = ("Recency", "mean"),
        Frequency = ("Frequency", "mean"),
        Monetary = ("Monetary", "mean"),
        Count = (cluster_col, "count")
    ).reset_index()

    total_customers = cluster_summary["Count"].sum()
    cluster_summary["Percent"] = 100 * cluster_summary["Count"] / total_customers

    labels = [
        (f"CLUSTER {row[cluster_col]}\n"
         f"{int(row.Recency)} days\n"
         f"{int(row.Frequency)} orders\n"
         f"{int(row.Monetary)} $\n"
         f"{row.Count} customers ({row.Percent:.2f}%)")
        for _, row in cluster_summary.iterrows()
    ]

    fig, ax = plt.subplots(figsize = (12, 6))
    squarify.plot(
        sizes = cluster_summary["Count"],
        label = labels,
        alpha = 0.8,
        color = plt.cm.Set3(range(len(cluster_summary))),
        text_kwargs = {"fontsize" : 10, "weight" : "bold"},
        ax = ax
    )
    ax.axis("off")
    ax.set_title(f"{len(cluster_summary)} Clusters – Treemap", fontsize = 16, weight = "bold")
    fig.tight_layout()
    return fig


# ================================================================
# --- CLUSTER QUALITY METRICS ---
# ================================================================

def compute_silhouette_scores(df_rfm : pd.DataFrame, rfm_scaled, cluster_col : str) -> dict :
    """
    Compute silhouette metrics for clustering.
    
    Returns:
        {
            "overall": float,
            "per_cluster": pd.Series,
            "per_customer": np.ndarray
        }
    """
    labels = df_rfm[cluster_col].values
    overall = silhouette_score(rfm_scaled, labels)
    per_customer = silhouette_samples(rfm_scaled, labels)
    per_cluster = pd.Series(per_customer).groupby(labels).mean()
    return {"overall": overall, "per_cluster": per_cluster, "per_customer": per_customer}


def plot_elbow_and_silhouette(rfm_scaled, k_range = range(2, 11)) :
    """Plot Elbow (inertia) and Silhouette scores for a range of k."""
    inertias = []
    silhouettes = []

    for k in k_range :
        kmeans = KMeans(n_clusters = k, random_state = 5)
        labels = kmeans.fit_predict(rfm_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(rfm_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize = (14, 5))

    # --- Elbow plot ---
    axes[0].plot(list(k_range), inertias, marker = "o")
    axes[0].set_title("Elbow Method (Inertia vs. k)", fontsize = 13, weight = "bold")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")

    # --- Silhouette plot ---
    axes[1].plot(list(k_range), silhouettes, marker = "o", color = "green")
    axes[1].set_title("Silhouette Score vs. k", fontsize = 13, weight = "bold")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    return fig

def plot_silhouette(df_rfm : pd.DataFrame, rfm_scaled, cluster_col : str):
    """Silhouette plot for clusters."""
    labels = df_rfm[cluster_col].values
    n_clusters = len(set(labels))

    sample_silhouette_values = silhouette_samples(rfm_scaled, labels)

    fig, ax = plt.subplots(figsize = (8, 6))
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor = color, alpha = 0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("Silhouette Plot for Clusters")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster")
    ax.axvline(x = np.mean(sample_silhouette_values), color = "red", linestyle = "--")
    return fig


def plot_silhouette_bar(per_cluster_scores : pd.Series) :
    """Barplot of average silhouette scores per cluster with distinct colors."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize = (8, 5))
    sns.barplot(
        x = per_cluster_scores.index.astype(str),
        y = per_cluster_scores.values,
        palette = "Set2",
        ax = ax
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Average Silhouette Score")
    ax.set_title("Silhouette Score by Cluster", fontsize = 13, weight = "bold")
    fig.tight_layout()
    return fig
