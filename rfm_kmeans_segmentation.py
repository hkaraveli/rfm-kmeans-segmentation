###############################################################
# Customer Segmentation with RFM and K-Means Clustering (FLO)
# Author: Halis Karaveli, 2025
###############################################################

"""
Business Problem:
    Segment customers for personalized marketing using both rule-based (RFM) and unsupervised (K-Means) clustering.

Dataset:
    2020-2021 omnichannel customer data (online + offline). See README for details.

Project Origin & Contribution:
    This project was inspired by a customer segmentation assignment from the MIUUL Data Science Bootcamp.
    The analysis has been significantly extended with additional unsupervised clustering (K-Means), new visualizations,
    and comprehensive commentary by Halis Karaveli. All code, documentation, and business interpretation are original,
    written in English, and tailored for public portfolio demonstration.
"""

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ---- 1. Data Load & Preparation ----
df = pd.read_csv("flo_data_20k.csv")
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
# Optional: add average order value as a feature
df["avg_order_value"] = df["customer_value_total"] / df["order_num_total"]
date_cols = [col for col in df.columns if "date" in col]
df[date_cols] = df[date_cols].apply(pd.to_datetime)
analysis_date = dt.datetime(2021, 6, 1)

# ---- 2. RFM Calculation ----
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).dt.days
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]
rfm["avg_order_value"] = df["avg_order_value"]

# RFM scores (quintile binning)
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
rfm["RF_SCORE"] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)
rfm["RFM_SCORE"] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str)

# Segment Mapping (Rule-based)
segment_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'cant_lose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
rfm['rfm_segment'] = rfm['RF_SCORE'].replace(segment_map, regex=True)

# ---- 3. RFM Segment Summary ----
print("\n[RFM Segment Means]\n")
print(rfm.groupby("rfm_segment")[["recency", "frequency", "monetary", "avg_order_value"]].agg(["mean", "count"]))

# ---- 4. K-Means Clustering ----
# Optional: try clustering on more features, e.g., ["recency", "frequency", "monetary", "avg_order_value"]
cluster_features = ["recency", "frequency", "monetary", "avg_order_value"]
X = rfm[cluster_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k (Elbow & Silhouette)
sse = []
silhouette_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sse.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K_range, sse, "bo-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("SSE")
plt.title("Elbow Method")
plt.subplot(1,2,2)
plt.plot(K_range, silhouette_scores, "go-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores")
plt.tight_layout()
plt.show()

# ---- 5. Fit K-Means ----
optimal_k = 7  # (change if different k is optimal)
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# Cluster Profiling (explain clusters)
print("\n[K-Means Cluster Means]\n")
print(rfm.groupby("kmeans_cluster")[["recency", "frequency", "monetary", "avg_order_value"]].agg(["mean", "count"]))

# Optional: assign cluster names based on business meaning
cluster_names = {
    0: "High-Value Active",
    1: "Dormant",
    2: "Average",
    3: "Super Loyal",
    4: "Ultra VIP",
    5: "At Risk",
    6: "Churned VIP"
}
rfm["kmeans_label"] = rfm["kmeans_cluster"].map(cluster_names)

# ---- 6. Visualize Clusters ----
plt.figure(figsize=(8,6))
scatter = plt.scatter(rfm["recency"], rfm["monetary"], c=rfm["kmeans_cluster"], cmap="tab10", alpha=0.6)
plt.xlabel("Recency (days)")
plt.ylabel("Monetary Value")
plt.title("K-Means Clusters (Recency vs. Monetary)")
plt.colorbar(scatter, label="Cluster")
plt.show()

# ---- 7. Export Results ----
rfm.to_csv("customer_rfm_kmeans_segments.csv", index=False)

# ---- 8. Marketing Use Cases ----
# a. Champions and loyal_customers, interested in "KADIN"
female_target_ids = df[
    (df["master_id"].isin(rfm[rfm["rfm_segment"].isin(["champions", "loyal_customers"])]["customer_id"])) &
    (df["interested_in_categories_12"].str.contains("KADIN"))
]["master_id"]
female_target_ids.to_csv("new_brand_target_customer_ids.csv", index=False)

# b. Men/children's discount campaign: Extract target customer IDs from the selected segments
male_kids_target_ids = df[
    (df["master_id"].isin(
        rfm[rfm["rfm_segment"].isin(["cant_lose", "hibernating", "new_customers"])]["customer_id"]
    )) &
    (df["interested_in_categories_12"].str.contains("ERKEK|COCUK"))
]["master_id"]

# Save the target IDs to CSV
male_kids_target_ids.to_csv("discount_target_customer_ids.csv", index=False)

# Inspect main statistics for target segment
target_info = df[df["master_id"].isin(male_kids_target_ids)]
print(target_info[["master_id", "order_num_total", "customer_value_total", "interested_in_categories_12"]].head(10))

# Statistical summary for key features
print("\n[customer_value_total - Discount Target Segment]\n", target_info["customer_value_total"].describe())
print("\n[order_num_total - Discount Target Segment]\n", target_info["order_num_total"].describe())

# Filter: Customers interested specifically in 'COCUK' category
kids_only = target_info[target_info["interested_in_categories_12"].apply(lambda x: "COCUK" in str(x))]
print("\n[Customers in 'COCUK' category]\n", kids_only[["master_id", "order_num_total", "customer_value_total", "interested_in_categories_12"]].head(10))
print("\n[customer_value_total - Only Kids Category]\n", kids_only["customer_value_total"].describe())

