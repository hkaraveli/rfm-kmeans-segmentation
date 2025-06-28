# Customer Segmentation with RFM and K-Means Clustering (FLO)

**Author:** Halis Karaveli  
**Date:** 2025

---

## Project Overview

This project demonstrates customer segmentation using both rule-based (RFM) and unsupervised (K-Means clustering) methods.  
The business context is based on a retail dataset provided during the MIUUL Data Science Bootcamp.  
The project has been **extended and re-implemented** by Halis Karaveli, featuring additional clustering, visualization, and explanatory analysis, all presented in English.

---

## Business Problem

FLO, a major retailer, aims to segment its customers to develop targeted marketing strategies.  
The goal is to analyze customer behavior, identify meaningful groups, and extract actionable insights to increase retention and value.

---

## Dataset

The dataset includes 2020-2021 omnichannel customer transactions (online & offline).  
**Columns:**
- `master_id`: Unique customer ID
- `order_channel`: Platform used (Android, iOS, Desktop, Mobile, Offline)
- `last_order_channel`: Channel of last purchase
- `first_order_date`, `last_order_date`: Dates of first/last purchase
- `last_order_date_online`, `last_order_date_offline`
- `order_num_total_ever_online`, `order_num_total_ever_offline`: Total purchase counts by channel
- `customer_value_total_ever_online`, `customer_value_total_ever_offline`: Total spend by channel
- `interested_in_categories_12`: Categories shopped in last 12 months

*Data is for demonstration purposes only.*

---

## Methods

- **RFM Segmentation:**  
  Rule-based customer segmentation by Recency, Frequency, and Monetary value.
- **K-Means Clustering:**  
  Data-driven clustering for deeper customer insight.
- **Visualization:**  
  Includes segment profiling, cluster analysis, and visual summaries.
- **Marketing Use Cases:**  
  Shows how segments/clusters can be used for campaign targeting.

---

## Project Origin & Acknowledgments

This repository is inspired by an educational exercise from the MIUUL Data Science Bootcamp.  
All code, documentation, and extended analysis in this repository are original and written by Halis Karaveli for public portfolio and educational demonstration.

---

## Usage

1. Download the dataset as `flo_data_20k.csv` and place in your working directory.
2. Run the main script `rfm_kmeans_segmentation.py`.
3. Review generated CSV files and visual outputs.

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib
