# Retail Inventory Demand Prediction Using Seasonal Analysis and Forecast Models

## Project Overview

Retail businesses need to maintain sufficient inventory to meet customer demand while avoiding unnecessary overstocking. Accurate demand forecasting can help businesses make better inventory planning and replenishment decisions.

This project develops a machine learning-based retail demand forecasting system using historical transaction data. The system performs data cleaning, exploratory data analysis, seasonal analysis, product-level time-series analysis, feature engineering, model training, evaluation, and future demand forecasting.

A Random Forest Regression model is used to predict future daily demand for selected retail products. The trained models are integrated into an interactive Streamlit dashboard where users can select a product and forecast demand for the next 1–30 days.

---

## Problem Statement

Historical retail transaction data contains useful patterns related to product demand, seasonality, day-of-week effects, and recent purchasing behavior. However, raw transaction data cannot be directly used for forecasting because it may contain cancelled transactions, duplicate records, invalid quantities, missing product information, and irregular product-level time series.

The objective of this project is to transform the raw retail transaction data into a clean product-level time series and develop a machine learning forecasting system that can estimate future daily demand.

---

## Objectives

The main objectives of this project are:

- Clean and preprocess the raw retail transaction dataset.
- Analyze transaction patterns and demand over time.
- Identify seasonal and day-of-week demand patterns.
- Analyze demand at the individual product level.
- Construct continuous daily product-level time series.
- Engineer time-based, lag-based, and rolling statistical features.
- Use a time-based train-test split for forecasting evaluation.
- Establish baseline forecasting methods.
- Train and evaluate a Random Forest Regression model.
- Analyze model errors and identify forecasting limitations.
- Generate recursive multi-day demand forecasts.
- Build an interactive Streamlit dashboard for business-oriented forecasting.

---

# Dataset

The project uses the **Online Retail II** dataset from the UCI Machine Learning Repository.

## Dataset Source

- **Dataset:** Online Retail II
- **Repository:** UCI Machine Learning Repository
- **Dataset ID:** 502
- **License:** CC BY 4.0
- **Citation:** Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
- **DOI:** https://doi.org/10.24432/C5CG6D

The dataset contains transaction records from a UK-based online retail business. The transaction data includes information about invoices, products, quantities, transaction dates, prices, customers, and countries.

## Dataset Used in This Project

The downloaded Excel file used in this project contains:

- **Rows:** 525,461
- **Columns:** 8
- **Unique invoices:** 28,816
- **Unique products:** 4,252
- **Unique customers:** 4,383
- **Unique countries:** 40

## Date Coverage

The actual downloaded dataset used in this project covers:

```text
01 December 2009 → 09 December 2010