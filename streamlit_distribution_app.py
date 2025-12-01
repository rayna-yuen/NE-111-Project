#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:54:18 2025

@author: raynayuen
"""
# streamlit_distribution_app.py

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Distribution Fitter", layout="wide")
st.title("📊 Distribution Fitter App")

# ------------------------------
# Sidebar: Data Input
# ------------------------------
st.sidebar.header("Data Input")
input_method = st.sidebar.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

data = []
if input_method == "Manual Entry":
    raw_data = st.sidebar.text_area("Enter numbers separated by commas:")
    if raw_data != "":
        data_list = raw_data.split(",")
        temp_data = []
        for d in data_list:
            try:
                temp_data.append(float(d))
            except:
                st.sidebar.error("Invalid input detected. Please enter numbers only.")
        data = np.array(temp_data)
elif input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            st.sidebar.error("No numeric columns found in CSV.")
        else:
            if len(numeric_cols) > 1:
                col_choice = st.sidebar.selectbox("Pick column:", numeric_cols)
            else:
                col_choice = numeric_cols[0]
            data = df[col_choice].dropna().values
            st.sidebar.success("Using column: " + str(col_choice))

# ------------------------------
# Data Warnings
# ------------------------------
if len(data) == 0:
    st.warning("No data yet!")
elif len(data) == 1:
    st.warning("Only one data point, fitting may fail.")
elif np.all(data == data[0]):
    st.warning("All values identical, distribution fitting may fail.")
elif np.any(data <= 0):
    st.warning("Some values are zero or negative, some fits may not work.")

# ------------------------------
# Distributions
# ------------------------------
distributions = ["norm", "gamma", "expon", "beta", "weibull_min", "weibull_max",
                 "lognorm", "uniform", "triang", "pareto"]

dist_choice = st.sidebar.selectbox("Select distribution to fit:", distributions)

fit_results = []

# Only try fitting if there is more than 1 data point
if len(data) > 1:
    for dname in distributions:
        dist_obj = getattr(stats, dname)
        try:
            params = dist_obj.fit(data)
            # calculate mean absolute error manually
            hist_vals, bin_edges = np.histogram(data, bins=25, density=True)
            bin_centers = []
            for i in range(len(bin_edges)-1):
                bin_centers.append((bin_edges[i] + bin_edges[i+1])/2)
            pdf_vals = dist_obj.pdf(np.array(bin_centers), *params)
            error = 0
            for i in range(len(hist_vals)):
                error += abs(hist_vals[i]-pdf_vals[i])
            error = error / len(hist_vals)
            fit_results.append({"Distribution": dname, "Dist": dist_obj, "Params": params, "Error": error})
        except:
            continue

# sort manually
for i in range(len(fit_results)):
    for j in range(i+1, len(fit_results)):
        if fit_results[i]["Error"] > fit_results[j]["Error"]:
            temp = fit_results[i]
            fit_results[i] = fit_results[j]
            fit_results[j] = temp

st.subheader("✅ Best-Fitting Distributions")
if len(fit_results) > 0:
    table_data = []
    for r in fit_results:
        table_data.append({"Distribution": r["Distribution"], "Mean Abs Error": r["Error"], "Params": r["Params"]})
    st.table(table_data)

    # CSV export
    csv_buffer = io.StringIO()
    df_export = pd.DataFrame(table_data)
    df_export.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Fit Results as CSV", csv_buffer.getvalue(), "fit_results.csv", "text/csv")

    # Show selected distribution
    selected_dist = None
    selected_params = None
    selected_error = None
    for r in fit_results:
        if r["Distribution"] == dist_choice:
            selected_dist = r["Dist"]
            selected_params = r["Params"]
            selected_error = r["Error"]
            break

    # Only plot if we have data and a valid distribution
    if selected_dist is not None and len(data) > 1:
        x_vals = np.linspace(min(data), max(data), 1000)
        try:
            y_vals = selected_dist.pdf(x_vals, *selected_params)
        except:
            y_vals = np.zeros_like(x_vals)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(data, bins=25, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        color = 'green' if fit_results[0]["Distribution"] == dist_choice else 'red'
        ax.plot(x_vals, y_vals, color=color, lw=2)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(dist_choice + " Fit")
        st.pyplot(fig)

        st.subheader("Manual Fitting")
        param_names = []
        if selected_dist.shapes is not None:
            param_names = selected_dist.shapes.split(",")
            for i in range(len(param_names)):
                param_names[i] = param_names[i].strip()

        sliders = {}
        for i in range(len(param_names)):
            sliders[param_names[i]] = st.slider(param_names[i],
                                                float(selected_params[i]*0.5),
                                                float(selected_params[i]*1.5),
                                                float(selected_params[i]),
                                                0.01)

        sliders["loc"] = st.slider("loc",
                                   float(selected_params[-2]*0.5),
                                   float(selected_params[-2]*1.5),
                                   float(selected_params[-2]),
                                   0.01)
        sliders["scale"] = st.slider("scale",
                                     float(selected_params[-1]*0.5),
                                     float(selected_params[-1]*1.5),
                                     float(selected_params[-1]),
                                     0.01)

        manual_params = []
        for p in param_names:
            manual_params.append(sliders[p])
        manual_params.append(sliders["loc"])
        manual_params.append(sliders["scale"])

        # plot manual fit
        try:
            y_manual = selected_dist.pdf(x_vals, *manual_params)
        except:
            y_manual = np.zeros_like(x_vals)

        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.hist(data, bins=25, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        ax2.plot(x_vals, y_manual, color='orange', lw=2)
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Density")
        ax2.set_title("Manual " + dist_choice + " Fit")
        st.pyplot(fig2)
