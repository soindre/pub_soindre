# Quantifying the Impact of Sampling Error in Social Vulnerability Index Construction

---

## Overview of the Analysis

### 1. Data Preparation
- Download American Community Survey (ACS) **Variance Replicate Tables (VRTs)** from [Census Bureau](https://www.census.gov/programs-surveys/acs/data/variance-tables.html)
- Construct composite variables (e.g., elderly population) using multiple ACS indicators
- Calculate **Margins of Error (MOE)** for these composite variables

### 2. Vulnerability Index Construction
- Build **81 versions** of a social vulnerability index
- Quantify and visualize the MOE for each index replicate
- Analyze how MOE varies across the vulnerability spectrum

### 3. Global Sensitivity Analysis
- Define parameter space (normalization, aggregation, MOE bounds, weights)
- Compute **Sobol indices** to assess sensitivity of results to each modeling choice
- Visualize rank variation and interaction effects

---

## Notes for Reviewers

- This repository **does not include the full codebase** or reusable software packages developed for this study, which are under separate development and licensing.
- It includes all **code** required to support the findings in the manuscript.

---

## Data Sources

All input data are **open source** and publicly available:

- **ACS Variance Replicate Tables (VRTs)**:  
  https://www.census.gov/programs-surveys/acs/data/variance-tables.html

Data used here have been downloaded, aggregated, and pre-processed for analysis. No proprietary datasets are used.

---
