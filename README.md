# Quantifying the Impact of Sampling Error in Social Vulnerability Index Construction


---

## Notes for Reviewers

- It includes all **code** required to support the findings in the manuscript.
- The full index construction and sensitivity analysis were executed on a high-performance computing (HPC) cluster using 150 CPU cores. 
- This repository does not include reusable software packages developed for this study, which are under separate development and licensing.

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

## Data Sources

- Input data are not included in this repository because they are openly available from official sources and can be accessed directly:
- ACS Variance Replicate Tables (VRTs):
  - https://www.census.gov/programs-surveys/acs/data/variance-tables.html
- Code used for analysis is included in this repository. It enables full reproduction of the study’s results, subject to available computing resources.

---
