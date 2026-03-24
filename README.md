# Federated $k$-means with Enhanced Balance and Privacy

## Overview

Welcome to the official repository for **Fabric**, a privacy-preserving federated and balanced $k$-means clustering algorithm for Vertical Federated Learning (VFL), and **Fabric-NoDP**, its non-private counterpart. Both methods are designed to address the core challenges of **communication efficiency** and **feature heterogeneity** in **Vertical Federated Learning (VFL)**.


In addition to the core algorithms, this repository includes several comparative methods:

- **Lloyd**
- **CDKM**
- **BCLS**
- **FCE**
- **VFLC**
- **DPFMPS**
- **Teb**

Each of these algorithms is implemented for experimentation and comparison in federated clustering contexts. 

## Datasets

For testing and evaluation purposes, the following datasets are provided in `.csv` format. You can download them and use with the provided MATLAB code:

1. [Census](https://archive.ics.uci.edu/ml/datasets/Adult)
2. [Diabetes](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
3. [Letter](https://www.kaggle.com/datasets/nishan192/letterrecognition-using-svm)
4. [Loan](https://www.kaggle.com/competitions/home-credit-default-risk/overview)
5. [Patient](https://doi.org/10.1007/s10888-014-9191-2)
6. [Student](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)
7. [Taxi](https://www.kaggle.com/c/nyc-taxi-trip-duration)
8. [YearPredictionMSD](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd)
9. [Census1990](https://proceedings.neurips.cc/paper/2019/file/fc192b0c0d270dbf41870a63a8c76c2f-Paper)
10. [HMDA](https://ffiec.cfpb.gov/data-browser/)

## Installation

To run the algorithms, make sure to clone the repository and set up the necessary dependencies. You can start by executing the `run.m` script, which will run the core functions of the repository.

### Required Tools
- MATLAB 2022

## Fabricmeans Overview

`Fabricmeans ` implements the **Federated $k$-means with Enhanced Balance and Privacy** algorithm, which is used to solve clustering problems in the context of **Vertical Federated Learning (VFL)** while maintaining communication efficiency and ensuring privacy. The function performs the following tasks:

- Initializes cluster centers.
- Iteratively updates the cluster assignments and centroids.
- Minimizes the objective function considering a balance loss term.
- Computes the total sum of squared errors (SSE) for cluster assignment quality.
- Incorporates parallel processing for handling large datasets efficiently.

### Function Signature
```matlab
[Y, minO, iter_num, sse, obj, balance_loss, elapsed_time] = Fabricmeans(X, label, c, block_size, rho, numWorkers, max_iters)
```



## Running the Code

To run any of the methods, simply execute the `run.m` script. The script is designed to allow for easy switching between different clustering algorithms and comparison methods. It runs all the core algorithms in separate folders, where you can also access specific function files for each algorithm.

### Example Usage:

```bash
cd fabricmeans
#Run the main script:
matlab -r "run"
```

