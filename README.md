# Fabricmeans
 Code for paper "ADMM for Federated 𝑘-means with Differential Privacy under Balance Constraints"

## Download data

To run the provided code, you will need to download the following datasets. After downloading, you can use the data in `.csv` format (DATASET_NAME.csv) within your MATLAB code.

Athlete https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results
Census https://archive.ics.uci.edu/ml/datasets/Adult
Spanish https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring/
NHANES https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset
Diabetes https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
Conhort https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records
Student https://analyse.kmi.open.ac.uk/open_dataset
Pricerunner https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering
Census1990 https://proceedings.neurips.cc/paper/2019/file/fc192b0c0d270dbf41870a63a8c76c2f-Paper
HMDA https://ffiec.cfpb.gov/data-browser/

## Start

There are four functions/apis available from the Fabricmeans folder (Fabricmeans, fabricmeans-DP, CDKM, and LLoyd). Try running "run.m". See run.m for how to use these two functions.

### Additional Methods for Comparison

In addition to the main **Fabricmeans** method, two balance comparison methods have been provided for benchmarking purposes:

- **BCLS**
  
- **BKM**

You can run these methods by running 'run.m' in the corresponding folder.

