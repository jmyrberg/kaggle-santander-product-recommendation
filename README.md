# Kaggle Competition: Santander Product Recommendation

This repository contains code used to achieve Top 8% spot in Kaggle competition [Santander Product Recommendation] (https://www.kaggle.com/c/santander-product-recommendation).

Several different models were tried
* Association rules
* Collaborative Filtering (item and user-based)
* LSTM Neural Network
* XGBoost

In the end, the best score was achieved with XGBoost and June 2016 training data only. The best submission is a weighted combination of:
* bag of 10 XGB models (weight of 0.5)
* best single XGB submission (weight of 0.25)
* second best XGB submission (weight of 0.25)

In this competition, seasonal effects played a big role. Therefore, it was important to choose training data correctly. Traditional models, such as association rules or collaborative filtering did not work well, at least when trained on all train data. LSTM's performance was similar to XGB's, but the first six training months was missing data, and even with different data imputation methods, the performance seemed to always be worse than XGB's.

