# AlixPartnersChallenge
 
This is the MATLAB script for the AlixPartners Challenge #1: P>N,
The crux of the challenge is to make inferences based on limited amounts of data. Specifically, in the dataset (available here www.alixpartners.com/insights-impact/insights/analytics-challenge/) there are 300 variables but our training dataset is only comprised of 250 rows.

My solution was to use K-fold cross validation with Partial Least Squares to reduce the dimensionality down to the optimum number of latents (calculated by the Mean Square Error of Prediction). Then run an optimised Support Vector Machine to classify the remaining 19,750 rows into the two classes (0 and 1).
AUC -ROC plots suggest an effectiveness of 0.999 with the training data.

