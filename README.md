# Vehicle Coupon Recommendation
This data was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios including the destination, current time, weather, passenger, etc., and then ask the person whether he will accept the coupon if he is the driver.

Given the high number of categorical variables, preprocessing was done separately from the data modelling for clarity.

For this analysis, three machine learning algorithms were used: Random Forest, Xgboost and Support Vector Machines.

The best model was xgboost which led to an accuracy of 80%. The data was resampled in order for it to be balanced.
