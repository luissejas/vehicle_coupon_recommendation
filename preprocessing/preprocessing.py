import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Reading the dataset
vehicle_recommendation_df = pd.read_csv(r"C:\Users\Lenovo\Dropbox\PC\Desktop\Files\Kaggle\in+vehicle+coupon+recommendation\in-vehicle-coupon-recommendation.csv")

# The dataset has a lot of categorical columns, we will need to find the dummy variables. This list will be used to filter the columns
# needed to find the dummy variables
columns_to_get_dummy = []

# ****Column Destination****

# Doing some adjustments to the records
vehicle_recommendation_df.replace({'destination':'No Urgent Place'}, 'other', inplace=True)
vehicle_recommendation_df.replace({'destination':'Home'}, 'home', inplace=True)
vehicle_recommendation_df.replace({'destination':'Work'}, 'work', inplace=True)

# Add as a column to get dummy later on
columns_to_get_dummy.append('destination')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# ****Column Passenger****

# Renaming the column to remove typo
vehicle_recommendation_df.rename(columns={'passanger':'passenger'}, inplace=True)

# Adjusting the values
vehicle_recommendation_df.replace({'passenger':'Alone'}, 'alone', inplace=True)
vehicle_recommendation_df.replace({'passenger':'Friend(s)'}, 'friend', inplace=True)
vehicle_recommendation_df.replace({'passenger':'Partner'}, 'partner', inplace=True)
vehicle_recommendation_df.replace({'passenger':'Kid(s)'}, 'kid', inplace=True)

#Adding the column into the dummy list
columns_to_get_dummy.append('passenger')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# ****Column Weather****

# Renaming the values to lower case
vehicle_recommendation_df.replace({'weather':'Sunny'}, 'sunny', inplace=True)
vehicle_recommendation_df.replace({'weather':'Snowy'}, 'snowy', inplace=True)
vehicle_recommendation_df.replace({'weather':'Rainy'}, 'rainy', inplace=True)

# Adding the column to the dummy list
columns_to_get_dummy.append('weather')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# ****Column Time****

vehicle_recommendation_df.replace({'time':'7AM'}, 'early_morning', inplace=True)
vehicle_recommendation_df.replace({'time':'10AM'}, 'late_morning', inplace=True)
vehicle_recommendation_df.replace({'time':'2PM'}, 'midday', inplace=True)
vehicle_recommendation_df.replace({'time':'6PM'}, 'afternoon', inplace=True)
vehicle_recommendation_df.replace({'time':'10PM'}, 'evening', inplace=True)

# Add to dummy list
columns_to_get_dummy.append('time')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# ****Column coupon****

# We can see here that there are two categories of restaurant which varies by cost, we will use cheap and expensive to distinguish them
vehicle_recommendation_df.replace({'coupon':'Coffee House'}, 'coffee', inplace=True)
vehicle_recommendation_df.replace({'coupon':'Restaurant(<20)'}, 'restaurant_cheap', inplace=True)
vehicle_recommendation_df.replace({'coupon':'Carry out & Take away'}, 'takeaway', inplace=True)
vehicle_recommendation_df.replace({'coupon':'Bar'}, 'bar', inplace=True)
vehicle_recommendation_df.replace({'coupon':'Restaurant(20-50)'}, 'restaurant_expensive', inplace=True)

# Add column as dummy
columns_to_get_dummy.append('coupon')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# ****Column Expiration ****

#Renaming the values to day and hour
vehicle_recommendation_df.replace({'expiration':'1d'}, 'day', inplace=True)
vehicle_recommendation_df.replace({'expiration':'2h'}, 'hour', inplace=True)

# Add to dummies list
columns_to_get_dummy.append('expiration')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# ****Column gender****

# Gender only needs to be put in lower case
vehicle_recommendation_df['gender'] = vehicle_recommendation_df['gender'].str.lower()

# Add to dummy list
columns_to_get_dummy.append('gender')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# **** Column Age ****

# To account age as a numerical variable, the values 50plus and below21 will be arbritarily changed to 60 and 15
vehicle_recommendation_df.replace({'age':'50plus'}, 60, inplace=True)
vehicle_recommendation_df.replace({'age':'below21'}, 15, inplace=True)

# Changing to integer
vehicle_recommendation_df['age'] = vehicle_recommendation_df['age'].astype('int16')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# **** Column MaritalStatus ****

# Converting all strings to lower case and changing the two word entries
vehicle_recommendation_df['maritalStatus'] = vehicle_recommendation_df['maritalStatus'].str.lower()
vehicle_recommendation_df.replace({'maritalStatus':'married partner'}, 'married', inplace=True)
vehicle_recommendation_df.replace({'maritalStatus':'unmarried partner'}, 'dating', inplace=True)

# Add column to dummy list
columns_to_get_dummy.append('maritalStatus')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# **** Column Education ****

#Renaming values
vehicle_recommendation_df.replace({'education':'Some college - no degree'}, 'college_incomplete', inplace=True)
vehicle_recommendation_df.replace({'education':'Bachelors degree'}, 'bachelor', inplace=True)
vehicle_recommendation_df.replace({'education':'Graduate degree (Masters or Doctorate)'}, 'graduate', inplace=True)
vehicle_recommendation_df.replace({'education':'Associates degree'}, 'associate', inplace=True)
vehicle_recommendation_df.replace({'education':'High School Graduate'}, 'high_school', inplace=True)
vehicle_recommendation_df.replace({'education':'Some High School'}, 'high_school_incomplete', inplace=True)

#Add to dummy list
columns_to_get_dummy.append('education')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# **** Column Occupation ****

vehicle_recommendation_df.replace({'occupation':['Unemployed', 'Student', 'Retired']}, 'no_job', inplace=True)
vehicle_recommendation_df.replace({'occupation':['Sales & Related','Management','Office & Administrative Support',
                                                 'Business & Financial', 'Legal','Computer & Mathematical']},'office', inplace=True)
vehicle_recommendation_df.replace({'occupation':['Education&Training&Library','Arts Design Entertainment Sports & Media',
                                                 'Community & Social Services','Personal Care & Service', 'Protective Service']},'service',inplace=True)
vehicle_recommendation_df.replace({'occupation':['Food Preparation & Serving Related','Healthcare Practicioners & Technical',
                                                 'Healthcare Support', 'Transportation & Material Moving','Architecture & Engineering',
                                                 'Life Physical Social Science','Construction & Extraction','Installation Maintenance & Repair',
                                                 'Production Occupations', 'Building & Grounds Cleaning & Maintenance',
                                                 'Farming Fishing & Forestry','Healthcare Practitioners & Technical']}, 'specialized', inplace=True)

# Add to dummy list
columns_to_get_dummy.append('occupation')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# **** Column Income ****

# The median salary in the US is about 54k a year. We will separate income into three buckets:low, average, high
vehicle_recommendation_df.replace({'income':['Less than $12500','$12500 - $24999', '$25000 - $37499']}, 'low', inplace=True)
vehicle_recommendation_df.replace({'income':['$37500 - $49999','$50000 - $62499', '$62500 - $74999','$75000 - $87499']}, 'average', inplace=True)
vehicle_recommendation_df.replace({'income':['$87500 - $99999', '$100000 or More']},'high', inplace=True)

# Add to dummy list
columns_to_get_dummy.append('income')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# **** Column Car ****

# The number of values are you too low, this column needs to be removed from the dataset
vehicle_recommendation_df.drop(columns=['car'], inplace=True)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# **** Column Bar ****

# Renaming the column to lower case
vehicle_recommendation_df.rename(columns={'Bar':'bar'}, inplace=True)

# Applying the transformations
vehicle_recommendation_df.replace({'bar':['never', 'less1']}, 0, inplace=True)
vehicle_recommendation_df.replace({'bar':'1~3'}, 2, inplace=True)
vehicle_recommendation_df.replace({'bar':'4~8'}, 6, inplace=True)
vehicle_recommendation_df.replace({'bar':'gt8'}, 10, inplace=True)
vehicle_recommendation_df.replace({'bar':np.NaN}, 0, inplace=True)

# Changing to integer
vehicle_recommendation_df['bar'] = vehicle_recommendation_df['bar'].astype('int16')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# **** Column Coffeehouse ****

#Renaming the column
vehicle_recommendation_df.rename(columns={'CoffeeHouse':'coffeehouse'}, inplace=True)

# Applying the same transformation as bar
vehicle_recommendation_df.replace({'coffeehouse':['never', 'less1']}, 0, inplace=True)
vehicle_recommendation_df.replace({'coffeehouse':'1~3'}, 2, inplace=True)
vehicle_recommendation_df.replace({'coffeehouse':'4~8'}, 6, inplace=True)
vehicle_recommendation_df.replace({'coffeehouse':'gt8'}, 10, inplace=True)
vehicle_recommendation_df.replace({'coffeehouse':np.NaN}, 0, inplace=True)

# Transforming the column
vehicle_recommendation_df['coffeehouse'] = vehicle_recommendation_df['coffeehouse'].astype('int16')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# **** Column CarryAway ****

# Renaming the column
vehicle_recommendation_df.rename(columns={'CarryAway':'carryaway'}, inplace=True)

# Applying the same transformation as bar
vehicle_recommendation_df.replace({'carryaway':['never', 'less1']}, 0, inplace=True)
vehicle_recommendation_df.replace({'carryaway':'1~3'}, 2, inplace=True)
vehicle_recommendation_df.replace({'carryaway':'4~8'}, 6, inplace=True)
vehicle_recommendation_df.replace({'carryaway':'gt8'}, 10, inplace=True)
vehicle_recommendation_df.replace({'carryaway':np.NaN}, 0, inplace=True)

# Transforming the column
vehicle_recommendation_df['carryaway'] = vehicle_recommendation_df['carryaway'].astype('int16')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# **** Column RestaurantLessThan 20 ****

# Renaming the column
vehicle_recommendation_df.rename(columns={'RestaurantLessThan20':'restaurant_cheap'}, inplace=True)

#Applying transformation
vehicle_recommendation_df.replace({'restaurant_cheap':['never', 'less1']}, 0, inplace=True)
vehicle_recommendation_df.replace({'restaurant_cheap':'1~3'}, 2, inplace=True)
vehicle_recommendation_df.replace({'restaurant_cheap':'4~8'}, 6, inplace=True)
vehicle_recommendation_df.replace({'restaurant_cheap':'gt8'}, 10, inplace=True)
vehicle_recommendation_df.replace({'restaurant_cheap':np.NaN}, 0, inplace=True)

# Transforming the column
vehicle_recommendation_df['restaurant_cheap'] = vehicle_recommendation_df['restaurant_cheap'].astype('int16')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# **** Column Restaurant20To50 ****

# Renaming the column
vehicle_recommendation_df.rename(columns={'Restaurant20To50':'restaurant_expensive'}, inplace=True)

#Applying transformation
vehicle_recommendation_df.replace({'restaurant_expensive':['never', 'less1']}, 0, inplace=True)
vehicle_recommendation_df.replace({'restaurant_expensive':'1~3'}, 2, inplace=True)
vehicle_recommendation_df.replace({'restaurant_expensive':'4~8'}, 6, inplace=True)
vehicle_recommendation_df.replace({'restaurant_expensive':'gt8'}, 10, inplace=True)
vehicle_recommendation_df.replace({'restaurant_expensive':np.NaN}, 0, inplace=True)

# Transforming the column
vehicle_recommendation_df['restaurant_expensive'] = vehicle_recommendation_df['restaurant_expensive'].astype('int16')

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# For the other columns, no preprocessing was needed, now we can start to get the dummies and generate the output

# First, we will normalize the data
scaler=MinMaxScaler()

# Normalizing the numerical columns
columns_to_normalize = ['temperature', 'age', 'bar', 'coffeehouse', 'carryaway', 'restaurant_cheap', 'restaurant_expensive']

for col in columns_to_normalize:
    vehicle_recommendation_df[col] = scaler.fit_transform(vehicle_recommendation_df[col].values.reshape(-1,1))

# Get dummies and generate output

# Getting dummies of categorical values and removing a category to prevent perfect collinearity
vehicle_recommendation_dummies_df = pd.get_dummies(vehicle_recommendation_df[columns_to_get_dummy], drop_first=True)

# We no longer need the original columns
vehicle_recommendation_df.drop(columns=columns_to_get_dummy, inplace=True)
vehicle_recommendation_df_consolidated = vehicle_recommendation_df.join(vehicle_recommendation_dummies_df)

# Exporting the df
vehicle_recommendation_df_consolidated.to_csv('vehicle_recommendation_consolidated.csv')


