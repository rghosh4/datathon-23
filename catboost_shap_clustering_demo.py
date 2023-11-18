# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:42:09 2023

@author: Riddhi
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
pio.renderers.default='browser'

## Updating the Directory and just running the Entire Script should work 
os.chdir("C:/Users/Riddhi/Downloads/")



# Load the dataset
df = pd.read_excel('datathon_data.xlsx')
#df.info()
#rows_with_nan_df = df[df.isna().any(axis=1)]
# Obtain rows with NaN values in the specified columns
rows_with_nan_df = df[df[['First_Open_Dt_Deposit', 'First_Open_Dt_Loan', 'First_Open_Dt_Card', 'First_Open_Dt_Mortgage']].isna().all(axis=1)]


# Get the row indexes from rows_with_nan_df
indexes_to_drop = rows_with_nan_df.index

# Drop the rows with missing values from the original DataFrame df
df = df.drop(indexes_to_drop)


###### Feature Engineering #########

# Define New Customer
df['new_customer'] = df[['First_Open_Dt_Deposit', 'First_Open_Dt_Loan', 'First_Open_Dt_Card', 'First_Open_Dt_Mortgage']].min(axis=1).ge(202301).astype(int)
df['new_customer'] = df['new_customer'].astype('category')



# Extract unique prefixes
prefixes = set(col.rsplit('_', 1)[0] for col in df.columns if col.endswith(tuple(['_' + str(i) for i in range(202303, 202308)])))

#Median  and Std dev for Numerical
# For each unique prefix, compute median and standard deviation and create respective  new columns
for prefix in prefixes:
    cols_of_interest = [col for col in df.columns if col.startswith(prefix)]
    df[prefix + '_median'] = df[cols_of_interest].median(axis=1)
    df[prefix + '_std_deviation'] = df[cols_of_interest].std(axis=1)

    
# Drop the specified columns like wealth product median and online enrolled median
columns_to_drop = ['Online_Enrolled_median', 'Online_Enrolled_std_deviation', 'wealth_product_median', 'wealth_product_std_deviation']
df = df.drop(columns=columns_to_drop, errors='ignore')  # ensures no error is raised if a column doesn't exist


# Modes for Categorical
# Specify unique prefixes 
prefixes1 = set(['wealth_product_', 'Online_Enrolled_'])

# For each unique prefix, compute mode and create new column
for prefix in prefixes1:
    cols_of_interest = [col for col in df.columns if col.startswith(prefix)]
    df[prefix + 'mode'] = df[cols_of_interest].mode(axis=1)[0]   
    


# Impute NaN values in Mode columns with 'N'
columns_to_impute = ['Online_Enrolled_mode', 'wealth_product_mode']
for column in columns_to_impute:
    df[column] = df[column].fillna('N')





# Denominator: Sum of 'Avg_Checking_Bal_median', 'Avg_Savings_Bal_median', and 'Avg_CD_Bal_median'
denominator = df['Avg_Checking_Bal_median'] + df['Avg_Savings_Bal_median'] + df['Avg_CD_Bal_median']

# Adjust the denominator: if it's 0, use 1 instead
denominator = denominator.where(denominator != 0, 1)


# New feature: 'Normalized Monthly Large Owed Amount'
numerator0 = df['Avg_Mortgage_Bal_median'] + df['Avg_Loan_Bal_median'] 
df['average_large_owed_median'] = numerator0 / denominator



# New feature: 'Normalized Monthly Small Owed Amount'
numerator1 = df['Avg_CreditCard_Bal_median'] + df['DebitCard_Spend_median']
df['average_small_owed_median'] = numerator1 / denominator



# New feature: 'Normalized Monthly Remote Deposit Amount'
numerator2 = df['Remote_Dep_Amt_median']
df['average_remote_gained_median'] = numerator2 / denominator



##(Customer Tenure)
# Define a function to calculate the difference in months
def months_difference(date1, date2):
    year_diff = (date1 // 100) - (date2 // 100)
    month_diff = (date1 % 100) - (date2 % 100)
    return year_diff * 12 + month_diff

# Calculate age in system in terms of months 
df['age_in_system_months'] = df[['First_Open_Dt_Deposit', 'First_Open_Dt_Loan', 'First_Open_Dt_Card', 'First_Open_Dt_Mortgage']].min(axis=1).apply(lambda x: months_difference(202308, x))


## Account Category
def categorize_account(row):
    # Define the accounts for easier referencing
    deposit_accounts = ['CD_Accts_median', 'Checking_Accts_median', 'Savings_Accts_median']
    loan_account = 'Loan_Accts_median'
    card_account = 'CreditCard_Accts_median'
    mortgage_account = 'Mortgage_Accts_median'
    
    has_deposit = any([row[acct] > 0 for acct in deposit_accounts])
    has_loan = row[loan_account] > 0
    has_card = row[card_account] > 0
    has_mortgage = row[mortgage_account] > 0

    # Define the conditions for each category
    conditions = [
        (has_deposit and not has_loan and not has_card and not has_mortgage, "deposit_only"),
        (not has_deposit and has_loan and not has_card and not has_mortgage, "loan_only"),
        (not has_deposit and not has_loan and has_card and not has_mortgage, "card_only"),
        (not has_deposit and not has_loan and not has_card and has_mortgage, "mortgage_only"),
        (has_deposit and has_loan and not has_card and not has_mortgage, "deposit_loan"),
        (has_deposit and not has_loan and has_card and not has_mortgage, "deposit_card"),
        (has_deposit and not has_loan and not has_card and has_mortgage, "deposit_mortgage"),
        (not has_deposit and has_loan and has_card and not has_mortgage, "loan_card"),
        (not has_deposit and has_loan and not has_card and has_mortgage, "loan_mortgage"),
        (not has_deposit and not has_loan and has_card and has_mortgage, "card_mortgage"),
        (has_deposit and has_loan and has_card and not has_mortgage, "deposit_loan_card"),
        (has_deposit and has_loan and not has_card and has_mortgage, "deposit_loan_mortgage"),
        (has_deposit and not has_loan and has_card and has_mortgage, "deposit_card_mortgage"),
        (not has_deposit and has_loan and has_card and has_mortgage, "loan_card_mortgage"),
        (has_deposit and has_loan and has_card and has_mortgage, "deposit_loan_card_mortgage"),
    ]
    
    # go through conditions and return the category when match met
    for condition, category in conditions:
        if condition:
            return category
    return "other"  # If no condition is met (should never reach here)

# Create the new feature
df['account_category'] = df.apply(categorize_account, axis=1)

# Sanity Check
df['account_category'].value_counts()

# Given value counts
value_counts = df['account_category'].value_counts()

# Find categories with counts less than or equal to 20 (Eyeballing/ Subjective)
low_cardinality_categories = value_counts[value_counts <= 20].index

# Update the 'account_category' to replace low cardinality categories with 'other'
df['account_category'] = df['account_category'].replace(low_cardinality_categories, 'other')




# Median of  month over month percatage change feature ... idea dropped
# months_range = [str(i) for i in range(202303, 202309)]  # Up to and including '202308'

# # For each unique prefix, compute month-over-month percentage difference and create a new column for median percentage difference
# for prefix in prefixes:
#     
#     cols_of_interest = [col for col in df.columns if col.startswith(prefix) and col.endswith(tuple(['_' + month for month in months_range]))]
    
#     # sort in chronological order
#     cols_of_interest.sort()
    
#     # only numeric columns for percentage calculations
#     numeric_cols = df[cols_of_interest].select_dtypes(include=['number'])
    
#     # check if any column in the row has a non-zero value in the specified range
#     nonzero_row_mask = (numeric_cols != 0).any(axis=1)
    
#     # add 0.01 to each column value if any column in the row has a non-zero value
#     numeric_cols.loc[nonzero_row_mask] += 0.01
    
#     # calculate month-over-month percentage difference for the numeric columns
#     percentage_difference_df = numeric_cols.pct_change(axis=1) * 100  
    
#     # calculate the median percentage difference for each row
#     median_percentage_difference = percentage_difference_df.median(axis=1)
    
#     # new column for the median percentage difference
#     df[prefix + '_median_pct_difference'] = median_percentage_difference




# potential responses (all the digital stuff)

cols = ['OnlineWallet_Dollars_median', 'DigLogins_TotalLogins_median', 'DigLogins_UniqDays_median', 
        'OnlineWallet_Tx_median', 'Alerts_Enrolled_median', 'Remote_Dep_Amt_median', 'Remote_Dep_Ct_median']

# subset
subset_df = df[cols]

# correlation matrix
correlation_matrix = subset_df.corr()

# correlation heatmap
fig = px.imshow(correlation_matrix,
                x=subset_df.columns,
                y=subset_df.columns,
                color_continuous_scale='darkmint')  

fig.update_layout(title='Correlation Heatmap')
fig.show()




# Initial Numeric Columns Sanity Check
columns_to_use = ['new_customer','account_category','Alerts_Enrolled_median',
'DebitCard_Tx_median',
'DebitCard_Tx_std_deviation',
'Checking_Accts_median',
'Checking_Accts_std_deviation',
'BranchTX_median',
'BranchTX_std_deviation',
'DebitCard_Spend_median',
'DebitCard_Spend_std_deviation',
'Avg_Savings_Bal_median',
'Avg_Savings_Bal_std_deviation',
'Avg_Mortgage_Bal_median',
'Avg_Mortgage_Bal_std_deviation',
'Avg_CreditCard_Bal_median',
'Avg_CreditCard_Bal_std_deviation',
'Loan_Accts_median',
'Loan_Accts_std_deviation',
'CD_Accts_median',
'CD_Accts_std_deviation',
'Avg_Loan_Bal_median',
'Avg_Loan_Bal_std_deviation',
'Avg_Checking_Bal_median',
'Avg_Checking_Bal_std_deviation',
'Savings_Accts_median',
'Savings_Accts_std_deviation',
'Avg_CD_Bal_median',
'Avg_CD_Bal_std_deviation',
'CreditCard_Accts_median',
'CreditCard_Accts_std_deviation',
'Mortgage_Accts_median',
'Mortgage_Accts_std_deviation',
'wealth_product_mode',
'average_large_owed_median',
'average_small_owed_median',
'average_remote_gained_median',
'age_in_system_months','age_range', 'Generation', 'Market']

# filter out only numeric columns 
numeric_cols = df[columns_to_use].select_dtypes(include=['float64', 'int64']).columns.tolist()

# subset
subset_df = df[numeric_cols]

# correlation matrix
correlation_matrix = subset_df.corr()

# correlation heatmap
fig = px.imshow(correlation_matrix,
                x=subset_df.columns,
                y=subset_df.columns,
                color_continuous_scale='viridis')  

fig.update_layout(title='Correlation Heatmap')
fig.show()



# Impute missing values in numeric columns with median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# Impute missing values in non-numeric columns with mode
non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
non_numeric_imputer = SimpleImputer(strategy='most_frequent')
df[non_numeric_cols] = non_numeric_imputer.fit_transform(df[non_numeric_cols])

# Noisy Data, no online enrollment also has some sporadic digital logins
# keep only Y's 

df = df[df['Online_Enrolled_mode'] == 'Y']



# Define the outcome columns
outcome_columns = ['OnlineWallet_Dollars_median', 'DigLogins_TotalLogins_median', 'DigLogins_UniqDays_median', 
                   'OnlineWallet_Tx_median', 'Alerts_Enrolled_median', 'Remote_Dep_Amt_median', 'Remote_Dep_Ct_median']


# remove 'DebitCard_Tx_std_deviation', 'DebitCard_Spend_median' ,'BranchTX_std_deviation', 'Avg_Savings_Bal_std_deviation',\
# 'Avg_CreditCard_Bal_std_deviation','Avg_Mortgage_Bal_median','Mortgage_Accts_median',
# to address multicollinearity 

# remove 'Online_Enrolled_mode as filtered out

# Define the columns to use as features
columns_to_use = [
'new_customer',
'account_category',
'Alerts_Enrolled_median',
'DebitCard_Tx_median',
'Checking_Accts_median',
'Checking_Accts_std_deviation',
'BranchTX_median',
'DebitCard_Spend_std_deviation',
'Avg_Savings_Bal_median',
'Avg_Mortgage_Bal_std_deviation',
'Avg_CreditCard_Bal_median',
'Loan_Accts_median',
'Loan_Accts_std_deviation',
'CD_Accts_median',
'CD_Accts_std_deviation',
'Avg_Loan_Bal_median',
'Avg_Loan_Bal_std_deviation',
'Avg_Checking_Bal_median',
'Avg_Checking_Bal_std_deviation',
'Savings_Accts_median',
'Savings_Accts_std_deviation',
'Avg_CD_Bal_median',
'Avg_CD_Bal_std_deviation',
'CreditCard_Accts_median',
'CreditCard_Accts_std_deviation',
'wealth_product_mode',
'average_large_owed_median',
'average_small_owed_median',
'average_remote_gained_median',
'age_in_system_months','age_range', 'Generation', 'Market']



# #Final Sanity Check
# # Filter out only numeric columns from the provided list
# numeric_cols = df[columns_to_use].select_dtypes(include=['float64', 'int64']).columns.tolist()

# # Create a subset of the DataFrame with only the numeric columns
# subset_df = df[numeric_cols]

# # Calculate the correlation matrix
# correlation_matrix = subset_df.corr()

# # Create a heatmap using Plotly
# fig = px.imshow(correlation_matrix,
#                 x=subset_df.columns,
#                 y=subset_df.columns,
#                 color_continuous_scale='viridis')  # Use 'viridis' colorscale

# fig.update_layout(title='Correlation Heatmap')
# fig.show()



import catboost as cb
from sklearn.metrics import mean_squared_error, r2_score


# Define the features and target
X = df[columns_to_use]
y = df[outcome_columns[1]]  #  can change /loop this for each outcome column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)



# CatBoost Pool 
train_dataset = cb.Pool(X_train, y_train, cat_features=X.select_dtypes(include=['object', 'category']).columns.tolist()) 
test_dataset = cb.Pool(X_test, y_test, cat_features=X.select_dtypes(include=['object', 'category']).columns.tolist())


# ##best model already found tuned parameters used below 
# ##no need to run the tuning step
# ## Define the CatBoost Regressor model
# model = cb.CatBoostRegressor(loss_function='RMSE', od_type='Iter',
#                             od_wait=30,
#                             verbose=True,task_type='CPU')

# learning_rate_samples = [sp_randFloat(0.001, 0.1).rvs() for _ in range(5)]
# depth_samples = [sp_randInt(2, 10).rvs() for _ in range(5)]



# # Define hyperparameters to be tuned
# grid = {
#     'iterations': [100, 150, 200,250],
#     'learning_rate': learning_rate_samples,
#     'depth': depth_samples,
#     'l2_leaf_reg': [0.2, 0.5, 1, 3, 5, 10]
# }

# # Execute grid search with cross-validation
# model.grid_search(grid, train_dataset,verbose=1)


# # Get the best score
# best_score = model.get_best_score()

# # Get the best hyperparameters
# best_params = model.get_params()

# print("Best Score:", best_score)
# print("Best Parameters:", best_params)


# Best Score: {'learn': {'RMSE': 9.473287399686951}}
# Best Parameters: {'loss_function': 'RMSE', 'od_wait': 30, 'od_type': 'Iter', 'verbose': True, 'task_type': 'CPU', 'depth': 9, 'iterations': 200, 'learning_rate': 0.07074500745159853, 'l2_leaf_reg': 0.2}

# # Evaluate the performance on the test set
# pred = model.predict(test_dataset)
# rmse = (np.sqrt(mean_squared_error(y_test, pred)))
# r2 = r2_score(y_test, pred)

# # Display the test performance
# print("Testing performance")
# print('RMSE: {:.2f}'.format(rmse))
# print('R2: {:.2f}'.format(r2))


# Testing performance
# RMSE: 14.34
# R2:  0.34


# best hyperparameters from the previous grid search
best_params = {'loss_function': 'RMSE', 'od_wait': 30, 'od_type': 'Iter', 'verbose': True, 'task_type': 'CPU', 'depth': 9, 'iterations': 200, 'learning_rate': 0.07074500745159853, 'l2_leaf_reg': 0.2}

#  CatBoost Regressor model with the best parameters above
best_model = cb.CatBoostRegressor(**best_params)

# train model using the best hyperparameters
best_model.fit(train_dataset)

# sanity check predictions/ model evaluations
pred_1 = best_model.predict(test_dataset)
rmse_1 = (np.sqrt(mean_squared_error(y_test, pred_1)))
r2_1 = r2_score(y_test, pred_1)

print("Test RMSE:", rmse_1)
print("Test R2:", r2_1)



# 'best_model' is the trained CatBoost model and 'X' is features DataFrame
sorted_feature_importance = best_model.feature_importances_.argsort()
plt.figure(figsize=(10, len(X.columns)))
plt.barh(X.columns[sorted_feature_importance], best_model.feature_importances_[sorted_feature_importance], color='turquoise')
plt.xlabel("CatBoost Feature Importance")
plt.show()

##### SHAP Stuff ##########

import shap

# 'best_model' is trained CatBoost model
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)


# Convert shap_values into an Explanation object
expl = shap.Explanation(values=shap_values, data=X_test, feature_names=X_test.columns)

# plot the bar chart
shap.plots.bar(expl,max_display =15)


#beeswarm plot
shap.summary_plot(shap_values, X_test)

################### Top 6 ########################

# mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

# sort the features based on the mean absolute SHAP values (descending)
sorted_indices = np.argsort(mean_abs_shap_values)[::-1]

# top 6 features
top_indices = sorted_indices[:6]
shap_values_top = shap_values[:, top_indices]
X_test_top = X_test.iloc[:, top_indices]


shap.summary_plot(shap_values_top, X_test_top)


############ Extract Branch Transaction Median #####################

feature_index = X_test.columns.get_loc('BranchTX_median')

shap_values_branchtx = shap_values[:, feature_index]
shap_values_branchtx = shap_values_branchtx.reshape(-1, 1)

X_test_branchtx = X_test[['BranchTX_median']]
shap.summary_plot(shap_values_branchtx, X_test_branchtx)




######################################################


# Tryout Waterfall plot for the first observation in X_test
expl = shap.Explanation(values=shap_values, 
                        data=X_test, 
                        feature_names=X_test.columns, 
                        base_values=explainer.expected_value)

shap.plots.waterfall(expl[0])



######################## SHAP deep dive for individual categories of the Categorical Variables ###########################

def plot_beeswarm_for_feature(X_test, expl, feature_name):
    # extract the feature values and SHAP values 
    feature_values = X_test[feature_name].reset_index(drop=True)
    feature_shap_values = expl[:, feature_name].values

    # reset indices for alignment
    feature_values_series = pd.Series(feature_shap_values).reset_index(drop=True)
    feature_values_mask = feature_values.reset_index(drop=True)

    # unique categories and their corresponding SHAP values
    unique_categories = feature_values.unique()
    new_shap_values = [feature_values_series[feature_values_mask == category].values 
                       for category in unique_categories]

    # pad the SHAP values with NaNs to make them the same length
    max_len = max(len(v) for v in new_shap_values)
    new_shap_values = [np.append(vs, [np.nan] * (max_len - len(vs))) for vs in new_shap_values]
    new_shap_values = np.array(new_shap_values).transpose()

    # new SHAP Explanation object with the new SHAP values
    feature_expl = shap.Explanation(values=new_shap_values, 
                                    data=np.array([[0] * len(unique_categories)] * max_len), 
                                    feature_names=list(unique_categories),
                                    base_values=np.array([0] * max_len))

    # Plot the beeswarm plot for the specified feature
    shap.plots.beeswarm(feature_expl, color_bar=False, show=True, max_display=len(unique_categories))

plot_beeswarm_for_feature(X_test, expl, 'Generation')
plot_beeswarm_for_feature(X_test, expl, 'Market')
plot_beeswarm_for_feature(X_test, expl, 'new_customer')
plot_beeswarm_for_feature(X_test, expl, 'age_range')
plot_beeswarm_for_feature(X_test, expl, 'Generation')
plot_beeswarm_for_feature(X_test, expl, 'Online_Enrolled_mode')






########################################################################################


# generate the HTML for the force plot -- very cool -- too intense for 10 mins
shap_html = shap.force_plot(explainer.expected_value, shap_values[:10, :], X_test.iloc[:10, :], show=False)

# convert the plot to HTML string format
shap_html_str = shap.getjs().replace('</head>', '<script src="https://cdn.jsdelivr.net/npm/@plotly/d3@3.5.17/d3.min.js"></script></head>') + shap_html.html()

# Save the plot
with open("shap_force_plot_digi_logins.html", "w") as f:
    f.write(shap_html_str)


# already computed shap_values and have an explainer object
# For a single instance
shap.decision_plot(explainer.expected_value, shap_values[150], X_test.iloc[150, :])

# for multiple instances (for example the first 10)
shap.decision_plot(explainer.expected_value, shap_values[:10, :], X_test.iloc[:10, :])



############### benchmarking  based on mean and grouped mean ###########################

import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


train_data = X_train.copy()
train_data['DigLogins_TotalLogins_median'] = y_train

# compute means on training data
means = train_data.groupby(['Generation', 'Market'])['DigLogins_TotalLogins_median'].mean().reset_index()

# merge the means with the test data to get the predictions
X_test_with_pred = pd.merge(X_test, means, on=['Generation', 'Market'], how='left')

# Handle any cases where there isn't a match 
#(i.e., combination of Generation and Market from test set wasn't in train set)
# for this, we use the global mean, but other strategies can be used as well
X_test_with_pred['DigLogins_TotalLogins_median'].fillna(y_train.mean(), inplace=True)

# Compute RMSE on test set
rmse = np.sqrt(mean_squared_error(y_test, X_test_with_pred['DigLogins_TotalLogins_median']))


#based on grouped means
print(rmse)
#17.370903677365394

# Calculate the mean of y_train
mean_y_train = y_train.mean()

# Predict all values in the test set with the mean of y_train
y_pred = [mean_y_train] * len(y_test)

# Compute RMSE on test set
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#baseline mean
print(rmse)
#17.60747557423354


#### Too Intense for Datathon

####################   Custering with SHAP values - Inspired by Aiden Cooper ####################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

# Compute 2D embedding of SHAP values
embedding = UMAP(n_components=2, n_neighbors=200, min_dist=0,random_state=123).fit_transform(shap_values)

# Get the quartile indicators for the 'DigLogins_TotalLogins_median' values in y_test
quartiles = np.percentile(y_test, [25, 50, 75])
colors = np.digitize(y_test, quartiles)

# Plot the embedding colored by the quartile of 'DigLogins_TotalLogins_median'
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap='viridis')
legend1 = plt.legend(*scatter.legend_elements(), title="Quartiles")
plt.gca().add_artist(legend1)
plt.title('UMAP embedding of SHAP values colored by quartile of DigLogins_TotalLogins_median')
plt.show()



########################## Prepping for DBScan Clustering ##################

from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Use NearestNeighbors to find the distance to the 5th nearest neighbor for each point
nn = NearestNeighbors(n_neighbors=5).fit(embedding)
distances, indices = nn.kneighbors(embedding)

# Sort and plot the distances
distances_to_5th_nearest = np.sort(distances[:, 4])
plt.plot(distances_to_5th_nearest)
plt.title('k-distance Graph')
plt.xlabel('Data Points sorted by distance to 5th nearest neighbor')
plt.ylabel('5th nearest neighbor distance')
plt.show()



chosen_eps = 0.25

# Grid search over min_samples
min_samples_values = range(5, 50, 5)
silhouette_scores = []

for min_samples in min_samples_values:
    labels = DBSCAN(eps=chosen_eps, min_samples=min_samples).fit(embedding).labels_
    
    # Compute silhouette score only if clusters are formed
    if len(np.unique(labels)) > 1:
        score = silhouette_score(embedding, labels)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(-1)  # No clusters were formed

# Plot Silhouette scores
plt.plot(min_samples_values, silhouette_scores, 'o-')
plt.title('Silhouette Scores for Different min_samples')
plt.xlabel('min_samples')
plt.ylabel('Silhouette Score')
plt.show()

# min_samples=20 / maybe 15 would be better

from sklearn.cluster import DBSCAN

# Apply DBSCAN clustering to the UMAP embeddings
s_labels = DBSCAN(eps=0.25, min_samples=20).fit(embedding).labels_

# Visualize the clusters on the UMAP plot
plt.figure(figsize=(10,8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=s_labels, cmap="viridis", s=15)
plt.colorbar(scatter, ticks=np.unique(s_labels))
plt.title('UMAP of SHAP Values with DBSCAN Clusters')
plt.xlabel('UMAP 1st Dimension')
plt.ylabel('UMAP 2nd Dimension')
plt.show()

# Interactive 

import plotly.express as px

# Convert UMAP embeddings and cluster labels to a DataFrame for easier plotting with Plotly
df_embedding = pd.DataFrame(embedding, columns=['UMAP 1st Dimension', 'UMAP 2nd Dimension'])

# Convert cluster labels to string for discrete colors
df_embedding['Cluster'] = s_labels.astype(str)

colors = px.colors.qualitative.Set1

# Create the scatter plot with discrete colors
fig = px.scatter(df_embedding, 
                 x='UMAP 1st Dimension', 
                 y='UMAP 2nd Dimension', 
                 color='Cluster',
                 color_discrete_sequence=colors,
                 labels={'Cluster': 'DBSCAN Cluster'},
                 title='UMAP of SHAP Values with DBSCAN Clusters')

# Show the figure
fig.show()

# Save the interactive plot as an HTML file
fig.write_html("DBSCAN_interactive_plot.html")



###################  Now Skope-Rules on the Clusters #########################

import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
collections.Iterator = collections.abc.Iterator
import six
import sys
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules

# Calculate quartiles for target variable
quartiles = np.percentile(y_test, [25, 50, 75])
quartile_labels = np.digitize(y_test, quartiles)



from sklearn.preprocessing import OneHotEncoder

# one-hot encoding for categorical columns
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_data = encoder.fit_transform(X_test.select_dtypes(include=['object']))
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(X_test.select_dtypes(include=['object']).columns))

# Combine the encoded data with the non-categorical columns of X_test
X_test_processed = pd.concat([X_test.select_dtypes(exclude=['object']).reset_index(drop=True), encoded_df], axis=1)


            
def convert_rules_to_names(rule, column_names):
    """Convert rules from SkopeRules that use indices (__C__i) to actual column names."""
    # sort indices in descending order to handle double-digit indices first
    indices_desc_order = sorted(range(len(column_names)), key=lambda x: -x)
    
    for idx in indices_desc_order:
        col_name = column_names[idx]
        rule = rule.replace(f'__C__{idx}', col_name)
    
    return rule

# Get the column names 
column_names = X_test_processed.columns

# empty DataFrame to hold results
df_rules = pd.DataFrame(columns=['Cluster', 'Quartile', 'Rule', 'Precision', 'Recall'])

# processed data with SkopeRules
for cluster in np.unique(s_labels):
    for q in range(4):
        # binary target for individual cluster and quartile
        yc = ((s_labels == cluster) & (quartile_labels == q)) * 1
        
        # skip if there are no samples in this cluster-quartile combination
        if sum(yc) == 0:
            continue

        # SkopeRules to identify rules with a maximum of two comparison terms
        sr = SkopeRules(max_depth=4,random_state=123).fit(X_test_processed, yc)

        # Check if any rules were found
        if sr.rules_:
            # Convert the rule to use column names for better interpretability
            rule_with_names = convert_rules_to_names(sr.rules_[0][0], column_names)
            
            # Append results to the DataFrame
            df_rules = df_rules.append({
                'Cluster': cluster,
                'Quartile': q,
                'Rule': rule_with_names,
                'Precision': sr.rules_[0][1][0],
                'Recall': sr.rules_[0][1][1]
            }, ignore_index=True)

# Filter the rules based on precision and recall criteria (custom) 
# this high precison and recall simulataneously is  impossible with plain clustering
df_filtered4 = df_rules[(df_rules['Precision'] > 0.6) & (df_rules['Recall'] > 0.6)]
df_filtered4.to_csv('filtered_rules.csv', index=False)



