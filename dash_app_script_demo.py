# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:07:20 2023

@author: Riddhi
"""

#after running, when you see something like <IPython.lib.display.IFrame at 0x2d209ae6ed0> 
# open the below in browser
#http://127.0.0.1:8050/


# Required Libraries
import os
import pandas as pd
from sklearn.impute import SimpleImputer
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re


# Updating the Directory and Reading the Data
os.chdir("C:/Users/Riddhi/Downloads/")
df = pd.read_excel('dataton_data.xlsx')

#df.info()
#rows_with_nan_df = df[df.isna().any(axis=1)]
# Obtain rows with NaN values in the specified columns
rows_with_nan_df = df[df[['First_Open_Dt_Deposit', 'First_Open_Dt_Loan', 'First_Open_Dt_Card', 'First_Open_Dt_Mortgage']].isna().all(axis=1)]

indexes_to_drop = rows_with_nan_df.index

# drop the rows with missing values from df
df = df.drop(indexes_to_drop)




df['new_customer'] = df[['First_Open_Dt_Deposit', 'First_Open_Dt_Loan', 'First_Open_Dt_Card', 'First_Open_Dt_Mortgage']].min(axis=1).ge(202301).astype(int)
df['new_customer'] = df['new_customer'].astype('category')
df['new_customer_feature'] = df['new_customer'].astype('category')

df['age_range'] = df['age_range'].str.split('.').str[1].str.strip()
df['Generation'] = df['Generation'].str.split('.').str[1].str.strip()




# Extract unique prefixes
prefixes = set(col.rsplit('_', 1)[0] for col in df.columns if col.endswith(tuple(['_' + str(i) for i in range(202303, 202308)])))

# For each unique prefix, compute median and create new column
for prefix in prefixes:
    cols_of_interest = [col for col in df.columns if col.startswith(prefix)]
    df[prefix + '_median'] = df[cols_of_interest].median(axis=1)
    df[prefix + '_std_deviation'] = df[cols_of_interest].std(axis=1)
#drop wealth product median and online enrolled median    
    
# drop the specified columns
columns_to_drop = ['Online_Enrolled_median', 'Online_Enrolled_std_deviation', 'wealth_product_median', 'wealth_product_std_deviation']
df = df.drop(columns=columns_to_drop, errors='ignore')  # ensures no error is raised if a column doesn't exist



#categorical
prefixes1 = set(['wealth_product_', 'Online_Enrolled_'])

# For each unique prefix, compute mode and create new column
for prefix in prefixes1:
    cols_of_interest = [col for col in df.columns if col.startswith(prefix)]
    df[prefix + 'mode'] = df[cols_of_interest].mode(axis=1)[0]   
    


# Impute NaN values in specified columns with 'N'
columns_to_impute = ['Online_Enrolled_mode', 'wealth_product_mode']
for column in columns_to_impute:
    df[column] = df[column].fillna('N')




# Numerator: Sum of 'Avg_Mortgage_Bal_median', 'Avg_CreditCard_Bal_median', 'Avg_Loan_Bal_median', and 'DebitCard_Spend_median'
numerator0 = df['Avg_Mortgage_Bal_median'] + df['Avg_Loan_Bal_median'] 

# Denominator: Sum of 'Avg_Checking_Bal_median', 'Avg_Savings_Bal_median', and 'Avg_CD_Bal_median'
denominator = df['Avg_Checking_Bal_median'] + df['Avg_Savings_Bal_median'] + df['Avg_CD_Bal_median']

# Adjust the denominator: if it's 0, use 1 instead
denominator = denominator.where(denominator != 0, 1)

# New feature: average_owed_median
df['average_large_owed_median'] = numerator0 / denominator


numerator1 = df['Avg_CreditCard_Bal_median'] + df['DebitCard_Spend_median']

# Adjust the denominator: if it's 0, use 1 instead
denominator = denominator.where(denominator != 0, 1)


df['average_small_owed_median'] = numerator1 / denominator




# Numerator: Sum of 'Avg_Mortgage_Bal_median', 'Avg_CreditCard_Bal_median', 'Avg_Loan_Bal_median', and 'DebitCard_Spend_median'
numerator2 = df['Remote_Dep_Amt_median']


# Adjust the denominator: if it's 0, use 1 instead
denominator = denominator.where(denominator != 0, 1)

df['average_remote_gained_median'] = numerator2 / denominator




# Define a function to calculate the difference in months
def months_difference(date1, date2):
    year_diff = (date1 // 100) - (date2 // 100)
    month_diff = (date1 % 100) - (date2 % 100)
    return year_diff * 12 + month_diff

# Calculate age in system in terms of months
df['age_in_system_months'] = df[['First_Open_Dt_Deposit', 'First_Open_Dt_Loan', 'First_Open_Dt_Card', 'First_Open_Dt_Mortgage']].min(axis=1).apply(lambda x: months_difference(202308, x))


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
    
    # Iterate through conditions and return the category when a condition is met
    for condition, category in conditions:
        if condition:
            return category
    return "other"  # If no condition is met 

# Create the new feature
df['account_category'] = df.apply(categorize_account, axis=1)


df['account_category'].value_counts()

# Given value counts
value_counts = df['account_category'].value_counts()

# categories with counts less than or equal to 13
low_cardinality_categories = value_counts[value_counts <= 20].index

# update 'account_category' column to replace low cardinality categories with 'other'
df['account_category'] = df['account_category'].replace(low_cardinality_categories, 'other')




# Impute missing values in numeric columns with median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# Impute missing values in non-numeric columns with mode
non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
non_numeric_imputer = SimpleImputer(strategy='most_frequent')
df[non_numeric_cols] = non_numeric_imputer.fit_transform(df[non_numeric_cols])


df["new_customer"] = df["new_customer"].astype(int)
df["existing_customer"] = 1 - df["new_customer"]


# Create got_wealth_product column
df['got_wealth_product'] = df['wealth_product_mode'].apply(lambda x: 1 if x == 'Y' else 0)
df['no_wealth_product'] = df['wealth_product_mode'].apply(lambda x: 1 if x == 'N' else 0)

# Create online_enrolled column
df['online_enrolled'] = df['Online_Enrolled_mode'].apply(lambda x: 1 if x == 'Y' else 0)
df['not_online_enrolled'] = df['Online_Enrolled_mode'].apply(lambda x: 1 if x == 'N' else 0)

q1 = df['DigLogins_TotalLogins_median'].quantile(0.25)
q2 = df['DigLogins_TotalLogins_median'].quantile(0.50)
q3 = df['DigLogins_TotalLogins_median'].quantile(0.75)

def categorize_by_quartile(value):
    if value <= q1:
        return 'Q1'
    elif value <= q2:
        return 'Q2'
    elif value <= q3:
        return 'Q3'
    else:
        return 'Q4'

df['Quartile_Category'] = df['DigLogins_TotalLogins_median'].apply(categorize_by_quartile)

# Create a dictionary mapping for quartile categories to integers
quartile_mapping = {
    'Q1': 1,
    'Q2': 2,
    'Q3': 3,
    'Q4': 4
}

# Map the 'Quartile_Category' column to its integer values
df['DigLogins_TotalLogins_median_quartiles'] = df['Quartile_Category'].map(quartile_mapping)

median_val = df['DigLogins_TotalLogins_median'].median()

# Create binary columns based on the median
df['DigLogins_TotalLogins_above_median'] = (df['DigLogins_TotalLogins_median'] > median_val).astype(int)
df['DigLogins_TotalLogins_below_median'] = (df['DigLogins_TotalLogins_median'] <= median_val).astype(int)

# Check the values in the new columns
print(df['DigLogins_TotalLogins_above_median'].value_counts())
print(df['DigLogins_TotalLogins_below_median'].value_counts())


# Check the new column values
#print(df['DigLogins_TotalLogins_median_quartiles'].value_counts())





# Define the outcome columns
outcome_columns = ['new_customer','got_wealth_product','no_wealth_product','online_enrolled','not_online_enrolled','existing_customer','DigLogins_TotalLogins_above_median','DigLogins_TotalLogins_below_median']

# Define the columns to use as features
columns_to_use = [
    'new_customer_feature',
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
    'Online_Enrolled_mode',
    'average_large_owed_median',
    'average_small_owed_median',
    'average_remote_gained_median',
    'age_in_system_months','age_range', 'Generation', 'Market']



# define a mapping from original column names to cleaner display names
column_name_mapping = {
    'new_customer': 'New Customer',
    'got_wealth_product': 'Got Wealth Product',
    'no_wealth_product': 'No Wealth Product',
    'online_enrolled': 'Online Enrolled',
    'not_online_enrolled': 'Not Online Enrolled',
    'existing_customer': 'Existing Customer',
    'DigLogins_TotalLogins_above_median': 'Digital Logins Above Median',
    'DigLogins_TotalLogins_below_median': 'Digital Logins Below Median',
    'new_customer_feature': 'New Customer Feature',
    'account_category': 'Account Category',
    'Alerts_Enrolled_median': 'No.of Alerts Enrolled',
    'DebitCard_Tx_median': 'Monthly No.of Debit Card Transactions',
    'Checking_Accts_median': 'No.of Checking Accounts',
    'Checking_Accts_std_deviation': 'Std Dev No. of Checking Accounts',
    'BranchTX_median': 'Monthly No. of Branch Transactions',
    'DebitCard_Spend_std_deviation': 'Std Dev of Monthly Debit Card Spend',
    'Avg_Savings_Bal_median': 'Monthly Avg Savings Balance',
    'Avg_Mortgage_Bal_std_deviation': 'Std Dev of Monthly Avg Mortgage Balance',
    'Avg_CreditCard_Bal_median': 'Monthly Avg Credit Card Balance',
    'Loan_Accts_median': 'No.of Loan Accounts',
    'Loan_Accts_std_deviation': 'Std Dev No. of Loan Accounts',
    'CD_Accts_median': 'No.of CD Accounts',
    'CD_Accts_std_deviation': 'Std Dev No. of CD Accounts',
    'Avg_Loan_Bal_median': 'Monthly Avg Loan Balance',
    'Avg_Loan_Bal_std_deviation': 'Std Dev of Monthly Avg Loan Balance',
    'Avg_Checking_Bal_median': 'Monthly Avg Checking Balance',
    'Avg_Checking_Bal_std_deviation': 'Std Dev of Avg Checking Balance',
    'Savings_Accts_median': 'No.of Savings Accounts',
    'Savings_Accts_std_deviation': 'Std Dev No. of Savings Accounts',
    'Avg_CD_Bal_median': 'Monthly Avg CD Balance',
    'Avg_CD_Bal_std_deviation': 'Std Dev of Monthly Avg CD Balance',
    'CreditCard_Accts_median': 'No.of Credit Card Accounts',
    'CreditCard_Accts_std_deviation': 'Std Dev No. of Credit Card Accounts',
    'wealth_product_mode': 'Wealth Products Indicator',
    'Online_Enrolled_mode': 'Online Enrollment Indicator',
    'average_large_owed_median': 'Normalized Monthly Large Owed Amount',
    'average_small_owed_median': 'Normalized Monthly Small Owed Amount',
    'average_remote_gained_median': 'Normalized Monthly Remote Deposit Amount',
    'age_in_system_months': 'Customer Tenure (Months)',
    'age_range': 'Age Range',
    'Generation': 'Generation',
    'Market': 'Market'
}

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# stylesheet for a cleaner appearance
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# initialize app with the external stylesheet
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# App Layout
app.layout = html.Div(style={'maxWidth': '1200px', 'margin': 'auto'}, children=[
    # Header
    html.Div([
        html.H1("Supervised Customer Segmentation Rules", style={'textAlign': 'center'})
    ]),
    
    # Row for filter, outcome selection, and sorting
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginTop': '20px', 'marginBottom': '20px'}, children=[
        
        html.Div(style={'flex': '1', 'marginRight': '10px'}, children=[
            html.Label('Filter Dataset by:', style={'marginBottom': '5px'}),
            dcc.Dropdown(
                id='filter-dropdown',
                options=[{'label': 'None', 'value': 'NONE'}] + [{'label': column_name_mapping[col], 'value': col} for col in column_name_mapping.keys() if col in outcome_columns],
                value='NONE',
                multi=False,
                clearable=True
            )
        ]),
        
        html.Div(style={'flex': '1', 'marginRight': '10px'}, children=[
            html.Label('Select Outcome Column:', style={'marginBottom': '5px'}),
            dcc.Dropdown(
                id='outcome-dropdown',
                options=[{'label': column_name_mapping.get(col, col), 'value': col} for col in outcome_columns],
                value=outcome_columns[0],
                multi=False,
                clearable=False
            )
        ]),
        
        html.Div(style={'flex': '1'}, children=[
            html.Label('Sort Rules By:', style={'marginBottom': '5px'}),
            dcc.RadioItems(
                id='sorting-selector',
                options=[
                    {'label': 'Precision', 'value': 'precision'},
                    {'label': 'F1-Score', 'value': 'f1'}
                ],
                value='precision',
                inline=True
            )
        ]),
    ]),
    
    # Features dropdown
    html.Div(style={'margin': '20px 0px'}, children=[
        html.Label('Select Features:', style={'marginBottom': '5px'}),
        dcc.Dropdown(
            id='features-dropdown',
            options=[{'label': 'Select All', 'value': 'ALL'}] + [{'label': column_name_mapping.get(col, col), 'value': col} for col in columns_to_use],
            value=columns_to_use,
            multi=True
        )
    ]),
    
    # Submit button
    html.Div(style={'textAlign': 'center', 'marginBottom': '20px'}, children=[
        html.Button(id='submit-button', n_clicks=0, children='Run Model',
                    style={
                        'padding': '5px 20px', 
                        'textAlign': 'center', 
                        'lineHeight': 'normal', 
                        'backgroundColor': '#3A873A', 
                        'border': 'none',
                        'color': 'white',
                        'marginBottom': '5px'
                    })
    ]),

    # Output area
    html.Div(id='rules-output', style={'borderTop': '1px solid #ddd', 'paddingTop': '20px'})
])

@app.callback(
    Output('features-dropdown', 'value'),
    [Input('features-dropdown', 'value')],
)
def select_all_features(selected_values):
    if 'ALL' in selected_values:
        return columns_to_use
    return selected_values        
        


# round_rule function to round numeric values in a rule
def round_rule(rule):
    # regex to find and round float numbers in the rule string
    return re.sub(r'(\d+\.\d+)', lambda x: str(round(float(x.group(1)), 2)), rule)

@app.callback(
    Output('rules-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('outcome-dropdown', 'value'),
     State('features-dropdown', 'value'),
     State('sorting-selector', 'value'),
     State('filter-dropdown', 'value')]
)

def update_rules(n_clicks, outcome_column, feature_columns, sort_by, filter_value):
    if n_clicks == 0:
        return []

    if filter_value and filter_value != 'NONE':
        filtered_df = df[df[filter_value] == 1]
    else:
        filtered_df = df.copy()
        
    y = filtered_df[outcome_column]
    X = filtered_df[feature_columns]

    # one-hot encoding for categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        encoder = OneHotEncoder(drop='first', sparse=False)
        encoded_data = encoder.fit_transform(X[categorical_columns])
        # just ensuring that encoded_df has the same index as X
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns), index=X.index)  

        # combining the encoded data with the non-categorical columns of X
        X = pd.concat([X.select_dtypes(exclude=['object']), encoded_df], axis=1)

    # ensuring that X and y have the same index
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    clf = SkopeRules(max_depth_duplication=3, max_depth=3, max_features=0.5,
                     max_samples_features=0.5, random_state=123, n_estimators=20,
                     feature_names=X.columns, recall_min=0.04, precision_min=0.6)

    clf.fit(X_train, y_train)
    
    if sort_by == 'precision':
        sorted_rules = sorted(clf.rules_, key=lambda x: x[1][0], reverse=True)
    else:  # If sort_by == 'f1'
        def f1_score_for_rule(rule_tuple):
            precision, recall, _ = rule_tuple[1]
            if (precision + recall) == 0:  # To avoid division by zero
                return 0
            return (2 * precision * recall) / (precision + recall)
        sorted_rules = sorted(clf.rules_, key=f1_score_for_rule, reverse=True)


    output_rules = []
    
    

    for rule_string, (precision, recall, _) in sorted_rules:
        # split the rule into individual conditions
        conditions = rule_string.split(" and ")
        
        # process each condition separately
        for i, condition in enumerate(conditions):
                # check if the condition is for binary categorical feature
                if (condition.endswith(" <= 0.5") or condition.endswith(" > 0.5")):
                    if "_median" not in condition.split(" <= ")[0] and "_median" not in condition.split(" > ")[0]:
                        if condition.endswith(" <= 0.5"):
                            conditions[i] = f"not in {condition.rsplit(' ', 2)[0]}"
                        elif condition.endswith(" > 0.5"):
                            conditions[i] = f"in {condition.rsplit(' ', 2)[0]}"    
        # join the conditions back into a rule string
        rule_string = " and ".join(conditions)
        
        # replace original column names with the mapped names
        for original_name, readable_name in column_name_mapping.items():
                rule_string = rule_string.replace(original_name, readable_name)

        # round the rule's numeric values
        rounded_rule = round_rule(rule_string)
        # formatted string for each rule with its precision and recall percentages
        rule_with_metrics = f"{rounded_rule} (Precision: {precision:.2%}, Recall: {recall:.2%})"
        
        output_rules.append(html.P(rule_with_metrics))
    
    return html.Div(output_rules)

if __name__ == '__main__':
    app.run_server(debug=True, open_browser=True)
