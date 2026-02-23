import pandas as pd # type: ignore

df = pd.read_csv('/Users/giancarlossanchez/Desktop/x/historical_data_2025Q3.txt', sep='|', header=None)

print(df.shape)
print(df.head())



columns = [
    'credit_score', 'first_payment_date', 'first_time_homebuyer', 'maturity_date',
    'msa', 'mip', 'units', 'occupancy_status', 'cltv', 'dti', 'upb',
    'ltv', 'interest_rate', 'channel', 'ppm_flag', 'product_type', 'state',
    'property_type', 'postal_code', 'loan_sequence_number', 'loan_purpose',
    'original_loan_term', 'number_of_borrowers', 'seller_name', 'servicer_name',
    'super_conforming_flag', 'pre_relief_refinance_loan_sequence_number',
    'program_indicator', 'relief_refinance_indicator', 'property_valuation_method',
    'io_indicator', 'mortgage_insurance_cancellation_indicator'
]

df.columns = columns
print(df.head())


# Check for missing values
print(df.isnull().sum())

# Check data types
print(df.dtypes)


# Dropped columns that are almost empty (largest amounts of missing values)

df.drop(columns=[
    'super_conforming_flag',
    'pre_relief_refinance_loan_sequence_number',
    'relief_refinance_indicator'
], inplace=True)

# Filled missing msa with 0 to retain metro area information which will be useful to compare its risk profile with rural areas.
df['msa'] = df['msa'].fillna(0)


# Dropped a single row since it only has one missing value 
df.dropna(subset=['servicer_name'], inplace=True)

# Confirmation
print(df.isnull().sum())
print(df.shape)



import matplotlib.pyplot as plt
import seaborn as sns




# This must come BEFORE the chart code
df = df[df['credit_score'] <= 850]
print(df.shape)

# Then credit score distribution chart 
plt.figure(figsize=(10,5))
sns.histplot(df['credit_score'], bins=50, color='steelblue')
plt.title('Credit Score Distribution')
plt.xlabel('Credit Score')
plt.ylabel('Count')
plt.show()



# This must come BEFORE the chart code
df = df[df['dti'] <= 65]

# Then DTI distribution chart 
plt.figure(figsize=(10,5))
sns.histplot(df['dti'], bins=50, color='steelblue')
plt.title('DTI Ratio')
plt.xlabel('DTI (%)')
plt.ylabel('Count')
plt.show()



# This must come BEFORE the chart code
df = df[df['interest_rate'] <= 10]

# Then Interest Rate distribution chart
plt.figure(figsize=(10,5))
sns.histplot(df['interest_rate'], bins=50, color='steelblue')
plt.title('Interest Rate')
plt.xlabel('Interest Rates')
plt.ylabel('Count')
plt.show()


# LTV distribution chart 
plt.figure(figsize=(10,5))
sns.histplot(df['ltv'], bins=50, color='steelblue')
plt.title('Loan-to-Value Distribution')
plt.xlabel('LTV (%)')
plt.ylabel('Count')
plt.show()


# Imported SQL 
import sqlite3
conn = sqlite3.connect('freddie_mac.db')
df.to_sql('loans', conn, if_exists='replace', index=False)
print("Data loaded into SQLite successfully!")
print(f"Total rows: {len(df)}")




# Query 1 = loan purpose analysis
query1 = """
SELECT loan_purpose,
       COUNT(*) as total_loans,
       ROUND(AVG(credit_score), 1) as avg_credit_score,
       ROUND(AVG(dti), 1) as avg_dti,
       ROUND(AVG(interest_rate), 2) as avg_interest_rate
FROM loans
GROUP BY loan_purpose
ORDER BY total_loans DESC
"""

result1 = pd.read_sql_query(query1, conn)
print(result1)

# PCN
# Purchase(buying a new home)
# Cash-out Refinance (Refinancing to take equity out as cash)
# No cash-out Refinance (Refinancing just to get a better a rate)





# Query 2 Average loan amount by state analysis 
query2 = """
SELECT state,
       COUNT(*) as total_loans,
       ROUND(AVG(upb), 0) as avg_loan_amount,
       ROUND(AVG(ltv), 1) as avg_ltv,
       ROUND(AVG(credit_score), 1) as avg_credit_score,
       ROUND(AVG(interest_rate), 2) as avg_interest_rate
FROM loans
WHERE state IN ('DC', 'MD', 'VA')
GROUP BY state
ORDER BY avg_loan_amount DESC
"""

result2 = pd.read_sql_query(query2, conn)
print(result2)





# Query 3 = high risk profile analysis

query3 = """
SELECT 
COUNT(*) as total_high_risk_loans, 
ROUND(AVG(interest_rate), 2) as avg_interest_rate, 
ROUND(AVG(ltv), 1) as avg_ltv,
ROUND(AVG(upb), 0) as avg_loan_amount,
ROUND(AVG(credit_score), 1) as avg_credit_score

FROM loans 
WHERE credit_score < 660
AND dti > 43
AND ltv > 90 
"""

result3 = pd.read_sql_query(query3, conn)
print(result3)




#Query 4 = high ltv loan analysis
query4 = """
SELECT state,
COUNT(*) as high_ltv_loans, 
ROUND(AVG(ltv), 1) as avg_ltv, 
ROUND(AVG(upb), 0) as avg_loan_amount

FROM loans 
WHERE state in ('VA', 'DC', 'MD')
AND ltv > 90
GROUP BY state
ORDER by avg_ltv

"""

result4 = pd.read_sql_query(query4, conn)
print(result4)







##---------------------- PREDICTIVE MODEL -------------------------## 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# high risk labels:  (1 = high risk & 0 = normal)



df['high_risk'] = ((df['credit_score'] < 660) & (df['dti'] > 43) & (df['ltv'] > 90)).astype(int)
print(df['high_risk'].value_counts())


# define features and target // (X)inputs the model learns from and (Y) what we are trying to predict


x = df[['credit_score', 'dti', 'ltv', 'interest_rate', 'upb']]
y = df['high_risk']

# train splits (80/20) 80% for training; 20% for testing 
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#Model training 
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)


#Model evaluation 
y_pred = model.predict(X_test)
print(classification_report(Y_test, y_pred))



