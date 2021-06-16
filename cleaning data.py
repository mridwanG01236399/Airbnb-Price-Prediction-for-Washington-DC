
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('listings.csv',encoding = 'utf-8')

col_names = df.columns.tolist()


urls = [i for i in col_names if 'url' in i] 

df.drop(urls, axis=1, inplace=True)


ids = ['scrape_id', 'host_id']

df.drop(ids, axis=1, inplace=True)


text = ['name', 'description', 'neighborhood_overview', 'description', 'host_name', 'host_verifications',
        'host_about', 'host_location', 'neighbourhood', 'neighbourhood_cleansed','amenities']


df.drop(text, axis=1, inplace=True)

more_than_50 = list(df.columns[df.isnull().mean() > 0.5])

df.drop(more_than_50, axis=1, inplace=True)

# Convert price type from string to float and remove $ sign and commas
df['price'] = df['price'].str.replace('$', '')
df['price'] = df['price'].str.replace(',', '').astype(float)


# Convert rates type from string to float and remove % sign
df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float)
df['host_response_rate'] = df['host_response_rate']*0.01
df['host_acceptance_rate'] = df['host_acceptance_rate'].str.replace('%', '').astype(float)
df['host_acceptance_rate'] = df['host_acceptance_rate']*0.01


## Fill numerical missing data with mean value
num_feat = df.select_dtypes(np.number)
num_col = num_feat.columns

imp_mean = Imputer(missing_values= np.nan, strategy= 'mean')
imp_mean = imp_mean.fit(num_feat)
df[num_col] = imp_mean.transform(df[num_col])
    

## Fill categorical missing data with most frequent value
cat_feat = df.select_dtypes(include='object')
cat_col = cat_feat.columns
    
most_freq = Imputer(missing_values= np.nan, strategy= 'most_frequent')
most_freq = most_freq.fit(cat_feat)
df[cat_col] = most_freq.transform(df[cat_col])


df['price'] = np.log10(df['price'])
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 
i = df[(df['price'] <= Q1 - 1.5 * IQR) | (df['price'] >= Q3 + 1.5 *IQR) ].index
df.drop(i , inplace=True)
i = df[(df['price'] >= 2.7) ].index
df.drop(i , inplace=True)
bool_col = ['instant_bookable', 'host_is_superhost', 'host_has_profile_pic', 
            'host_identity_verified']
df[bool_col] = df[bool_col].replace({'t': 1, 'f': 0}).astype(int)


from datetime import date 
dates = ['last_scraped', 'host_since', 'calendar_last_scraped', 
             'first_review', 'last_review']
for col in dates:
    df[col] = pd.to_datetime(df.loc[:, col], format = '%Y-%m-%d')
df['host_since_days'] = (df.last_scraped - df.host_since).dt.days
df['first_reviewed'] = (df.last_scraped - df.first_review).dt.days
df['last_reviewed'] = (df.last_scraped - df.last_review).dt.days

