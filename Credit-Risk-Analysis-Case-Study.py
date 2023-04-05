#!/usr/bin/env python
# coding: utf-8

# ## CREDIT RISK ANALYSIS EDA CASE STUDY

# ### Business Objectives
# This case study aims to identify patterns which indicate if a client has difficulty paying their installments which may be used for taking actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc. This will ensure that the consumers capable of repaying the loan are not rejected. Identification of such applicants using EDA is the aim of this case study.
#  
# In other words, the company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.  The company can utilise this knowledge for its portfolio and risk assessment.

# ### <font color = blue>Analysis of Application_Data </font>

# #### <font color=maroon>Loading data and normal routine check </font>

# In[1]:


#importing data from CSV file into pandas dataframe
get_ipython().system('pip install matplotlib')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
application_data = pd.read_csv('application_data.csv')
application_data.head()


# In[2]:


application_data.shape


# **This dataset for application data has:**
# 
# - <font color=navy>**307511 rows**</font>
# - <font color=navy>**122 columns**</font>

# In[3]:


application_data.dtypes.value_counts()


# **<font color = navy>We can see that there are:**</font>
# - **<font color = navy>65 columns with dtype=float64**</font>
# - **<font color = navy>41 columns with dtype=int64**</font>
# - **<font color = navy>16 columns with dtype=object</font>**

# In[4]:


# Get the count,size and unique values in each column of application data
application_data.agg(['count','size','nunique'])


# #### <font color=maroon>Checking Distribution of Target Variable</font>

# In[5]:


defaulters=application_data[application_data.TARGET==1]
nondefaulters=application_data[application_data.TARGET==0]


# In[6]:


sns.countplot(application_data.TARGET)
plt.xlabel("TARGET Value")
plt.ylabel("Count of TARGET")
plt.title("Distribution of TARGET Variable")
plt.show()


# **From this information, we see this is an imbalanced dataset. There are far more loans that were repaid on time than loans that were not repaid.**
# 
# **More than 25000 loans were repaid, Less than 5000 loans were not repaid.**

# In[7]:


percentage_defaulters=(len(defaulters)*100)/len(application_data)
percentage_nondefaulters=(len(nondefaulters)*100)/len(application_data)

print("The Percentage of people who have paid their loan is:",round(percentage_nondefaulters,2),"%")
print("The Percentage of people who have NOT paid their loan is:",round(percentage_defaulters,2),"%")
print("The Ratio of Data Imbalance is:",round(len(nondefaulters)/len(defaulters),2))


# #### <font color=maroon>Identifying missing values in each column</font>

# In[8]:


#Function to calculate meta-data to identify % of data is missing in each column
def meta_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    unique = data.nunique()
    datatypes = data.dtypes
    return pd.concat([total, percent, unique, datatypes], axis=1, keys=['Total', 'Percent', 'Unique', 'Data_Type']).sort_values(by="Percent", ascending=False)


# In[9]:


#calculating meta-data for application_data
app_meta_data=meta_data(application_data)
app_meta_data.head(20)


# #### <font color=maroon>Dropping columns with High Missing Values</font>

# In[10]:


#dropping columns with more than 57% missing values 
#Selected 57% because we don't want to drop EXT_SOURCE_1 which is an important variable
cols_to_keep=list(app_meta_data[(app_meta_data.Percent<57)].index)
application_data=application_data[cols_to_keep]
application_data.describe()


# #### <font color=maroon>Checking columns with very less missing values</font>

# In[11]:


#Checking columns with very less missing values
low_missing=pd.DataFrame(app_meta_data[(app_meta_data.Percent>0)&(app_meta_data.Percent<15)])
low_missing


# #### <font color=maroon>Explanation for treatment of columns with low missing values</font>
# 
# 1. AMT_REQ_CREDIT_BUREAU_HOUR
# 2. AMT_REQ_CREDIT_BUREAU_DAY
# 3. AMT_REQ_CREDIT_BUREAU_WEEK
# 4. AMT_REQ_CREDIT_BUREAU_MON
# 5. AMT_REQ_CREDIT_BUREAU_QRT
# 6. AMT_REQ_CREDIT_BUREAU_YEAR
# 
# **We can impute missing values in these columns above with 0s and assume that no enquiry was made during the time reflected in null rows.**
# 
# 
# 1. NAME_TYPE_SUITE - we should leave empty values as it is or impute it with "Others A" or "Others B" depending on what they mean.
# 2. OBS_30_CNT_SOCIAL_CIRCLE & related fields. 
# 3. EXT_SOURCE_2 
# 4. AMT_GOODS_PRICE 
# 5. CNT_FAM_MEMBERS
# 6. AMT_ANNUITY
# 7. DAYS_LAST_PHONE_CHANGE
# 
# 
# **We should not add any additional info in missing values of these columns above as it would lead to noise and exaggeration.**

# #### <font color=maroon>Let's take look at all the columns names for different data types </font>

# In[12]:


application_data.select_dtypes('object').columns


# In[13]:


application_data.select_dtypes('float64').columns


# In[14]:


application_data.select_dtypes('int64').columns


# **A lot of the int columns look like Flags, let check their unique values**

# In[15]:


application_data.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)


# **<font color=navy>Notice a lot of "int" columns have 2 unique values. These are flags or Flag type varibles. Which have no use in bivariate analysis. These can be converted to Yes/No values for categorical analysis.</font>**

# In[16]:


#columns to convert
cols_to_convert=list(app_meta_data[(app_meta_data.Unique==2)&(app_meta_data.Data_Type=="int64")].index)

#function to conver columns
def convert_data(application_data, cols_to_convert):
    for y in cols_to_convert:
        application_data.loc[:,y].replace((0, 1), ('N', 'Y'), inplace=True)
    return application_data

#calling the function for application_data
convert_data(application_data, cols_to_convert)
application_data.TARGET.replace(('N', 'Y'), (0, 1), inplace=True)
application_data.dtypes.value_counts()


# ### <font color=maroon>Univariate Analyis on Categorical Columns</font>

# In[17]:


defaulters=application_data[application_data.TARGET==1]

nondefaulters=application_data[application_data.TARGET==0]


# **Getting a list of columns with dtype=object, to identify columns for categorical analysis**

# In[18]:


application_data.select_dtypes('object').columns


# In[19]:


## FUNCTION TO PLOT CHARTS

def plot_charts(var, label_rotation,horizontal_layout):
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15,20))
    
    s1=sns.countplot(ax=ax1,x=defaulters[var], data=defaulters, order= defaulters[var].value_counts().index,)
    ax1.set_title('Distribution of '+ '%s' %var +' for Defaulters', fontsize=10)
    ax1.set_xlabel('%s' %var)
    ax1.set_ylabel("Count of Loans")
    if(label_rotation):
        s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
    s2=sns.countplot(ax=ax2,x=nondefaulters[var], data=nondefaulters, order= nondefaulters[var].value_counts().index,)
    if(label_rotation):
        s2.set_xticklabels(s2.get_xticklabels(),rotation=90)
    ax2.set_xlabel('%s' %var)
    ax2.set_ylabel("Count of Loans")
    ax2.set_title('Distribution of '+ '%s' %var +' for Non-Defaulters', fontsize=10)
    plt.show()


# In[20]:


plot_charts('NAME_CONTRACT_TYPE', label_rotation=False,horizontal_layout=True)


# > We observe that the number of **Cash loans** is much higher than the number of **Revolving loans** for both Target = 0 and Target = 1

# In[21]:


plot_charts('CODE_GENDER', label_rotation=False,horizontal_layout=True)


# > We observe that the number of **Females** taking loans is much higher than the number of **Males** for both Target = 0 and Target = 1

# In[22]:


plot_charts('FLAG_OWN_REALTY', label_rotation=False,horizontal_layout=True)
plot_charts('FLAG_OWN_CAR', label_rotation=False,horizontal_layout=True)


# We observe that the number of most people applying for loan **do not own a car.** 
# 
# > We also observe that the **ratio of people who own a car is higher for non-defaulters**

# In[23]:


plot_charts('REG_CITY_NOT_LIVE_CITY', label_rotation=False,horizontal_layout=True)
plot_charts('REG_CITY_NOT_WORK_CITY', label_rotation=True,horizontal_layout=True)


# We observe that the Ratio of people whose **Registration City is not the same as live city or work city** is higher in case of defaulters are compared to defaulters.
# > It tells us that people who live or work in a city different than the registration city are more likely to have payment difficulties.

# In[24]:


plot_charts('NAME_HOUSING_TYPE', label_rotation=True,horizontal_layout=True)


# > Observation:
# 1. Most people live in a House/Apartment
# 2. Ratio of People who live **With Parents** is more for defaulter than non-defaulters. It tells us that applicant who live with parents have a higher chance of having payment difficulties.

# In[25]:


plot_charts('NAME_FAMILY_STATUS', label_rotation=True,horizontal_layout=True)


# Ratio of **Single/Unmarried** people is more in the left graph.
# > Single/Unmarried people are more likely to have payment difficulties

# In[26]:


plot_charts('NAME_EDUCATION_TYPE', label_rotation=True,horizontal_layout=True)


# > While the category with highest count remains same. 
# 1. This chart tells us that people with Academic Degree rarely take loans and are rarely defaulters. So they are potentially good customers.
# 2. People with higher education are less likely to have payment difficulties. The Ratio is higher for non-defaulters than defaulters.

# In[27]:


plot_charts('NAME_INCOME_TYPE', label_rotation=True,horizontal_layout=True)


# > Commercial associates, Pensioner, State Servants have a higher ratio to total in non-defaulters.

# In[28]:


plot_charts('WALLSMATERIAL_MODE', label_rotation=True,horizontal_layout=True)


# > This interesting chart tells us that most defaulters have houses made of stone and brick while most non-defaulters have houses made of Panel

# In[29]:


plot_charts('ORGANIZATION_TYPE', label_rotation=True,horizontal_layout=False)


# In[30]:


plot_charts('FLAG_WORK_PHONE', label_rotation=True,horizontal_layout=True)


# In[31]:


plot_charts('NAME_INCOME_TYPE', label_rotation=True,horizontal_layout=True)


# In[32]:


plot_charts('OCCUPATION_TYPE', label_rotation=True,horizontal_layout=True)


# ### <font color=maroon>Univariate & Bivariate Analyis on Numeric Columns</font>

# **Getting a list of columns with dtype=object, to identify columns for analysis**

# In[33]:


application_data.select_dtypes('float64').columns


# In[34]:


application_data.select_dtypes('int64').columns


# ### <font color=maroon>REMOVING OUTLIERS </font>

# In[35]:


## FUNCTION FOR PLOTTING BOX PLOT AND HISTOGRAM

def plot_boxhist(var):
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    s=sns.boxplot(y=defaulters[var]);
    plt.title('Box Plot of '+ '%s' %var +' for Defaulters', fontsize=10)
    plt.xlabel('%s' %var)
    plt.ylabel("Count of Loans")
    plt.subplot(1, 2, 2)
    s=plt.hist(x=defaulters[var]);
    plt.xlabel('%s' %var)
    plt.ylabel("Count of Loans")
    plt.title('Histogram of '+ '%s' %var +' for Defaulters', fontsize=10)
plt.show()


# In[36]:


plot_boxhist('AMT_INCOME_TOTAL')


# > We can see that there are some outliers and the graph looks like this to accomodate those outliers.

# In[37]:


#Removing all entries above 99 percentile
application_data=application_data[application_data.AMT_INCOME_TOTAL<np.nanpercentile(application_data['AMT_INCOME_TOTAL'], 99)]

#update dataframes
defaulters=application_data[application_data.TARGET==1] 
nondefaulters=application_data[application_data.TARGET==0]

plot_boxhist('AMT_INCOME_TOTAL')


# > This tell us that most people with payment have incomes in the lower range between 100000 to 200000 which some on the higher end some on the lower

# In[38]:


plot_boxhist('AMT_CREDIT')


# In[39]:


#Removing all entries above 99 percentile
application_data=application_data[application_data.AMT_CREDIT<np.nanpercentile(application_data['AMT_CREDIT'], 99)]

#update dataframes
defaulters=application_data[application_data.TARGET==1] 
nondefaulters=application_data[application_data.TARGET==0]

plot_boxhist('AMT_CREDIT')


# > we observe that the credit amount lies between 250000 to around 500000 for defaulters

# In[40]:


plot_boxhist('AMT_ANNUITY')


# In[41]:


#Removing all entries above 99 percentile
application_data=application_data[application_data.AMT_ANNUITY<np.nanpercentile(application_data['AMT_ANNUITY'], 90)]

#update dataframes
defaulters=application_data[application_data.TARGET==1] 
nondefaulters=application_data[application_data.TARGET==0]

plot_boxhist('AMT_ANNUITY')


# In[42]:


#Deriving new metric Age from Days Birth
application_data['AGE'] = application_data['DAYS_BIRTH'] / -365
plt.hist(application_data['AGE']);
plt.title('Histogram of age in years.');


# > Age seems to be fairly distributed

# In[43]:


sns.boxplot(y=application_data['DAYS_EMPLOYED']);
plt.title('Length of days employed before loan.');


# In[44]:


application_data['DAYS_EMPLOYED'].describe()


# > There is an outlier here. The max value is 365243 days which is not practically possible. 
# > This might be an error and we can replace this value with null

# In[45]:


application_data['DAYS_EMPLOYED']=application_data['DAYS_EMPLOYED'].replace(365243, np.nan)
application_data['DAYS_EMPLOYED'].describe()


# In[46]:


#Deriving variable "Years Employed" from days employed
application_data['YEARS_EMPLOYED'] = (application_data['DAYS_EMPLOYED']/-365)

#update dataframes
defaulters=application_data[application_data.TARGET==1] 
nondefaulters=application_data[application_data.TARGET==0]


# In[47]:


plot_boxhist('YEARS_EMPLOYED')


# > A large number of entries have 0 which means, a lot of people don't work.

# In[48]:


application_data.groupby(['NAME_INCOME_TYPE']).agg({'YEARS_EMPLOYED': ['mean', 'median', 'count', 'max'], 'AGE': ['median']})


# > We see that Pensioners comprise a lot of non-working people, which is normal. Working people seemed to have worked for many years.

# In[49]:


application_data.groupby(['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE']).agg({'AMT_INCOME_TOTAL': ['mean', 'median', 'count', 'max']})


# > We can see that most of the loans are taken by working people with secondary education.

# ### <font color=maroon>Binning of Continuous Variables</font>

# In[50]:


application_data['AMT_INCOME_TOTAL'].describe()


# In[51]:


defaulters.loc[:,'INCOME_BRACKET']=pd.qcut(application_data.loc[:,'AMT_INCOME_TOTAL'],q=[0,0.10,0.35,0.50,0.90,1], labels=['Very_low','Low','Medium','High','Very_high'])
nondefaulters.loc[:,'INCOME_BRACKET']=pd.qcut(application_data.loc[:,'AMT_INCOME_TOTAL'],q=[0,0.10,0.35,0.50,0.90,1], labels=['Very_low','Low','Medium','High','Very_high'])


# ### <font color=Maroon>Analysis of Continuous variables for TARGET=1 and TARGET=0</font>

# In[52]:


plot_charts('INCOME_BRACKET', label_rotation=True,horizontal_layout=True)


# In[53]:


defaulters.loc[:,'Rating1']=pd.cut(application_data.loc[:,'EXT_SOURCE_1'],[0,0.10,0.35,0.50,0.80,1], labels=['Very_low','Low','Medium','High','Very_high'])
nondefaulters.loc[:,'Rating1']=pd.cut(application_data.loc[:,'EXT_SOURCE_1'],[0,0.10,0.35,0.50,0.80,1], labels=['Very_low','Low','Medium','High','Very_high'])


# In[54]:


plot_charts('Rating1', label_rotation=True,horizontal_layout=True)


# **A large number of defaulters have very Low rating, while a large number of non-defaulters have a high rating.**
# 

# In[55]:


defaulters.loc[:,'Rating2']=pd.cut(application_data.loc[:,'EXT_SOURCE_2'],[0,0.10,0.35,0.50,0.80,1], labels=['Very_low','Low','Medium','High','Very_high'])
nondefaulters.loc[:,'Rating2']=pd.cut(application_data.loc[:,'EXT_SOURCE_2'],[0,0.10,0.35,0.50,0.80,1], labels=['Very_low','Low','Medium','High','Very_high'])


# In[56]:


plot_charts('Rating2', label_rotation=True,horizontal_layout=True)


# **A large number of defaulters have Low rating, while a large number of non-defaulters have a high rating.**

# In[57]:


defaulters.loc[:,'Rating3']=pd.cut(application_data.loc[:,'EXT_SOURCE_3'], [0,0.10,0.35,0.50,0.80,1], labels=['Very_low','Low','Medium','High','Very_high'])
nondefaulters.loc[:,'Rating3']=pd.cut(application_data.loc[:,'EXT_SOURCE_3'], [0,0.10,0.35,0.50,0.80,1], labels=['Very_low','Low','Medium','High','Very_high'])


# In[58]:


plot_charts('Rating3', label_rotation=True,horizontal_layout=True)


# **A large number of defaulters have very Low rating, while a large number of non-defaulters have a high rating.**
# 

# In[59]:


defaulters.loc[:,'AMT_ANNUITY_BINS']=pd.qcut(application_data.loc[:,'AMT_ANNUITY'], [0,0.30,0.50,0.85,0.1], labels=['Low','Medium','High','Very_High'])
nondefaulters.loc[:,'AMT_ANNUITY_BINS']=pd.qcut(application_data.loc[:,'AMT_ANNUITY'], [0,0.30,0.50,0.85,1], labels=['Low','Medium','High','Very_High'])


# In[60]:


plot_charts('AMT_ANNUITY_BINS', label_rotation=False,horizontal_layout=True)


# **maxinum number of defaulters have Low_annuity Values, while maximum number of non-defaulters have high annuity**

# In[61]:


age_data = application_data.loc[:,['TARGET', 'DAYS_BIRTH']]
age_data.loc[:,'YEARS_BIRTH'] = application_data.loc[:,'DAYS_BIRTH']/ -365
# Bin the age data
age_data.loc[:,'YEARS_BINNED'] = pd.cut(age_data.loc[:,'YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)


# In[62]:


age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups


# In[63]:


plt.figure(figsize = (8, 8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');


# > Maximum Failure to Repay is in Age Group 20-25

# ### <font color=Navy> Bi-Variate Analysis of Variables</font>

# In[64]:


#selecting columns for correlation, removing cols for floor and house ec

cols=['EXT_SOURCE_1', 'EXT_SOURCE_3', 'EXT_SOURCE_2',
       'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'CNT_FAM_MEMBERS',
       'DAYS_LAST_PHONE_CHANGE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL',
       'DAYS_REGISTRATION', 'REGION_POPULATION_RELATIVE','CNT_CHILDREN', 'HOUR_APPR_PROCESS_START',
       'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT',
       'DAYS_ID_PUBLISH', 'DAYS_EMPLOYED', 'DAYS_BIRTH']


# In[65]:


defaulters_1=defaulters[cols]
defaulters_correlation = defaulters_1.corr()
round(defaulters_correlation, 3)


# In[66]:


defaulters_correlation.head(10).index


# In[67]:


c1=defaulters_correlation.unstack()
c1.sort_values(ascending=False).drop_duplicates().head(10)


# In[68]:


c1.sort_values(ascending=False).drop_duplicates().tail(10)


# In[69]:


# figure size
plt.figure(figsize=(30,20))

# heatmap
sns.heatmap(defaulters_correlation, cmap="YlGnBu", annot=True)
plt.show()


# #### 5 most positive correlations
# 1. AMT_CREDIT - AMT_GOODS_PRICE
# 2. REGION_RATING_CLIENT_W_CITY - REGION_RATING_CLIENT
# 3. CNT_CHILDREN - CNT_FAM_MEMBERS
# 4. AMT_CREDIT - AMT_ANNUITY
# 5. AMT_GOODS_PRICE - AMT_ANNUITY

# #### 5 most negative correlations
# 1. HOUR_APPR_PROCESS_START - REGION_RATING_CLIENT_W_CITY
# 2. REGION_RATING_CLIENT - HOUR_APPR_PROCESS_START
# 3. REGION_POPULATION_RELATIVE - REGION_RATING_CLIENT
# 4. REGION_RATING_CLIENT_W_CITY - REGION_POPULATION_RELATIVE
# 5. EXT_SOURCE_1 - DAYS_BIRTH

# In[70]:


nondefaulters_1=nondefaulters[cols]
nondefaulters_correlation = nondefaulters_1.corr()
round(nondefaulters_correlation, 3)


# In[71]:


nondefaulters_correlation.head(10).index


# In[72]:


c2=nondefaulters_correlation.unstack()
c2.sort_values(ascending=False).drop_duplicates().head(10)


# In[73]:


c2.sort_values(ascending=False).drop_duplicates().tail(10)


# In[74]:


# figure size
plt.figure(figsize=(30,20))

# heatmap
sns.heatmap(nondefaulters_correlation, cmap="YlGnBu", annot=True)
plt.show()


# ### <font color=navy> Analysis of Previous Application Dataset</font>

# In[75]:


#importing data from CSV file into pandas dataframe

previous_data = pd.read_csv('previous_application.csv')
previous_data.head()


# In[76]:


previous_data.shape


# <font color=navy>**This dataset for application data has:**</font>
# - <font color=navy>**1670214 rows**</font>
# - <font color=navy>**37 columns**</font>

# In[77]:


application_data.dtypes.value_counts()


# **<font color = navy>We can see that there are:**</font>
# - **<font color = navy>65 columns with dtype=float64**</font>
# - **<font color = navy>41 columns with dtype=int64**</font>
# - **<font color = navy>16 columns with dtype=object</font>**

# In[78]:


previous_data.columns


# In[79]:


previous_data.info()


# In[80]:


previous_data.NAME_CONTRACT_STATUS.unique()


# In[81]:


import matplotlib
sns.countplot(previous_data.NAME_CONTRACT_STATUS)
plt.xlabel("Contract Status")
plt.ylabel("Count of Contract Status")
plt.title("Distribution of Contract Status")
plt.show()


# #### Identifying missing values and filtering out columns with high missing values

# In[82]:


prev_meta_data=meta_data(previous_data)
prev_meta_data.reset_index(drop=False).head(20)


# In[83]:


#dropping columns with more than 55% missing values 
cols_to_keep=list(prev_meta_data[(prev_meta_data.Percent<55)].index)
previous_data=previous_data[cols_to_keep]
previous_data.describe()


# In[84]:


#Checking columns with very less missing values
low_missing=pd.DataFrame(prev_meta_data[(prev_meta_data.Percent>0)&(prev_meta_data.Percent<15)])
low_missing


# > Both of these columns should not be imputed with any values

# In[85]:


cols_to_convert=list(prev_meta_data[(prev_meta_data.Unique==2)&((prev_meta_data.Data_Type=="int64")|(prev_meta_data.Data_Type=="float64"))].index)
cols_to_convert


# In[86]:


def convert_data(previous_data, cols_to_convert):
    for y in cols_to_convert:
        previous_data.loc[:,y].replace((0, 1), ('N', 'Y'), inplace=True)
    return previous_data
convert_data(previous_data, cols_to_convert)
previous_data.dtypes.value_counts()


# In[87]:


approved=previous_data[previous_data.NAME_CONTRACT_STATUS=='Approved']
refused=previous_data[previous_data.NAME_CONTRACT_STATUS=='Refused']
canceled=previous_data[previous_data.NAME_CONTRACT_STATUS=='Canceled']
unused=previous_data[previous_data.NAME_CONTRACT_STATUS=='Unused Offer']


# In[88]:


percentage_approved=(len(approved)*100)/len(previous_data)
percentage_refused=(len(refused)*100)/len(previous_data)
percentage_canceled=(len(canceled)*100)/len(previous_data)
percentage_unused=(len(unused)*100)/len(previous_data)

print("The Percentage of people whose loans have been Approved is:",round(percentage_approved,2),"%")
print("The Percentage of people whose loans have been Refused is:",round(percentage_refused,2),"%")
print("The Percentage of people whose loans have been Canceled is:",round(percentage_canceled,2),"%")
print("The Percentage of people whose loans have been Unused is:",round(percentage_unused,2),"%")


# In[89]:


def plot_3charts(var, label_rotation,horizontal_layout):
    if(horizontal_layout):
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(15,30))
    
    s1=sns.countplot(ax=ax1,x=refused[var], data=refused, order= refused[var].value_counts().index,)
    ax1.set_title("Refused", fontsize=10)
    ax1.set_xlabel('%s' %var)
    ax1.set_ylabel("Count of Loans")
    if(label_rotation):
        s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
    
    s2=sns.countplot(ax=ax2,x=approved[var], data=approved, order= approved[var].value_counts().index,)
    if(label_rotation):
        s2.set_xticklabels(s2.get_xticklabels(),rotation=90)
    ax2.set_xlabel('%s' %var)
    ax2.set_ylabel("Count of Loans")
    ax2.set_title("Approved", fontsize=10)
    
    
    s3=sns.countplot(ax=ax3,x=canceled[var], data=canceled, order= canceled[var].value_counts().index,)
    ax3.set_title("Canceled", fontsize=10)
    ax3.set_xlabel('%s' %var)
    ax3.set_ylabel("Count of Loans")
    if(label_rotation):
        s3.set_xticklabels(s3.get_xticklabels(),rotation=90)
    plt.show()


# In[90]:


previous_data.select_dtypes('object').columns


# In[91]:


plot_3charts('PRODUCT_COMBINATION', label_rotation=True,horizontal_layout=True)


# - We observe most number of loans were approved for POS household with interest.
# - Most number of refused loans were of Cash X-Sell: Low Product combination
# - Most Canceled loans were Cash loans

# In[92]:


plot_3charts('NAME_YIELD_GROUP', label_rotation=True,horizontal_layout=True)


# - Most approved loans were from **Middle** Yield Goup
# - Most refused loans were from Yield Goups Not specified

# In[93]:


plot_3charts('NAME_PORTFOLIO', label_rotation=True,horizontal_layout=True)


# - Most approved loans were **POS**
# - Most refused loans were **Cash**

# In[94]:


plot_3charts('CHANNEL_TYPE', label_rotation=True,horizontal_layout=True)


# - Most approved loans were from **Country-wide** Channel
# - Most refused loans were from **Credit and Cash Offices** Channel

# In[95]:


plot_3charts('NAME_PRODUCT_TYPE', label_rotation=True,horizontal_layout=True)


# In[96]:


plot_3charts('NAME_PAYMENT_TYPE', label_rotation=True,horizontal_layout=True)


# In[97]:


plot_3charts('NAME_CONTRACT_TYPE', label_rotation=True,horizontal_layout=True)


# In[98]:


plot_3charts('NAME_CLIENT_TYPE', label_rotation=True,horizontal_layout=True)


# In[99]:


sns.countplot(x=approved['NAME_CLIENT_TYPE'], data=previous_data)


# ## <font color=navy>Removing Outliers</font>

# In[100]:


fig, ax = plt.subplots(figsize = (30, 8))
plt.subplot(1, 2, 1)
sns.boxplot(y=approved['AMT_ANNUITY']);
plt.subplot(1, 2, 2)
plt.hist(approved['AMT_ANNUITY'])
plt.title('AMT_ANNUITY')
plt.show()


# In[ ]:





# In[101]:


approved = approved[approved.AMT_ANNUITY < np.nanpercentile(approved['AMT_ANNUITY'], 99)]

fig, ax = plt.subplots(figsize=(40, 10))
fig.suptitle('AMT_ANNUITY')

plt.subplot(1, 2, 1)
sns.boxplot(y=approved['AMT_ANNUITY'])

plt.subplot(1, 2, 2)
plt.hist(approved['AMT_ANNUITY'])


# In[102]:


fig, ax = plt.subplots(figsize = (30, 8))
plt.subplot(1, 2, 1)
sns.boxplot(y=approved['AMT_CREDIT']);
plt.subplot(1, 2, 2)
plt.hist(approved['AMT_CREDIT'])
plt.title('AMT_CREDIT')
plt.show()


# In[103]:



approved = approved[approved.AMT_CREDIT < np.nanpercentile(approved['AMT_CREDIT'], 90)]

fig, ax = plt.subplots(figsize=(30, 8))

plt.subplot(1, 2, 1)
sns.boxplot(y=approved['AMT_CREDIT']);
plt.title('AMT_CREDIT boxplot on data within 99 percentile');

plt.subplot(1, 2, 2)
plt.hist(approved['AMT_CREDIT'])
plt.title('AMT_CREDIT')
fig.suptitle('AMT_CREDIT boxplot and histogram on data within 90 percentile', fontsize=16);


# In[104]:


fig, ax = plt.subplots(figsize = (30, 8))
plt.subplot(1, 2, 1)
sns.boxplot(y=approved['AMT_GOODS_PRICE']);
plt.subplot(1, 2, 2)
plt.hist(approved['AMT_GOODS_PRICE'])
plt.title('AMT_GOODS_PRICE')
plt.show()


# In[105]:


approved = approved[approved.AMT_GOODS_PRICE < np.nanpercentile(approved['AMT_GOODS_PRICE'], 90)]

fig, ax = plt.subplots(figsize=(30, 8))

plt.subplot(1, 2, 1)
sns.boxplot(y=approved['AMT_GOODS_PRICE'])
plt.title('AMT_GOODS_PRICE boxplot on data within 90 percentile');

plt.subplot(1, 2, 2)
plt.hist(approved['AMT_GOODS_PRICE'])
plt.title('AMT_GOODS_PRICE')

fig.suptitle('AMT_GOODS_PRICE boxplot and histogram on data within 90 percentile', fontsize=16);


# ### BIVARIATE ANALYSIS OF VARIABLES

# In[106]:


cols_approved=['AMT_DOWN_PAYMENT', 'RATE_DOWN_PAYMENT', 'DAYS_TERMINATION', 'DAYS_LAST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_FIRST_DUE', 'DAYS_FIRST_DRAWING', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'CNT_PAYMENT', 'AMT_CREDIT', 'DAYS_DECISION', 'AMT_APPLICATION']
approved_num=approved[cols_approved]


# In[107]:


cols_refused=['AMT_DOWN_PAYMENT', 'RATE_DOWN_PAYMENT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'CNT_PAYMENT', 'AMT_CREDIT', 'DAYS_DECISION', 'AMT_APPLICATION']
refused_num=refused[cols_refused]


# In[108]:


#calculating correlation for approved
approved_correlation = approved_num.corr()
round(approved_correlation, 3)


# In[109]:


c1=approved_correlation.unstack()
c1.sort_values(ascending=False).drop_duplicates().head(10)


# In[110]:


c1.sort_values(ascending=False).drop_duplicates().tail(10)


# In[111]:


# figure size
plt.figure(figsize=(30,20))

# heatmap
sns.heatmap(approved_correlation, cmap="YlGnBu", annot=True)
plt.show()


# In[112]:


#calculating correlation for approved
refused_correlation = refused_num.corr()
round(refused_correlation, 3)


# In[113]:


# figure size
plt.figure(figsize=(30,20))

# heatmap
sns.heatmap(refused_correlation, cmap="YlGnBu", annot=True)
plt.show()


# In[114]:


c2=refused_correlation.unstack()
c2.sort_values(ascending=False).drop_duplicates().head(10)


# In[115]:


c2.sort_values(ascending=False).drop_duplicates().tail(10)


# In[116]:


def has_terminated(x):
    if x < 0:
        return 'Loan Terminated'
    else:
        return 'Loan Open'
    
approved['CURRENT_STATUS'] = approved['DAYS_TERMINATION'].apply(has_terminated) 


# In[134]:


plt.figure(figsize=(5,5))
sns.countplot(x=approved['CURRENT_STATUS'], data=approved)
plt.show()


# # Conclusion

#  - Banks should focus more on contract type ‘Student’ ,’pensioner’ and ‘Businessman’ with housing ‘type other than ‘Co-op          apartment’ and 'office appartment' for successful payments.
# - Banks should focus less on income types maternity leave and working as they have most number of unsuccessful payments In loan   purpose ‘Repairs’:
# 
#   -   a. Although having higher number of rejection in loan purposes with 'Repairs' we can observe difficulties in payment.
#   -   b. There are few places where loan payment diffuculty is significantly high.
#   -   c. Bank should continue to be cautious while giving loan for this purpose.
#   
#   
# - Bank can focus mostly on housing type with parents , House or apartment and municipal apartment with purpuse of education, buying land, buying a garage, purchase of electronic equipment and some other purposes with target0 significantly more than target1 for successful payments.
# 
# -  Banks can offer more offers to clients who are students and pensioners as they take all offers and are more likely to pay back
# 
# -  CODE_GENDER: Men are at relatively higher default rate
# -  NAME_FAMILY_STATUS : People who have civil marriage or who are single default a lot.
# -  NAME_EDUCATION_TYPE: People with Lower Secondary & Secondary education
# -  NAME_INCOME_TYPE: Clients who are either at Maternity leave OR Unemployed default a lot.
# -  REGION_RATING_CLIENT: People who live in Rating 3 has highest defaults.
# -  OCCUPATION_TYPE: Avoid Low-skill Laborers, Drivers and Waiters/barmen staff, Security staff, Laborers and Cooking staff as      the default rate is huge.
# -  ORGANIZATION_TYPE: Organizations with highest percent of loans not repaid are Transport: type 3 (16%), Industry: type 13        (13.5%), Industry: type 8 (12.5%) and Restaurant (less than 12%). Self-employed people have relative high defaulting rate,      and      thus should be avoided to be approved for loan or provide loan with higher interest rate to mitigate the risk of        defaulting.
# -  DAYS_BIRTH: Avoid young people who are in age group of 20-40 as they have higher probability of defaulting
# -  DAYS_EMPLOYED: People who have less than 5 years of employment have high default rate.
# -  CNT_CHILDREN & CNT_FAM_MEMBERS: Client who have children equal to or more than 9 default 100% and hence their applications      are to be rejected.
# -  AMT_GOODS_PRICE: When the credit amount goes beyond 3M, there is an increase in defaulters.

# ### The following attributes indicate that people from these category tend to default but then due to the number of people and the amount of loan, the bank could provide loan with higher interest to mitigate any default risk thus preventing business loss:

# -  NAME_HOUSING_TYPE: High number of loan applications are from the category of people who live in Rented apartments & living with parents and hence offering the loan would mitigate the loss if any of those default.
# -  AMT_CREDIT: People who get loan for 300-600k tend to default more than others and hence having higher interest specifically for this credit range would be ideal.
# -  AMT_INCOME: Since 90% of the applications have Income total less than 300,000 and they have high probability of defaulting, they could be offered loan with higher interest compared to other income category.
# -  CNT_CHILDREN & CNT_FAM_MEMBERS: Clients who have 4 to 8 children has a very high default rate and hence higher interest should be imposed on their loans.
# -  NAME_CASH_LOAN_PURPOSE: Loan taken for the purpose of Repairs seems to have highest default rate. A very high number applications have been rejected by bank or refused by client in previous applications as well which has purpose as repair or other. This shows that purpose repair is taken as high risk by bank and either they are rejected, or bank offers very high loan interest rate which is not feasible by the clients, thus they refuse the loan. The same approach could be followed in future as well.

# ## Other suggestions:

# - 90% of the previously cancelled client have actually repayed the loan. Record the reason for cancellation which might help the bank to determine and negotiate terms with these repaying customers in future for increase business opportunity.
# - 88% of the clients who were refused by bank for loan earlier have now turned into a repaying client. Hence documenting the reason for rejection could mitigate the business loss and these clients could be contacted for further loans.
#  
