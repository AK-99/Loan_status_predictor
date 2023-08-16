#!/usr/bin/env python
# coding: utf-8

# # Keras API Project  on loan status predictor
# 
# ## The Data
# 
# We will be using a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club
# 
# 
# ### The Objective
# 
# 
# Can we create a model that can predict whether or not a borrower will repay their loan given historical data on loans issued and information on whether or not the borrower defaulted (charge-off)? By doing this, we will be able to determine whether or not a potential consumer will be able to repay the loan in the future. When assessing the effectiveness of your model, consider categorization metrics!
# 
# 

# ## Importing libraries
# 
# ####

# In[14]:


import pandas as pd


# In[15]:


data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')


# In[16]:


print(data_info.loc['revol_util']['Description'])


# In[17]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[18]:


feat_info('mort_acc')


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# might be needed depending on your version of Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


df = pd.read_csv('lending_club_loan_two.csv')


# In[21]:


df.info()


# 
# 
# 
# 
# -----
# ------
# 
# # Section 1: Exploratory Data Analysis
# 
# **OVERALL OBJECTIVE: Recognize the critical factors, see summary statistics, and display the data.**
# 
# 
# ----

# In[22]:


sns.countplot(x='loan_status',data=df)


# **Creating a histogram of the loan_amnt column.**

# In[23]:


plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False,bins=40)
plt.xlim(0,45000)


# **Exploring correlation between the continuous feature variables. Calculating the correlation between all continuous numeric variables using .corr() method.**

# In[24]:


df.corr()


# **Visualizing this using a heatmap.**
# 
# 

# In[25]:


# CODE HERE


# In[26]:


plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.ylim(10, 0)


# **There is almost perfect correlation with the "installment" feature.**

# In[27]:


feat_info('installment')


# In[28]:


feat_info('loan_amnt')


# In[29]:


sns.scatterplot(x='installment',y='loan_amnt',data=df,)


# **Creating a boxplot showing the relationship between the loan_status and the Loan Amount.**

# In[30]:


sns.boxplot(x='loan_status',y='loan_amnt',data=df)


# **Calculating the summary statistics for the loan amount, grouped by the loan_status.**

# In[31]:


df.groupby('loan_status')['loan_amnt'].describe()


# **Exploring the Grade and SubGrade columns that LendingClub attributes to the loans.**

# In[32]:


sorted(df['grade'].unique())


# In[33]:


sorted(df['sub_grade'].unique())


# **Creating  a countplot per grade. Set the hue to the loan_status label.**

# In[34]:


sns.countplot(x='grade',data=df,hue='loan_status')


# **Displaying a count plot per subgrade.**

# In[35]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )


# In[36]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' ,hue='loan_status')


# **It looks like F and G subgrades don't get paid back that often. Isloating those and recreate the countplot just for those subgrades.**

# In[37]:


f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')


# **Creating a new column called 'load_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".**

# In[38]:


df['loan_status'].unique()


# In[39]:


df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})


# In[40]:


df[['loan_repaid','loan_status']]


# In[41]:


df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# ---
# ---
# # Section 2: Data PreProcessing
# 
# **Section Objective: Remove or fill any missing data. Remove unnecessary or repetitive features. Convert categorical string features to dummy variables.**
# 
# 

# In[42]:


df.head()


# # Missing Data
# 
# **Exploring this missing data columns.**

# In[43]:


len(df)


# **Creating a Series that displays the total count of missing values per column.**

# In[44]:


df.isnull().sum()


# **Converting this Series to be in term of percentage of the total DataFrame**

# In[ ]:


100* df.isnull().sum()/len(df)


# **Examining emp_title and emp_length to see whether it will be okay to drop them.**

# In[ ]:


feat_info('emp_title')
print('\n')
feat_info('emp_length')


# In[ ]:


df['emp_title'].nunique()


# In[ ]:


df['emp_title'].value_counts()


# **Realistically there are too many unique job titles to try to convert this to a dummy variable feature. Therefore removing that emp_title column.**

# In[ ]:


df = df.drop('emp_title',axis=1)


# **Create a count plot of the emp_length feature column.**

# In[ ]:


sorted(df['emp_length'].dropna().unique())


# In[ ]:


emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']


# In[ ]:


plt.figure(figsize=(12,4))

sns.countplot(x='emp_length',data=df,order=emp_length_order)


# **Plot of countplot with a hue separating Fully Paid vs Charged Off**

# In[ ]:


plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')


# In[ ]:


emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']


# In[ ]:


emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']


# In[ ]:


emp_len = emp_co/emp_fp


# In[ ]:


emp_len


# In[ ]:


emp_len.plot(kind='bar')


# **Charge off rates are extremely similar across all employment lengths. Therefore dropping the emp_length column.**

# In[ ]:


df = df.drop('emp_length',axis=1)


# **Revisiting the DataFrame to see what feature columns still have missing data.**

# In[ ]:


df.isnull().sum()


# **Review the title column vs the purpose column.**

# In[ ]:


df['purpose'].head(10)


# In[ ]:


df['title'].head(10)


# **The title column is simply a string subcategory/description of the purpose column. Therefore droping the title column.**

# In[ ]:


df = df.drop('title',axis=1)


# 
# 
# **Finding out what the mort_acc feature represents**

# In[ ]:


feat_info('mort_acc')


# **Creating a value_counts of the mort_acc column.**

# In[ ]:


df['mort_acc'].value_counts()


# In[ ]:


print("Correlation with the mort_acc column")
df.corr()['mort_acc'].sort_values()


# **Looks like the total_acc feature correlates with the mort_acc , this makes sense! using fillna() approach. We will group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry. To get the result below:**

# In[ ]:


print("Mean of mort_acc column per total_acc")
df.groupby('total_acc').mean()['mort_acc']


# **filling in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above. This involves using an .apply() method with two columns.**
# 
# 

# In[ ]:


total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# In[ ]:


total_acc_avg[2.0]


# In[ ]:


def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


# In[ ]:


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)


# In[ ]:


df.isnull().sum()


# **revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data. removing the rows that are missing those values in those columns with dropna().**

# In[ ]:


df = df.dropna()


# In[ ]:


df.isnull().sum()


# ## Categorical Variables and Dummy Variables
# 
# **We're done working with the missing data! Now we just need to deal with the string values due to the categorical columns.**
# 

# In[ ]:


df.select_dtypes(['object']).columns


# 
# 
# 
# ### term feature
# 
# **Converting the term feature into either a 36 or 60 integer numeric data type using .apply() or .map().**

# In[ ]:


df['term'].value_counts()


# In[ ]:


# Or just use .map()
df['term'] = df['term'].apply(lambda term: int(term[:3]))


# ### grade feature
# 
# **We already know grade is part of sub_grade, so just drop the grade feature.**

# In[ ]:


df = df.drop('grade',axis=1)


# **Converting the subgrade into dummy variables.**

# In[ ]:


subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)


# In[ ]:


df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)


# In[ ]:


df.columns


# In[ ]:


df.select_dtypes(['object']).columns


# ### verification_status, application_type,initial_list_status,purpose 
# 

# In[ ]:


dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)


# ### home_ownership
# **Review the value_counts for the home_ownership column.**

# In[ ]:


df['home_ownership'].value_counts()


# In[ ]:


df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)


# ### address
# **TASK:  feature engineering a zip code column from the address in the data set. Creating a column called 'zip_code' that extracts the zip code from the address column.**

# In[ ]:


df['zip_code'] = df['address'].apply(lambda address:address[-5:])


# **Now making this zip_code column into dummy variables using pandas. Concatenating the result and drop the original zip_code column along with dropping the address column.**

# In[ ]:


dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)


# ### issue_d 
# 
# **This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date, drop this feature.**

# In[ ]:


df = df.drop('issue_d',axis=1)


# ### earliest_cr_line
# 

# In[ ]:


df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)


# In[ ]:


df.select_dtypes(['object']).columns


# ## Train Test Split

# **Importing train_test_split from sklearn.**

# In[ ]:


from sklearn.model_selection import train_test_split


# **droping the load_status column we created earlier, since its a duplicate of the loan_repaid column. using the loan_repaid column since its already in 0s and 1s.**

# In[ ]:


df = df.drop('loan_status',axis=1)


# In[ ]:


X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values


# **Performing a train/test split with test_size=0.2 and a random_state of 101.**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# ## Normalizing the Data
# 
# **Using a MinMaxScaler to normalize the feature data X_train and X_test.**

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train)


# In[ ]:


X_test = scaler.transform(X_test)


# # Creating the Model
# 
# **importing the necessary Keras functions.**

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm


# 

# In[ ]:


# CODE HERE
model = Sequential()

# Choose whatever number of layers/neurons you want.




# In[ ]:


model = Sequential()


# input layer
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')


# **Fitting the model to the training data for at least 25 epochs. Also add in the validation data for later plotting.**

# In[ ]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          )


# **saving**

# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


model.save('full_data_project_model.h5')  


# # Section 3: Evaluating Model Performance.
# 
# **Plotting out the validation loss versus the training loss.**

# In[ ]:


losses = pd.DataFrame(model.history.history)


# In[ ]:


losses[['loss','val_loss']].plot()


# **Creating predictions from the X_test set and display a classification report and confusion matrix for the X_test set.**

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


predictions = model.predict_classes(X_test)


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


confusion_matrix(y_test,predictions)


# **Using the predictions on a given customer below, an finding out whether to give loan**

# In[ ]:


import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[ ]:


model.predict_classes(new_customer.values.reshape(1,78))


# **Now checking, did this person actually end up paying back their loan**

# In[ ]:


df.iloc[random_ind]['loan_repaid']

