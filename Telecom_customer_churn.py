#!/usr/bin/env python
# coding: utf-8

# # Analysis on Telcom Customer Churn

# Context:
# -----------
# 
# "Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." 
# 
# Content:
# ------------
# 
# Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
# 
# The data set includes information about:
# ------------------------------------------------------------
# 
# Customers who left within the last month – the column is called Churn
# 
# Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# 
# Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# 
# Demographic info about customers – gender, age range, and if they have partners and dependents

# # Exploratory Data Analysis

# In[1]:


#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading the file
df=pd.read_csv("telecom_customer_churn.csv")
#pd.options.display.max_columns = 30 
#pd.options.display.max_rows = None 


# 
# Context
# "Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]
# 
# Content
# Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
# 
# The data set includes information about:
# 
# Customers who left within the last month – the column is called Churn
# Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# Demographic info about customers – gender, age range, and if they have partners and dependents
# 

# In[3]:


#first 5 rows are obtained
df.head()


# In[4]:


#shape gives the no:of rows and columns
df.shape


# In[5]:


#gives the general info about the dataset like the non-null counts and the datatype of each column
df.info()


# In[6]:


df.dtypes
#total charges is seen to be object.


# In[7]:


#shows the list of columns
df.columns


# In[8]:


#to find whether there is any null values for each column
df.isnull().sum()
#no null values


# In[9]:


#df.TotalCharges = df.TotalCharges.astype('float64')
#gives errors as this column is having spaces instead of null values .So need to convert to 0's and type cast accordingly


# In[10]:


#showing the rows having spaces for the column total charges
df[df["TotalCharges"]==" "][["TotalCharges"]]


# In[11]:


# Replacing the mssing value into 0 and converting the object into float value.
df['TotalCharges'] = df['TotalCharges'].replace(" ", 0).astype('float32')


# In[12]:


#checking whether the column is converted into float type
df.info()


# In[13]:


#summary of numerical columns
df.describe()


# In[14]:


#summary of categorical columns
df.describe(include = object)


# In[15]:


#The columns having "No internet service" are replaced to "No"
#The column "MultipleLines" having "No phone service" are replaced to "No"

replace_cols=["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
for i in replace_cols:
    df[i]  = df[i].replace('No internet service' , 'No')
df["MultipleLines"]=df["MultipleLines"].replace("No phone service","No")


# In[16]:


#to check if duplicated rows are present
df.duplicated().sum()


# In[17]:


y = pd.crosstab(df["Churn"],columns = "Frequency")
print(y)
#no: of customers churned = 1869
#no: of customers not churned = 5174


# In[18]:


#bar plot showing the customers who churned and who didn't
y_bar = y.plot(kind="bar")
y_percent = y/len(df)*100
print(round(y_percent,2))
#27% churned
#73% not churned


# In[19]:


#categorical columns and numerical columns
categorical_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","Churn"]
numerical_cols = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]


# # Hypothesis Generation

# Possible Questions or variables to be checked:
# 
# 1)tenure - which category of people (people with high tenure or low tenure) are getting churned.We need to know if recently joining cstomers are churning or not
# 
# 2)MonthlyCharges - if the monthly charges are high, there is a chance for churning.We need to analyse whether monthly charges are high or not
# 
# 3)TotalCharges - Same as monthly charge, total charge should increase accoding to monthly charges
# 
# 4)SeniorCitizen - need to check whether senior citizens are more tending to churn
# 
# 5)PaymentMethod - to check whether payment method is creating any transaction issues which is causing churning.Which among them is causing issue
# 
# 6)PaperlessBilling - to see how many customers using paperless billing and analyse it with respect to churning 
# 
# 7)There are multiple services that company is providing like phone,internet,multiple lines, etc.check which particular service or which all services is giving more churning

# **KDE PLOT on tenure, MonthlyCharges and TotalCharges.**

# In[20]:


"""
checking the churn status of other numerical fields using kde plot
we can see that recent joiners have a churning tendency more and high monthly charges leads to churning
"""
def kde(feature):
    plt.figure(figsize=(9,4))
    plt.title("kde plot for {}".format(feature))
    ax0=sns.kdeplot(df[df["Churn"]=="Yes"][feature],color="red",label= "Churn - Yes")
    ax1=sns.kdeplot(df[df["Churn"]=="No"][feature],color="green",label="Churn - No")
kde("tenure")
kde("MonthlyCharges")
kde("TotalCharges")


# # Tenure

# In[21]:


#Univariate Analysis
#histogram
sns.distplot(df["tenure"])


# In[22]:


# there is a good no: of people with less than 10 months of tenure approximately 26%
df[df["tenure"]<10]["tenure"].count()/len(df)*100


# In[23]:


#summary of tenure
df["tenure"].describe()


# In[24]:


#dividing tenure into 3 categories for further analysisanalysis
#tenure>=60 months-->highest
#tenure 20 to 60 months-->medium
#tenure 0 to 20 months--->lowest
df["tenure_groups"] = np.where(df["tenure"]>=60,"highest",np.where(df["tenure"]<20,"lowest","medium"))


# In[25]:


sns.countplot(df["tenure_groups"],data=df)
pd.crosstab(df["tenure_groups"],columns="frequency")


# In[26]:


#Multivariate Analysis
#checking which tenure period gives more churning.Around 44% among the lowest tenure group has churned
tenure_Crosstab = pd.crosstab(df.tenure_groups, columns=df.Churn)
row_tot = tenure_Crosstab.sum(axis=1)
tenure_Crosstab_prop = round(tenure_Crosstab.div(row_tot,axis=0)*100)
print("---------------------------------------------------------------------------------------------------------------------------")
print("The proportion of churning in different tenure groups namley lowest,medium and highest in the order of their tenure period is: ")
print("---------------------------------------------------------------------------------------------------------------------------")
print(tenure_Crosstab_prop)
tenure_Crosstab_prop.plot(kind = 'bar' ,rot=0 , figsize = [16,5])

#lowest tenure period gives more churning


# **tenure vs Monthly charges and total charges**

# In[27]:


#as tenure is less and monthly or total charges increases, churning happens
g=sns.PairGrid(df,x_vars=["MonthlyCharges","TotalCharges"],y_vars="tenure",hue="Churn",palette="coolwarm",height=8)
g.map(plt.scatter,alpha=0.5)
plt.legend(loc=(-0.3,0.6))


# **Summary:
# low tenure is a reason for churning.This means that new joining customers are getting churned.**

# # MonthlyCharges

# In[28]:


#univarate analysis
#summary of Monthly Charges
df["MonthlyCharges"].describe()


# In[29]:


#histogram showing the distribution of monthly charges
sns.distplot(df["MonthlyCharges"])


# In[30]:


#we can see that as monthly charges increases, churning increases
sns.boxplot(x="Churn",y="MonthlyCharges",data=df,palette="coolwarm")


# **Monthly Charges vs Multiple Lines**

# In[31]:


df.MultipleLines.value_counts()


# In[32]:


"""
multiple lines with high monthly charges is showing high churning rate.
Whether or not the person has multiple lines, if he has high monthly charges, he has a tendency to churn.

"""
print(sns.boxplot(x="MultipleLines",y="MonthlyCharges",hue="Churn",data=df,palette="coolwarm"))


# **Monthly Charges vs Internet Service**

# In[33]:


#Fibre optic services have a high monthly charge when compared to others and so is the churn rate
sns.boxplot(x="InternetService",y="MonthlyCharges",hue="Churn",data=df,palette="coolwarm")


# **Monthly Charges vs Phone Service**

# In[34]:


#churning is there for people having phone service and high monthly charges
sns.boxplot(x="PhoneService",y="MonthlyCharges",hue="Churn",data=df,palette="coolwarm")


# **Monthly Charges vs Total Charges**

# In[35]:


plt.figure(figsize=(13,8))
sns.scatterplot(x="MonthlyCharges",y="TotalCharges",data = df,palette="coolwarm",hue = "Churn")
# using monthly charges for further analysis instead of total charges as both are proportional and taking anyone of this would be only required 


# **Summary:As monthly charges and total charges increases, churning increases**

# # Senior Citizen

# In[36]:


#We can infer that there are less senior citizen people(1142 senior citizens) joined when compared to youngsters
sns.countplot(x="SeniorCitizen",data=df)
pd.crosstab(df["SeniorCitizen"],columns="frequency")


# In[37]:


#here among the senior citzens,around 42% has churned where as youngsters have churned less(among youngsters, 24% only churned)
SeniorCitizen_Crosstab = pd.crosstab(df.SeniorCitizen, columns=df.Churn)
row_tot = SeniorCitizen_Crosstab.sum(axis=1)
print("------------------------------------------------------------------------------------")
SeniorCitizen_Crosstab_prop = round(SeniorCitizen_Crosstab.div(row_tot,axis=0)*100)
print("Percentage of people who got attrited among the senior citizen and youngsters: ")
print("------------------------------------------------------------------------------------")
print(SeniorCitizen_Crosstab_prop)
SeniorCitizen_Crosstab_prop.plot(kind = 'bar' ,rot=0 , figsize = [16,5])


# In[38]:


#senior citizen vs payment method


# In[39]:


#senior citizens have opted electronic check more when compared to other payment methods.
#So we need to know if there was any issue regarding electronic check
sns.barplot(x="SeniorCitizen",y="PaymentMethod",data=df)


# In[40]:


#The average monthly charges were around 90 dollars for senior citizens who have churned
#whereas the average is less for people who haven't churned around 65 dollars
sns.boxplot(x="SeniorCitizen",y="MonthlyCharges",data=df)


# **Summary:
#     Senior citizens are comparitively very less.ie, around 16%.Among these 16%, around 48% are churned .
#     When checked their monthly charges, it looks comparitively higher for people who churned among the senior citizens
#     Also, the payment method used was electronic check.We need to further analyse whether electronic check is creating an issue for them causing churning**

# # All other services including:
# **OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies**
# 

# In[41]:


replace_cols=['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
'StreamingTV', 'StreamingMovies']

#To display these columns together with subplots using for loop

x=0
y=0
num=0
plt.tight_layout()
fig, axes =plt.subplots(2,3,figsize=(15,8))
for x in range(2):
    for y in range(3):
        sns.countplot(x=replace_cols[num],data=df,hue = "Churn",ax=axes[x,y],palette="coolwarm")
        num +=1
#for people who have opted the services, the churning rate(shown in pink) is not higher with respect to the 
#churning rate of people who haven't opted in an overall view


# In[42]:


#univariate crosstab
df_service=df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
'StreamingTV', 'StreamingMovies']]
#def crosstab(df_service):

for idx, variable in enumerate(df_service.columns):
    #univariate crosstab
    other_services_Crosstab = pd.crosstab(df_service[variable],columns = df.Churn)
    row_tot = other_services_Crosstab.sum(axis=1)
    other_services_Crosstab_prop = round(other_services_Crosstab.div(row_tot,axis=0)*100)
    print("Showing the percentage of churn happened for people opting {}".format(variable))
    print("-----------------------------------------------------------------------------------")
    print(other_services_Crosstab_prop)
    print("-----------------------------------------------------------------------------------")
#churn yes and service yes is checked here
#for people who have opted the services, the churning rate is not higher as expected


# **Summary:**
#     
# **As the churning rate is less for yes category of the services, we consider this to be not effecting churning**

# # Contract

# In[43]:


#people with monthly contract showing high churning rate.
sns.countplot(x="Contract",data=df,hue="Churn",palette="coolwarm")


# In[44]:


#monthly charges is high for for all kind of contracts 
sns.boxplot(x="Contract",y="MonthlyCharges",hue="Churn",data=df,palette="coolwarm")


# **summary:Month to month contract gives churning.The reason might be they can leave the service if they are not interested.So people thinking that in mind might have chosen monthly contract.Churning is very low for one year and two year contract even if the monthly charges are high**

# # PaperlessBilling

# In[45]:


#around 60% of the customers have chosen paperless billing
round(pd.crosstab(df.PaperlessBilling,columns="frequency")/len(df)*100,2)


# In[46]:


sns.countplot(x="PaperlessBilling",data=df,palette="coolwarm")


# In[47]:


"""
Churn rate is more for people opted paperless billing.
"""
sns.countplot(x="PaperlessBilling",hue="Churn",data=df,palette="coolwarm")
print("Among the people who chose paperless billing",round(1400/(1400+2771)*100),"% are churned")
print("Among the people who didn't choose paperless billing",round(469/(469+2403)*100),"% are only churned")
df.groupby(["PaperlessBilling","Churn"])["PaperlessBilling"].agg(["count"])


# **Summary:Paperless billing is very common for customers and people opted this are having high churning**

# # Payment Method

# In[48]:


#Checking the count of different payment methods
print(df["PaymentMethod"].value_counts())

sns.countplot(x="PaymentMethod",data=df,palette="coolwarm")
plt.tight_layout()
plt.xticks(rotation=45)
#electronic check is more used


# In[49]:



PaymentMethod_Crosstab = pd.crosstab(df.PaymentMethod, columns=df.Churn)
row_tot = PaymentMethod_Crosstab.sum(axis=1)
PaymentMethod_Crosstab_prop = round(PaymentMethod_Crosstab.div(row_tot,axis=0)*100)
print("Percentage of people who got attrited among the senior citizen and youngsters: ")
print(PaymentMethod_Crosstab_prop)
PaymentMethod_Crosstab_prop.plot(kind = 'bar' ,rot=0 , figsize = [16,5])


#Electronic check payment method is giving more churning.

#among the people who have opted electronic check, around 45% are churned


# **Summary:Electronic check is causing more churning of customers even if it is more preferred.This might be because of loading issues due to traffic or there might be other complaints.**

# In[50]:


#Using a heatmap to find the correlation between the numerical columns
tc=df.corr()
sns.heatmap(tc,xticklabels=True,annot=True,cmap="coolwarm")
#here we see tenure and monthly charges show the highest churning


# # Summary on EDA

# Variables causing Churning:
# 
# 1)tenure
# 
# 2)Monthly Charges
# 
# 3)Total Charges
# 
# 4)Internet Service-Fibre optic service
# 
# 5)Senior Citizen due to monthly charges and payment method which is electronic check
# 
# 6)payment method-electronic check
# 
# 7)Contract-Month to Month
# 
# 8)PaperlessBilling

# # Converting Categorical columns to numerical columns

# In[51]:


#Creating dummy values for the categorical columns for visualization and modelling purpose
dummy=["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling","PaymentMethod","Churn"]
df=pd.get_dummies(df, prefix=dummy, columns=dummy,drop_first=True)#to get either one of the columns.ie, yes or no
df.head()


# In[52]:


contract = pd.get_dummies(df["Contract"])
contract[["Month-to-month"]].head()


# In[53]:


df = pd.concat([df,contract],axis=1)


# In[54]:


df.rename(columns = {'Month-to-month' : 'Month_to_month', 'PaymentMethod_Electronic check' : 'PaymentMethod_Electronic_check','InternetService_Fiber optic':'InternetService_Fiber_optic'}, inplace = True)


# In[55]:


#Splitting dataset into X using the important features
X=df[["TotalCharges","tenure","SeniorCitizen","MonthlyCharges",'Month_to_month']]
#target column(dependent coumn) is taken i.e churn/not churn
y = df["Churn_Yes"]


# In[56]:


X


# In[57]:


X.dtypes


# # Train Test Split

# In[58]:


#Splitting the whole dataset into 2:Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# # Feature Scaling

# In[59]:


"""
Using a standard scaler to scale all columns into a small range which makes prediction more easier and 
decreses the chances of the model getting biased.

"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train) #Fitting and transforming the X train dataset
X_test=sc.transform(X_test)#Transforming the test dataset


# # Modelling Using Classification Models and Checking Performance

# In[60]:


#Classification Model-1
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
#fitting the train values( both X and y )
classifier.fit(X_train,y_train)
#Predicting the y valus for the X_test
y_pred=classifier.predict(X_test)



#Performance check using Confusion matrix,accuracy score and Classification Report
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

print(confusion_matrix(y_pred,y_test))
print("\n") #correct prediction= 1425+250, wrong prediction=135+303

print(classification_report(y_pred,y_test))
print("\n") # to get the values of precision,recall,f1-score and support

print("accuracy score of Logistic Regression : ",round(accuracy_score(y_pred,y_test),2))
#Accuracy=( no:of correct predictions)/(Total no:of predictions)


# In[61]:


import pickle


# In[62]:


pkl_file = open("classifiertelcom.pkl", "wb") # create a binary file, open it and then only we can save the model

# wb -> opening file in writing mode
pickle.dump(classifier, pkl_file) # saving of the trained model into pkl file

# close the file
pkl_file.close()


# In[63]:


get_ipython().system('ls')


# In[64]:


classifier.predict([[30,10,0,40,1]])


# In[ ]:




