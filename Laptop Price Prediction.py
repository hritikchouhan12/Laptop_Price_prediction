#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("laptop_data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


#to count duplicate rows present in our dataset for better analysis
df.duplicated().sum()


# In[7]:


#Missing values in each column 
df.isnull().sum()


# In[8]:


#remove column unnamed:0 as it does not require
df.drop(columns=['Unnamed: 0'],inplace=True)


# In[9]:


df.head()


# In[10]:


#Replace Kg and GB from weight and Ram column to convert its data type
df["Ram"]=df["Ram"].str.replace('GB','')
df["Weight"]=df["Weight"].str.replace('kg','')


# In[11]:


df.head()


# In[12]:


#Still datatype is object of column weight and Ram, So we need to change it in float and int
df["Ram"]=df["Ram"].astype('int32')
df["Weight"]=df["Weight"].astype('float32')


# In[13]:


df.info()


# In[14]:


#Now we do univariate and bivariate analysis on data
#So,first we do univariate analysis
import seaborn as sns


# In[15]:


#plot distribution of price in density of laptop
sns.distplot(df["Price"])


# In[16]:


#number of laptops of different companies
df['Company'].value_counts().plot(kind='bar')


# In[17]:


#Average price of each brand to see the impact of brands on price
import matplotlib.pyplot as plt
sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[18]:


#count number of laptops of different types of laptop
df["TypeName"].value_counts().plot(kind='bar')


# In[19]:


#How Prices varies with types of laptop
sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[20]:


sns.distplot(df["Inches"])


# In[21]:


sns.scatterplot(x=df["Inches"],y=df["Price"])
#There is not a strong relationship between price and Inches but ya there is relation between both


# In[22]:


#lets come to screenresolution, it has no any specific datatype so its value is varying for different laptops
#lets see its value
df['ScreenResolution'].value_counts()
#it has hidden information that whether laptop is touchscreen or not, its screenresolution value


# In[23]:


df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[24]:


df.head()


# In[25]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[26]:


sns.barplot(x=df['Touchscreen'],y=df['Price'])
#price varies according to touchscreen, so it is helpful column to predict price


# In[27]:


#Extract IPS display information from screenresolution column
df['Ips']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[28]:


df.head()


# In[29]:


df['Ips'].value_counts().plot(kind='bar')


# In[30]:


sns.barplot(x=df['Ips'],y=df['Price'])


# In[31]:


#Now, Extracting screensize from screenresolution
new=df['ScreenResolution'].str.split('x',n=1,expand=True)


# In[32]:


df["X_res"]=new[0];
df["Y_res"]=new[1];


# In[33]:


df.head()


# In[34]:


#Y_res is fine but X_res is not right, we need to extract it using a regular expression
df["X_res"]=df["X_res"].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[35]:


df.head()


# In[36]:


df["X_res"]=df["X_res"].astype('int')
df["Y_res"]=df["Y_res"].astype('int')


# In[37]:


df.info()


# In[38]:


#Find co-relation of each column with price
df.corr()['Price']


# In[39]:


#Pixel per Inches is very important property of laptop which vary with price
#X_Res, Y_res,Inches separately does not make sense
df['PPi']=((((df['X_res']**2)+(df['Y_res']**2))**0.5)/df['Inches']).astype('float')


# In[40]:


df.head()


# In[41]:


df.corr()['Price']


# In[42]:


df.drop(columns=['ScreenResolution'],inplace=True)


# In[43]:


df.head()


# In[44]:


df.drop(columns=['X_res','Y_res','Inches'],inplace=True)


# In[45]:


df.head()


# In[46]:


df['Cpu'].value_counts()


# In[47]:


#extract processor information from CPU column
df['Cpu Name']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[48]:


df.head()


# In[49]:


#Fetch Processor from CPU name column
def fetch_processor(text):
    if text=="Intel Core i3" or text=="Intel Core i5" or text=="Intel Core i7":
        return text
    else:
        if text.split()[0]=='Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD processor'


# In[50]:


df['Cpu Brand']=df['Cpu Name'].apply(fetch_processor)


# In[51]:


df.head()


# In[52]:


df['Cpu Brand'].value_counts().plot(kind='bar')


# In[53]:


sns.barplot(x=df['Cpu Brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[54]:


df.drop(columns=['Cpu','Cpu Name'],inplace=True)


# In[55]:


df.head()


# In[56]:


#lets move to Ram now.So, Ram has good corelation with price
df['Ram'].value_counts().plot(kind='bar')


# In[57]:


#there is linear relationship of Ram with price
sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[58]:


#lets focus on memory column
df["Memory"].value_counts()
#there are different categories of memory, so we need here good feature engineering


# In[59]:


# here, we have to create different columns according to type of memory like SSD,HDD,hybrid,flash storage
#preprocessing
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


# In[60]:


df.head()


# In[61]:


df.drop(columns=["Memory"],inplace=True)


# In[62]:


df.head()


# In[63]:


df.corr()['Price']


# In[64]:


#it seems very less relation with hybrid and flash_storage. So ,we drop it
df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)


# In[65]:


df.head()


# In[66]:


#GPU and Operating system Opsys column is left, so we need to also focus on this
df['Gpu'].value_counts()


# In[67]:


#we extract GPU brand from GPU
df['Gpu Brand']=df['Gpu'].apply(lambda x:x.split()[0])


# In[68]:


df.head()


# In[69]:


df['Gpu Brand'].value_counts()


# In[70]:


df=df[df['Gpu Brand']!='ARM']


# In[71]:


df['Gpu Brand'].value_counts()


# In[72]:


#lets see how graphics card vary with price
sns.barplot(x=df['Gpu Brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[73]:


df.drop(columns=['Gpu'],inplace=True)


# In[74]:


df.head()


# In[75]:


df['OpSys'].value_counts()


# In[76]:


sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[83]:


def cat_os(inp):
    if inp=='Windows 10' or inp=='Windows 7' or inp=='Windows 10 S':
        return 'Windows'
    elif inp=='macOS' or inp=='Mac OS X':
        return 'Mac'
    else:
        return 'others/No OS/Linux'


# In[89]:


df['Os']=df['OpSys'].apply(cat_os)


# In[85]:


df.head()


# In[ ]:


df.drop(columns=df['OpSys'],inplace=True)


# In[87]:


df.head()


# In[88]:


sns.barplot(x=df['Os'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[91]:


sns.distplot(df['Weight'])


# In[92]:


sns.scatterplot(x=df['Weight'],y=df['Price'])


# In[93]:


df.corr()['Price']


# In[94]:


sns.heatmap(df.corr())


# In[95]:


#our target column is price here, our target column is skewed 
#SO we use log transformation
sns.distplot(np.log(df['Price']))


# In[96]:


x=df.drop(columns=['Price'])
y=np.log(df['Price'])


# In[97]:


x


# In[98]:


y


# In[99]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=2)


# In[100]:


x_train


# In[101]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error


# In[103]:


#import different model of machine learning
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR


# # Linear Regression

# In[107]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[108]:


df.head()


# # Ridge Regression

# In[110]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Lasso Regression

# In[112]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # KNN

# In[113]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Decision Tree

# In[114]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # SVM

# In[116]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Random Forest

# In[117]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Extra Tress

# In[118]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = ExtraTreesRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # AdaBoost

# In[120]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=15,learning_rate=1.0)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Gradient Boost
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
# # XGB Regressor

# In[124]:


pip install xgboost


# In[125]:


from xgboost import XGBRegressor
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Stacking

# In[126]:


from sklearn.ensemble import VotingRegressor,StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')


estimators = [
    ('rf', RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),
    ('gbdt',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),
    ('xgb', XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5))
]

step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Exporting the Model

# In[128]:


import pickle

pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[129]:


df


# In[ ]:




