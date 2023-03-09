#!/usr/bin/env python
# coding: utf-8

# # Data science blog using web API data from Kaggle

# ## 1. System Setup

# ### 1.1 Credits
# - Kaggle API for retrieving data https://www.kaggle.com/datasets/thedevastator/airbnb-prices-in-european-cities
# - Tutorial for nomalising data https://www.digitalocean.com/community/tutorials/normalize-data-in-python
# - Kaggle submission using the airbnb dataset https://www.kaggle.com/code/ibabarx/airbnb-prices-in-european-cities#notebook-container

# ### 1.2 Setup Packages

# In[95]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import os
#import kaggle
#import streamlit as st
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector


# ### 1.3 Setup color palette
# There is one palette that will be used throughout the case,
# PaletteC is a 10 color palette for all the 10 Cities,

# In[8]:


paletteC = ['#A6CEE3','#FF7F00', '#CAB2D6', '#33A02C', '#FDBF6F', '#E31A1C', '#6A3D9A', '#FB9A99', '#B2DF8A', '#1F78B4']
sns.color_palette(paletteC)


# In[9]:


sns.set_theme(style='whitegrid', palette=paletteC)


# ## 2. Import Data

# ### 2.1 Import and unpack from API

# In[132]:


# Importing databases
get_ipython().system('kaggle datasets download -d thedevastator/airbnb-prices-in-european-cities')


# Kaggle data is taken from the API in zip format. The following code block will unzip the datasets so they can be used in our project.

# In[133]:


get_ipython().system('unzip airbnb-prices-in-european-cities.zip -d Datasets')


# Move extracted CSV files into dataframes.

# In[12]:


amsterdam_weekdays = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/amsterdam_weekdays.csv')
amsterdam_weekends = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/amsterdam_weekends.csv')
athens_weekdays = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/athens_weekdays.csv')
athens_weekends = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/athens_weekends.csv')
barcelona_weekdays = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/barcelona_weekdays.csv')
barcelona_weekends = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/barcelona_weekends.csv')
berlin_weekdays = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/berlin_weekdays.csv')
berlin_weekends = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/berlin_weekends.csv')
budapest_weekdays = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/budapest_weekdays.csv')
budapest_weekends = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/budapest_weekends.csv')
lisbon_weekdays = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/lisbon_weekdays.csv')
lisbon_weekends = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/lisbon_weekends.csv')
london_weekdays = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/london_weekdays.csv')
london_weekends = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/london_weekends.csv')
paris_weekdays = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/paris_weekdays.csv')
paris_weekends = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/paris_weekends.csv')
rome_weekdays = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/rome_weekdays.csv')
rome_weekends = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/rome_weekends.csv')
vienna_weekdays = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/vienna_weekdays.csv')
vienna_weekends = pd.read_csv('/Users/laurenmeenhorst/Desktop/DataScienceBlogForHVA/Datasets/vienna_weekends.csv')


# In[13]:


def combine(csv_1, col_1, csv_2, col_2, city):
    csv_1['week_time'] = col_1
    csv_2['week_time'] = col_2
    merged = pd.concat([csv_1, csv_2])
    merged['city'] = city
    return merged


# #### Merge week and weekend data

# In[14]:


Amsterdam = combine(amsterdam_weekdays,'weekdays',amsterdam_weekends,'weekends','amsterdam')
Athens = combine(athens_weekdays,'weekdays',athens_weekends,'weekends','athens')
Barcelona = combine(barcelona_weekdays,'weekdays',barcelona_weekends,'weekends','barcelona')
Berlin = combine(berlin_weekdays,'weekdays',berlin_weekends,'weekends','berlin')
Budapest = combine(budapest_weekdays,'weekdays',budapest_weekends,'weekends','budapest')
Lisbon = combine(lisbon_weekdays,'weekdays',lisbon_weekends,'weekends','lisbon')
London = combine(london_weekdays,'weekdays',london_weekends,'weekends','london')
Paris = combine(paris_weekdays,'weekdays',paris_weekends,'weekends','paris')
Rome = combine(rome_weekdays,'weekdays',rome_weekends,'weekends','rome')
Vienna = combine(vienna_weekdays,'weekdays',vienna_weekends,'weekends','vienna')


# #### Merge Cities

# In[15]:


cities = [Amsterdam, Athens, Barcelona, Berlin, Budapest, Lisbon, London, Paris, Rome, Vienna]
city_names = [city.city.unique()[0].capitalize() for city in cities]
europe_data = pd.concat(cities, ignore_index=True)
europe_data.drop(columns = ['Unnamed: 0'], inplace=True)


# ### 2.2 Initial data inspection of European Airbnb data
# 1. realSum, The total price of the Airbnb listing. (Numeric)
# 2. room_type, The type of room being offered (e.g. private, shared, etc.). (Categorical)
# 3. room_shared, Whether the room is shared or not. (Boolean)
# 4. room_private, Whether the room is private or not. (Boolean)
# 5. person_capacity, The maximum number of people that can stay in the room. (Numeric)
# 6. host_is_superhost, Whether the host is a superhost or not. (Boolean)
# 7. multi, Whether the listing is for multiple rooms or not. (Boolean)
# 8. biz, Whether the listing is for business purposes or not. (Boolean)
# 9. cleanliness_rating, The cleanliness rating of the listing. (Numeric)
# 10. guest_satisfaction_overall, The overall guest satisfaction rating of the listing. (Numeric)
# 11. bedrooms, The number of bedrooms in the listing. (Numeric)
# 12. dist, The distance from the city centre. (Numeric)
# 13. metro_dist, The distance from the nearest metro station. (Numeric)
# 14. lng, The longitude of the listing. (Numeric)
# 15. lat, The latitude of the listing. (Numeric)

# In[16]:


# Take a look at the top rows
europe_data.head()


# In[17]:


# Take a look at a random sample
europe_data.sample(10)


# In[18]:


# Take a look at feature information
europe_data.info()


# In[19]:


europe_data.isnull().sum()


# We can see there are no empty values in our dataset.
# 
# The features attr_index, atrr_index_norm, rest_index and rest_index_norm are not included in the datasets' documentation. This makes these features hard to interoperate. Do to this we will not be including them in our project.

# In[21]:


europe_data.drop(columns = ['attr_index', 'attr_index_norm', "rest_index", 'rest_index_norm'], inplace=True)


# In[17]:


# Statistical description of our data
europe_data.describe(include='all')


# ### 2.3 Initial data cleaning - removing outliers
# Now we want to look for outliers, we expect the outliers to be in the realSum, as we saw with the .describe() that the max was 18545 and the min was 35

# In[22]:


fig, ax = plt.subplots(5,2, sharex=True, figsize=(15,30))
fig.suptitle('Boxplots of realSum & city')

for i, city in enumerate(europe_data['city'].unique()):
    row = i // 2
    col = i % 2
    sns.boxplot(ax=ax[row, col], data=europe_data[europe_data['city'] == city], y='realSum')
    ax[row, col].set_title(city)

plt.tight_layout()
plt.show()


# We determined that there are outliers in the realSum. Our decision was that an outlier is a value in realSum that is at least 1000 above the below outlier, by looking at the boxplots above we can determine the following values as non-outliers and add them to a new dataset.

# In[44]:


cities_2 = [Amsterdam[Amsterdam['realSum'] < 2000], Athens[Athens['realSum'] < 500], Barcelona[Barcelona['realSum'] < 1000], Berlin[Berlin['realSum'] < 800], Budapest[Budapest['realSum'] < 550], Lisbon[Lisbon['realSum'] < 650], London[London['realSum'] < 1500], Paris[Paris['realSum'] < 1200], Rome[Rome['realSum'] < 550], Vienna[Vienna['realSum'] < 750]]


# In[45]:


europe_data_2 = pd.concat(cities_2, ignore_index=True)


# In[46]:


europe_data_2.describe()


# Now we will take a look at the boxplots after the outliers have been removed:

# In[31]:


fig, ax = plt.subplots(5,2, sharex=True, figsize=(15,30))
fig.suptitle('Boxplots of realSum & city')

for i, city in enumerate(europe_data_2['city'].unique()):
    row = i // 2
    col = i % 2
    sns.boxplot(ax=ax[row, col], data=europe_data_2[europe_data_2['city'] == city], y='realSum')
    ax[row, col].set_title(city)

plt.tight_layout()
plt.show()


# ## 3. Exploratory Data analysis
# With the outliers removed we can look at the influence of the variables on the realSum.

# ### 3.1 Comparing the effect of time of week on prices

# In[34]:

plt.figure

# Create layout of our plots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 3))

# boxplot of realsum by week time
sns.boxplot(y='realSum', data=europe_data_2,x='week_time',ax = axs[0])
axs[0].tick_params(axis='y', labelsize=15)
axs[0].tick_params(axis='x', labelsize=15)

# layered hist of realsum by weektime
europe_data_2.groupby('week_time')['realSum'].plot(kind='hist', alpha=0.15, bins=15,ax=axs[1])

# kernel density estimate of realsum for weekdays and weekends
sns.kdeplot(data=europe_data_2[europe_data_2['week_time'] == 'weekdays']['realSum'], label='weekdays',ax=axs2)
sns.kdeplot(data=europe_data_2[europe_data_2['week_time'] == 'weekends']['realSum'], label='weekends',ax=axs2)
plt.subplots_adjust(hspace=0.65)
plt.show()


# In the previous plots we can see that week time had almost no influence on realsum.
# 
# We will split the figure by city in order to check wether the effect differs per city.

# In[35]:


# Rank cities
ranks = europe_data.groupby('city')['realSum'].mean().sort_values()[::-1].index

plt.figure(figsize=(15, 8))
ax = plt.subplot()
plt.axis([0,8,0,2000])

sns.boxplot(data=europe_data, x="city", y="realSum", hue="week_time",
            fliersize=0.5, linewidth=1, order=ranks)
plt.ylabel('Total price')
ax.set_xticklabels(ranks)
plt.legend(loc=1)
plt.show()
plt.clf()


# We can see that the effect stays minor.

# ### 3.2 Frequency distribution of numeric features

# In[47]:


# List all numerical features, ignore booleans
numerical_features = list(europe_data_2.select_dtypes(include=['int64','float64']).columns[i] for i in [2,5,6,7,8,9,10,11,12,13,14,15])


# In[48]:


# Define a plotter function, so we can plot all the features in one go
def plotter_numerical (feature, color, row):
    sns.histplot(data = europe_data_2[feature], ax=axes[row, 0], kde=True, color=color,line_kws={'color': 'Yellow'})
    axes[row,0].set_title(str(feature)+" Frequency (HISTPLOT)")
    axes[row,1].boxplot(Amsterdam[feature])
    axes[row,1].set_title(str(feature)+" Distribution (BOXPLOT)")

plt.figure
fig, axes = plt.subplots(nrows=12, ncols=2, figsize=(15, 50))
for i in range(12):
    plotter_numerical( numerical_features[i] , '#000000' , i)

plt.subplots_adjust(hspace=0.50)
plt.show()


# **Conclusions from the prev figure**
# - The people capacity in descending order of frequency is 2,4,3,6 and 5 for european Airbnb listings.
# - European Airbnb listings have a cleanliness rating overall. The distribution can be considered left skewed.
# - Overall customer satisfaction seems to follow the same pattern as cleanlines rating.
# - Most listings are between 0 and 7 km from the city centre.
# - Most listings are within 3 km of the nearest metro station

# ### 3.3 Scatterplots of numeric features including realsum trendline

# In[51]:


# Define a plotter function, so we can plot all the features in one go
def plotter_numerical_scatter (feature, color, x, y):
    axes[x,y].scatter(y=europe_data_2["realSum"], x=europe_data_2[feature],c=color)
    trend_line = np.poly1d(np.polyfit(europe_data_2[feature],europe_data_2["realSum"], 1))
    axes[x,y].plot(europe_data_2[feature], trend_line(europe_data_2[feature]), "r--")
    axes[x,y].set_ylabel("Price")
    axes[x,y].set_xlabel(feature)

# define the layout for our plots
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 17.5))
x = 0
y = 0
for i in range(12):
    plotter_numerical_scatter(numerical_features[i] , '#ADD8E6',x,y)
    y  = y + 1
    if y == 3:
        x = x + 1
        y = 0

plt.subplots_adjust(hspace=0.5)


# **Conclusions from the prev figure**
# - No bedrooms, person capacity,attraction index and restaurant index show a positive trend-line as their value increases.
# - Cleanliness rating and guest satisfaction seem to have a neutral trend-line which is unexpected.
# - Dist from centre and metro have a slight negative impact on price as they increase.

# ### 3.3 Analysis of categorical and binary features

# In[57]:


categorical_features = ['room_type','room_shared','room_private','host_is_superhost','multi','biz','week_time']


# In[59]:


# Define a plotter function, so we can plot all the features in one go
def plotter_categorical_bar_and_box (feature, color, row):
    axes[row,0].bar(x = list(europe_data_2[feature].value_counts().index), height=list(europe_data_2[feature].value_counts().values),color=color)
    axes[row,0].set_ylabel("Counts")
    axes[row,0].set_title(str(feature)+" COUNTS (BARPLOT)")

    sns.boxplot(data=europe_data_2,x = feature,y = 'realSum',ax=axes[row,1])
    axes[row,1].set_ylabel("Price")
    axes[row,1].set_title(str(feature)+" RELATION WITH REALSUM")


plt.figure
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 35))

for i in range(7):
    plotter_categorical_bar_and_box(categorical_features[i] , '#ADD8E6' , i)

plt.subplots_adjust(hspace=0.50)
plt.show()


# ## 4. Data Cleaning / Pre-processing

# In[60]:


europe_data_2.columns


# ### 4.1 Change boolean True/False to 1/0

# In[65]:


europe_data_2.replace({False: 0, True: 1},inplace=True)
europe_data_2.head()


# ### 4.2 Replace categorical vars with dummy variables
# We want to use our categorical values in our model so we will have to replace them with dummies.

# In[67]:


europe_data_2_categorical_dummies = pd.get_dummies(europe_data_2[['room_type','week_time','city']],drop_first=True)
europe_data_3 = pd.concat([europe_data_2_categorical_dummies, europe_data_2.drop(columns=['room_type','week_time', 'city'])], axis=1)


# In[69]:


europe_data_3.head()


# There are slight variances in median realSum of different cities in order to use this to train our model we have made a dummy for the different cities.

# ### 4.3 Check feature correlation

# In[74]:


plt.figure
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25, 15))
sns.heatmap(europe_data_3.corr(),cmap=sns.color_palette("Paired",20),annot=True,ax=axes)


# - Room shared and room private have a perfect correlation with room_type_shared and private which makes these features redundant.
# - rest_index_norm and attr_index_norm are normalised features. We also have the non normalised versions of these features so we don't need them right now.

# ### 4.4 Remove unused cols

# In[75]:


europe_data_3.drop(['Unnamed: 0', 'room_shared', 'room_private', 'rest_index_norm','attr_index_norm'], axis=1)


# ### 4.5 Normalising the features
# Data is commonly rescaled to fall between 0 and 1, because machine learning algorithms tend to perform better, or converge faster, when the different features are on a smaller scale. Before training machine learning models on data, it’s common practice to normalize the data first to potentially get better, faster results. Normalization also makes the training process less sensitive to the scale of the features, resulting in better coefficients after training.

# In[85]:


Standard_Scaler = StandardScaler()


# In[86]:


features_to_normalise= ['person_capacity',
                        'cleanliness_rating',
                        'guest_satisfaction_overall',
                        'bedrooms',
                        'dist',
                        'metro_dist',
                        'attr_index',
                        'rest_index',
                        'lng',
                        'lat']
features_not_to_normalise = ['room_type_Private room',
                         'room_type_Shared room',
                         'week_time_weekends',
                         'city_athens',
                         'city_barcelona',
                         'city_berlin',
                         'city_budapest',
                         'city_lisbon',
                         'city_london',
                         'city_paris',
                         'city_rome',
                         'city_vienna',
                         'realSum',
                         'host_is_superhost',
                         'multi',
                         'biz',]


# In[88]:


normalised_features = pd.DataFrame(Standard_Scaler.fit_transform(europe_data_3[features_to_normalise]), columns=features_to_normalise)
normalised_features.head()


# In[89]:


europe_data_final = pd.concat([normalised_features.reset_index(drop=True),  europe_data_3[features_not_to_normalise].reset_index(drop=True)], axis=1)
europe_data_final.head()


# ## 6. Building a model

# ### 6.1 create inputs and outputs

# In[91]:


X_train , X_test , Y_train , Y_test = \
    train_test_split(europe_data_final.drop(columns=['realSum']),
                     europe_data_final['realSum'],
                     random_state=4,
                     test_size=0.15,
                     stratify=europe_data_final[['week_time_weekends', 'city_athens','city_barcelona', 'city_berlin', 'city_budapest','city_lisbon', 'city_london', 'city_paris', 'city_rome', 'city_vienna']])


# ### 6.2 Sequential feature selection

# In[96]:


def sequential_feature_selection(model,X_train,Y_train,X_test):
    sfs = SequentialFeatureSelector(model,  direction='backward', scoring='r2', cv=5)
    sfs.fit(X_train, Y_train)
    X_train_selected = sfs.transform(X_train)
    X_test_selected = sfs.transform(X_test)
    return X_train_selected, X_test_selected


# ### 6.3 Linear regression

# In[107]:


LR = LinearRegression()


# In[108]:


X_train_selected, X_test_selected = sequential_feature_selection(LinearRegression(),X_train,Y_train,X_test)


# In[113]:


LR.fit(X_train_selected, Y_train)


# In[116]:


LR_TrainSet_Prediction = LR.predict(X_train_selected)
LR_TestSet_Prediction = LR.predict(X_test_selected)

train_score = LR.score(X_train_selected, Y_train)
test_score = LR.score(X_test_selected, Y_test)
print(train_score, test_score)


# In[121]:


plt.scatter(Y_test, LR_TestSet_Prediction, color='#FC814A', alpha=.2)
plt.axis([0,1500,-200,1000])
plt.ylabel('Predicted Airbnb prices')
plt.xlabel('Actual Airbnb prices')
plt.title('Predicted vs Actual Airbnb prices')
plt.show()
plt.close()


# In[124]:


# Take a look at a random sample
europe_data_final.sample(1)


# In[129]:


airbnb_apartment_paris = [[-0.88964, 0.639872, 0.044035, -0.241981, 0.292298, -0.630045, 0.316806, 0.097607, -0.522647, 0.60589,0,1]]
#438.997111,
predicted_airbnb_apartment_ams = LR.predict(airbnb_apartment_paris)
print('Test apartment = entire apartment, no room shared, private room, 2 person capacity, not a superhost, multiple rooms, not intended for business, cleanliness rating of 9.5, overal guest rating of 9.2, has one bedroom  is 500 meters from city center, 1km from metro, longitude of 5, latitude of 52, listing for weekdays, in Paris')
print('-------------------------------------------------------------------------')
print("Predicted realSum for appartment: €%.2f" % predicted_airbnb_apartment_ams)


# In[ ]:





#%%
