# import the necessary libraries
import streamlit as st  # for creating the web app
import pandas as pd  # for data manipulation
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
#import geopandas as gpd

# Setting the title of the tab and the favicon
st.set_page_config(page_title='European Airbnb pricing dashboard', page_icon=':house:', layout='centered')

# load the data
@st.cache_data
def load_data():
    europe_data_init = pd.read_csv('europe_data_init.csv')
    return europe_data_init

# load the data
europe_data_init = load_data()
europe_data_init.drop(['Unnamed: 0'], axis=1)

# Inserting image
image = Image.open('airbnb.png')
st.image(image, use_column_width=True)

# Setting the title on the page with some styling
st.markdown(
    "<h1 style='text-align: center'>European Airbnb pricing dashboard</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>",
    unsafe_allow_html=True)

# Putting in personal details with some styling
st.markdown(
    "<body style='text-align: center'> <b>Created by HVA Data Science group 3</b></br><a href=https://github.com/LaurenWritesCode/DataScienceBlogForHVA>- Project repository on GitHub</a><hr style='height:2px;border-width:0;color:gray;background-color:gray'></body>",
    unsafe_allow_html=True)

# Project information
st.header('Introduction')
st.markdown('Write some text')
st.markdown('Write some text')
st.markdown("<hr style='height:2px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)

# create the sidebar
st.sidebar.header('User Input Features')
# Select box for cities
InputCity = st.sidebar.selectbox("Select your city", (europe_data_init["city"].unique()))
# Select box for room types
InputRoomtype = st.sidebar.selectbox("Select your room type", (europe_data_init["room_type"].unique()))

# Select based on input val
Select = europe_data_init[europe_data_init["city"] == InputCity]
st.dataframe(Select)


# Select city to display
selected_city = st.selectbox("Select a city", europe_data_init['city'].unique())

if not selected_city:
    st.warning("Please select a city.")
else:
    # Filter data by selected city
    europe_data_for_plot = europe_data_init[europe_data_init['city'] == selected_city]

    # Create layout of our plots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
    fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 3))

    # Boxplot of realSum by week time
    sns.boxplot(y='realSum', data=europe_data_for_plot, x='week_time', ax=axs[0])
    axs[0].tick_params(axis='y', labelsize=15)
    axs[0].tick_params(axis='x', labelsize=15)

    # Layered hist of realSum by weektime
    europe_data_for_plot.groupby('week_time')['realSum'].plot(kind='hist', alpha=0.15, bins=15, ax=axs[1])

    # Kernel density estimate of realSum for weekdays and weekends
    sns.kdeplot(data=europe_data_for_plot[europe_data_for_plot['week_time'] == 'weekdays']['realSum'], label='weekdays', ax=axs2)
    sns.kdeplot(data=europe_data_for_plot[europe_data_for_plot['week_time'] == 'weekends']['realSum'], label='weekends', ax=axs2)
    plt.subplots_adjust(hspace=0.65)

    # Show plots
    st.pyplot(fig)
    st.pyplot(fig2)



#st.sidebar.markdown("### Scatter Chart: Explore Relationship Between Measurements :")

# List all numerical features, ignore booleans
numerical_features = list(europe_data_init.select_dtypes(include=['int64','float64']).columns[i] for i in [2,5,6,7,8,9,10,11,12,13,14,15])
# Select box for numerical features
InputDotPlotFeature = st.sidebar.selectbox("Select the feature you want to see", (numerical_features))
# Checkbox for showing the trendline
show_trendline = st.sidebar.checkbox("Show trendline", value=True)
# Set plot title
fig_title = f"Scatterplot of {InputDotPlotFeature} vs. Real Sum"
# Create a scatterplot
fig, ax = plt.subplots()
ax.scatter(europe_data_init[InputDotPlotFeature], europe_data_init["realSum"])
# Add a trendline if the checkbox is selected
if show_trendline:
    z = np.poly1d(np.polyfit(europe_data_init[InputDotPlotFeature], europe_data_init["realSum"], 1))
    ax.plot(europe_data_init[InputDotPlotFeature], z(europe_data_init[InputDotPlotFeature]), "r--")
# Add axis labels and a title
ax.set_xlabel(InputDotPlotFeature)
ax.set_ylabel('Real Sum')
ax.set_title(fig_title)
# Display the plot
st.pyplot(fig)

# Create slider for realSum
min_val = int(europe_data_init['realSum'].min())
max_val = int(europe_data_init['realSum'].max())
realSum_val = st.slider("Select realSum", min_val, max_val, min_val, step=100)

# Filter data based on realSum value
filtered_data = europe_data_init[europe_data_init['realSum'] == realSum_val]

# Add week_time labels
filtered_data['week_time_flt'] = filtered_data['week_time'].map({0: 'Weekdays', 1: 'Weekends'})

# Create map
fig = px.scatter_mapbox(filtered_data, lat="lat", lon="lng", hover_name="city",
                        hover_data=["room_type", "guest_satisfaction_overall"],
                        size='realSum', color='realSum',
                        color_continuous_scale= ['#5F8D4E', '#E0FF4F', '#EC9708', '#ED7014', '#F0544F', '#8b0000'],
                        range_color=[min(europe_data_init['realSum']), max(europe_data_init['realSum'])],
                        animation_frame="week_time_flt", zoom=3, height=500)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig["layout"].pop("updatemenus")

# Show map
st.plotly_chart(fig)

st.write("""
# Europe Data Cost Map
This is a map of the cost of airbnb's in the European city. The longitude and latitude of each city are taken to get the exact AirBnB data, where the color value (realSum) colors from green (cheap), to red (expensive). 
The resulting map shows the distribution of the 'realSum' column for each city and has a slider for each week time (weekend or weekday).
To view the map for a specific week time, move the slider to 'weekdays' or 'weekends'.""")

