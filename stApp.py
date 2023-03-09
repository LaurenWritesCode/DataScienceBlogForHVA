# import the necessary libraries
import streamlit as st  # for creating the web app
import pandas as pd  # for data manipulation
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px

# Setting the title of the tab and the favicon
st.set_page_config(page_title='European Airbnb pricing dashboard', page_icon=':house:', layout='wide')

# Adding a sidebar with an introduction to our dashboard
with st.sidebar.container():
    image = Image.open('airbnb.png')
    st.image(image, use_column_width=True)
    st.markdown("# Exploring Airbnb Data in 10 Popular European Cities")
    st.markdown("Welcome to our Streamlit dashboard, where we explore Airbnb data from 10 popular European cities. "
                "Airbnb has revolutionized the hospitality industry, "
                "providing travelers with a unique and affordable way to experience new destinations. ")
    st.markdown("Our analysis includes data from the cities of Amsterdam, Athens, Barcelona, Berlin, Budapest, "
                "Lisbon, London, Paris, Rome, and Vienna.")
    st.markdown("Our goal is to provide an interactive and informative platform for anyone interested in exploring "
                "Airbnb trends in Europe. Whether you're a traveler looking to plan your next trip or a data "
                "enthusiast interested in exploring trends.")

# Define palette
paletteC = ['#ff395b','#A6CEE3', '#CAB2D6', '#33A02C', '#FDBF6F', '#E31A1C', '#6A3D9A', '#FB9A99', '#B2DF8A', '#1F78B4']
sns.color_palette(paletteC)

# load the data
@st.cache_data
def load_data():
    europe_data_init = pd.read_csv('europe_data_init.csv')
    return europe_data_init

# load the data
europe_data_init = load_data()
europe_data_init.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

# Inserting image
image = Image.open('airbnb-pretty-1.jpeg')
st.image(image, use_column_width=True)

# Setting the title on the page with some styling
st.markdown(
    "<h1 style='text-align: center'>European Airbnb pricing dashboard</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>",
    unsafe_allow_html=True)

# Putting in personal details with some styling
st.markdown(
    "<body style='text-align: center'> <b>Created by HVA Data Science group 3</b></br><a href=https://github.com/LaurenWritesCode/DataScienceBlogForHVA>- Project repository on GitHub</a><hr style='height:2px;border-width:0;color:gray;background-color:gray'></body>",
    unsafe_allow_html=True)

st.header('Airbnb data by city')
# Select box for cities
input_city = st.selectbox("Select the city you would like to learn about", europe_data_init["city"].unique())
# Filter data based on selected city
selected_data = europe_data_init[europe_data_init["city"] == input_city]
# Select the columns you want to display
display_columns = ["realSum",
                   "room_type",
                   "room_shared",
                   "room_private",
                   "person_capacity",
                   "multi",
                   "biz",
                   "cleanliness_rating",
                   "guest_satisfaction_overall",
                   "bedrooms",
                   "dist",
                   "metro_dist",
                   "city"]
# Display selected columns of the filtered dataframe
st.dataframe(selected_data[display_columns])
st.subheader('The features')
st.markdown('1. realSum, The total price of the Airbnb listing. (Numeric)\n'
            '2. room_type, The type of room being offered (e.g. private, shared, etc.). (Categorical)\n'
            '3. room_shared, Whether the room is shared or not. (Boolean)\n'
            '4. room_private, Whether the room is private or not. (Boolean)\n'
            '5. person_capacity, The maximum number of people that can stay in the room. (Numeric)\n'
            '6. host_is_superhost, Whether the host is a superhost or not. (Boolean)\n'
            '7. multi, Whether the listing is for multiple rooms or not. (Boolean)\n'
            '8. biz, Whether the listing is for business purposes or not. (Boolean)\n'
            '9. cleanliness_rating, The cleanliness rating of the listing. (Numeric)\n'
            '10. guest_satisfaction_overall, The overall guest satisfaction rating of the listing. (Numeric)\n'
            '11. bedrooms, The number of bedrooms in the listing. (Numeric)\n'
            '12. dist, The distance from the city centre. (Numeric)\n'
            '13. metro_dist, The distance from the nearest metro station. (Numeric)\n')

# Add a horizontal line
st.markdown("<hr style='height:2px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)
# Plots to show realsum difference for weekends/weekdays
st.header('Comparing the effect of time of week on prices')
# Rank cities
ranks = europe_data_init.groupby('city')['realSum'].mean().sort_values()[::-1].index

# Create plot
fig, ax = plt.subplots(figsize=(15, 8))
plt.axis([0, 8, 0, 2000])

sns.boxplot(data=europe_data_init, x="city", y="realSum", hue="week_time",
            fliersize=0.5, linewidth=1, order=ranks, palette=paletteC)
plt.ylabel('Total price')
ax.set_xticklabels(ranks)
plt.legend(loc=1)

# Show plot
st.pyplot(fig)

# Select city to display
selected_city = st.selectbox("Select a city to view more specific visuals", europe_data_init['city'].unique())
if not selected_city:
    st.warning("Please select a city.")
else:
    # Filter data by selected city
    europe_data_for_plot = europe_data_init[europe_data_init['city'] == selected_city]

    # Create layout of our plots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
    fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 3))

    # Boxplot of realSum by week time
    sns.boxplot(y='realSum', data=europe_data_for_plot, x='week_time', ax=axs[0], palette=paletteC)
    axs[0].tick_params(axis='y', labelsize=15)
    axs[0].tick_params(axis='x', labelsize=15)

    # Layered hist of realSum by weektime
    colors = [paletteC[0], paletteC[1]]  # choose colors for each week_time
    europe_data_for_plot.groupby('week_time')['realSum'].plot(kind='hist', alpha=0.5, bins=15, ax=axs[1], color=colors)
    axs[1].legend()

    # Kernel density estimate of realSum for weekdays and weekends
    sns.kdeplot(data=europe_data_for_plot[europe_data_for_plot['week_time'] == 'weekdays']['realSum'], label='weekdays', ax=axs2, color='#ff395b')
    sns.kdeplot(data=europe_data_for_plot[europe_data_for_plot['week_time'] == 'weekends']['realSum'], label='weekends', ax=axs2, color='#A6CEE3')
    axs2.legend()
    plt.subplots_adjust(hspace=0.65)

    # Show plots
    st.pyplot(fig)
    st.pyplot(fig2)

st.markdown("In the previous plots we can see that week time had almost no influence on realsum.")
st.markdown("<hr style='height:2px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)

# Frequency distribution of numeric features
st.header('Frequency distribution of numeric features')
# List all numerical features, ignore booleans
numerical_features = ['person_capacity',
                      'cleanliness_rating',
                      'guest_satisfaction_overall',
                      'bedrooms',
                      'dist',
                      'metro_dist',
                      'attr_index',
                      'attr_index_norm',
                      'rest_index',
                      'rest_index_norm',
                      'lng',
                      'lat']

# Define a plotter function, so we can plot all the features in one go
def plotter_numerical(feature, color, row):
    sns.histplot(data=europe_data_init[feature], ax=axes[row, 0], kde=True, color=color, line_kws={'color': 'Yellow'}, palette=paletteC)
    axes[row, 0].set_title(f"{feature} Frequency (HISTPLOT)")
    # Create a transparent boxplot with a blue mean line
    sns.boxplot(data=europe_data_init, x=feature, ax=axes[row, 1], color="white", palette=paletteC)
    sns.despine(ax=axes[row, 1], left=True)
    axes[row, 1].set_title(f"{feature} Distribution (BOXPLOT)")

# Create the plot
fig, axes = plt.subplots(nrows=12, ncols=2, figsize=(15, 45))
for i, feature in enumerate(numerical_features):
    plotter_numerical(feature, '#000000', i)

plt.subplots_adjust(hspace=0.50)
# Display the plot in Streamlit
st.pyplot(fig)

conclusions = """
<div style='background-color: #fff2f4; padding: 20px; border-radius: 10px;'>
    <h3 style='text-align: center;'>Conclusions from the figure:</h3>
    <p style='text-align: justify;'>Based on the data, European Airbnb listings have the following descending order of people capacity: 2, 4, 3, 6, and 5. This suggests that most listings are suitable for couples or small groups of travelers.</p>
    <p style='text-align: justify;'>The distribution of cleanliness ratings for European Airbnb listings is left-skewed, indicating that most listings have high cleanliness ratings. This is a positive indication of the quality of the accommodations offered by Airbnb hosts.</p>
    <p style='text-align: justify;'>Customer satisfaction also appears to follow a similar pattern as the cleanliness rating. This suggests that cleanliness is an important factor in determining the overall satisfaction of Airbnb guests.</p>
    <p style='text-align: justify;'>The majority of listings are located within a range of 0 to 7 kilometers from the city center. This indicates that Airbnb hosts are aware of the importance of proximity to city centers in attracting travelers.</p>
    <p style='text-align: justify;'>Most listings are also situated within 3 kilometers of the nearest metro station. This suggests that Airbnb hosts are aware of the convenience of public transportation for travelers and are taking this factor into consideration when choosing the location of their listings.</p>
</div>
"""

st.markdown(conclusions, unsafe_allow_html=True)

# Scatterplot showing effect of numerical features on realsum
st.header('Scatterplot showing effect of numerical features on realsum')
# Select box for numerical features
InputDotPlotFeature = st.selectbox("Select the feature you want to see", (numerical_features))
# Checkbox for showing the trendline
show_trendline = st.checkbox("Show trendline", value=True)
# Set plot title
fig_title = f"Scatterplot of {InputDotPlotFeature} vs. Real Sum"
# Create a scatterplot
fig, ax = plt.subplots()
ax.scatter(europe_data_init[InputDotPlotFeature], europe_data_init["realSum"], color = "#A6CEE3")
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

st.markdown("<div style='background-color: #fff2f4; padding: 20px; border-radius: 10px;'>"
            "<h3 style='text-align: center;'>Conclusions from the figure:</h3>"
            "<p style='text-align: justify;'>No bedrooms, person capacity, attraction index, and restaurant index show a positive trend-line as their value increases.</p>"
            "<p style='text-align: justify;'>Cleanliness rating and guest satisfaction seem to have a neutral trend-line which is unexpected.</p>"
            "<p style='text-align: justify;'>Distance from centre and metro have a slight negative impact on price as they increase.</p>"
            "<p style='text-align: justify;'>These observations suggest that certain features may have a stronger correlation with price than others. For example, larger bedrooms and higher person capacity may lead to higher prices, while being further from the city centre or metro station may lead to lower prices.</p>"
            "<p style='text-align: justify;'>It's also interesting that cleanliness rating and guest satisfaction don't seem to have a strong correlation with price, which could suggest that other factors, such as location or amenities, may be more important to customers.</p>"
            "<p style='text-align: justify;'>Overall, these insights can help inform our understanding of the Airbnb market in Europe and guide future analyses or decision-making.</p>"
            "</div>", unsafe_allow_html=True)

st.markdown("<hr style='height:2px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)

# Analysis of categorical and binary features
st.header('Analysis of categorical and binary features')

categorical_features = ['room_type','room_shared','room_private','host_is_superhost','multi','biz','week_time']
# Define a plotter function, so we can plot all the features in one go
def plotter_categorical_bar_and_box(feature, color, row):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
    sns.barplot(x=list(europe_data_init[feature].value_counts().index), y=list(europe_data_init[feature].value_counts().values), color=color, ax=axes[0])
    axes[0].set_ylabel("Counts")
    axes[0].set_title(str(feature) + " COUNTS (BARPLOT)")
    sns.boxplot(data=europe_data_init, x=feature, y='realSum', ax=axes[1], palette=paletteC)
    axes[1].set_ylabel("Price")
    axes[1].set_title(str(feature) + " RELATION WITH REALSUM")
    st.pyplot(fig)

# Create a figure with multiple subplots
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 35))

# Iterate over each of the categorical features and plot them using the plotter function
for i in range(7):
    plotter_categorical_bar_and_box(categorical_features[i], paletteC[i], i)

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.50)

st.markdown("<hr style='height:2px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)

# Convert realSum column to integer type
europe_data_init['realSum'] = europe_data_init['realSum'].astype(int)

# Write page title and description
st.write("""
# Europe Data Cost Map
This map shows the cost of Airbnb rentals in European cities, with colors ranging from green (cheap) to red (expensive).
The longitude and latitude of each city are used to get the exact Airbnb data. The map includes 
a slider to select either weekdays or weekends and another slider to filter by price.
""")

# Add a slider to filter by realSum
realSum_range = st.slider("Select a price range",
                          int(europe_data_init["realSum"].min()),
                          int(europe_data_init["realSum"].max()),
                          (int(europe_data_init["realSum"].min()),
                           int(europe_data_init["realSum"].max())))
filtered_data = europe_data_init[(europe_data_init["realSum"] >= realSum_range[0]) & (europe_data_init["realSum"] <= realSum_range[1])]

# Define the map plot
fig = px.scatter_mapbox(filtered_data, lat="lat", lon="lng", hover_name="city",
                        hover_data=["room_type", "guest_satisfaction_overall"],
                        size='realSum', color='realSum',
                        color_continuous_scale= ['#5F8D4E', '#E0FF4F', '#EC9708', '#ED7014', '#F0544F', '#8b0000'],
                        range_color=[realSum_range[0], realSum_range[1]],
                        #animation_frame="week_time_flt", zoom=3, height=500
                        )

fig.update_layout(mapbox_style="open-street-map", margin= {"r":0,"t":0,"l":0,"b":0}, mapbox_center = {'lat':52, 'lon':5})
fig["layout"].pop("updatemenus")

# Display the map
st.plotly_chart(fig, use_container_width=True)

#%%
