import streamlit as st
st.set_page_config(page_title="Airbnb NYC Dashboard", layout="wide")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
import statsmodels.api as sm
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv("AB_NYC_2019.csv")

    df = df.dropna(subset=['name', 'host_name']).copy()
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['last_review'] = df['last_review'].fillna("No Reviews")
    df['price_per_night'] = df['price'] / df['minimum_nights']
    df = df[df['price_per_night'] < 1000]

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
    room_map = {rtype: idx for idx, rtype in enumerate(df['room_type'].unique())}
    df['room_type_encoded'] = df['room_type'].map(room_map)
    df['availability_rate'] = df['availability_365'] / 365

    times_sq = (40.7580, -73.9855)
    central_park = (40.785091, -73.968285)
    dumbo = (40.7033, -73.9881)

    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    df['dist_to_times_square'] = haversine_distance(df['latitude'], df['longitude'], *times_sq)
    df['dist_to_central_park'] = haversine_distance(df['latitude'], df['longitude'], *central_park)
    df['dist_to_dumbo'] = haversine_distance(df['latitude'], df['longitude'], *dumbo)

    return df

df = load_data()

st.title("🏙️ Airbnb Price Trends - NYC")
st.markdown("Explore pricing, availability, and listing patterns across New York City.")

st.sidebar.header("🔎 Filter Listings")
room_options = df['room_type'].unique()
selected_room = st.sidebar.selectbox("Room Type", room_options)
min_price, max_price = st.sidebar.slider("Price Range ($)", 0, 1000, (50, 300))

filtered_df = df[
    (df['room_type'] == selected_room) &
    (df['price'] >= min_price) &
    (df['price'] <= max_price)
]

st.subheader("📍 Map of Filtered Listings")
st.map(filtered_df[['latitude', 'longitude']])

st.subheader("💸 Price Distribution by Location")
fig_scatter = px.scatter(
    filtered_df, x="longitude", y="latitude", color="price",
    hover_data=["neighbourhood", "price", "minimum_nights"],
    color_continuous_scale="Plasma", title="Price by Geo-Location"
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("🏘️ Room Type Distribution")
room_counts = df['room_type'].value_counts()
st.bar_chart(room_counts)

st.subheader("📆 Availability Rate by Room Type")
fig_hist = px.histogram(df, x="availability_rate", nbins=20, color="room_type",
                        title="Availability Rate Distribution")
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("📊 Summary Statistics")
st.dataframe(filtered_df.describe(include='all'))

st.subheader("📍 Price vs. Distance to NYC Landmarks (Combined View)")
filtered_price = df[df['price'] < 1000]

def lowess_line(x, y):
    lowess = sm.nonparametric.lowess(y, x, frac=0.3)
    return lowess[:, 0], lowess[:, 1]

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Distance to Times Square", "Distance to Central Park", "Distance to DUMBO"),
    shared_yaxes=True, horizontal_spacing=0.07
)

x1, y1 = lowess_line(filtered_price['dist_to_times_square'], filtered_price['price'])
fig.add_trace(go.Scatter(x=filtered_price['dist_to_times_square'], y=filtered_price['price'],
                         mode='markers', marker=dict(size=3, opacity=0.5, color='blue'), name="Times Sq"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', line=dict(color='red'), name="Trendline"),
              row=1, col=1)

x2, y2 = lowess_line(filtered_price['dist_to_central_park'], filtered_price['price'])
fig.add_trace(go.Scatter(x=filtered_price['dist_to_central_park'], y=filtered_price['price'],
                         mode='markers', marker=dict(size=3, opacity=0.5, color='green'), name="Central Park"),
              row=1, col=2)
fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines', line=dict(color='black'), name="Trendline"),
              row=1, col=2)

x3, y3 = lowess_line(filtered_price['dist_to_dumbo'], filtered_price['price'])
fig.add_trace(go.Scatter(x=filtered_price['dist_to_dumbo'], y=filtered_price['price'],
                         mode='markers', marker=dict(size=3, opacity=0.5, color='orange'), name="DUMBO"),
              row=1, col=3)
fig.add_trace(go.Scatter(x=x3, y=y3, mode='lines', line=dict(color='blue'), name="Trendline"),
              row=1, col=3)

fig.update_layout(
    height=500, width=1100,
    title_text="💸 Price vs. Distance to NYC Landmarks",
    showlegend=False
)

fig.update_xaxes(title_text="Distance (km)", row=1, col=1)
fig.update_xaxes(title_text="Distance (km)", row=1, col=2)
fig.update_xaxes(title_text="Distance (km)", row=1, col=3)
fig.update_yaxes(title_text="Price (per night $)", row=1, col=1)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📌 Select a Landmark to View Price vs. Distance")
landmark = st.selectbox("Choose a landmark", ["Times Square", "Central Park", "DUMBO"])

if landmark == "Times Square":
    fig_landmark = px.scatter(
        df[df['price'] < 1000],
        x="dist_to_times_square", y="price",
        trendline="lowess", opacity=0.5,
        labels={"dist_to_times_square": "Distance to Times Square (km)", "price": "Price ($)"},
        title="Price vs. Distance to Times Square"
    )
elif landmark == "Central Park":
    fig_landmark = px.scatter(
        df[df['price'] < 1000],
        x="dist_to_central_park", y="price",
        trendline="lowess", opacity=0.5,
        labels={"dist_to_central_park": "Distance to Central Park (km)", "price": "Price ($)"},
        title="Price vs. Distance to Central Park"
    )
else:
    fig_landmark = px.scatter(
        df[df['price'] < 1000],
        x="dist_to_dumbo", y="price",
        trendline="lowess", opacity=0.5,
        labels={"dist_to_dumbo": "Distance to DUMBO (km)", "price": "Price ($)"},
        title="Price vs. Distance to DUMBO"
    )

st.plotly_chart(fig_landmark, use_container_width=True)
