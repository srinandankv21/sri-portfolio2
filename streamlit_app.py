import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz

# Set the style for plots
plt.style.use('seaborn')

# Streamlit App Title and Description
st.title('Energy Consumption Clustering for Houses using Fuzzy C-Means')

st.write("""
This app allows you to cluster houses based on their **Max** and **Min Energy Consumption per Day** using Fuzzy C-means clustering. 
You can adjust the number of houses and clusters using the sidebar.
The energy data is scaled between 0 and 10.
""")

# --- Sidebar Inputs ---
st.sidebar.header("User Input Parameters")

# Slider to select the number of houses
num_houses = st.sidebar.slider("Select the number of houses", min_value=50, max_value=1000, value=300, step=50)

# Slider to select the number of clusters
num_clusters = st.sidebar.slider("Select the number of clusters", min_value=2, max_value=10, value=3)

# Set random seed for reproducibility
np.random.seed(42)

# --- Generate Sample Data ---
# Max and Min Energy Consumed per Day (kWh)
max_energy_consumed = np.random.randint(50, 500, size=num_houses)  # Max energy between 50 to 500 kWh
min_energy_consumed = np.random.randint(50, 500, size=num_houses)  # Min energy between 50 to 500 kWh

# Total Energy Consumed for the Month (assuming 30 days in a month)
total_energy_consumed = np.random.randint(3000, 15000, size=num_houses)  # Total monthly energy

# Create a DataFrame
df = pd.DataFrame({
    'Max Energy Consumed per Day (kWh)': max_energy_consumed,
    'Min Energy Consumed per Day (kWh)': min_energy_consumed,
    'Total Energy Consumed for the Month (kWh)': total_energy_consumed
})

# Display the first few rows of the data
st.subheader('Sample Data')
st.write(df.head())

# --- Data Scaling ---
st.subheader('Scaling Data between 0 and 10')
# Scaling the 'Max' and 'Min' energy consumed between 0 and 10
scaler = MinMaxScaler(feature_range=(0, 10))
scaled_data = scaler.fit_transform(df[['Max Energy Consumed per Day (kWh)', 'Min Energy Consumed per Day (kWh)']])

# Convert the scaled data back into a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=['Max Energy Consumed per Day (Scaled)', 'Min Energy Consumed per Day (Scaled)'])
st.write(scaled_df.head())

# --- Fuzzy C-means Clustering ---
st.subheader(f'Fuzzy C-means Clustering with {num_clusters} Clusters')

# Apply Fuzzy C-Means (FCM)
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    scaled_data.T, num_clusters, 2, error=0.005, maxiter=1000, init=None)

# Assign each data point to the cluster with the highest membership value
cluster_labels = np.argmax(u, axis=0)
df['Cluster'] = cluster_labels

# Get the centroids (cluster centers)
centroids = cntr

# --- Visualization ---
st.subheader('Visualizations')

# 1. Scatter Plot of Max and Min Energy Consumption Before Clustering
st.write("### Max and Min Energy Consumption (Before Clustering, Scaled)")
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(scaled_df['Max Energy Consumed per Day (Scaled)'], 
            scaled_df['Min Energy Consumed per Day (Scaled)'], 
            color='grey', alpha=0.6)
ax1.set_title('Max and Min Energy Consumption (Before Clustering, Scaled)')
ax1.set_xlabel('Max Energy Consumed per Day (Scaled)')
ax1.set_ylabel('Min Energy Consumed per Day (Scaled)')
st.pyplot(fig1)

# 2. Scatter Plot of Clusters with Different Colors
st.write(f"### Fuzzy C-means Clustering of Max and Min Energy Consumption (Clusters = {num_clusters})")
fig2, ax2 = plt.subplots(figsize=(8, 6))
colors = plt.cm.get_cmap('tab10', num_clusters)

for cluster in range(num_clusters):
    clustered_data = scaled_df[df['Cluster'] == cluster]
    ax2.scatter(clustered_data['Max Energy Consumed per Day (Scaled)'], 
                clustered_data['Min Energy Consumed per Day (Scaled)'], 
                label=f'Cluster {cluster + 1}', alpha=0.6)
    
# Plot centroids
ax2.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')
ax2.set_title(f'Fuzzy C-means Clustering of Max and Min Energy Consumption (Clusters = {num_clusters})')
ax2.set_xlabel('Max Energy Consumed per Day (Scaled)')
ax2.set_ylabel('Min Energy Consumed per Day (Scaled)')
ax2.legend()
st.pyplot(fig2)

# --- Membership Levels Plot ---
st.subheader('Membership Levels for Each Cluster')

# Plot membership levels for the first 20 data points (to avoid overcrowding)
st.write("### Membership Levels for the First 20 Data Points")
fig3, ax3 = plt.subplots(figsize=(10, 6))
for i in range(num_clusters):
    ax3.plot(u[i, :20], label=f'Membership for Cluster {i + 1}', marker='o')

ax3.set_title('Membership Levels for the First 20 Data Points')
ax3.set_xlabel('Data Point Index')
ax3.set_ylabel('Membership Level')
ax3.legend()
st.pyplot(fig3)

# Optional: Display final data with clusters
st.subheader('Clustered Data')
st.write(df.head())

# Optional: Download clustered data
@st.cache
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df)

st.download_button(
    label="Download Clustered Data as CSV",
    data=csv,
    file_name=f'fuzzy_clustered_energy_consumption_{num_houses}_houses.csv',
    mime='text/csv',
)
