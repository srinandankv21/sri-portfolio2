import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz

# Set the style for plots
#plt.style.use('seaborn')

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
num_houses = st.sidebar.slider("Select the number of houses", min_value=50, max_value=30000, value=300, step=50)

# Slider to select the number of clusters
num_clusters = st.sidebar.slider("Select the number of clusters", min_value=2, max_value=10, value=3)

# Set random seed for reproducibility
np.random.seed(42)

# --- Generate Sample Data ---
# Max and Min Energy Consumed per Day (kWh)
max_energy_consumed = np.random.randint(50, 500, size=num_houses)  # Max energy between 50 to 500 kWh
min_energy_consumed = np.random.randint(50, 500, size=num_houses)  # Min energy between 50 to 500 kWh

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
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    scaled_data.T, num_clusters, 2, error=0.005, maxiter=1000)

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

# Add membership levels to the DataFrame
for i in range(num_clusters):
    df[f'membership_cluster_{i}'] = u[i, :]

# Plot membership levels for the first cluster using Seaborn
#st.write(f"### Membership Levels for Cluster 0")
#fig3, ax3 = plt.subplots(figsize=(8, 6))
#sns.scatterplot(x='Max Energy Consumed per Day (kWh)', y='Min Energy Consumed per Day (kWh)', 
 #               hue='membership_cluster_0', data=df, palette='coolwarm')
#ax3.set_title('Membership Levels for Cluster 0')
#ax3.set_xlabel('Max Energy Consumed per Day (kWh)')
#ax3.set_ylabel('Min Energy Consumed per Day (kWh)')
#st.pyplot(fig3)

# Plot membership levels for other clusters
for i in range(num_clusters):
    st.write(f"### Membership Levels for Cluster {i}")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Max Energy Consumed per Day (kWh)', y='Min Energy Consumed per Day (kWh)', 
                    hue=f'membership_cluster_{i}', data=df, palette='coolwarm')
    ax.set_title(f'Membership Levels for Cluster {i}')
    ax.set_xlabel('Max Energy Consumed per Day (kWh)')
    ax.set_ylabel('Min Energy Consumed per Day (kWh)')
    st.pyplot(fig)

# Display final data with clusters
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

# --- Additional Visualizations ---
st.subheader('Additional Visualizations')

# 1. Histogram of Max Energy Consumed per Day
st.write("### Histogram of Max Energy Consumed per Day")
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.histplot(df['Max Energy Consumed per Day (kWh)'], bins=30, kde=True, ax=ax1)
ax1.set_title('Distribution of Max Energy Consumed per Day')
ax1.set_xlabel('Max Energy Consumed per Day (kWh)')
ax1.set_ylabel('Frequency')
st.pyplot(fig1)

# 2. Histogram of Min Energy Consumed per Day
st.write("### Histogram of Min Energy Consumed per Day")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.histplot(df['Min Energy Consumed per Day (kWh)'], bins=30, kde=True, ax=ax2)
ax2.set_title('Distribution of Min Energy Consumed per Day')
ax2.set_xlabel('Min Energy Consumed per Day (kWh)')
ax2.set_ylabel('Frequency')
st.pyplot(fig2)

# 3. Bar Chart of Average Energy Consumption per Month by Cluster
st.write("### Bar Chart of Average Total Energy Consumed per Month by Cluster")
# Assume 'Cluster' column is available and cluster labels are generated
df['Cluster'] = np.random.randint(0, 3, size=len(df))  # Random clusters for demonstration
average_monthly_energy_by_cluster = df.groupby('Cluster')['Total Energy Consumed for the Month (kWh)'].mean().reset_index()

fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.barplot(x='Cluster', y='Total Energy Consumed for the Month (kWh)', data=average_monthly_energy_by_cluster, ax=ax3)
ax3.set_title('Average Total Energy Consumed per Month by Cluster')
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Average Total Energy Consumed for the Month (kWh)')
st.pyplot(fig3)

# 4. Pair Plot for Max and Min Energy Consumed
st.write("### Pair Plot for Max and Min Energy Consumed")
fig4 = sns.pairplot(df[['Max Energy Consumed per Day (kWh)', 'Min Energy Consumed per Day (kWh)']])
st.pyplot(fig4)

