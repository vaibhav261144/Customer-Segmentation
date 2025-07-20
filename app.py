import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime as dt

st.set_page_config(page_title="Customer RFM Segmentation", layout="wide")
st.title("📊 Customer Segmentation using RFM & KMeans Clustering")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("online retail.csv", encoding='ISO-8859-1')
    df.dropna(subset=['Invoice', 'Customer ID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Total'] = df['Quantity'] * df['Price']
    return df

df = load_data()

# Sidebar filters
min_date = df['InvoiceDate'].min().date()
max_date = df['InvoiceDate'].max().date()
st.sidebar.date_input("📅 Filter by date range", [min_date, max_date], min_value=min_date, max_value=max_date)

# RFM Calculation
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'Invoice': 'nunique',
    'Total': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm = rfm[rfm['Monetary'] > 0]

# Scale
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Label Segments
labels = {
    0: 'At Risk',
    1: 'Hibernating',
    2: 'Champions',
    3: 'Loyal Customers'
}
rfm['Segment'] = rfm['Cluster'].map(labels)

# Segment overview
st.subheader("📋 Segment Summary")
st.dataframe(rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2).sort_values('Monetary', ascending=False))

# Distribution
st.subheader("📈 Segment Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(data=rfm, x='Segment', order=rfm['Segment'].value_counts().index, palette='Set2')
ax1.set_ylabel("Customer Count")
st.pyplot(fig1)

# Heatmap
st.subheader("🔥 RFM Heatmap by Segment")
rfm_heat = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
fig2, ax2 = plt.subplots()
sns.heatmap(rfm_heat.T, annot=True, cmap="YlGnBu")
st.pyplot(fig2)

# Download segmented data
st.subheader("⬇️ Download Segmented Data")
csv = rfm.reset_index().to_csv(index=False)
st.download_button("Download as CSV", csv, "rfm_segments.csv", "text/csv")
