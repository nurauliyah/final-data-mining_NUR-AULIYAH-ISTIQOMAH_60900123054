import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("Customer Segmentation Clustering App")
st.write("Upload dataset CSV kamu untuk melakukan K-Means Clustering.")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview Dataset")
    st.dataframe(df.head())

    # Pilih jumlah cluster
    k = st.slider("Pilih Jumlah Cluster (k)", min_value=2, max_value=10, value=3)

    # Normalisasi data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    # K-Means
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(scaled)
    df["Cluster"] = labels

    st.write("### Dataset dengan Label Cluster")
    st.dataframe(df)

    # Plot cluster
    st.write("### Visualisasi Cluster")
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df["Annual_Income"],
        df["Spending_Score"],
        c=df["Cluster"]
    )
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    st.pyplot(fig)

    # Download CSV hasil clustering
    st.write("### Download Dataset Hasil Clustering")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="clustered_customer_segmentation.csv",
        mime="text/csv"
    )
else:
    st.info("Silakan upload file CSV terlebih dahulu.")
