import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Page Configuration
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
st.title("🛍️ Mall Customer Clustering App")

# 1. File Upload
uploaded_file = st.sidebar.file_uploader("Upload Mall_Customers.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview", df.head())
    
    # Feature selection (Annual Income & Spending Score are standard for this dataset)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Sidebar Options
    clustering_method = st.sidebar.selectbox("Choose Clustering Method", ["K-Means", "Hierarchical"])
    n_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 5)

    # 2. K-Means Clustering
    if clustering_method == "K-Means":
        st.subheader("K-Means Clustering Analysis")
        
        # Elbow Method for Visualization
        if st.checkbox("Show Elbow Method"):
            wcss = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i, init='k-means++', random_state=42)
                km.fit(X)
                wcss.append(km.inertia_)
            fig_elbow, ax = plt.subplots()
            ax.plot(range(1, 11), wcss, marker='o')
            ax.set_title('The Elbow Method')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('WCSS')
            st.pyplot(fig_elbow)

        # Model Building
        model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        y_labels = model.fit_predict(X)
        df['Cluster'] = y_labels

    # 3. Hierarchical Clustering
    else:
        st.subheader("Hierarchical (Agglomerative) Clustering")
        
        # Dendrogram Visualization
        if st.checkbox("Show Dendrogram"):
            fig_dendro, ax = plt.subplots(figsize=(10, 5))
            linkage_matrix = linkage(X, method='ward')
            dendrogram(linkage_matrix, ax=ax)
            ax.set_title('Customer Dendrogram')
            st.pyplot(fig_dendro)

        # Model Building
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        y_labels = model.fit_predict(X)
        df['Cluster'] = y_labels

    # 4. Final Visualization
    st.write(f"### Resulting Clusters (n={n_clusters})")
    fig_clusters, ax = plt.subplots()
    sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], hue=df['Cluster'], palette='viridis', s=100, ax=ax)
    ax.set_title(f'Clusters using {clustering_method}')
    st.pyplot(fig_clusters)
    
    # Show Segmented Data
    st.write("### Segmented Customers", df)
else:
    st.info("Please upload the 'Mall_Customers.csv' file to begin.")
