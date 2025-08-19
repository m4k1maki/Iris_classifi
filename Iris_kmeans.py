# kmeans_iris.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=columns)

print("Kích thước dữ liệu:", df.shape)
print(df.head())
print(df.describe())

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('equal')
labels = ['Setosa', 'Versicolor', 'Virginica']
sizes = [50, 50, 50]
ax.pie(sizes, labels=labels, autopct='%1.2f%%')
plt.title("Phân bố dữ liệu Iris")
plt.show()

df.hist(figsize=(8, 6))
plt.suptitle("Biểu đồ Histogram các thuộc tính Iris")
plt.show()

sns.pairplot(df, hue="class")
plt.show()

X = df.drop("class", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker="o", linestyle="--")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Số cụm (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", max_iter=300, n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nGán nhãn cụm:")
print(df.head())

# Đánh giá bằng Silhouette Score
score = silhouette_score(X_scaled, df["Cluster"])
print(f"Silhouette Score: {score:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="Cluster", palette="viridis", s=100, alpha=0.7)
plt.title("Clusters of Iris Data")
plt.show()
