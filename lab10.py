import streamlit as st
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit UI Configuration
st.set_page_config(page_title="Big Data E-Commerce Analysis", layout="wide")
st.title("🛍️ Big Data Analysis for Women's Clothing E-Commerce")
st.markdown("""This app performs big data analysis, including data generation, cleaning, EDA, regression, clustering, and classification using PySpark and MLlib.""")

# Function to generate dataset
def generate_dataset():
    np.random.seed(42)
    data = {
        "Age": np.random.randint(18, 70, 23486),
        "Title": ["Great product" if i % 2 == 0 else "Not satisfied" for i in range(23486)],
        "Review Text": ["Loved it!" if i % 3 == 0 else "Could be better" for i in range(23486)],
        "Rating": np.random.randint(1, 6, 23486),
        "Recommended IND": np.random.choice([0, 1], 23486),
        "Positive Feedback Count": np.random.randint(0, 100, 23486),
        "Division Name": np.random.choice(["General", "Petite", "Tall"], 23486),
        "Department Name": np.random.choice(["Dresses", "Tops", "Bottoms"], 23486),
        "Class Name": np.random.choice(["Casual", "Formal", "Sportswear"], 23486),
    }
    return pd.DataFrame(data)

# Generate and display dataset
data = generate_dataset()
st.write("## 📊 Generated Dataset")
st.dataframe(data.head(10))

# Initialize PySpark session
spark = SparkSession.builder.appName("BigDataApp").getOrCreate()
spark_df = spark.createDataFrame(data)

# Handling Missing Values
spark_df = spark_df.fillna({"Title": "Unknown", "Review Text": "No Review"})
st.write("## 🔄 Handling Missing Values")
st.write(spark_df.toPandas().head())

# Exploratory Data Analysis
st.write("## 📈 Exploratory Data Analysis")
avg_age = spark_df.select(mean(col("Age"))).collect()[0][0]
common_rating = spark_df.groupBy("Rating").count().orderBy(col("count").desc()).first()[0]
st.write(f"**Average Age of Reviewers:** {avg_age:.2f}")
st.write(f"**Most Common Rating:** {common_rating}")

# Plot Rating Distribution
fig, ax = plt.subplots()
sns.countplot(data=data, x="Rating", palette="coolwarm", ax=ax)
ax.set_title("Distribution of Ratings")
st.pyplot(fig)

# Feature Engineering
vector_assembler = VectorAssembler(inputCols=["Age", "Rating", "Positive Feedback Count"], outputCol="features")
data_vectorized = vector_assembler.transform(spark_df)

# Regression Model
st.write("## 🔢 Regression Analysis")
lr = LinearRegression(featuresCol="features", labelCol="Positive Feedback Count")
model = lr.fit(data_vectorized)
st.write(f"**Regression Model Coefficients:** {model.coefficients}")
st.write(f"**Intercept:** {model.intercept}")

# Clustering
st.write("## 🔍 Clustering Analysis")
kmeans = KMeans(featuresCol="features", k=3)
model = kmeans.fit(data_vectorized)
st.write("### Cluster Centers:")
for i, center in enumerate(model.clusterCenters()):
    st.write(f"Cluster {i+1}: {center}")

# Classification
st.write("## ✅ Classification Analysis")
dt = DecisionTreeClassifier(featuresCol="features", labelCol="Recommended IND")
model = dt.fit(data_vectorized)
predictions = model.transform(data_vectorized)
evaluator = MulticlassClassificationEvaluator(labelCol="Recommended IND", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
st.write(f"**Classification Accuracy:** {accuracy:.2f}")

# Sidebar Navigation
st.sidebar.title("Navigation")
selected_section = st.sidebar.radio("Go to", ["Dataset", "EDA", "Regression", "Clustering", "Classification"])

if selected_section == "Dataset":
    st.write("## 📜 Dataset Overview")
    st.dataframe(data)
elif selected_section == "EDA":
    st.write("## 📊 Exploratory Data Analysis")
    st.pyplot(fig)
elif selected_section == "Regression":
    st.write("## 🔢 Regression Model")
    st.write(f"**Coefficients:** {model.coefficients}")
elif selected_section == "Clustering":
    st.write("## 🔍 Clustering Model")
    for i, center in enumerate(model.clusterCenters()):
        st.write(f"Cluster {i+1}: {center}")
elif selected_section == "Classification":
    st.write("## ✅ Classification Accuracy")
    st.write(f"**Accuracy:** {accuracy:.2f}")

st.success("🚀 Big Data Processing Completed!")
