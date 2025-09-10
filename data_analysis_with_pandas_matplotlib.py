# =====================================================
# Analyzing Data with Pandas and Visualizing with Matplotlib
# Author: Robert Kibugi
# Date: September 2025
# =====================================================

# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Enable inline plotting if using Jupyter
# %matplotlib inline

# =====================================================
# Task 1: Load and Explore the Dataset
# =====================================================

try:
    # Load Iris dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame  # Convert to pandas DataFrame
    
    print("Dataset loaded successfully.\n")
except FileNotFoundError as e:
    print("Error: Dataset file not found.", e)
except Exception as e:
    print("Unexpected error:", e)

# Display first few rows
print("First five rows of the dataset:")
print(df.head())

# Inspect dataset structure
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# No missing values in Iris dataset, but as best practice:
df = df.dropna()  # Would drop missing values if any

# =====================================================
# Task 2: Basic Data Analysis
# =====================================================

# Compute basic statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Grouping by species and computing mean of numerical columns
grouped_means = df.groupby("target").mean()
print("\nMean values per species (0=setosa, 1=versicolor, 2=virginica):")
print(grouped_means)

# Mapping target codes to species names for clarity
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Observation example
print("\nObservation: Virginica generally has the largest petal length and width compared to other species.")

# =====================================================
# Task 3: Data Visualization
# =====================================================

sns.set(style="whitegrid")  # Better plot style

# ---- 1. Line Chart (simulating trend with cumulative sum of sepal length) ----
plt.figure(figsize=(8,5))
df["sepal length (cm)"].cumsum().plot(kind="line")
plt.title("Cumulative Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Cumulative Sepal Length (cm)")
plt.show()

# ---- 2. Bar Chart: Average Petal Length per Species ----
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal length (cm)", data=df, ci=None, palette="muted")
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# ---- 3. Histogram: Distribution of Sepal Width ----
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# ---- 4. Scatter Plot: Sepal Length vs Petal Length ----
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="deep")
plt.title("Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# =====================================================
# Findings & Observations
# =====================================================

print("\nKey Findings:")
print("- Setosa has distinctly smaller petal length and width compared to other species.")
print("- Versicolor and Virginica overlap in sepal size, but Virginica tends to have larger petals overall.")
print("- The histogram shows sepal width distribution is roughly normal, centered around ~3 cm.")
print("- Scatter plot confirms species can be separated visually by petal length and sepal length.")