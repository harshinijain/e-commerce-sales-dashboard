import pandas as pd

df = pd.read_csv("dataset/SampleSuperstore.csv")

print(df.head())
print(df.isnull().sum())
df = df.dropna()
df = df.drop_duplicates()
df.to_csv("cleaned_data.csv")
print(df["Sales"].sum())
print(df["Profit"].sum())
print(
df.groupby("City")["Sales"]
.sum()
.sort_values(ascending=False)
.head(10)
)
import matplotlib.pyplot as plt

df.groupby("Category")["Sales"].sum().plot(kind="bar")

plt.title("Sales by Category")
plt.ylabel("Sales")
plt.show()
df.groupby("Category")["Profit"].sum().plot(kind="bar")

plt.title("Profit by Category")
plt.ylabel("Profit")
plt.show()
df.groupby("Region")["Sales"].sum().plot(kind="bar")

plt.title("Sales by Region")
plt.ylabel("Sales")
plt.show()
df.groupby("Region")["Profit"].sum().plot(kind="bar")

plt.title("Profit by Region")
plt.ylabel("Profit")
plt.show()