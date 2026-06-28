import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv("cleaned_data.csv")

print(df.head())
print(df.columns)
df = pd.get_dummies(
    df,
    columns=[
        "Category",
        "Region",
        "Segment"
    ],
    drop_first=True
)
print(df.head())
print(df.columns)
X = df[
    [
        "Quantity",
        "Discount",
        "Profit",
        "Category_Office Supplies",
        "Category_Technology",
        "Region_East",
        "Region_South",
        "Region_West",
        "Segment_Corporate",
        "Segment_Home Office"
    ]
]

y = df["Sales"]

y = df["Sales"]
print(X.head())

print(y.head())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
model = LinearRegression()

model.fit(X_train, y_train)

print("AI Model Trained Successfully!")
predictions = model.predict(X_test)

print(predictions[:10])
mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R² Score:", r2)
results = pd.DataFrame({
    "Actual Sales": y_test,
    "Predicted Sales": predictions
})

print(results.head(10))