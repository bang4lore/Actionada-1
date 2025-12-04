import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv("daily_transactions.csv")
df = df.dropna(subset=["Category"])
df["Merchant"] = df["Merchant"].fillna("UNKNOWN")
df["Date"] = pd.to_datetime(df["Date"])
df["day_of_week"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

y = df["Category"]
X = df[["Merchant", "Transaction_Amount", "Transaction_Type", "day_of_week", "month", "is_weekend"]]

preprocessor = ColumnTransformer([
    ("text", CountVectorizer(), "Merchant"),
    ("num", StandardScaler(), ["Transaction_Amount"]),
    ("cat", OneHotEncoder(), ["Transaction_Type", "day_of_week", "month", "is_weekend"])
])

clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
