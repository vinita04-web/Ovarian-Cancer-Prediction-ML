import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel("data/Supplementary data 3.xlsx")

X = df.drop("TYPE", axis=1)
y = df["TYPE"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print(importance.head(10))