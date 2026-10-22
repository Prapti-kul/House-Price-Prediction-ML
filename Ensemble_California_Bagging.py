
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score

#-------------------------------------
# Step 1 : Load the data
#-------------------------------------

df = pd.read_csv("california_housing.csv")
print("Shape of Dataset : ", df.shape)
print("First few records : ", df.head())

#-------------------------------------
# Step 2 : Separate Features and Labels
#-------------------------------------

X = df.drop("target", axis=1)
Y = df["target"]

print("Shape of X : ", X.shape)
print("Shape of Y : ", Y.shape)

#-------------------------------------
# Step 3 : Split dataset for training and testing
#-------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#-------------------------------------
# Step 4.1 : Create the base model
#-------------------------------------

base_model = DecisionTreeRegressor(random_state=42)

#-------------------------------------
# Step 4.2 : Create the Bagging model
#-------------------------------------

model = BaggingRegressor(
    estimator=base_model,
    n_estimators=10,
    random_state=42
)

#-------------------------------------
# Step 5 : Train the model
#-------------------------------------

model = model.fit(X_train, Y_train)

#-------------------------------------
# Step 6 : Test the model
#-------------------------------------

Y_pred = model.predict(X_test)

#-------------------------------------
# Step 7 : Evaluate the model
#-------------------------------------

print("MSE : ", mean_squared_error(Y_test, Y_pred))
print("R2 : ", r2_score(Y_test, Y_pred))

#-------------------------------------
# Step 8 : Save the trained Bagging model
#-------------------------------------

joblib.dump(model, "bagging_model.pkl")

#-------------------------------------
# Step 9 : Save Bagging model results
#-------------------------------------

results = pd.DataFrame({
    "Model": ["Bagging Regressor"],
    "MSE": [mean_squared_error(Y_test, Y_pred)],
    "R2": [r2_score(Y_test, Y_pred)]
})

results.to_csv("bagging_results.csv", index=False)

predictions = pd.DataFrame({
    "Actual": Y_test.values,
    "Predicted": Y_pred
})

predictions.to_csv("bagging_predictions.csv", index=False)

print("Bagging model and results saved successfully!")
