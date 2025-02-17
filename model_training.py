# model_training.py
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


data = pd.read_csv('data/merged_data.csv')


X = data.drop(['price', 'date', 'item'], axis=1)
y = data['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


dt_model = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
dt_model.fit(X_train, y_train)


knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

print("Decision Tree:")
evaluate(dt_model, X_test, y_test)

print("\nKNN:")
evaluate(knn_model, X_test, y_test)


joblib.dump(dt_model, 'models/grocery_model.joblib')