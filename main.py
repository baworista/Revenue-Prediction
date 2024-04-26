from pathlib import Path
import urllib.request
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    '''Save images to the folder with high resolution'''
    #images_path = f"images/{fig_id}.{fig_extension}"
    images_path = Path("images")/f"{fig_id}.{fig_extension}" #Кросс-платформа
    Path("images").mkdir(parents=True, exist_ok=True)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(images_path, format=fig_extension, dpi=resolution)


def load_file_from_git(file_name: str): #  file: str - type hint
    '''Load training and test datasets. If it's needed, create datasets folder''' #black & isort; isort - imports; black - all code
    tarball_path = Path("datasets")/f"{file_name}"
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/baworista/DataRepository/main/{file_name}"
        urllib.request.urlretrieve(url, tarball_path)
    return pd.read_csv(Path(f"datasets/{file_name}"))


def save_plot_acc(plotname: str, y_train_prepared, y_predict): #http://bokeh.org/ instead matplot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train_prepared, y_train_prepared, color='blue', label='Actual')
    plt.scatter(y_train_prepared, y_predict, color='red', label='Predicted', alpha=0.1)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.legend()
    save_fig(plotname)


raw_train = load_file_from_git("existing_customers.csv")
print(raw_train.head(10))
print(raw_train.info())
raw_test = load_file_from_git("new_customers.csv")
print(raw_test.head(10))
print(raw_test.info())

# Delete unused columns from training dataset and show info about data
train = raw_train.drop(['customer_id', 'join_ip'], axis="columns") #"columns" instead of 1
test = raw_test.drop(['customer_id', 'join_ip'], axis="columns") #"columns" instead of 1
pd.set_option('display.max_columns', None)

# print("Train set")
# print(train.shape)
# print(train.info())
# print(train.head())
#
# print("\nTest set")
# print(train.shape)
# print(test.head())


# Show different data info
country_counts = train['age'].value_counts()
top_20_countries = country_counts.head(20)
plt.figure(figsize=(14, 8))
top_20_countries.plot(kind='bar', color='red')
plt.title('Ilość wystąpień wieku')
plt.xlabel('Wiek')
plt.ylabel('Ilość wystąpień')
plt.xticks(rotation=90)
save_fig("Ilość_wystąpień_wieku")

country_counts = train['join_ip_country'].value_counts()
top_20_countries = country_counts.head(20)
plt.figure(figsize=(14, 8))
top_20_countries.plot(kind='bar', color='orange')
plt.title('Ilość wystąpień krajów')
plt.xlabel('Kraj')
plt.ylabel('Ilość wystąpień')
plt.xticks(rotation=90)
save_fig("Ilość_wystąpień_krajów")

revenue_by_country = train.groupby('join_ip_country')['revenue'].sum().sort_values(ascending=False)
top_20_countries_revenue = revenue_by_country.head(20)
plt.figure(figsize=(14, 8))
top_20_countries_revenue.plot(kind='bar', color='skyblue')
plt.title('Suma przychodów w poszczególnych krajach')
plt.xlabel('Kraj')
plt.ylabel('Suma przychodów')
plt.xticks(rotation=90)
save_fig("Suma_przychodów_w_poszczególnych_krajach")


# Correlation matrix and its plot
train['gender'] = train['gender'].map({'M': 0, 'F': 1})
train['join_campain'] = train['join_campain'].astype(float)

test['gender'] = test['gender'].map({'M': 0, 'F': 1})
test['join_campain'] = test['join_campain'].astype(float)

train_corr = train.drop(['join_ip_country'], axis=1)

corr_matrix = train_corr.corr()
print("Correlation matrix to revenue")
print(corr_matrix['revenue'].sort_values(ascending=False))

numeric_columns = train_corr.select_dtypes(include='number').columns.tolist()
scatter_matrix(train_corr[numeric_columns], figsize=(12, 12))
save_fig("Macierz korelacji")


# Preprocessing

# X_train = train.iloc[:, :-1]
# y_train = train.iloc[:, -1:]

# new_order = ['age', 'gender', 'join_campain', 'price_first_item_purchased', 'join_ip_country', 'join_pages_visited', 'join_campain', 'join_GDP_cap', 'revenue']
# train = train.reindex(columns=new_order)

print(train)

y_train = train['revenue']
X_train = train.drop(['revenue'], axis=1)

y_train = pd.DataFrame(y_train)
X_train = pd.DataFrame(X_train)


# X_test = test.iloc[:, :-1]
# y_test = test.iloc[:, -1:]

non_numeric_columns = X_train.select_dtypes(exclude='number').columns.tolist()
print("non_numeric_columns: ")
print(non_numeric_columns)
numeric_columns = X_train.select_dtypes(include='number').columns.tolist()
print("numeric_columns: ")
print(numeric_columns)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

non_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('non_num', non_numeric_transformer, non_numeric_columns),
    ])

X_train_prepared = preprocessor.fit_transform(X_train)
imputer = SimpleImputer(strategy='median')
y_train_prepared = imputer.fit_transform(y_train)
y_train_prepared = np.ravel(y_train_prepared)

# non_numeric_columns = X_test.select_dtypes(exclude='number').columns.tolist()
# print("non_numeric_columns: ")
# print(non_numeric_columns)
# numeric_columns = X_test.select_dtypes(include='number').columns.tolist()
# print("numeric_columns: ")
# print(numeric_columns)

# X_test_prepared = preprocessor.fit_transform(X_test)
# y_test_prepared = imputer.fit_transform(y_test)
# y_test_prepared = np.ravel(y_test_prepared)

# LinearRegression Model
model = LinearRegression()

model.fit(X_train_prepared, y_train_prepared)

y_predict = model.predict(X_train_prepared)
save_plot_acc("Linear_Regression_Actual_Predicted", y_train_prepared, y_predict)

print("Linear regressor RMSE: ", root_mean_squared_error(y_train_prepared, y_predict))
print("Linear regressor MAPE: ", mean_absolute_percentage_error(y_train_prepared, y_predict))


# y_predict = model.predict(X_test_prepared)
# save_plot_acc("Linear_Regression_Actual_Predicted", y_test_prepared, y_predict)
#
# print("Linear regressor RMSE: ", root_mean_squared_error(y_test_prepared, y_predict))
# print("Linear regressor MAPE: ", mean_absolute_percentage_error(y_test_prepared, y_predict))



# Get coefficients
coefficients = model.coef_

# Calculate importance (absolute value of coefficients)
feature_importance = np.abs(coefficients)

# Print or visualize feature importances
for i, importance in enumerate(feature_importance):
    print(f"Feature {i}: Importance = {importance}")


# RandomForest Model
param_grid = {
    'n_estimators': [130],
    'max_depth': [20]
}

grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train_prepared, y_train_prepared)
print("Najlepsze parametry:", grid_search_rf.best_params_)
model = grid_search_rf.best_estimator_

y_predict = model.predict(X_train_prepared)

save_plot_acc("Random_Forest_Actual_Predicted", y_train_prepared, y_predict)
print("Random forest regressor RMSE: ", root_mean_squared_error(y_train_prepared, y_predict))
print("Random forest regressor MAPE: ", mean_absolute_percentage_error(y_train_prepared, y_predict))


# Support vector regression model
X_train_small = X_train_prepared[:1000]
y_train_small = y_train_prepared[:1000]

model =  SVR(kernel="poly", gamma=5, epsilon=5, C=5)

model.fit(X_train_small, y_train_small)

y_predict = model.predict(X_train_small)

save_plot_acc("SVR_Actual_Predicted", y_train_small, y_predict)
print("SVR regressor RMSE: ", root_mean_squared_error(y_train_small, y_predict))
print("SVR regressor MAPE: ", mean_absolute_percentage_error(y_train_small, y_predict))


# Multi-Layer Perceptron(Worse than LinearRegression and RandomForest)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_prepared, y_train_prepared)

print(X_train_prepared.shape)

norm_layer = tf.keras.layers.Normalization()


model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(25, activation="relu"),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)
norm_layer.adapt(X_valid)

history = model.fit(X_train_prepared, y_train_prepared, epochs=80,
                    validation_data=(X_valid, y_valid))

#Show keras learning curves plot
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[15, 80], ylim=[0, 10000], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc="lower left")
save_fig("keras_learning_curves_plot")

y_predict = model.predict(X_train_prepared)
save_plot_acc("MLP_Actual_Predicted", y_train_prepared, y_predict)
print("SVR regressor MLP: ", root_mean_squared_error(y_train_prepared, y_predict))
print("SVR regressor MLP: ", mean_absolute_percentage_error(y_train_prepared, y_predict))


