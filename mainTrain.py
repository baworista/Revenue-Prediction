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
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    '''Save images to the folder with high resolution'''
    # images_path = Path("images")/f"{fig_id}.{fig_extension}"
    # Path("images").mkdir(parents=True, exist_ok=True)
    # if tight_layout:
    #     plt.tight_layout()
    # plt.savefig(images_path, format=fig_extension, dpi=resolution)


def load_file_from_git(file_name: str): #  file: str - type hint
    '''Load training and test datasets. If it's needed, create datasets folder''' #black & isort; isort - imports; black - all code
    tarball_path = Path("datasets")/f"{file_name}"
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/baworista/DataRepository/main/{file_name}"
        urllib.request.urlretrieve(url, tarball_path)
    return pd.read_csv(Path(f"datasets/{file_name}"))


def plot_prediction_analysis(y_true, y_pred, fig_id="prediction_analysis"):
    """
    Generate plots to analyze the predictions made by models.

    Parameters:
        y_true (array-like): Array of true target values.
        y_pred (array-like): Array of predicted target values.
        fig_id (str): Identifier for saving the figures.
    """
    # Scatter Plot of Predicted vs. Actual Values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Scatter Plot of Predicted vs. Actual Values")
    plt.grid(True)
    save_fig(fig_id + "_scatter_plot_actual_vs_predicted")

    # Residual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_true - y_pred, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.axhline(y=0, color='r', linestyle='-')
    plt.grid(True)
    save_fig(fig_id + "_residual_plot")

    # Distribution of Residuals
    plt.figure(figsize=(10, 6))
    residuals = y_true - y_pred
    plt.hist(residuals, bins=30, alpha=0.5)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    plt.grid(True)
    save_fig(fig_id + "_distribution_of_residuals")

    plt.show()

raw_train = load_file_from_git("existing_customers.csv")
print("Raw training data loaded")
print(raw_train.head(2))
print(raw_train.info())

raw_test = load_file_from_git("new_customers.csv")
print("Raw test data loaded")
print(raw_test.head(2))
print(raw_test.info())

# Delete unused columns from training dataset and show info about data
train = raw_train.drop(['customer_id', 'join_ip'], axis="columns") #"columns" instead of 1
test = raw_test.drop(['customer_id', 'join_ip'], axis="columns") #"columns" instead of 1
pd.set_option('display.max_columns', None)

print("Train set")
print(train.shape)

print("\nTest set")
print(test.shape)


# # Show different data info
# age_counts = train['age'].value_counts()
# top_20_countries = age_counts.head(20)
# plt.figure(figsize=(14, 8))
# top_20_countries.plot(kind='bar', color='red')
# plt.title('Number of age occurrences')
# plt.xlabel('Age')
# plt.ylabel('Number of occurrences')
# plt.xticks(rotation=90)
# save_fig("Number_of_age_occurrences")
#
# country_counts = train['join_ip_country'].value_counts()
# top_20_countries = country_counts.head(20)
# plt.figure(figsize=(14, 8))
# top_20_countries.plot(kind='bar', color='orange')
# plt.title('Number of country occurrences')
# plt.xlabel('Country')
# plt.ylabel('Number of occurrences')
# plt.xticks(rotation=90)
# save_fig("Number_of_country_occurrences")
#
# revenue_by_country = train.groupby('join_ip_country')['revenue'].sum().sort_values(ascending=False)
# top_20_countries_revenue = revenue_by_country.head(20)
# plt.figure(figsize=(14, 8))
# top_20_countries_revenue.plot(kind='bar', color='skyblue')
# plt.title('Total revenues in individual countries')
# plt.xlabel('Country')
# plt.ylabel('Total revenue')
# plt.xticks(rotation=90)
# save_fig("Total_revenues_in_individual_countries")


# Change bool type to 1 or 0 and
train['gender'] = train['gender'].map({'M': 0, 'F': 1})
train['join_campain'] = train['join_campain'].astype(float)

test['gender'] = test['gender'].map({'M': 0, 'F': 1})
test['join_campain'] = test['join_campain'].astype(float)


# Correlation matrix and its plot
train_corr = train.drop(['join_ip_country'], axis=1)

corr_matrix = train_corr.corr()
print("Correlation matrix to revenue")
print(corr_matrix['revenue'].sort_values(ascending=False))

numeric_columns = train_corr.select_dtypes(include='number').columns.tolist()
scatter_matrix(train_corr[numeric_columns], figsize=(12, 12))
save_fig("Correlation matrix")


# Preprocessing
# Separating data to use it in models
y_train = train['revenue']
X_train = train.drop(['revenue'], axis=1)

y_train = pd.DataFrame(y_train)
X_train = pd.DataFrame(X_train)

print("Separated data print")
print(y_train.info)
print(X_train.info)

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
y_train_prepared = numeric_transformer.fit_transform(y_train)

# print(X_train_prepared[:5])
# print(y_train_prepared[:5])

# Flattering to make from 2D to 1D
y_train_prepared = np.ravel(y_train_prepared)

# Split data into training and test sets
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train_prepared, y_train_prepared, test_size=0.2, random_state=42)

print("Prepared data shapes")
print(X_train_split.shape)
print(X_test_split.shape)
print(y_train_split.shape)
print(y_test_split.shape)


# LinearRegression Model
regressionModel = LinearRegression()

regressionModel.fit(X_train_prepared, y_train_prepared)

y_predict = regressionModel.predict(X_train_prepared)

plot_prediction_analysis(y_train_prepared, y_predict, "Linear_Regression_Actual_Predicted")
print("Linear regressor RMSE: ", root_mean_squared_error(y_train_prepared, y_predict))
print("Linear regressor MAPE: ", mean_absolute_percentage_error(y_train_prepared, y_predict))


# # Get coefficients
# coefficients = model.coef_
#
# # Calculate importance (absolute value of coefficients)
# feature_importance = np.abs(coefficients)
#
# # Print or visualize feature importances
# for i, importance in enumerate(feature_importance):
#     print(f"Feature {i}: Importance = {importance}")


# RandomForest Model
param_grid = {
    'n_estimators': [130],
    'max_depth': [20]
}

grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train_prepared, y_train_prepared)
print("Najlepsze parametry:", grid_search_rf.best_params_)
RFmodel = grid_search_rf.best_estimator_

y_predict = RFmodel.predict(X_train_prepared)

plot_prediction_analysis(y_train_prepared, y_predict, "Random_Forest_Actual_Predicted")
print("Random forest regressor RMSE: ", root_mean_squared_error(y_train_prepared, y_predict))
print("Random forest regressor MAPE: ", mean_absolute_percentage_error(y_train_prepared, y_predict))


# Support vector regression model
X_train_small = X_train_prepared[:3000]
y_train_small = y_train_prepared[:3000]

SVRmodel =  SVR(kernel="poly", gamma=5, epsilon=5, C=5)

SVRmodel.fit(X_train_small, y_train_small)

y_predict = SVRmodel.predict(X_train_small)

plot_prediction_analysis(y_train_small, y_predict, "SVR_Actual_Predicted")
print("SVR regressor RMSE: ", root_mean_squared_error(y_train_small, y_predict))
print("SVR regressor MAPE: ", mean_absolute_percentage_error(y_train_small, y_predict))


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and memory growth is set.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")


# Multi-Layer Perceptron(Worse than LinearRegression and RandomForest)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_prepared, y_train_prepared)

norm_layer = tf.keras.layers.Normalization()


MLPmodel = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(25, activation="relu"),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
MLPmodel.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)
norm_layer.adapt(X_valid)

history = MLPmodel.fit(X_train_prepared, y_train_prepared, epochs=30,
                    validation_data=(X_valid, y_valid))

#Show keras learning curves plot
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[15, 80], ylim=[0, 10000], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc="lower left")
save_fig("keras_learning_curves_plot")

y_predict = MLPmodel.predict(X_train_prepared)
print(y_train_prepared.shape)
y_predict = np.ravel(y_predict)
print(y_predict.shape)
plot_prediction_analysis(y_train_prepared, y_predict, "MLP_Actual_Predicted")
print("SVR regressor MLP: ", root_mean_squared_error(y_train_prepared, y_predict))
print("SVR regressor MLP: ", mean_absolute_percentage_error(y_train_prepared, y_predict))


# Create a VotingRegressor with the individual regressors
named_estimators = [
    ("LinearRegression", regressionModel),
    ("RandomForest", RFmodel),
    ("SVR", SVRmodel),
    ("MLP", MLPmodel),
]

voting_reg = VotingRegressor(named_estimators)

# Fit the VotingRegressor to the training data
voting_reg.fit(y_train_prepared, y_train_prepared)

# Predict using the VotingRegressor
y_pred_voting = voting_reg.predict(y_train_prepared)

# Evaluate the VotingRegressor
print("Voting Regressor RMSE: ", root_mean_squared_error(y_test_split, y_pred_voting))
print("Voting Regressor MAPE: ", mean_absolute_percentage_error(y_test_split, y_pred_voting))

# Plot prediction analysis for VotingRegressor
plot_prediction_analysis(y_train_prepared, y_pred_voting, "Voting_Regressor")






# # LinearRegression Model
#
# regressionModel = LinearRegression()
#
# regressionModel.fit(X_train_split, y_train_split)
#
# y_predict = regressionModel.predict(X_train_split)
# print("train Linear regressor RMSE: ", root_mean_squared_error(y_train_split, y_predict))
# print("train Linear regressor MAPE: ", mean_absolute_percentage_error(y_train_split, y_predict))
#
# y_test_predict = regressionModel.predict(X_test_split)
# print("test Linear regressor RMSE: ", root_mean_squared_error(y_test_split, y_test_predict))
# print("test Linear regressor MAPE: ", mean_absolute_percentage_error(y_test_split, y_test_predict))
#
# plot_prediction_analysis(y_test_split, y_test_predict, "Linear_Regression_Actual_Predicted")

# # Get coefficients
# coefficients = RFmodel.coef_
#
# # Calculate importance (absolute value of coefficients)
# feature_importance = np.abs(coefficients)
#
# # Print or visualize feature importances
# for i, importance in enumerate(feature_importance):
#     print(f"Feature {i}: Importance = {importance}")


# # Multi-Layer Perceptron(Worse than LinearRegression and RandomForest)
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X_train_prepared, y_train_prepared)
#
# norm_layer = tf.keras.layers.Normalization()
#
#
# MLPmodel = tf.keras.Sequential([
#     norm_layer,
#     tf.keras.layers.Dense(100, activation="relu"),
#     tf.keras.layers.Dense(50, activation="relu"),
#     tf.keras.layers.Dense(25, activation="relu"),
#     tf.keras.layers.Dense(1)
# ])
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# MLPmodel.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
# norm_layer.adapt(X_train)
# norm_layer.adapt(X_valid)
#
# history = MLPmodel.fit(X_train_prepared, y_train_prepared, epochs=2,
#                     validation_data=(X_valid, y_valid))
#
# #Show keras learning curves plot
# pd.DataFrame(history.history).plot(
#     figsize=(8, 5), xlim=[15, 80], ylim=[0, 10000], grid=True, xlabel="Epoch",
#     style=["r--", "r--.", "b-", "b-*"])
# plt.legend(loc="lower left")
# save_fig("keras_learning_curves_plot")
#
# y_predict = MLPmodel.predict(X_train_prepared)
# print(y_train_prepared.shape)
# y_predict = np.ravel(y_predict)
# print(y_predict.shape)
#
#
# plot_prediction_analysis(y_train_prepared, y_predict, "MLP_Actual_Predicted")
# print("SVR regressor MLP: ", root_mean_squared_error(y_train_prepared, y_predict))
# print("SVR regressor MLP: ", mean_absolute_percentage_error(y_train_prepared, y_predict))
