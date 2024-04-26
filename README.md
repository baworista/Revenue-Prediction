Overview:
Performing data loading, preprocessing, model training, and evaluation for a revenue prediction task using machine learning models such as Linear Regression, 
Random Forest Regressor, Support Vector Regression (SVR), and a Multi-Layer Perceptron (MLP) using TensorFlow.

Key Libraries Used:
pathlib: For file path handling.
urllib.request: For downloading files from a URL.
pandas: For data manipulation and analysis.
sklearn: For machine learning utilities like model training, preprocessing, and evaluation.
matplotlib: For data visualization.
numpy: For numerical computations.
tensorflow: For building and training neural network models.

Functions:
save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
Saves plots with high resolution in the "images" folder.
load_file_from_git(file_name: str):
Loads datasets from a specified GitHub repository if not already downloaded.
save_plot_acc(plotname: str, y_train_prepared, y_predict):
Generates and saves scatter plots for actual vs predicted values.

Workflow:
Loads training and test datasets from a GitHub repository.
Performs data preprocessing including dropping columns, handling missing values, and encoding categorical features.
Visualizes data distribution and correlations using bar plots and correlation matrices.

Trains machine learning models:
Linear Regression
Random Forest Regressor (with hyperparameter tuning using GridSearchCV)
Support Vector Regression (SVR)
Multi-Layer Perceptron (MLP) using TensorFlow
Evaluates model performance using metrics such as Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE).
Saves visualizations and model evaluation plots in the "images" folder.
