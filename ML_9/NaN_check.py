import numpy as np
import pandas as pd
X_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_train_sc.csv")
y_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_train.csv")
X_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_test_sc.csv")
y_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_test.csv")

# Check for NaN values
nan_check_X_train = np.isnan(X_train)
nan_check_y_train = np.isnan(y_train)
nan_check_X_test = np.isnan(X_test)
nan_check_y_test = np.isnan(y_test)

print("NaN values in X_train:", nan_check_X_train.any())
print("NaN values in y_train:", nan_check_y_train.any())
print("NaN values in X_test:", nan_check_X_test.any())
print("NaN values in y_test:", nan_check_y_test.any())

# Check for infinite values
inf_check_X_train = np.isinf(X_train)
inf_check_y_train = np.isinf(y_train)
inf_check_X_test = np.isinf(X_test)
inf_check_y_test = np.isinf(y_test)

print("Infinite values in X_train:", inf_check_X_train.any())
print("Infinite values in y_train:", inf_check_y_train.any())
print("Infinite values in X_test:", inf_check_X_test.any())
print("Infinite values in y_test:", inf_check_y_test.any())
