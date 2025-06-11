import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap
import joblib
import os

def train_and_save_explainer():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    employee_data_path = os.path.join(data_dir, "employee_data.csv")
    
    try:
        employee_df = pd.read_csv(employee_data_path)
    except FileNotFoundError:
        print(f"Error: {employee_data_path} not found. Please ensure your employee_data.csv is in the 'data' directory.")
        return

    # Define features and target
    features = ['experience', 'gender', 'role']
    target = 'salary'

    # Check if all required columns exist
    if not all(col in employee_df.columns for col in features + [target]):
        print("Error: Missing required columns in employee_data.csv. Please ensure 'experience', 'gender', 'role', and 'salary' columns are present.")
        return

    # Convert target to numeric, coercing errors to NaN and then dropping
    employee_df[target] = pd.to_numeric(employee_df[target], errors='coerce')
    employee_df.dropna(subset=[target], inplace=True) # Drop rows where salary is NaN
    if employee_df.empty:
        print("Error: No valid salary data after cleaning. Cannot train model.")
        return

    # Impute missing 'experience' with its median
    if 'experience' in employee_df.columns:
        if employee_df['experience'].isnull().any():
            median_experience = employee_df['experience'].median()
            employee_df['experience'].fillna(median_experience, inplace=True)
            print(f"Warning: Missing 'experience' values imputed with median: {median_experience}")
    
    # Drop any rows with NaN in feature columns after imputation and encoding
    employee_df.dropna(subset=features, inplace=True)
    if employee_df.empty:
        print("Error: No valid feature data after cleaning. Cannot train model.")
        return

    # One-hot encode categorical features
    X = pd.get_dummies(employee_df[features], columns=['gender', 'role'], drop_first=True)
    
    # Convert boolean columns (from one-hot encoding) to int (0 or 1)
    X = X.astype(int)

    y = employee_df[target]

    # Ensure all columns in X are numeric. This implicitly handles if pd.get_dummies creates non-numeric columns.
    X = X.apply(pd.to_numeric, errors='coerce')
    X.dropna(inplace=True) # Drop rows with any remaining NaNs in features after numeric conversion

    if X.empty:
        print("Error: No valid feature data after encoding and numeric conversion. Cannot train model.")
        return

    # Align y with X after dropping rows from X
    y = y.loc[X.index]

    # Split data (optional for simple explainer generation, but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Debug: Check dtypes and nulls in X_train before explainer
    print("DEBUG: X_train dtypes before SHAP explainer:")
    print(X_train.dtypes)
    print("DEBUG: X_train null sums before SHAP explainer:")
    print(X_train.isnull().sum())

    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model, X_train) # Use X_train as background data for explainer

    # Save the explainer
    explainer_path = os.path.join(models_dir, "shap_explainers.pkl")
    joblib.dump(explainer, explainer_path)
    print(f"SHAP Explainer saved to: {explainer_path}")

if __name__ == "__main__":
    train_and_save_explainer() 