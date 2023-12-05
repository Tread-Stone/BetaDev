import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# Define a function to parse complex fields like 'torque'
def parse_torque(torque_str):
    # Implement your parsing logic here
    # Example: Extract numerical part and convert to float
    try:
        value = float(torque_str.split()[0])
    except (ValueError, IndexError):
        value = 0
    return value


def preprocess_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Fill missing values
    df.fillna(
        {
            "numerical_column": df["numerical_column"].mean(),
            "categorical_column": df["categorical_column"].mode()[0],
        },
        inplace=True,
    )

    # One-hot encode categorical data
    df = pd.get_dummies(df, columns=["categorical_column1", "categorical_column2"])

    # Label encode other categorical fields if necessary
    label_encoder = LabelEncoder()
    df["another_categorical_column"] = label_encoder.fit_transform(
        df["another_categorical_column"]
    )

    # Normalize numerical data
    scaler = StandardScaler()
    df[["numerical_column1", "numerical_column2"]] = scaler.fit_transform(
        df[["numerical_column1", "numerical_column2"]]
    )

    # Parse complex fields like 'torque'
    df["torque"] = df["torque"].apply(parse_torque)

    # Feature engineering (optional)
    df["new_feature"] = df["feature1"] / df["feature2"]

    # Split the dataset
    X = df.drop("target_column", axis=1)
    y = df["target_column"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    file_path = "../assets/Car details v3.csv"
    X_train, X_test, y_train, y_test = preprocess_data(file_path)
