import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


np.random.seed(42)
data = {
    'Age': np.random.randint(20, 70, 1000),
    'Income': np.random.randint(20000, 200000, 1000),
    'Home': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], 1000),
    'Emp_length': np.random.randint(0, 40, 1000),
    'Intent': np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL'], 1000),
    'Amount': np.random.randint(1000, 50000, 1000),
    'Rate': np.random.uniform(5.0, 25.0, 1000),
    'Status': np.random.randint(0, 2, 1000),
    'Percent_income': np.random.uniform(0.0, 1.0, 1000),
    'Default': np.random.choice(['Y', 'N'], 1000),
    'Cred_length': np.random.randint(2, 31, 1000)
}
df = pd.DataFrame(data)


df['TARGET'] = np.random.randint(0, 2, 1000)


def explore_data(df):
    print(df.head())
    print(df.info())
    print(df.describe())
    missing_values = df.isnull().sum()
    print("Missing values:\n", missing_values[missing_values > 0])

def preprocess_data(df):
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])))
    df_imputed.columns = df.select_dtypes(include=[np.number]).columns
    
    
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna('Missing', inplace=True)
        df[col] = le.fit_transform(df[col])
    
    
    df_encoded = pd.concat([df_imputed, df.select_dtypes(include=['object'])], axis=1)
    
    return df_encoded


def split_data(df):
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')
    
    return accuracy, precision, recall, f1, roc_auc


def plot_feature_importance(model, X_train):
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importance')
    plt.show()


def main():
    explore_data(df)
    df_processed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_processed)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, X_train)

if __name__ == "__main__":
    main()
