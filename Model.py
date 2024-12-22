import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer

def visualize_results(pca, fa, feature_names):
    # Scree Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca.explained_variance_)+1), pca.explained_variance_, marker='o', linestyle='--')
    plt.title('Scree Plot (Eigenvalues)')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

    # Factor Loadings Table
    loadings = pd.DataFrame(fa.loadings_, columns=[f'Factor {i+1}' for i in range(fa.loadings_.shape[1])], index=feature_names)
    print("Factor Loadings:")
    print(loadings)

def Exploratory_Data_Analysis(df):
    # Number of people who responded yes by profession

    df_yes = df[df['responded'] == 'yes']
    profession_counts = df_yes['profession'].value_counts().sort_values()
    plt.figure(figsize=(10, 7))
    bars = plt.bar(profession_counts.index, profession_counts.values, color='skyblue')
    
    # Add value annotations on top of bars
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(int(bar.get_height())),
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.xticks(rotation=90)
    plt.xlabel("Profession")
    plt.ylabel("Count")
    plt.title("Value Count of People by Profession (Sorted by Count)")
    plt.tight_layout()
    plt.show()

    # Value count by method of contact.

    contact_counts = df['contact'].value_counts().sort_values()
    plt.figure(figsize=(10, 7))
    bars = plt.bar(contact_counts.index, contact_counts.values, color='skyblue')

    # Add value annotations on top of bars
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(int(bar.get_height())),
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.xticks(rotation=90)
    plt.xlabel("Contact")
    plt.ylabel("Count")
    plt.title("Value Count of People by Contact (Sorted by Count)")
    plt.tight_layout()
    plt.show()
    
    
def model_evaluation(X_train, X_val, y_train, y_val, models):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for model_name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results[model_name] = {
            'Mean CV Accuracy': np.mean(scores),
            'Validation Accuracy': accuracy_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred, average='weighted'),
            'Recall': recall_score(y_val, y_pred, average='weighted'),
            'F1 Score': f1_score(y_val, y_pred, average='weighted')
        }

    results_df = pd.DataFrame(results).T
    return results_df

def main():
    # Step 1: Load the datasets
    train_data = pd.read_csv('marketing_training.csv')
    test_data = pd.read_csv('marketing_test.csv')

    # Step 2: Data Cleaning
    if 'Unnamed: 0' in train_data.columns:
        train_data = train_data.drop('Unnamed: 0', axis=1)
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop('Unnamed: 0', axis=1)

    
    eda = Exploratory_Data_Analysis(train_data)

    common_columns = [col for col in train_data.columns if col in test_data.columns]
    target_column = 'responded'
    train_data = train_data[common_columns + [target_column]]
    test_data = test_data[common_columns]

    imputer = SimpleImputer(strategy='most_frequent')
    train_data.iloc[:, :-1] = imputer.fit_transform(train_data.iloc[:, :-1])
    test_data.iloc[:, :] = imputer.transform(test_data)

    for col in train_data.columns:
        if train_data[col].dtype == 'object':
            le = LabelEncoder()
            train_data[col] = le.fit_transform(train_data[col])
            if col in test_data.columns:
                test_data[col] = le.transform(test_data[col])

    scaler = StandardScaler()
    X = scaler.fit_transform(train_data.drop('responded', axis=1))
    y = train_data['responded']


    # Step 3: PCA and Factor Analysis
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced to {X_pca.shape[1]} components")

    fa = FactorAnalyzer(n_factors=5, rotation='varimax')
    fa.fit(X)
    X_fa = fa.transform(X)
    print(f"Factor Analysis reduced to {fa.loadings_.shape[1]} factors")

    visualize_results(pca, fa, train_data.drop('responded', axis=1).columns)

    # Step 4: Model Development
    X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }

    results_df = model_evaluation(X_train, X_val, y_train, y_val, models)
    print("Model Performance Metrics:")
    print(results_df)

if __name__ == "__main__":
    main()
