"""
Fayton Uncertainty Model - Developed by konikhzed
GitHub: https://github.com/konikhzed/Fayton
License: MIT (https://opensource.org/license/mit/)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris, load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class GoldenUncertaintyModel(BaseEstimator, ClassifierMixin):
    """
    Golden Uncertainty Model with φ-based architecture
    
    Parameters:
    uncertainty_level (float): Level of uncertainty to inject (default=0.1)
    adaptive_noise (bool): Whether to adapt noise to feature scales (default=True)
    random_state (int): Random seed for reproducibility
    """
    
    def __init__(self, uncertainty_level=0.1, adaptive_noise=True, random_state=None):
        # Golden ratio and its properties
        self.φ = (1 + np.sqrt(5)) / 2  # ≈1.618
        self.φ_inv = 1 / self.φ  # ≈0.618
        self.φ_angle = 2 * np.pi * self.φ_inv  # Golden angle ≈137.5°
        
        self.uncertainty_level = uncertainty_level
        self.adaptive_noise = adaptive_noise
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Watermark - Konikhzed's signature
        self.creator = "konikhzed"
        self.model_signature = f"GU-{self.creator}-v1.0"
    
    def add_uncertainty(self, X):
        """Add golden ratio-based uncertainty to input data"""
        # Basic noise with golden ratio inverse
        noise = np.random.normal(0, self.uncertainty_level * self.φ_inv, X.shape)
        
        # Create golden spiral pattern
        n_samples = X.shape[0]
        angles = np.arange(n_samples) * self.φ_angle
        spiral = np.sin(angles[:, None] * self.φ_inv)
        
        # Adaptive scaling to feature variances
        if self.adaptive_noise:
            feature_scales = np.std(X, axis=0)
            spiral = spiral * feature_scales * self.uncertainty_level
        
        return X + noise + spiral
    
    def fit(self, X, y):
        """Fit the model to training data with uncertainty injection"""
        # Check and validate input
        X, y = check_X_y(X, y)
        
        # Add golden uncertainty
        X_uncertain = self.add_uncertainty(X)
        
        # Initialize and train classifier
        self.classifier_ = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state
        )
        self.classifier_.fit(X_uncertain, y)
        
        # Store metadata
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        check_is_fitted(self)
        X = check_array(X)
        return self.classifier_.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        check_is_fitted(self)
        X = check_array(X)
        return self.classifier_.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return accuracy, f1
    
    def verify_attribution(self):
        """Verify proper attribution of the model"""
        print(f"### This model is developed by {self.creator} ###")
        print(f"### Please cite: https://github.com/konikhzed/Fayton ###")
        print(f"### Model Signature: {self.model_signature} ###\n")

def run_full_pipeline(dataset_name="Iris", uncertainty_levels=[0.05, 0.1, 0.2], random_state=42):
    """
    Run full evaluation pipeline for the model
    
    Parameters:
    dataset_name (str): Name of dataset ('Iris' or 'Breast Cancer')
    uncertainty_levels (list): List of uncertainty levels to test
    random_state (int): Random seed for reproducibility
    """
    # Load dataset
    if dataset_name == "Iris":
        data = load_iris()
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
    else:
        raise ValueError("Unsupported dataset. Choose 'Iris' or 'Breast Cancer'")
    
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    print(f"\n{'='*50}")
    print(f"Evaluating on {dataset_name} Dataset")
    print(f"{'='*50}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Store results
    results = []
    
    # Test different uncertainty levels
    for level in uncertainty_levels:
        print(f"\n🔧 Testing uncertainty level: {level}")
        
        # Create and train model
        model = GoldenUncertaintyModel(
            uncertainty_level=level,
            adaptive_noise=True,
            random_state=random_state
        )
        
        # Verify attribution
        model.verify_attribution()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        accuracy, f1 = model.evaluate(X_test, y_test)
        
        # Store results
        results.append({
            'Uncertainty Level': level,
            'Accuracy': accuracy,
            'F1 Score': f1
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_df, 
        x='Uncertainty Level', 
        y='Accuracy',
        marker='o',
        markersize=8,
        linewidth=2.5
    )
    plt.title(f'Fayton Model Performance on {dataset_name}', fontsize=14)
    plt.xlabel('Uncertainty Level')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'fayton_{dataset_name}_performance.png', dpi=300)
    plt.show()
    
    return results_df

# Example usage
if __name__ == "__main__":
    # Run on Iris dataset
    iris_results = run_full_pipeline(
        dataset_name="Iris",
        uncertainty_levels=[0.05, 0.1, 0.15, 0.2],
        random_state=42
    )
    
    print("\nResults for Iris Dataset:")
    print(iris_results)
    
    # Run on Breast Cancer dataset
    cancer_results = run_full_pipeline(
        dataset_name="Breast Cancer",
        uncertainty_levels=[0.05, 0.1, 0.2, 0.3],
        random_state=42
    )
    
    print("\nResults for Breast Cancer Dataset:")
    print(cancer_results)
