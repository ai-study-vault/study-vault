"""
Permutation Feature Importance Implementation
Module for calculating feature importance by permuting features and measuring performance drop.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union, Optional
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import warnings


class PermutationImportance:
    """
    Calculate permutation feature importance for any model.
    
    This is a model-agnostic, global explanation method that measures how much
    a model's performance drops when a single feature's values are randomly shuffled.
    
    Attributes:
        model: Fitted scikit-learn model
        X_test: Test features (numpy array or pandas DataFrame)
        y_test: Test targets
        metric: Performance metric ('r2', 'accuracy', 'mse')
        n_repeats: Number of permutation repeats for stable estimates
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        model,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: np.ndarray,
        metric: str = 'auto',
        n_repeats: int = 10,
        random_state: int = 42
    ):
        """
        Initialize PermutationImportance calculator.
        
        Args:
            model: Fitted scikit-learn model
            X_test: Test feature matrix
            y_test: Test target vector
            metric: 'auto', 'r2', 'accuracy', 'mse'
            n_repeats: Number of permutation repeats
            random_state: Seed for reproducibility
        """
        self.model = model
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.metric = metric
        self.n_repeats = n_repeats
        self.random_state = random_state
        
        # Extract feature names
        if isinstance(X_test, pd.DataFrame):
            self.feature_names = X_test.columns.tolist()
            self.X_test = X_test.values
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
        
        # Determine metric if auto
        if metric == 'auto':
            # Infer from model type
            if hasattr(model, 'predict_proba'):
                self.metric = 'accuracy'
            else:
                self.metric = 'r2'
        
        self._baseline_score = None
    
    def _get_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate performance metric."""
        if self.metric == 'r2':
            return r2_score(y_true, y_pred)
        elif self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'mse':
            return -mean_squared_error(y_true, y_pred)  # Negative so higher is better
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    @property
    def baseline_score(self) -> float:
        """Get baseline model score on test set."""
        if self._baseline_score is None:
            y_pred = self.model.predict(self.X_test)
            self._baseline_score = self._get_score(self.y_test, y_pred)
        return self._baseline_score
    
    def calculate(self) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate permutation feature importance.
        
        Returns:
            Dictionary with keys:
                - 'importances_mean': Mean importance for each feature
                - 'importances_std': Standard deviation of importance
                - 'importances_all': All importance values (n_features x n_repeats)
                - 'baseline_score': Model's baseline performance
                - 'feature_names': Names of features
        """
        n_features = self.X_test.shape[1]
        importances = np.zeros((n_features, self.n_repeats))
        
        baseline = self.baseline_score
        
        for feature_idx in range(n_features):
            for repeat in range(self.n_repeats):
                # Shuffle feature
                X_permuted = self.X_test.copy()
                rng = np.random.RandomState(self.random_state + repeat)
                rng.shuffle(X_permuted[:, feature_idx])
                
                # Get permuted score
                y_pred = self.model.predict(X_permuted)
                permuted_score = self._get_score(self.y_test, y_pred)
                
                # Calculate importance (drop in performance)
                importance = baseline - permuted_score
                importances[feature_idx, repeat] = importance
        
        return {
            'importances_mean': np.mean(importances, axis=1),
            'importances_std': np.std(importances, axis=1),
            'importances_all': importances,
            'baseline_score': baseline,
            'feature_names': self.feature_names,
            'metric': self.metric
        }
    
    def get_summary(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get sorted summary of feature importance.
        
        Args:
            top_n: Number of top features to show
            
        Returns:
            DataFrame with features sorted by importance
        """
        results = self.calculate()
        
        df = pd.DataFrame({
            'Feature': results['feature_names'],
            'Importance': results['importances_mean'],
            'Std': results['importances_std'],
            'Lower_CI': results['importances_mean'] - 2 * results['importances_std'],
            'Upper_CI': results['importances_mean'] + 2 * results['importances_std']
        })
        
        df = df.sort_values('Importance', ascending=False).reset_index(drop=True)
        return df.head(top_n)
    
    def plot_importance(self, top_n: int = 15, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot feature importance with error bars.
        
        Args:
            top_n: Number of features to plot
            figsize: Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        results = self.calculate()
        sorted_idx = np.argsort(results['importances_mean'])[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(sorted_idx))
        
        ax.barh(y_pos, 
                results['importances_mean'][sorted_idx],
                xerr=results['importances_std'][sorted_idx],
                color='#D4A843',
                alpha=0.8,
                error_kw={'elinewidth': 2, 'capsize': 5})
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([results['feature_names'][i] for i in sorted_idx])
        ax.set_xlabel(f'Importance (Drop in {self.metric})', fontweight='bold')
        ax.set_title('Permutation Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Styling
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        return fig, ax


# Example usage function
def example_diabetes():
    """Example: Calculate PFI on diabetes dataset."""
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Load data
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate PFI
    pfi = PermutationImportance(model, X_test, y_test, n_repeats=10)
    
    # Print results
    print("=" * 60)
    print("PERMUTATION FEATURE IMPORTANCE - DIABETES DATASET")
    print("=" * 60)
    print(f"\nBaseline R² Score: {pfi.baseline_score:.4f}\n")
    print(pfi.get_summary(top_n=10))
    
    # Plot
    pfi.plot_importance(top_n=10)
    
    return pfi


if __name__ == "__main__":
    example_diabetes()