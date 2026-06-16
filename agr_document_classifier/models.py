from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Simplified models with stronger regularization to prevent overfitting
POSSIBLE_CLASSIFIERS = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': [
            {
                'C': loguniform(1e-3, 10),
                'solver': ['lbfgs'],
                'penalty': ['l2'],
                'class_weight': ['balanced'],
                'warm_start': [False]
            },
            {
                'C': loguniform(1e-3, 10),
                'solver': ['liblinear', 'saga'],
                'penalty': ['l1', 'l2'],
                'class_weight': ['balanced'],
                'warm_start': [False]
            }
        ]
    },
    'LinearSVC': {
        # Scalable linear SVM (liblinear/saga) — handles the high-dimensional
        # sparse BoW matrix cheaply, unlike the libsvm-based SVC. No predict_proba;
        # the classify pipeline falls back to decision_function -> sigmoid.
        'model': LinearSVC(random_state=42, max_iter=5000),
        'params': [
            {
                'C': loguniform(1e-3, 10),
                'penalty': ['l2'],
                'loss': ['squared_hinge', 'hinge'],
                'dual': [True],
                'class_weight': ['balanced'],
            },
            {
                'C': loguniform(1e-3, 10),
                'penalty': ['l1'],
                'loss': ['squared_hinge'],
                'dual': [False],
                'class_weight': ['balanced'],
            },
        ]
    },
    'LGBMClassifier': {
        # LightGBM gradient boosting — sparse-native and typically >= XGBoost.
        # subsample_freq=1 makes the sampled `subsample` (bagging) actually apply.
        'model': LGBMClassifier(random_state=42, verbose=-1, subsample_freq=1),
        'params': {
            'n_estimators': randint(50, 200),
            'num_leaves': randint(15, 63),
            'max_depth': [-1, 3, 5, 7],
            'learning_rate': loguniform(0.01, 0.3),
            'min_child_samples': randint(5, 40),
            'subsample': uniform(0.6, 0.4),         # bagging fraction (0.6-1.0)
            'colsample_bytree': uniform(0.5, 0.5),  # feature fraction (0.5-1.0)
            'reg_alpha': loguniform(1e-3, 10),      # L1
            'reg_lambda': loguniform(1e-3, 10),     # L2
            'class_weight': ['balanced', None],
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': randint(50, 200),  # Reduced from 1000
            'max_depth': list(range(3, 12, 2)),  # Shallower trees
            'min_samples_split': randint(5, 30),  # Increased minimum
            'min_samples_leaf': randint(3, 15),  # Increased minimum
            'max_features': ['sqrt', 'log2', 0.3, 0.5],  # Limit features
            'criterion': ['gini', 'entropy'],
            'min_impurity_decrease': uniform(0.0, 0.05),  # Add min impurity
            'bootstrap': [True],
            'class_weight': ['balanced', None]
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': randint(50, 150),  # Reduced iterations
            'learning_rate': loguniform(0.01, 0.3),  # Lower learning rates
            'max_depth': list(range(2, 8)),  # Shallower trees
            'min_samples_split': randint(5, 20),
            'min_samples_leaf': randint(3, 10),
            'subsample': uniform(0.6, 0.4),  # Subsampling for regularization
            'max_features': ['sqrt', 'log2', 0.5]
        }
    },
    'XGBClassifier': {
        # n_jobs=1: XGBoost otherwise grabs every core, and under the parallel
        # RandomizedSearchCV (n_jobs=-1) that becomes workers x cores threads,
        # each allocating per-thread buffers sized by the wide BoW feature matrix
        # (2**18 cols) -> OOM. RF/GBM/LGBM default to single-threaded, so only XGB
        # oversubscribed. Let the outer search own the parallelism.
        'model': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1),
        'params': {
            'n_estimators': randint(50, 150),  # Reduced iterations
            'learning_rate': loguniform(0.01, 0.3),  # Lower learning rates
            'max_depth': list(range(2, 8)),  # Shallower trees
            'min_child_weight': randint(1, 10),  # Minimum sum of instance weight
            'subsample': uniform(0.6, 0.4),  # Subsampling
            'colsample_bytree': uniform(0.5, 0.5),  # Feature subsampling
            'reg_alpha': loguniform(1e-3, 10),  # L1 regularization
            'reg_lambda': loguniform(1e-3, 10),  # L2 regularization
            'gamma': uniform(0, 0.5)  # Minimum loss reduction
        }
    },
    'MLPClassifier': {
        'model': MLPClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.2, random_state=42),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],  # Simpler architectures
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': loguniform(1e-3, 1),  # Stronger L2 regularization
            'learning_rate': ['adaptive'],
            'learning_rate_init': loguniform(1e-4, 1e-2),
            'batch_size': ['auto', 32, 64],
            'beta_1': [0.9],  # Fixed Adam parameter
            'beta_2': [0.999],  # Fixed Adam parameter
        }
    },
    'SVC': {
        'model': SVC(probability=True, cache_size=500, random_state=42),
        'params': {
            'C': loguniform(1e-2, 10),  # Stronger regularization
            'kernel': ['linear', 'rbf'],  # Removed poly and sigmoid
            'gamma': ['scale', 'auto'] + list(loguniform(1e-4, 1e-1).rvs(20)),
            'shrinking': [True],
            'class_weight': ['balanced'],  # Always balanced
            'decision_function_shape': ['ovr']
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10],  # Shallower trees
            'min_samples_split': [10, 20, 30],  # Higher minimums
            'min_samples_leaf': [5, 10, 15],  # Higher minimums
            'max_features': ['sqrt', 'log2', 0.5],  # Limit features
            'min_impurity_decrease': [0.0, 0.01, 0.02],  # Add min impurity
            'class_weight': ['balanced', None]
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [5, 7, 9, 11, 15],  # More neighbors for smoother decision boundary
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto'],
            'p': [1, 2],  # Manhattan and Euclidean distance
            'leaf_size': [30, 50]  # For tree-based algorithms
        }
    },
    'SGDClassifier': {
        'model': SGDClassifier(random_state=42, max_iter=1000, early_stopping=True, validation_fraction=0.2),
        'params': {
            'loss': ['log_loss', 'modified_huber'],  # Only probabilistic losses
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': loguniform(1e-5, 1e-1),  # Regularization strength
            'l1_ratio': uniform(0, 1),  # For elasticnet
            'learning_rate': ['optimal', 'adaptive'],
            'eta0': [0.01, 0.1],  # Initial learning rate
            'class_weight': ['balanced']  # Always balanced
        }
    }
}
