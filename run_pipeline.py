"""
Credit Evaluation System - Machine Learning Pipeline with SHAP Explainability
ENHANCED: Added SHAP for model interpretability alongside performance and energy metrics
"""

import numpy as np
import pandas as pd
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow
TENSORFLOW_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    tf.random.set_seed(42)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not available. DNN model will be skipped.")

# Try to import SHAP
SHAP_AVAILABLE = True
try:
    import shap
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not available. Explainability features will be skipped.")
    print("To enable SHAP: pip install shap")

np.random.seed(42)

# DVFS Energy Model Parameters
ENERGY_BASE_POWER = 15.0
ENERGY_IDLE_POWER = 5.0
ENERGY_FIXED_OVERHEAD = 0.1
DVFS_POWER_EXPONENT = 2.5


def get_cpu_info():
    """Get CPU information for power modeling"""
    try:
        import psutil
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'available': True
        }
    except ImportError:
        return {'cpu_count': 4, 'cpu_percent': 50.0, 'available': False}


def compute_energy_estimate(time_seconds, dvfs_level, cpu_info=None):
    """Compute energy estimate based on DVFS level"""
    if cpu_info is None:
        cpu_info = get_cpu_info()
    
    adjusted_time = time_seconds / dvfs_level
    dynamic_power = ENERGY_BASE_POWER * (dvfs_level ** DVFS_POWER_EXPONENT)
    static_power = ENERGY_IDLE_POWER * (1 + 0.1 * (dvfs_level - 1.0))
    total_power = dynamic_power + static_power
    
    if cpu_info['available']:
        utilization_factor = cpu_info['cpu_percent'] / 100.0
        total_power = static_power + (dynamic_power * utilization_factor)
    
    energy_total = total_power * adjusted_time + ENERGY_FIXED_OVERHEAD
    return energy_total, adjusted_time, total_power


def generate_synthetic_data(n_samples=2000, n_features=20, random_state=42):
    """Generate synthetic credit evaluation data"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_features)
    logits = X @ weights
    probabilities = 1 / (1 + np.exp(-logits))
    y = (probabilities > 0.5).astype(int)
    
    # Create feature names
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    return X, y, feature_names


def load_or_generate_data():
    """Load or generate data for training"""
    X, y, feature_names = generate_synthetic_data(n_samples=2000, n_features=20, random_state=42)
    return X, y, feature_names


def preprocess_loan_dataset(df):
    """Preprocess real loan dataset"""
    df = df.copy()
    
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
    
    if 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
        y = df['Loan_Status'].values
        df = df.drop('Loan_Status', axis=1)
    else:
        raise ValueError("Dataset must contain 'Loan_Status' column")
    
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace('3+', '3')
        df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)
    
    X = df.values
    feature_names = df.columns.tolist()
    return X, y, feature_names


def split_data_properly(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split data with NO DATA LEAKAGE
    Returns: X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler
    """
    # Split test set first
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )
    
    # Split train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    # FIT scaler on training data ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def compute_shap_values(model, model_name, X_train, X_test, feature_names):
    """
    Compute SHAP values for model interpretability
    Returns: shap_values, expected_value, explainer
    """
    if not SHAP_AVAILABLE:
        return None, None, None
    
    try:
        print(f"    Computing SHAP values for {model_name}...")
        start_time = perf_counter()
        
        # Select appropriate SHAP explainer based on model type
        if model_name in ['Decision Tree', 'Random Forest']:
            # TreeExplainer: Fast and exact for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Handle binary classification (returns list of arrays)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]
                
        elif model_name == 'Logistic Regression':
            # LinearExplainer: Fast for linear models
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)
            expected_value = explainer.expected_value
            
        elif model_name == 'Deep Neural Network':
            # DeepExplainer: For neural networks
            background = X_train[:100]  # Use subset as background
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(X_test[:100])  # Limit for speed
            expected_value = explainer.expected_value
            
        else:  # SVM or other models
            # KernelExplainer: Model-agnostic (slower but works for any model)
            background = shap.sample(X_train, 50)  # Small sample for speed
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test[:100])  # Limit samples
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]
        
        elapsed_time = perf_counter() - start_time
        print(f"    SHAP computation completed in {elapsed_time:.2f}s")
        
        return shap_values, expected_value, explainer
        
    except Exception as e:
        print(f"    WARNING: SHAP computation failed for {model_name}: {e}")
        return None, None, None


def create_shap_visualizations(results, feature_names, save_dir='static'):
    """
    Create SHAP visualizations for all models
    """
    if not SHAP_AVAILABLE:
        print("Skipping SHAP visualizations (SHAP not available)")
        return {}
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    shap_plot_files = {}
    
    model_names = [k for k in results.keys() if k not in ['y_test', 'X_test', 'feature_names']]
    
    for model_name in model_names:
        if 'shap_values' not in results[model_name] or results[model_name]['shap_values'] is None:
            continue
        
        try:
            shap_values = results[model_name]['shap_values']
            expected_value = results[model_name]['expected_value']
            
            # Get corresponding X_test (handle size mismatch)
            X_test = results['X_test']
            if len(shap_values) < len(X_test):
                X_test_shap = X_test[:len(shap_values)]
            else:
                X_test_shap = X_test
            
            model_safe_name = model_name.replace(' ', '_').lower()
            
            # 1. SHAP Summary Plot (Global Feature Importance)
            print(f"    Creating SHAP summary plot for {model_name}...")
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_shap, 
                            feature_names=feature_names, 
                            show=False, max_display=15)
            plt.title(f'{model_name} - Feature Importance (SHAP)', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            summary_file = f'{save_dir}/shap_summary_{model_safe_name}.png'
            plt.savefig(summary_file, dpi=100, bbox_inches='tight')
            plt.close()
            shap_plot_files[f'{model_safe_name}_summary'] = os.path.basename(summary_file)
            
            # 2. SHAP Bar Plot (Feature Importance Ranking)
            print(f"    Creating SHAP bar plot for {model_name}...")
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_shap,
                            feature_names=feature_names,
                            plot_type='bar', show=False, max_display=15)
            plt.title(f'{model_name} - Feature Importance Ranking', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            bar_file = f'{save_dir}/shap_bar_{model_safe_name}.png'
            plt.savefig(bar_file, dpi=100, bbox_inches='tight')
            plt.close()
            shap_plot_files[f'{model_safe_name}_bar'] = os.path.basename(bar_file)
            
            # 3. SHAP Force Plot (First Prediction Explanation)
            print(f"    Creating SHAP force plot for {model_name}...")
            try:
                shap.force_plot(expected_value, shap_values[0], 
                              X_test_shap[0], 
                              feature_names=feature_names,
                              matplotlib=True, show=False)
                plt.title(f'{model_name} - Individual Prediction Explanation', 
                         fontsize=12, fontweight='bold')
                force_file = f'{save_dir}/shap_force_{model_safe_name}.png'
                plt.savefig(force_file, dpi=100, bbox_inches='tight')
                plt.close()
                shap_plot_files[f'{model_safe_name}_force'] = os.path.basename(force_file)
            except Exception as e:
                print(f"      Force plot failed: {e}")
            
            # 4. SHAP Dependence Plot (Top Feature)
            if len(feature_names) > 0:
                print(f"    Creating SHAP dependence plot for {model_name}...")
                try:
                    # Find most important feature
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    top_feature_idx = np.argmax(mean_abs_shap)
                    
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(top_feature_idx, shap_values, X_test_shap,
                                       feature_names=feature_names, show=False)
                    plt.title(f'{model_name} - Feature Interaction: {feature_names[top_feature_idx]}',
                             fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    dep_file = f'{save_dir}/shap_dependence_{model_safe_name}.png'
                    plt.savefig(dep_file, dpi=100, bbox_inches='tight')
                    plt.close()
                    shap_plot_files[f'{model_safe_name}_dependence'] = os.path.basename(dep_file)
                except Exception as e:
                    print(f"      Dependence plot failed: {e}")
            
        except Exception as e:
            print(f"    ERROR creating SHAP plots for {model_name}: {e}")
            continue
    
    return shap_plot_files


def create_dnn_model(input_dim):
    """Create a lightweight Deep Neural Network for credit evaluation"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def perform_cross_validation(X_train, y_train, n_folds=5):
    """Perform k-fold cross-validation on TRAINING DATA ONLY"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    cv_results = {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for name, model in models.items():
        cv_acc = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
        cv_f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1', n_jobs=-1)
        
        cv_results[name] = {
            'accuracy_scores': cv_acc,
            'accuracy_mean': cv_acc.mean(),
            'accuracy_std': cv_acc.std(),
            'f1_scores': cv_f1,
            'f1_mean': cv_f1.mean(),
            'f1_std': cv_f1.std()
        }
    
    return cv_results


def train_and_evaluate_models(X, y, feature_names, dvfs_level=1.0, compute_shap=True):
    """
    Complete training pipeline with SHAP explainability
    """
    print("\n" + "="*70)
    print("TRAINING PIPELINE WITH SHAP EXPLAINABILITY")
    print("="*70)
    
    # Split data properly
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_data_properly(X, y)
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_results = perform_cross_validation(X_train, y_train)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print(f"{'='*70}")
        
        # TRAINING
        start_time = perf_counter()
        model.fit(X_train, y_train)
        train_time = perf_counter() - start_time
        print(f"  Training completed in {train_time:.4f}s")
        
        # VALIDATION
        start_time = perf_counter()
        y_val_pred = model.predict(X_val)
        val_inference_time = perf_counter() - start_time
        
        # TEST
        start_time = perf_counter()
        y_test_pred = model.predict(X_test)
        test_inference_time = perf_counter() - start_time
        
        # Energy calculation
        energy, adj_time, power = compute_energy_estimate(test_inference_time, dvfs_level)
        
        # SHAP computation
        shap_values, expected_value, explainer = None, None, None
        if compute_shap and SHAP_AVAILABLE:
            shap_values, expected_value, explainer = compute_shap_values(
                model, name, X_train, X_test, feature_names
            )
        
        # Store results
        results[name] = {
            'model': model,
            'train_time': train_time,
            'cv_accuracy_mean': cv_results[name]['accuracy_mean'],
            'cv_accuracy_std': cv_results[name]['accuracy_std'],
            'cv_f1_mean': cv_results[name]['f1_mean'],
            'cv_f1_std': cv_results[name]['f1_std'],
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
            'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
            'val_f1': f1_score(y_val, y_val_pred, zero_division=0),
            'val_confusion_matrix': confusion_matrix(y_val, y_val_pred),
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'inference_time': test_inference_time,
            'adjusted_inference_time': adj_time,
            'energy_estimate': energy,
            'predictions': y_test_pred,
            'shap_values': shap_values,
            'expected_value': expected_value,
            'explainer': explainer
        }
        
        print(f"  Test Accuracy: {results[name]['accuracy']:.4f}")
        print(f"  Energy: {energy:.4f}J")
        if shap_values is not None:
            print(f"  SHAP: ✓ Computed")
    
    # Deep Neural Network
    if TENSORFLOW_AVAILABLE:
        print(f"\n{'='*70}")
        print("Training: Deep Neural Network")
        print(f"{'='*70}")
        
        start_time = perf_counter()
        dnn_model = create_dnn_model(X_train.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = dnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        train_time_dnn = perf_counter() - start_time
        print(f"  Training completed in {train_time_dnn:.4f}s")
        
        # Validation
        y_val_pred_proba = dnn_model.predict(X_val, verbose=0).flatten()
        y_val_pred = (y_val_pred_proba > 0.5).astype(int)
        
        # Test
        start_time = perf_counter()
        y_test_pred_proba = dnn_model.predict(X_test, verbose=0).flatten()
        y_test_pred = (y_test_pred_proba > 0.5).astype(int)
        test_inference_time_dnn = perf_counter() - start_time
        
        energy_dnn, adj_time_dnn, power_dnn = compute_energy_estimate(test_inference_time_dnn, dvfs_level)
        
        # SHAP for DNN (optional, can be slow)
        shap_values_dnn, expected_value_dnn, explainer_dnn = None, None, None
        if compute_shap and SHAP_AVAILABLE:
            shap_values_dnn, expected_value_dnn, explainer_dnn = compute_shap_values(
                dnn_model, 'Deep Neural Network', X_train, X_test, feature_names
            )
        
        results['Deep Neural Network'] = {
            'model': dnn_model,
            'train_time': train_time_dnn,
            'cv_accuracy_mean': 0,
            'cv_accuracy_std': 0,
            'cv_f1_mean': 0,
            'cv_f1_std': 0,
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
            'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
            'val_f1': f1_score(y_val, y_val_pred, zero_division=0),
            'val_confusion_matrix': confusion_matrix(y_val, y_val_pred),
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'inference_time': test_inference_time_dnn,
            'adjusted_inference_time': adj_time_dnn,
            'energy_estimate': energy_dnn,
            'predictions': y_test_pred,
            'test_probabilities': y_test_pred_proba,
            'shap_values': shap_values_dnn,
            'expected_value': expected_value_dnn,
            'explainer': explainer_dnn
        }
        
        print(f"  Test Accuracy: {results['Deep Neural Network']['accuracy']:.4f}")
        print(f"  Energy: {energy_dnn:.4f}J")
        if shap_values_dnn is not None:
            print(f"  SHAP: ✓ Computed")
    
    # Store metadata
    results['y_test'] = y_test
    results['X_test'] = X_test
    results['feature_names'] = feature_names
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70 + "\n")
    
    return results


def create_visualizations(results, save_dir='static'):
    """Create all visualizations including SHAP"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    plot_files = {}
    model_names = [k for k in results.keys() if k not in ['y_test', 'X_test', 'feature_names']]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    colors = colors[:len(model_names)]
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # Standard visualizations (accuracy, precision, recall, etc.)
    # ... [Previous visualization code remains the same] ...
    
    # 1. Accuracy Comparison
    plt.figure()
    accuracies = [results[m]['accuracy'] for m in model_names]
    bars = plt.bar(model_names, accuracies, color=colors, alpha=0.8)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    plt.xticks(rotation=15, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    plot_files['accuracy'] = 'accuracy_comparison.png'
    
    # 2. Precision and Recall
    plt.figure()
    x = np.arange(len(model_names))
    width = 0.35
    
    precisions = [results[m]['precision'] for m in model_names]
    recalls = [results[m]['recall'] for m in model_names]
    
    plt.bar(x - width/2, precisions, width, label='Precision', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, recalls, width, label='Recall', color='#e74c3c', alpha=0.8)
    
    plt.ylabel('Score', fontsize=12)
    plt.title('Precision and Recall Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names, rotation=15, ha='right')
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/precision_recall.png', dpi=100, bbox_inches='tight')
    plt.close()
    plot_files['precision_recall'] = 'precision_recall.png'
    
    # 3. Inference Time
    plt.figure()
    inference_times = [results[m]['adjusted_inference_time'] * 1000 for m in model_names]
    bars = plt.bar(model_names, inference_times, color=colors, alpha=0.8)
    plt.ylabel('Inference Time (milliseconds)', fontsize=12)
    plt.title('Model Inference Time Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/inference_time.png', dpi=100, bbox_inches='tight')
    plt.close()
    plot_files['inference_time'] = 'inference_time.png'
    
    # 4. Energy Consumption
    plt.figure()
    energies = [results[m]['energy_estimate'] for m in model_names]
    bars = plt.bar(model_names, energies, color=colors, alpha=0.8)
    plt.ylabel('Estimated Energy (Joules)', fontsize=12)
    plt.title('Model Energy Consumption Estimate', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/energy_consumption.png', dpi=100, bbox_inches='tight')
    plt.close()
    plot_files['energy'] = 'energy_consumption.png'
    
    # 5. Energy vs Accuracy Trade-off
    plt.figure()
    accuracies = [results[m]['accuracy'] for m in model_names]
    energies = [results[m]['energy_estimate'] for m in model_names]
    
    plt.scatter(energies, accuracies, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
    
    for i, model in enumerate(model_names):
        plt.annotate(model, (energies[i], accuracies[i]),
                    xytext=(10, 10), textcoords='offset points', fontsize=9, ha='left')
    
    plt.xlabel('Estimated Energy (Joules)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Energy vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/energy_accuracy_tradeoff.png', dpi=100, bbox_inches='tight')
    plt.close()
    plot_files['tradeoff'] = 'energy_accuracy_tradeoff.png'
    
    # 6. Confusion Matrices
    num_models = len(model_names)
    if num_models == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    elif num_models == 3:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
    
    axes = axes.ravel() if num_models > 1 else [axes]
    
    for idx, model in enumerate(model_names):
        cm = results[model]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=False, square=True)
        axes[idx].set_title(f'{model}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
    
    if num_models == 3:
        axes[3].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=100, bbox_inches='tight')
    plt.close()
    plot_files['confusion'] = 'confusion_matrices.png'
    
    # SHAP Visualizations
    print("\nGenerating SHAP visualizations...")
    feature_names = results.get('feature_names', [])
    shap_plots = create_shap_visualizations(results, feature_names, save_dir)
    plot_files.update(shap_plots)
    
    return plot_files


def save_results_csv(results, save_dir='static'):
    """Save results to CSV including SHAP availability"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = [k for k in results.keys() if k not in ['y_test', 'X_test', 'feature_names']]
    
    data = []
    for model in model_names:
        r = results[model]
        has_shap = 'Yes' if r.get('shap_values') is not None else 'No'
        
        data.append({
            'Model': model,
            'CV_Accuracy': f"{r['cv_accuracy_mean']:.4f} ± {r['cv_accuracy_std']:.4f}" if r['cv_accuracy_mean'] > 0 else 'N/A',
            'Val_Accuracy': f"{r['val_accuracy']:.4f}",
            'Test_Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1_Score': f"{r['f1']:.4f}",
            'Training_Time_s': f"{r['train_time']:.4f}",
            'Inference_Time_ms': f"{r['adjusted_inference_time'] * 1000:.4f}",
            'Energy_Estimate_J': f"{r['energy_estimate']:.4f}",
            'SHAP_Available': has_shap
        })
    
    df = pd.DataFrame(data)
    csv_path = f'{save_dir}/model_comparison_results.csv'
    df.to_csv(csv_path, index=False)
    
    return csv_path


# Main execution for testing
if __name__ == '__main__':
    print("Testing Credit Evaluation System with SHAP...")
    
    # Load data
    X, y, feature_names = load_or_generate_data()
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train and evaluate
    results = train_and_evaluate_models(X, y, feature_names, dvfs_level=1.0, compute_shap=True)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_files = create_visualizations(results, save_dir='static')
    print(f"Created {len(plot_files)} visualizations")
    
    # Save results
    csv_path = save_results_csv(results, save_dir='static')
    print(f"Results saved to: {csv_path}")
    
    print("\n✅ Testing complete!")