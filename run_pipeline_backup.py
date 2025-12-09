"""
Credit Evaluation System - Machine Learning Pipeline
UPDATED: Fixed data leakage, added proper 60/20/20 split and cross-validation
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
    return X, y


def load_or_generate_data():
    """Load or generate data for training"""
    return generate_synthetic_data(n_samples=2000, n_features=20, random_state=42)


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
    return X, y


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


def train_and_evaluate_models(X, y, dvfs_level=1.0):
    """
    Complete training pipeline with NO DATA LEAKAGE
    Includes: 60/20/20 split, cross-validation, train, validate, test
    """
    # Split data properly (NO DATA LEAKAGE)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_data_properly(X, y)
    
    # Cross-validation (training data only)
    cv_results = perform_cross_validation(X_train, y_train)
    
    # Train and evaluate models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # TRAINING
        start_time = perf_counter()
        model.fit(X_train, y_train)
        train_time = perf_counter() - start_time
        
        # VALIDATION
        start_time = perf_counter()
        y_val_pred = model.predict(X_val)
        val_inference_time = perf_counter() - start_time
        
        # TESTING
        start_time = perf_counter()
        y_test_pred = model.predict(X_test)
        test_inference_time = perf_counter() - start_time
        
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = model.decision_function(X_test)
        
        energy, adj_time, power = compute_energy_estimate(test_inference_time, dvfs_level)
        
        results[name] = {
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
            'test_probabilities': y_test_proba
        }
    
    # Add DNN if TensorFlow available
    if TENSORFLOW_AVAILABLE:
        dnn_model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        start_time = perf_counter()
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        dnn_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
        train_time_dnn = perf_counter() - start_time
        
        y_val_pred_proba = dnn_model.predict(X_val, verbose=0).flatten()
        y_val_pred = (y_val_pred_proba > 0.5).astype(int)
        
        start_time = perf_counter()
        y_test_pred_proba = dnn_model.predict(X_test, verbose=0).flatten()
        y_test_pred = (y_test_pred_proba > 0.5).astype(int)
        test_inference_time_dnn = perf_counter() - start_time
        
        energy_dnn, adj_time_dnn, power_dnn = compute_energy_estimate(test_inference_time_dnn, dvfs_level)
        
        results['Deep Neural Network'] = {
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
            'test_probabilities': y_test_pred_proba
        }
    
    results['y_test'] = y_test
    return results


def create_visualizations(results, save_dir='static'):
    """Create all visualizations"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    plot_files = {}
    model_names = [k for k in results.keys() if k != 'y_test']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    colors = colors[:len(model_names)]
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
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
    
    return plot_files


def save_results_csv(results, save_dir='static'):
    """Save results to CSV"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = [k for k in results.keys() if k != 'y_test']
    
    data = []
    for model in model_names:
        r = results[model]
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
            'Energy_Estimate_J': f"{r['energy_estimate']:.4f}"
        })
    
    df = pd.DataFrame(data)
    csv_path = f'{save_dir}/model_comparison_results.csv'
    df.to_csv(csv_path, index=False)
    
    return csv_path