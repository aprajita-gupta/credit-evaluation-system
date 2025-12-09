"""
Fix for 500 error when running with 3 models
Updates visualization code to handle variable number of models
"""

import os
import sys

if not os.path.exists('run_pipeline.py'):
    print("ERROR: run_pipeline.py not found")
    print("Make sure you're in the credit-evaluation-system folder")
    sys.exit(1)

print("Updating run_pipeline.py to handle 3 models...")

with open('run_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Dynamic color palette
old_colors = """    model_names = [k for k in results.keys() if k != 'y_test']
    
    # 1. Accuracy Comparison
    plt.figure()
    accuracies = [results[m]['accuracy'] for m in model_names]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = plt.bar(model_names, accuracies, color=colors, alpha=0.8)"""

new_colors = """    model_names = [k for k in results.keys() if k != 'y_test']
    
    # Dynamic color palette
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    colors = colors[:len(model_names)]  # Use only as many colors as needed
    
    # 1. Accuracy Comparison
    plt.figure()
    accuracies = [results[m]['accuracy'] for m in model_names]
    bars = plt.bar(model_names, accuracies, color=colors, alpha=0.8)"""

if old_colors in content:
    content = content.replace(old_colors, new_colors)
    print("✓ Fixed color palette")
else:
    print("⚠ Color palette already updated or not found")

# Fix 2: Dynamic confusion matrix grid
old_confusion = """    # 6. Confusion Matrices (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, model in enumerate(model_names):
        cm = results[model]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=False, square=True)
        axes[idx].set_title(f'{model}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
    
    plt.tight_layout()
    confusion_file = f'{save_dir}/confusion_matrices.png'
    plt.savefig(confusion_file, dpi=100, bbox_inches='tight')
    plt.close()
    plot_files['confusion'] = 'confusion_matrices.png'"""

new_confusion = """    # 6. Confusion Matrices (dynamic grid based on number of models)
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
    
    # Hide the extra subplot if we have 3 models
    if num_models == 3:
        axes[3].set_visible(False)
    
    plt.tight_layout()
    confusion_file = f'{save_dir}/confusion_matrices.png'
    plt.savefig(confusion_file, dpi=100, bbox_inches='tight')
    plt.close()
    plot_files['confusion'] = 'confusion_matrices.png'"""

if old_confusion in content:
    content = content.replace(old_confusion, new_confusion)
    print("✓ Fixed confusion matrix grid")
else:
    print("⚠ Confusion matrix already updated or not found")

# Write updated file
with open('run_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ SUCCESS! File has been updated.")
print("\nThe app will now work correctly with 3 models.")
print("\nRestart your Flask app:")
print("  Press CTRL+C to stop the current app")
print("  Then run: python app.py")