"""
Quick fix for TensorFlow import issue
Run this to make TensorFlow optional in your existing installation
"""

import os
import sys

# Check if run_pipeline.py exists
if not os.path.exists('run_pipeline.py'):
    print("ERROR: run_pipeline.py not found in current directory")
    print("Make sure you're in the credit-evaluation-system folder")
    sys.exit(1)

print("Reading run_pipeline.py...")
with open('run_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already patched
if 'TENSORFLOW_AVAILABLE' in content:
    print("✅ File is already patched! TensorFlow is optional.")
    print("You can now run: python app.py")
    sys.exit(0)

print("Patching file to make TensorFlow optional...")

# Find the TensorFlow imports and replace them
old_import = """import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)"""

new_import = """# Try to import TensorFlow, but make it optional
TENSORFLOW_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.random.set_seed(42)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not available. DNN model will be skipped.")
    print("To use all 4 models, install TensorFlow with: pip install tensorflow")

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)"""

content = content.replace(old_import, new_import)

# Fix create_dnn_model function
old_dnn_def = """def create_dnn_model(input_dim):
    \"\"\"
    Create a lightweight Deep Neural Network for credit evaluation
    \"\"\"
    model = keras.Sequential(["""

new_dnn_def = """def create_dnn_model(input_dim):
    \"\"\"
    Create a lightweight Deep Neural Network for credit evaluation
    \"\"\"
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = keras.Sequential(["""

content = content.replace(old_dnn_def, new_dnn_def)

# Fix DNN training section
old_dnn_train = """    # ===== Deep Neural Network =====
    print("Training Deep Neural Network...")
    start_time = perf_counter()
    dnn_model = create_dnn_model(X_train_scaled.shape[1])"""

new_dnn_train = """    # ===== Deep Neural Network =====
    if TENSORFLOW_AVAILABLE:
        print("Training Deep Neural Network...")
        start_time = perf_counter()
        dnn_model = create_dnn_model(X_train_scaled.shape[1])"""

content = content.replace(old_dnn_train, new_dnn_train)

# Add else clause before storing test labels
old_end_dnn = """    
    # Store test labels for later use
    results['y_test'] = y_test"""

new_end_dnn = """    else:
        print("Skipping Deep Neural Network (TensorFlow not available)")
        print("Install TensorFlow to use all 4 models: pip install tensorflow")
    
    # Store test labels for later use
    results['y_test'] = y_test"""

content = content.replace(old_end_dnn, new_end_dnn)

# Write the patched file
print("Writing patched file...")
with open('run_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ SUCCESS! File has been patched.")
print()
print("TensorFlow is now optional. You can run:")
print("  python app.py")
print()
print("The system will work with 3 models (Logistic Regression, SVM, Decision Tree)")
print("DNN will be skipped if TensorFlow is not installed.")