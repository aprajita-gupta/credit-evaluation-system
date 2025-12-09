# Credit Evaluation System with Energy-Aware DVFS Integration

A complete web application for comparing machine learning models on credit scoring tasks with real-time energy consumption analysis.

## Features

- **Multiple ML Models**: Compare Logistic Regression, SVM, Decision Tree, and Deep Neural Network
- **Energy Analysis**: Software-based DVFS simulation showing energy/performance trade-offs
- **Built-in Dataset**: Synthetic credit dataset generator included
- **Custom Data Support**: Upload your own CSV files
- **Interactive Visualizations**: Accuracy, precision/recall, inference time, energy consumption, and confusion matrices
- **User-Friendly Interface**: Simple web UI designed for non-experts
- **Downloadable Results**: Export comparison metrics to CSV

## Requirements

- Python 3.8 or higher
- pip package manager
- Modern web browser (Chrome, Firefox, Safari, Edge)
- No special hardware needed - runs on any standard development machine

## Installation & Setup

### Step 1: Create Project Directory

```bash
mkdir credit-evaluation-system
cd credit-evaluation-system
```

Copy all project files into this directory:
- app.py
- run_pipeline.py
- requirements.txt
- templates/index.html
- templates/results.html

### Step 2: Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Installation takes 2-5 minutes depending on your internet connection.

### Step 4: Run the Application

```bash
python app.py
```

You should see output like:
```
 * Running on http://0.0.0.0:5000
 * Running on http://127.0.0.1:5000
```

### Step 5: Open in Browser

Open your web browser and navigate to:
```
http://localhost:5000
```

## Using the Application

### Quick Start (Using Built-in Dataset)

1. Open the application in your browser
2. Leave all default settings
3. Click "Run Analysis"
4. Wait 30-60 seconds for models to train
5. View comprehensive results with graphs and metrics

### Using Custom Data

1. Prepare a CSV file with:
   - A column named "default" containing labels (0 or 1)
   - At least one feature column
   - Example format:
     ```
     feature1,feature2,feature3,default
     0.5,1.2,0.8,0
     0.3,0.9,1.1,1
     ```

2. Check "Upload my own CSV file" checkbox
3. Click "Choose CSV File" and select your file
4. Click "Run Analysis"

### Adjusting Energy Settings

The "Processing Speed Level" slider simulates DVFS (Dynamic Voltage and Frequency Scaling):

- **0.5 - 0.9**: Lower speed, reduced energy consumption
- **1.0**: Balanced (default)
- **1.1 - 2.0**: Higher speed, increased energy consumption

This parameter affects:
- Inference time (execution speed)
- Energy consumption estimates
- Does NOT affect model accuracy or training

## Understanding the Results

### Summary Metrics
- **Best Accuracy**: Model with highest correct prediction rate
- **Lowest Energy**: Most energy-efficient model
- **Fastest Inference**: Model with shortest prediction time

### Detailed Comparison Table
Shows all metrics for each model:
- **Accuracy**: Overall correctness (0-1 scale)
- **Precision**: Correct positive predictions / all positive predictions
- **Recall**: Correct positive predictions / all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Inference Time**: How long predictions take (milliseconds)
- **Energy**: Estimated energy consumption (Joules)

### Visualizations
1. **Accuracy Comparison**: Bar chart of model accuracies
2. **Precision and Recall**: Side-by-side comparison
3. **Inference Time**: Execution speed comparison
4. **Energy Consumption**: Energy efficiency analysis
5. **Energy vs Accuracy Trade-off**: Scatter plot showing optimal models
6. **Confusion Matrices**: Detailed classification performance

### Downloading Results
Click "Download Results CSV" to save all metrics for further analysis.

## Troubleshooting

### Port Already in Use
If port 5000 is busy, edit app.py and change the last line:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Module Not Found Errors
Ensure virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

Then reinstall dependencies:
```bash
pip install -r requirements.txt
```

### Slow Training
Training typically takes 30-90 seconds. If slower:
- Close other applications
- Use smaller datasets
- Reduce dataset size in run_pipeline.py (change n_samples parameter)

### CSV Upload Errors
Ensure your CSV file:
- Has a "default" column with 0/1 values
- Has at least one feature column
- Is properly formatted (no missing values)

## Project Structure

```
credit-evaluation-system/
│
├── app.py                  # Flask backend and routes
├── run_pipeline.py         # ML pipeline, training, and visualization
├── requirements.txt        # Python dependencies
│
├── templates/
│   ├── index.html         # Main page UI
│   └── results.html       # Results display page
│
├── static/                # Generated plots (auto-created)
│   ├── *.png             # Visualization images
│   └── *.csv             # Results CSV
│
└── uploads/              # User-uploaded files (auto-created)
```

## Technical Details

### Machine Learning Models
- **Logistic Regression**: Linear classifier with L2 regularization
- **SVM**: RBF kernel with default parameters
- **Decision Tree**: Max depth of 10 to prevent overfitting
- **DNN**: 3 hidden layers (64→32→16 neurons) with dropout

### DVFS Energy Model
Energy is calculated using the formula:
```
Energy (J) = Power × Time + Overhead
Power = Base_Power × (DVFS_level)^2.5
Time = Measured_Time / DVFS_level
```

This provides realistic energy estimates based on actual execution times.

### Data Processing
- 80/20 train-test split
- Standard scaling (zero mean, unit variance)
- Stratified sampling to preserve class distribution
- Random seed (42) for reproducibility

## Stopping the Application

Press `Ctrl+C` in the terminal where the app is running.

To deactivate the virtual environment:
```bash
deactivate
```

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure Python version is 3.8 or higher

## License

This project is provided as-is for educational and research purposes.