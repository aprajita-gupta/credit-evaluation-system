"""
Credit Evaluation System with Energy-Aware DVFS Integration
Flask Backend Application
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from werkzeug.utils import secure_filename
from run_pipeline import (
    generate_synthetic_data,
    train_and_evaluate_models,
    create_visualizations,
    save_results_csv,
    preprocess_loan_dataset
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Store latest results in memory
latest_results = None


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """Run model training and evaluation"""
    global latest_results
    
    try:
        # Get parameters from request
        dvfs_level = float(request.form.get('dvfs_level', 1.0))
        use_uploaded = request.form.get('use_uploaded', 'false') == 'true'
        
        # Validate DVFS level
        if dvfs_level < 0.5 or dvfs_level > 2.0:
            return jsonify({'error': 'DVFS level must be between 0.5 and 2.0'}), 400
        
        # Load data
        if use_uploaded and 'datafile' in request.files:
            file = request.files['datafile']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not file.filename.endswith('.csv'):
                return jsonify({'error': 'Only CSV files are supported'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load custom dataset
            import pandas as pd
            try:
                df = pd.read_csv(filepath)
                
                # Check for either 'default' or 'Loan_Status' column
                if 'Loan_Status' in df.columns:
                    # Real loan dataset format
                    X, y = preprocess_loan_dataset(df)
                elif 'default' in df.columns:
                    # Simple format
                    if len(df.columns) < 2:
                        return jsonify({'error': 'Dataset must have at least one feature column besides the label'}), 400
                    X = df.drop('default', axis=1).values
                    y = df['default'].values
                else:
                    return jsonify({'error': 'Dataset must contain either a "default" or "Loan_Status" column for labels'}), 400
                
            except Exception as e:
                return jsonify({'error': f'Failed to load CSV: {str(e)}'}), 400
            
        else:
            # Use synthetic data
            X, y = generate_synthetic_data(n_samples=2000, n_features=20, random_state=42)
        
        # Clear old plots
        for file in os.listdir(app.config['STATIC_FOLDER']):
            if file.endswith('.png'):
                os.remove(os.path.join(app.config['STATIC_FOLDER'], file))
        
        # Train and evaluate models
        results = train_and_evaluate_models(X, y, dvfs_level=dvfs_level)
        
        # Create visualizations
        plot_files = create_visualizations(results, save_dir=app.config['STATIC_FOLDER'])
        
        # Save results CSV
        csv_path = save_results_csv(results, save_dir=app.config['STATIC_FOLDER'])
        
        # Store results
        latest_results = {
            'results': results,
            'plot_files': plot_files,
            'csv_file': os.path.basename(csv_path),
            'dvfs_level': dvfs_level
        }
        
        return jsonify({
            'success': True,
            'message': 'Analysis completed successfully',
            'redirect': '/results'
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/results')
def results():
    """Display results page"""
    global latest_results
    
    if latest_results is None:
        return render_template('results.html', no_results=True)
    
    return render_template('results.html', 
                         results=latest_results['results'],
                         plot_files=latest_results['plot_files'],
                         csv_file=latest_results['csv_file'],
                         dvfs_level=latest_results['dvfs_level'])


@app.route('/download_csv')
def download_csv():
    """Download results CSV"""
    global latest_results
    
    if latest_results is None:
        return "No results available", 404
    
    csv_path = os.path.join(app.config['STATIC_FOLDER'], latest_results['csv_file'])
    return send_file(csv_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)