"""
Credit Evaluation System with SHAP Explainability
Flask Backend Application - Enhanced with Explainable AI
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
from run_pipeline_with_shap import (
    load_or_generate_data,
    train_and_evaluate_models,
    create_visualizations,
    save_results_csv,
    SHAP_AVAILABLE
)

app = Flask(__name__)

# Global storage for latest results
latest_results = None

# Ensure static directory exists
os.makedirs('static', exist_ok=True)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', shap_available=SHAP_AVAILABLE)


@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """
    Run the ML analysis pipeline with SHAP explainability
    Returns JSON with status and results
    """
    global latest_results
    
    try:
        # Get parameters from request
        data = request.get_json()
        dvfs_level = float(data.get('dvfs_level', 1.0))
        compute_shap = data.get('compute_shap', True)
        
        print(f"\n{'='*70}")
        print(f"Starting Analysis")
        print(f"  DVFS Level: {dvfs_level}")
        print(f"  Compute SHAP: {compute_shap}")
        print(f"  SHAP Available: {SHAP_AVAILABLE}")
        print(f"{'='*70}\n")
        
        # Load data
        print("Loading dataset...")
        X, y, feature_names = load_or_generate_data()
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features\n")
        
        # Train and evaluate models with SHAP
        results = train_and_evaluate_models(
            X, y, feature_names, 
            dvfs_level=dvfs_level, 
            compute_shap=compute_shap and SHAP_AVAILABLE
        )
        
        # Print results summary
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY:")
        model_names = [k for k in results.keys() if k not in ['y_test', 'X_test', 'feature_names']]
        print(f"Total models trained: {len(model_names)}")
        for model_name in model_names:
            r = results[model_name]
            print(f"\n{model_name}:")
            print(f"  Accuracy: {r['accuracy']:.4f}")
            print(f"  Energy: {r['energy_estimate']:.6f} J")
            print(f"  Inference Time: {r['adjusted_inference_time']*1000:.4f} ms")
            if r.get('shap_values') is not None:
                print(f"  SHAP: ✓ Available")
            else:
                print(f"  SHAP: ✗ Not computed")
        print(f"{'='*70}\n")
        
        # Create visualizations (including SHAP)
        print("\nGenerating visualizations...")
        plot_files = create_visualizations(results, save_dir='static')
        print(f"Visualizations created: {len(plot_files)} files")
        
        # Count SHAP plots
        shap_plots = [k for k in plot_files.keys() if 'shap' in k]
        print(f"  Standard plots: {len(plot_files) - len(shap_plots)}")
        print(f"  SHAP plots: {len(shap_plots)}")
        
        # Save results to CSV
        print("\nSaving results to CSV...")
        csv_file = save_results_csv(results, save_dir='static')
        print(f"Results saved to: {csv_file}")
        
        # Store results globally (remove large objects to save memory)
        results_summary = {}
        for model_name in model_names:
            results_summary[model_name] = {
                k: v for k, v in results[model_name].items()
                if k not in ['model', 'explainer', 'shap_values']  # Remove large objects
            }
            # Add flag for SHAP availability
            results_summary[model_name]['has_shap'] = results[model_name].get('shap_values') is not None
        
        latest_results = {
            'results': results_summary,
            'plot_files': plot_files,
            'csv_file': os.path.basename(csv_file),
            'dvfs_level': dvfs_level,
            'shap_enabled': compute_shap and SHAP_AVAILABLE,
            'feature_names': results.get('feature_names', [])
        }
        
        print(f"\n{'='*70}")
        print("✅ Analysis completed successfully!")
        print(f"{'='*70}\n")
        
        return jsonify({
            'status': 'success',
            'message': 'Analysis completed successfully with SHAP explainability!',
            'shap_available': SHAP_AVAILABLE,
            'shap_computed': compute_shap and SHAP_AVAILABLE,
            'redirect': '/results'
        })
        
    except Exception as e:
        print(f"\n{'!'*70}")
        print(f"ERROR: {str(e)}")
        print(f"{'!'*70}\n")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'status': 'error',
            'message': f'Error during analysis: {str(e)}'
        }), 500


@app.route('/results')
def results():
    """Display results page with SHAP visualizations"""
    global latest_results
    
    if latest_results is None:
        return render_template('results.html', no_results=True)
    
    # Separate SHAP plots from standard plots
    standard_plots = {}
    shap_plots = {}
    
    for key, value in latest_results['plot_files'].items():
        if 'shap' in key:
            shap_plots[key] = value
        else:
            standard_plots[key] = value
    
    return render_template('results.html',
                         results=latest_results['results'],
                         plot_files=standard_plots,  # Keep plot_files for backward compatibility
                         standard_plots=standard_plots,
                         shap_plots=shap_plots,
                         csv_file=latest_results['csv_file'],
                         dvfs_level=latest_results['dvfs_level'],
                         shap_enabled=latest_results['shap_enabled'],
                         shap_available=SHAP_AVAILABLE)


@app.route('/explainability')
def explainability():
    """SHAP explainability page"""
    global latest_results
    
    if latest_results is None:
        return render_template('explainability.html', 
                             no_results=True,
                             shap_available=SHAP_AVAILABLE)
    
    # Get only SHAP plots
    shap_plots = {k: v for k, v in latest_results['plot_files'].items() if 'shap' in k}
    
    # Organize by model
    models_shap = {}
    for key, filename in shap_plots.items():
        # Extract model name from key (e.g., 'logistic_regression_summary')
        parts = key.split('_')
        if len(parts) >= 2:
            # Handle multi-word model names
            if 'neural' in key:
                model_name = 'Deep Neural Network'
            elif 'logistic' in key:
                model_name = 'Logistic Regression'
            elif 'svm' in key:
                model_name = 'SVM'
            elif 'decision' in key:
                model_name = 'Decision Tree'
            elif 'random' in key:
                model_name = 'Random Forest'
            else:
                continue
            
            if model_name not in models_shap:
                models_shap[model_name] = {}
            
            # Determine plot type
            if 'summary' in key and 'bar' not in key:
                models_shap[model_name]['summary'] = filename
            elif 'bar' in key:
                models_shap[model_name]['bar'] = filename
            elif 'force' in key:
                models_shap[model_name]['force'] = filename
            elif 'dependence' in key:
                models_shap[model_name]['dependence'] = filename
    
    return render_template('explainability.html',
                         models_shap=models_shap,
                         results=latest_results['results'],
                         shap_enabled=latest_results['shap_enabled'],
                         shap_available=SHAP_AVAILABLE,
                         feature_names=latest_results.get('feature_names', []))


@app.route('/download/<filename>')
def download_file(filename):
    """Download generated files"""
    file_path = os.path.join('static', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404


@app.route('/dvfs')
def dvfs():
    """DVFS configuration page"""
    return render_template('dvfs.html')


@app.route('/analytics')
def analytics():
    """Analytics dashboard page"""
    return render_template('analytics.html')


@app.route('/about_shap')
def about_shap():
    """Information page about SHAP"""
    return render_template('about_shap.html', shap_available=SHAP_AVAILABLE)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Credit Evaluation System - SHAP Enhanced Version")
    print("="*70)
    
    if SHAP_AVAILABLE:
        print("\n✅ SHAP is available - Explainability features enabled")
    else:
        print("\n⚠️  SHAP is not installed - Explainability features disabled")
        print("   To enable SHAP: pip install shap")
    
    print("\n🌐 Access the application at:")
    print("   → http://localhost:5000")
    print("   → http://127.0.0.1:5000")
    print("\n💡 Press CTRL+C to stop the server")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)