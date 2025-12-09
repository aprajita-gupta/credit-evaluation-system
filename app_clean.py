"""
Credit Evaluation System - Clean Production Version
Flask web application for credit risk assessment with DVFS energy analysis
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
from run_pipeline import (
    load_or_generate_data,
    train_and_evaluate_models,
    create_visualizations,
    save_results_csv
)

app = Flask(__name__)

# Global storage for latest results
latest_results = None

# Ensure static directory exists
os.makedirs('static', exist_ok=True)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """
    Run the ML analysis pipeline
    Returns JSON with status and results
    """
    global latest_results
    
    try:
        # Get parameters from request
        data = request.get_json()
        selected_models = data.get('models', ['lr', 'svm', 'dt', 'dnn'])
        dvfs_level = float(data.get('dvfs_level', 1.0))
        
        print(f"\n{'='*60}")
        print(f"Starting Analysis with DVFS Level: {dvfs_level}")
        print(f"Selected Models: {selected_models}")
        print(f"{'='*60}\n")
        
        # Load data
        print("Loading dataset...")
        X, y = load_or_generate_data()
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features\n")
        
        # Train and evaluate models
        results = train_and_evaluate_models(X, y, dvfs_level=dvfs_level)
        
        # DEBUG: Print what's in results
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY:")
        print(f"Total keys in results: {len(results)}")
        for key in results.keys():
            if key != 'y_test':
                print(f"\n{key}:")
                print(f"  Accuracy: {results[key]['accuracy']:.4f}")
                print(f"  Energy: {results[key]['energy_estimate']:.6f} J")
                print(f"  Inference Time: {results[key]['adjusted_inference_time']*1000:.4f} ms")
        print(f"{'='*60}\n")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        plot_files = create_visualizations(results, save_dir='static')
        print("Visualizations created successfully!")
        
        # Save results to CSV
        print("Saving results to CSV...")
        csv_file = save_results_csv(results, save_dir='static')
        print(f"Results saved to: {csv_file}")
        
        # Store results globally
        latest_results = {
            'results': results,
            'plot_files': plot_files,
            'csv_file': csv_file,
            'dvfs_level': dvfs_level
        }
        
        print(f"\n{'='*60}")
        print("Analysis completed successfully!")
        print(f"{'='*60}\n")
        
        return jsonify({
            'status': 'success',
            'message': 'Analysis completed successfully!',
            'redirect': '/results'
        })
        
    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"ERROR: {str(e)}")
        print(f"{'!'*60}\n")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'status': 'error',
            'message': f'Error during analysis: {str(e)}'
        }), 500


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


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Credit Evaluation System - Starting Server")
    print("="*60)
    print("\n📍 Access the application at:")
    print("   → http://localhost:5000")
    print("   → http://127.0.0.1:5000")
    print("\n💡 Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    # Run with debug=False to prevent file watching issues
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)