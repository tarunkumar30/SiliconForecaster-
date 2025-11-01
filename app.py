"""
Flask app to upload a CSV, load a trained model + scaler (joblib .pkl files),
make predictions for every row, and display results in an HTML table.

Place your 'best_model.pkl' and 'scaler_model.pkl' inside a folder named
`models/` at the repository root, or next to this file. The app attempts
both locations.

Comments throughout explain each step.
"""
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib

# Configuration
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
MODEL_PATHS = [
    os.path.join(APP_ROOT, 'models', 'best_model.pkl'),
    os.path.join(APP_ROOT, 'best_model.pkl'),
]
SCALER_PATHS = [
    os.path.join(APP_ROOT, 'models', 'scaler_model.pkl'),
    os.path.join(APP_ROOT, 'scaler_model.pkl'),
]
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.environ.get('FLASK_SECRET', 'change-me-for-prod')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def find_and_load(path_list, friendly_name):
    """Try a list of possible paths and return the loaded object.

    Raises FileNotFoundError if none of the paths exist.
    """
    for p in path_list:
        if os.path.exists(p):
            print(f"Loading {friendly_name} from: {p}")
            return joblib.load(p)
    raise FileNotFoundError(f"Could not find {friendly_name}. Tried: {path_list}")


def prepare_features(df, scaler):
    """Determine the feature columns to send to scaler/model.

    If the scaler exposes `feature_names_in_`, use that order. Otherwise
    we default to using all numeric columns present in the uploaded CSV.
    """
    # If scaler was a scikit-learn transformer, it may have feature_names_in_
    if hasattr(scaler, 'feature_names_in_'):
        features = list(scaler.feature_names_in_)
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Uploaded CSV is missing required feature columns: {missing}")
        X = df[features]
    else:
        # Fall back to numeric columns only
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if not features:
            raise ValueError("No numeric features found in the uploaded CSV.")
        X = df[features]
    return X, features


@app.route('/')
def index():
    """Render the upload page."""
    return render_template('index_alt.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle CSV upload, run predictions, and render results."""
    # Check that the file part exists
    if 'file' not in request.files:
        flash('No file part in request')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        try:
            # Read CSV into DataFrame (attempt to infer separators and encodings)
            df = pd.read_csv(upload_path)
        except Exception as e:
            flash(f'Could not read CSV: {e}')
            return redirect(url_for('index'))

        # Load scaler and model (deferred until upload to allow app to start without models)
        try:
            scaler = find_and_load(SCALER_PATHS, 'scaler')
        except FileNotFoundError as e:
            flash(str(e))
            return redirect(url_for('index'))

        try:
            model = find_and_load(MODEL_PATHS, 'model')
        except FileNotFoundError as e:
            flash(str(e))
            return redirect(url_for('index'))

        # Prepare features and run scaler/model
        try:
            X_raw, feature_cols = prepare_features(df, scaler)
        except Exception as e:
            flash(str(e))
            return redirect(url_for('index'))

        try:
            X_scaled = scaler.transform(X_raw)
        except Exception as e:
            flash(f'Error while transforming features with scaler: {e}')
            return redirect(url_for('index'))

        try:
            preds = model.predict(X_scaled)
        except Exception as e:
            flash(f'Error while running model.predict: {e}')
            return redirect(url_for('index'))

        # Optionally add probabilities for binary classifiers
        proba_col = None
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
                # If binary, keep probability for positive class
                if proba.shape[1] == 2:
                    proba_col = proba[:, 1]
                else:
                    # For multiclass, keep max probability
                    proba_col = proba.max(axis=1)
        except Exception:
            proba_col = None

        # Build results DataFrame for display
        results = df.copy()
        results['prediction'] = preds
        if proba_col is not None:
            results['prediction_probability'] = np.round(proba_col, 4)

        # Convert to HTML table in template safely
        table_html = results.to_html(classes=['table', 'table-striped', 'table-bordered'], index=False, justify='center')

        return render_template('results_alt.html', table_html=table_html, filename=filename)

    else:
        flash('Allowed file types: csv')
        return redirect(url_for('index'))


if __name__ == '__main__':
    # For local development only. Production should use Gunicorn or similar.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
