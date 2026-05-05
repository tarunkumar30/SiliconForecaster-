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
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import io
import base64

# Use Agg backend to prevent display issues in headless environments
matplotlib.use('Agg')

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


def flash_invalid_file_error(error=None):
    """Store a generic popup message for invalid or malformed CSV uploads.

    The actual exception is logged server-side for debugging, but users only
    see a simple and friendly message.
    """
    if error is not None:
        app.logger.exception('Invalid or malformed CSV upload')
    set_popup_message('You have chosen the wrong file. Please select a valid file to proceed.')


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


def validate_and_predict(features_df, model, scaler):
    """
    Enhanced validation to handle extreme outliers and zeros.
    Prevents infinity and overflow errors by clipping extreme values.
    """
    predictions = []
    feat_names = scaler.feature_names_in_
    X_to_predict = features_df[feat_names].values

    for idx in range(len(X_to_predict)):
        row = X_to_predict[idx]

        # 1. Check for all-zero rows
        if np.all(np.abs(row) < 1e-6):
            predictions.append(1)  # Force FAIL
            continue

        try:
            # 2. Transform the row
            scaled_row = scaler.transform(row.reshape(1, -1))

            # 3. CLIP extreme values to prevent float32 overflow in models
            # This handles the 'infinity' error by capping values at reasonable limits
            scaled_row = np.clip(scaled_row, -1e6, 1e6)

            pred = model.predict(scaled_row)[0]
            predictions.append(pred)
        except Exception as e:
            # If scaling or prediction still fails due to numerical issues
            print(f"Error processing row {idx}: {e}")
            predictions.append(1)  # Default to fail on error

    return np.array(predictions)


def generate_charts(predictions):
    """Generate bar and pie charts for pass/fail predictions.
    
    Returns base64 encoded strings for the charts.
    """
    # Count pass/fail predictions
    # Map -1 to 0 (Pass) and 1 to 1 (Fail) for consistency
    predictions_binary = np.where(predictions == -1, 0, 1)
    pred_counts = pd.Series(predictions_binary).value_counts().sort_index()
    
    # Map 0/1 to Pass/Fail labels
    labels = ['Pass' if x == 0 else 'Fail' for x in pred_counts.index]
    values = pred_counts.values
    colors = ["#4ade80", "#f87171"]  # Green for Pass, Red for Fail
    
    # Create figure with subplots for bar and pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1e1e1e')
    
    # Bar chart
    ax1.bar(labels, values, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Count', color='white', fontsize=12, fontweight='bold')
    ax1.set_title('Pass/Fail Distribution (Bar Chart)', color='white', fontsize=14, fontweight='bold')
    ax1.set_facecolor('#2d2d2d')
    ax1.tick_params(axis='both', colors='white')
    ax1.grid(axis='y', alpha=0.3, color='white')
    
    # Add value labels on bars
    for i, v in enumerate(values):
        ax1.text(i, v + 0.5, str(v), ha='center', color='white', fontweight='bold')
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'color': 'white', 'fontsize': 11, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax2.set_title('Pass/Fail Distribution (Pie Chart)', color='white', fontsize=14, fontweight='bold')
    ax2.set_facecolor('#2d2d2d')
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', facecolor='#1e1e1e', edgecolor='none', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"


def set_popup_message(message):
    """Store a one-time popup message in session for display on the next page."""
    session['popup_message'] = message


@app.route('/')
def index():
    """Render the upload page."""
    # Require login: if no user in session, send to login page
    if not session.get('user'):
        return redirect(url_for('login'))
    popup_message = session.pop('popup_message', None)
    return render_template('index_alt.html', popup_message=popup_message)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Simple login page. On success redirect to the Predict page (index)."""
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        # Define credentials
        admin_credentials = {'Tarun': 'Tarun@2001'}
        employee_credentials = {
            'Mani': 'Mani@2004',
            'Chandu': 'Chandu@2005',
            'Soni': 'Soni@2003'
        }

        if username in admin_credentials and password == admin_credentials[username]:
            session['user'] = 'admin'
            set_popup_message('You have logged in successfully')
            return redirect(url_for('index'))
        elif username in employee_credentials and password == employee_credentials[username]:
            session['user'] = username
            set_popup_message('Your login credentials will be sent to admin')
            return redirect(url_for('index'))
        else:
            set_popup_message('Invalid username or password. Please contact admin for login credentials')
            return redirect(url_for('login'))

    popup_message = session.pop('popup_message', None)
    return render_template('login.html', popup_message=popup_message)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle CSV upload, run predictions, and render results."""
    # Check that the file part exists
    if 'file' not in request.files:
        set_popup_message('You have chosen the wrong file. Please select a valid file to proceed.')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        set_popup_message('You have chosen the wrong file. Please select a valid file to proceed.')
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
            flash_invalid_file_error(e)
            return redirect(url_for('index'))

        # Load scaler and model (deferred until upload to allow app to start without models)
        try:
            scaler = find_and_load(SCALER_PATHS, 'scaler')
        except FileNotFoundError as e:
            app.logger.exception('Model or scaler load failed')
            set_popup_message('Server configuration error. Please contact support.')
            return redirect(url_for('index'))

        try:
            model = find_and_load(MODEL_PATHS, 'model')
        except FileNotFoundError as e:
            app.logger.exception('Model or scaler load failed')
            set_popup_message('Server configuration error. Please contact support.')
            return redirect(url_for('index'))

        # Prepare features
        try:
            X_raw, feature_cols = prepare_features(df, scaler)
        except Exception as e:
            flash_invalid_file_error(e)
            return redirect(url_for('index'))

        # Use the enhanced validation function to make predictions
        try:
            preds = validate_and_predict(df, model, scaler)
        except Exception as e:
            flash_invalid_file_error(e)
            return redirect(url_for('index'))

        # Optionally add probabilities for binary classifiers
        proba_col = None
        try:
            if hasattr(model, 'predict_proba'):
                # Process probabilities row by row with the same clipping strategy
                probabilities = []
                feat_names = scaler.feature_names_in_
                X_to_predict = df[feat_names].values
                
                for idx in range(len(X_to_predict)):
                    row = X_to_predict[idx]
                    
                    # Check for all-zero rows
                    if np.all(np.abs(row) < 1e-6):
                        probabilities.append(0.0)  # Low probability for fail cases
                        continue
                    
                    try:
                        # Transform and clip
                        scaled_row = scaler.transform(row.reshape(1, -1))
                        scaled_row = np.clip(scaled_row, -1e6, 1e6)
                        
                        proba = model.predict_proba(scaled_row)[0]
                        # If binary, keep probability for positive class (index 1)
                        if len(proba) == 2:
                            probabilities.append(proba[1])
                        else:
                            # For multiclass, keep max probability
                            probabilities.append(proba.max())
                    except Exception:
                        probabilities.append(0.0)
                
                proba_col = np.array(probabilities)
        except Exception:
            proba_col = None

        # Build results DataFrame for display
        results = df.copy()
        
        # Map predictions to Pass/Fail labels
        # Assuming: -1 = Pass, 1 = Fail (based on your training code)
        results['prediction'] = preds
        results['prediction_label'] = results['prediction'].apply(
            lambda x: 'Pass' if x == -1 else 'Fail'
        )
        
        if proba_col is not None:
            results['prediction_probability'] = np.round(proba_col, 4)

        # Generate charts for pass/fail distribution
        chart_image = generate_charts(preds)

        # Convert to HTML table in template safely
        table_html = results.to_html(classes=['table', 'table-striped', 'table-bordered'], index=False, justify='center')

        return render_template('results_alt.html', table_html=table_html, filename=filename, chart_image=chart_image)

    else:
        set_popup_message('You have chosen the wrong file. Please select a valid file to proceed.')
        return redirect(url_for('index'))


if __name__ == '__main__':
    # For local development only. Production should use Gunicorn or similar.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)