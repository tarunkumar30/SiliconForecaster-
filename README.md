# Flask ML Prediction App

## Description

This repository contains a minimal Flask application that allows users to upload a CSV file and receive predictions from a pretrained model and scaler saved with `joblib`.

## Included Files

- `app.py` - Main Flask application. Loads the model and scaler (`joblib`) and provides an upload form and prediction results.
- `templates/index.html` - Upload form.
- `templates/results.html` - Results page displaying predictions in a table.
- `static/styles.css` - Minimal stylesheet.
- `requirements.txt` - Python dependencies.
- `Procfile` - Deployment configuration for Render / Heroku using `gunicorn`.
- `.gitignore`

## Local Development

1. Create a virtual environment and install the requirements (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Place your saved model files in a `models/` folder at the repository root:

```
models/best_model.pkl
models/scaler_model.pkl
```

3. Run the app locally:

```powershell
python app.py
```

4. Open `http://127.0.0.1:5000` in your browser and upload a CSV file.

## Authentication

This application requires user authentication. Upon accessing the app, you will be redirected to a login page. Use the following test credentials to log in:

- **Administrator**:
  - Username: `Tarun`
  - Password: `Tarun@2001`

- **Users**:
  - Username: `Mani`, Password: `Mani@2004`
  - Username: `Chandu`, Password: `Chandu@2005`
  - Username: `Soni`, Password: `Soni@2003`

After logging in, you can upload CSV files for prediction.

## Deployment Notes

- Render: Connect your GitHub repository and it will detect the `Procfile`. Set the build command to `pip install -r requirements.txt`; the app will run using `gunicorn app:app`.
- GitHub: Push your repository. If model files are large, consider using Git LFS or placing them in cloud storage and updating `app.py` to download them at startup.

## Troubleshooting

- If the scaler includes `feature_names_in_`, the uploaded CSV must contain those exact columns.
- If you encounter a `FileNotFoundError` for the model or scaler, verify the files exist at the expected paths.

## Alternate UI Theme

This repository includes an alternate dark/minimal theme with the following files:

- `templates/index_alt.html`
- `templates/results_alt.html`
- `static/styles_alt.css`

To preview the alternate theme locally, temporarily change the template names in `app.py` render calls (for example, render `index_alt.html` instead of `index.html`). This preserves the alternate look without removing the default colorful theme.
