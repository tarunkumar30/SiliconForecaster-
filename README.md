# Flask ML prediction app

This repository provides a minimal Flask app that lets you upload a CSV file
and get predictions from your pre-trained model and scaler saved with joblib.

Files added:

- `app.py` - main Flask application. Loads model and scaler (joblib) and exposes an upload form and prediction results.
- `templates/index.html` - upload form
- `templates/results.html` - results page that shows predictions in a table
- `static/styles.css` - tiny stylesheet
- `requirements.txt` - Python dependencies
- `Procfile` - for Render / Heroku (uses gunicorn)
- `.gitignore`

How to use (local development)

1. Create a virtual environment and install requirements (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Place your saved model files in a `models/` folder at repo root:

```
models/best_model.pkl
models/scaler_model.pkl
```

3. Run the app locally:

```powershell
python app.py
```

4. Open http://127.0.0.1:5000 in your browser and upload a CSV.

Deployment notes

- Render: connect your GitHub repo and it will detect `Procfile`. Set build command to `pip install -r requirements.txt` and it will run using `gunicorn app:app`.
- GitHub: push your repo. If model files are large consider Git LFS or place them in cloud storage and update `app.py` to download at startup.

Troubleshooting

- If the scaler has `feature_names_in_` saved, the uploaded CSV must include those columns exactly.
- If you see a FileNotFoundError for model/scaler, ensure the files exist at the paths described above.

Alternate UI theme

- This repo includes an alternate dark/minimal theme. Files:
	- `templates/index_alt.html`
	- `templates/results_alt.html`
	- `static/styles_alt.css`

To preview the alternate theme locally, temporarily change the template names in `app.py` render calls (e.g. render `index_alt.html` instead of `index.html`). This keeps the alternate look available without removing the default colorful theme.
