# FitPulse Analytics

A Streamlit-based fitness app analytics prototype for exploratory analysis, predictive model comparison, and responsible-use review.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements.txt
python -m streamlit run app.py
```

If your shell has an Anaconda `base` environment active, deactivate it before activating this project's `.venv`. Use `python -m streamlit` to ensure Streamlit runs from the active Python environment.

## Features
- Executive summary and exploratory charts
- Data quality checks and non-destructive extreme-record sensitivity analysis
- Held-out comparison of mean baseline, linear regression, and random forest
- Group error diagnostics by gender, location, and age band
- Responsible analytics guidance and business considerations

The included dataset is a sample input. Confirm its provenance, field definitions, and permissions before external or operational use. Outputs are exploratory, not clinical advice.
