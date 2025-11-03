# Spanish → English PDF Translator

This small Streamlit app extracts Spanish text from PDFs and translates it to English using the Helsinki-NLP Marian MT model.

Quick start (recommended)

1. Ensure Python 3.11 is available. The project was tested with Python 3.11.x because some PyTorch wheels are not available for newer Python versions.

2. Create and activate the virtual environment (from the project root):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

4. Stop the background Streamlit process (if started in background):

```bash
kill $(cat streamlit.pid) && rm streamlit.pid
```

Notes
- `requirements.txt` references the PyTorch CPU wheel index so pip can find `torch` CPU wheels (the `--find-links` directive must appear before the `torch` line).
- If you prefer to use your system Python (3.12+), you may need to change the `torch` spec to a version that provides wheels for that Python or install PyTorch manually following instructions at https://pytorch.org/get-started/locally/.

Files of interest
- `app.py` — the Streamlit app source.
- `requirements.txt` — Python dependencies (we moved the PyTorch wheel link above the torch spec).

If you want, I can add a small systemd service, Dockerfile, or a GitHub Actions workflow to run tests or deploy.
# translator2
Enhanced from v1
