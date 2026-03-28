# MarketLens (Vercel UI Version)

This project now runs as a Vercel-compatible web app with:

- Static frontend UI (`index.html`, `styles.css`, `app.js`)
- Python API endpoints (`api/analyze.py`, `api/health.py`)
- Existing analysis engine reused from `stock.py`

## Local Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set optional API keys:

```bash
NEWS_API_KEY=your_key
GNEWS_API_KEY=your_key
```

4. Run locally with Vercel CLI or any static server + Python function emulation.

## Deploy to Vercel

1. Push this repository to GitHub.
2. Import it into Vercel.
3. Add environment variables in the Vercel project settings:
   - `NEWS_API_KEY`
   - `GNEWS_API_KEY`
4. Deploy. The app serves at `/` and API at:
   - `/api/analyze?ticker=GOOG&period=6mo`
   - `/api/health`

## Notes

- Streamlit UI is no longer required for the deployed app.
- `stock.py` remains the analysis core used by the API layer.

