from datetime import datetime, timezone

from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "marketlens",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

