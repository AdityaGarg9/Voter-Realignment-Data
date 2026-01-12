import os
import ast
import base64
import json
from pathlib import Path
from typing import Optional, List

import pandas as pd
from flask import Flask, request, render_template_string, redirect, url_for, session

from google.cloud import storage
from google.oauth2 import service_account

# ----------------------------
# App / Security Configuration
# ----------------------------
app = Flask(__name__)



GCS_BUCKET = os.environ.get("GCS_BUCKET")
GCP_SA_KEY_B64 = os.environ.get("GCP_SA_KEY_B64")

if not GCS_BUCKET or not GCP_SA_KEY_B64:
    raise RuntimeError("GCS_BUCKET and GCP_SA_KEY_B64 must be set in Render.")

CACHE_DIR = Path("/tmp/data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STATE_COLUMN = "combination"

VALID_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
    "DC"
}

# ----------------------------
# Google Cloud Storage Helpers
# ----------------------------
_storage_client = None

def get_storage_client():
    global _storage_client
    if _storage_client:
        return _storage_client

    sa_info = json.loads(base64.b64decode(GCP_SA_KEY_B64))
    creds = service_account.Credentials.from_service_account_info(sa_info)
    _storage_client = storage.Client(credentials=creds)
    return _storage_client

def ensure_local_csv(k: int, s: int) -> str:
    filename = f"filtered_results_k{k}s{s}.csv"
    local_path = CACHE_DIR / filename

    if local_path.exists():
        return str(local_path)

    client = get_storage_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(filename)

    if not blob.exists():
        raise FileNotFoundError(f"{filename} not found in GCS bucket.")

    blob.download_to_filename(str(local_path))
    return str(local_path)

# ----------------------------
# Data Helpers
# ----------------------------
def parse_combo(cell) -> Optional[tuple]:
    try:
        parsed = ast.literal_eval(str(cell))
        if isinstance(parsed, (tuple, list)):
            return tuple(sorted(str(x).upper() for x in parsed))
    except Exception:
        pass
    return None

def detect_year_col(columns):
    if "YEAR" in columns:
        return "YEAR"
    for c in columns:
        if c.lower() == "year":
            return c
    return None

def get_transposed_matches(df: pd.DataFrame, states: list[str], s: int) -> pd.DataFrame:
    target = tuple(sorted(states))
    df["_combo_norm"] = df[STATE_COLUMN].apply(parse_combo)
    matches = df[df["_combo_norm"] == target].drop(columns="_combo_norm")

    if matches.empty:
        return pd.DataFrame()

    matches = matches.drop(columns=["proportional_stable_bool", "proportional_stable_boolean"], errors="ignore")

    matches = matches.rename(columns={
        "vote_balance": "Balance",
        "POPULAR VOTE: D": "Popular votes won by Democrats in this combination",
        "POPULAR VOTE:D": "Popular votes won by Democrats in this combination",
        "POPULAR VOTE: R": "Popular votes won by Republicans in this combination",
        "POPULAR VOTE:R": "Popular votes won by Republicans in this combination",
        "combination": "Combination",
        "TOTAL_DEM_VOTES": "Electoral College Votes won by Democrats in current system",
        "TOTAL_REP_VOTES": "Electoral College Votes won by Republicans in current system",
        "chg_dem": "Votes Gained by Democrats under the new system for this specific state combination",
        "chg_rep": "Votes Gained by Republicans under the new system for this specific state combination",
        "balance_prop": "Change in votes from one party to another, new system vs. current system",
        "DIFF": "Margin of victory in the current system",
        "WINNER": "Winning Party in the current system",
        "VOTES_TO_WIN": "Margin"
    })

    year_col = detect_year_col(matches.columns)
    if year_col:
        matches = matches.sort_values(year_col)

    t = matches.T

    if year_col:
        t.columns = matches[year_col].astype(str).tolist()
        t = t.drop(index=year_col)

    desired_order = [
        "Combination",
        "Winning Party in the current system",
        "Margin",
        "Balance",
        "Popular votes won by Democrats in this combination",
        "Popular votes won by Republicans in this combination",
        "Electoral College Votes won by Democrats in current system",
        "Electoral College Votes won by Republicans in current system",
        "Votes Gained by Democrats under the new system for this specific state combination",
        "Votes Gained by Republicans under the new system for this specific state combination",
        "Change in votes from one party to another, new system vs. current system",
        "Margin of victory in the current system"
    ]

    t = t.loc[[r for r in desired_order if r in t.index]]
    return t

# ----------------------------
# Templates
# ----------------------------
BASE_CSS = """
body { font-family: Arial, sans-serif; margin: 24px; }
.header { margin-bottom: 12px; }
.title { font-size: 28px; font-weight: 700; }
.subtitle { font-size: 14px; color: #555; margin-top: 4px; }
.card { max-width: 900px; padding: 16px; border: 1px solid #ddd; border-radius: 10px; }
input { padding: 8px; width: 100%; margin: 6px 0 12px; }
button { padding: 10px 14px; cursor: pointer; }
.error { color: #b00020; margin-top: 8px; }
table { border-collapse: collapse; margin-top: 14px; width: 100%; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
.row { display: flex; gap: 10px; flex-wrap: wrap; }
.cell { flex: 1; min-width: 140px; }
.intro { max-width: 900px; line-height: 1.5; margin: 10px 0 14px 0; }
"""

PAGE_KS = f"""
<!doctype html>
<html>
<head>
  <title>Voter Realignment Project</title>
  <style>{BASE_CSS}</style>
</head>
<body>
<div class="header">
  <div class="title">Voter Realignment Project</div>
  <div class="subtitle">Aditya Garg, 2025</div>
</div>

<p class="intro">
The current electoral college system in the United States is critical to the outcome of the U.S. Presidential System.
We propose a novel bipartisan framework for allocating electoral college votes proportionally.
</p>

<div class="card">
<form method="post">
<label>How many election cycles starting with 2020?</label>
<input name="k" type="number" min="1" max="6" required>

<label>How many states per combination?</label>
<input name="s" type="number" min="3" max="6" required>

<button type="submit">Continue</button>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
</form>
</div>
</body>
</html>
"""

PAGE_STATES = f"""
<!doctype html>
<html>
<head>
<title>Voter Realignment Project</title>
<style>{BASE_CSS}</style>
</head>
<body>

<div class="header">
  <div class="title">Voter Realignment Project</div>
  <div class="subtitle">Aditya Garg, 2025</div>
</div>

<div class="card">
<h2>Enter {{ s }} states</h2>
<form method="post">
<div class="row">
{% for i in range(1, s+1) %}
  <div class="cell">
    <label>State {{ i }}</label>
    <input name="state_{{ i }}" maxlength="2" required>
  </div>
{% endfor %}
</div>

<button type="submit">Lookup</button>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
</form>

{% if table_html %}
<h3>Results</h3>
{{ table_html | safe }}
{% endif %}
</div>
</body>
</html>
"""

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        k = int(request.form["k"])
        s = int(request.form["s"])
        session["k"] = k
        session["s"] = s
        return redirect(url_for("states"))
    return render_template_string(PAGE_KS)

@app.route("/states", methods=["GET", "POST"])
def states():
    k = session.get("k")
    s = session.get("s")
    error = None
    table_html = None

    if request.method == "POST":
        states_in = [request.form[f"state_{i}"].upper() for i in range(1, s+1)]

        if len(set(states_in)) != s:
            error = "Duplicate states entered."
        elif any(st not in VALID_STATES for st in states_in):
            error = "Invalid state abbreviation."
        else:
            try:
                df = pd.read_csv(ensure_local_csv(k, s))
                result = get_transposed_matches(df, states_in, s)
                if result.empty:
                    error = "No matching rows found."
                else:
                    table_html = result.to_html(index=True, escape=True)
            except Exception as e:
                error = str(e)

    return render_template_string(PAGE_STATES, s=s, error=error, table_html=table_html)

# ----------------------------
# Entrypoint (Render uses Gunicorn)
# ----------------------------
if __name__ == "__main__":
    app.run()
