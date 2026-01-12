from flask import Flask, request, render_template_string, redirect, url_for, session
import pandas as pd
import ast
import os

app = Flask(__name__)
app.secret_key = "replace-with-a-random-secret"  # needed for session

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

STATE_COLUMN = "combination"  # your csv column with tuples like ('CA','NY','TX')

VALID_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
    "DC"
}

# ---------- Helpers ----------
def csv_path_for(k: int, s: int) -> str:
    # Change this naming rule if your files are named differently
    filename = f"filtered_results_k{k}s{s}_final.csv"
    return os.path.join(DATA_DIR, filename)

def normalize_from_csv(cell, expected_len: int | None = None):
    parsed = ast.literal_eval(str(cell))

    if not isinstance(parsed, (tuple, list)):
        raise ValueError(f"Unexpected value in '{STATE_COLUMN}': {cell}")

    if expected_len is not None and len(parsed) != expected_len:
        raise ValueError(
            f"Unexpected value in '{STATE_COLUMN}': {cell} "
            f"(expected {expected_len} states, got {len(parsed)})"
        )

    return tuple(sorted(x.strip().upper() for x in parsed))


def get_transposed_matches(df: pd.DataFrame, states: list[str]) -> pd.DataFrame:
    target = tuple(sorted(states))

    matches = (
        df[df["_combo_norm"] == target]
        .drop(columns=["_combo_norm"])
        .sort_values("YEAR")
    )

    if matches.empty:
        return pd.DataFrame()

    matches = matches.drop(columns=["proportional_stable_bool"], errors="ignore")   
    
    matches = matches.rename(columns={
        "vote_balance": "Balance",
        "POPULAR VOTE: D": "Popular votes won by Democrats",
        "POPULAR VOTE: R": "Popular votes won by Republicans",
        "combination":"Combination",
        "balance_change":"Balance Change",
        "TOTAL_DEM_VOTES": "Electoral College Votes won by Democrats in current system",
        "TOTAL_REP_VOTES": "Electoral College Votes won by Republicans in current system",
        "chg_dem": "Votes Gained by Democrats under the new system for this specific state combination",
        "chg_rep": "Votes Gained by Republicans under the new system for this specific state combination",
        "WINNER": "Winning Party in the current system",
        "VOTES_TO_WIN": "Margin",
        "proportional_stable":"Is the combination proportional stable?"
    }) 

    # Transpose
    t = matches.T

    # Use YEAR values as column headers
    years = matches["YEAR"].astype(str).tolist()
    t.columns = years

    # Drop redundant YEAR row
    if "YEAR" in t.index:
        t = t.drop(index="YEAR")

    # Desired display order of rows (index)
    desired_order = [
        "Combination",
        "Is the combination proportional stable?",
        "Winning Party in the current system",
        "Margin",
        "Balance",
        "Balance Change",
        "DIFF",
        "balance_prop"
        "Popular votes won by Democrats in this combination",
        "Popular votes won by Republicans in this combination",
        "Electoral College Votes won by Democrats in current system",
        "Electoral College Votes won by Republicans in current system",
        "Votes Gained by Democrats under the new system for this specific state combination",
        "Votes Gained by Republicans under the new system for this specific state combination"
        
    ]
    
    # Keep only rows that exist, in desired order, then append any leftovers
    existing = [r for r in desired_order if r in t.index]
    

    t = t.loc[existing]
    
    return t



# ---------- Templates ----------
BASE_CSS = """

  body { font-family: Arial, sans-serif; margin: 24px; }
  .header { margin-bottom: 14px; }
  .title { font-size: 28px; font-weight: 700; }
  .subtitle { font-size: 14px; color: #555; margin-top: 4px; }
  .card { max-width: 820px; padding: 16px; border: 1px solid #ddd; border-radius: 10px; }
  input { padding: 8px; width: 100%; margin: 6px 0 12px; }
  button { padding: 10px 14px; cursor: pointer; }
  .error { color: #b00020; margin-top: 8px; }
  .meta { color: #555; font-size: 0.95em; margin-top: 6px; }
  a { color: #0b57d0; text-decoration: none; }
  table { border-collapse: collapse; margin-top: 14px; width: 100%; }
  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
  thead th { background: #f6f6f6; }
  .row { display: flex; gap: 10px; }
  .intro { max-width: 820px; line-height: 1.5; margin: 10px 0 14px 0; color: #333;}

"""

PAGE_KS = """
<!doctype html>
<html>
<head>
  <title>Voter Realignment Project</title>
   <style>
   {{ BASE_CSS }}
   </style>
</head>
<body>
  <div class="header">
    <div class="title">Voter Realignment Project</div>
    <div class="subtitle">Aditya Garg, 2025</div>
  </div>

  <p class="intro">
    The current electoral college system in the United States is critical to the outcome of the U.S. Presidential System.
    While there are widely acknowledged flaws in the system, it has been hard to change because alternative systems seem
    to benefit one party over the other, thus finding resistance from the party that stands to lose from a new system.
    We propose a novel path for allocating electoral college votes. The goal of our method is to allocate electoral
    college votes in proportion to the popular vote. We have developed a bipartisan framework for implementation.
  </p>

  <p class="intro">
    We suggest that states form small coalitions to adapt to the new system, in a way that does not change the outcome
    of the overall election. Our proposed method is neutral for both parties. It balances the effects of electoral votes
    gained and lost in each individual state through forming coalitions of states. This framework therefore allows both
    major parties to obtain electoral votes from states where they lose the popular vote without altering the overall
    result of the election. Our proposed framework is robust over various election cycles and sizes of coalitions.
    In summary, it is a bipartisan framework for electoral reform.
  </p>

  <div class="card">
    <h2>Choose analysis settings</h2>
    <form method="post">
      <label>How many election cycles would you like to consider starting with 2024?</label>
      <input name="k" type="number" min="3" max="6"
             placeholder="Enter a number from 3 to 6"
             required value="{{ k_val or '' }}" />

      <label>How many states do you want to consider in each combination?</label>
      <input name="s" type="number" min="3" max="6"
             placeholder="Enter a number from 3 to 6"
             required value="{{ s_val or '' }}" />

      <button type="submit">Continue</button>

      {% if error %}
        <div class="error">{{ error }}</div>
      {% endif %}
    </form>
  </div>
</body>
</html>
"""


PAGE_STATES = """
<!doctype html>
<html>
<head>
  <title>Voter Realignment Project</title>
  <style>
  {{ BASE_CSS }}
  </style>
</head>
<body>

  <div class="header">
    <div class="title">Voter Realignment Project</div>
    <div class="subtitle">Aditya Garg, 2025</div>
  </div>

  <p class="intro">
    The current electoral college system in the United States is critical to the outcome of the U.S. Presidential System.
    While there are widely acknowledged flaws in the system, it has been hard to change because alternative systems seem
    to benefit one party over the other, thus finding resistance from the party that stands to lose from a new system.
    We propose a novel path for allocating electoral college votes. The goal of our method is to allocate electoral
    college votes in proportion to the popular vote. We have developed a bipartisan framework for implementation.
  </p>

  <p class="intro">
    We suggest that states form small coalitions to adapt to the new system, in a way that does not change the outcome
    of the overall election. Our proposed method is neutral for both parties. It balances the effects of electoral votes
    gained and lost in each individual state through forming coalitions of states. This framework therefore allows both
    major parties to obtain electoral votes from states where they lose the popular vote without altering the overall
    result of the election. Our proposed framework is robust over various election cycles and sizes of coalitions.
    In summary, it is a bipartisan framework for electoral reform.
  </p>

  <div class="intro">
  <strong>Terminology:</strong>
  <ul>
    <li><strong>Margin</strong>: number of votes that need to change from one party to another, to change the outcome of the overall election. This applies to the overall election results and is independent of the state combination</li>
    <li><strong>Balance</strong>: difference between the total number of electoral votes won by the two major parties in the current winner-take-all system, for that particular state(s) combination</li>
    <li><strong>Balance-change</strong>: change in the margin of victory if all the states in a combination were to switch from winner-take-all to proportional. Applies to each combination separately</li>

  </ul>
</div>

  

  <div class="card">
    <h2>Enter {{ s }} states (2-letter abbreviations)</h2>

    <div class="meta">
      Loaded file: <b>{{ filename }}</b> —
      <a href="{{ url_for('index') }}">change settings</a>
    </div>

    <form method="post">
      <div class="row" style="flex-wrap: wrap;">
        {% for i in range(1, s + 1) %}
          <div style="flex: 1; min-width: 120px;">
            <label>State {{ i }}</label>
            <input
              name="state_{{ i }}"
              placeholder="CA"
              maxlength="2"
              required
              value="{{ request.form.get('state_' ~ i, '') }}"
            />
          </div>
        {% endfor %}
      </div>

      <button type="submit">Lookup</button>

      {% if error %}
        <div class="error">{{ error }}</div>
      {% endif %}
    </form>

    {% if table_html %}
      <h3>Results</h3>
      {{ table_html | safe }}
    {% endif %}
  </div>

</body>
</html>
"""



# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        k_raw = (request.form.get("k", "") or "").strip()
        s_raw = (request.form.get("s", "") or "").strip()

        # Validate integers
        try:
            k = int(k_raw)
        except ValueError:
            return render_template_string(PAGE_KS, BASE_CSS=BASE_CSS,error="Error: Please enter an integer from 3 to 6.", k_val=k_raw, s_val=s_raw)

        try:
            s = int(s_raw)
        except ValueError:
            return render_template_string(PAGE_KS, BASE_CSS=BASE_CSS,error="Error: Please enter an integer from 3 to 5.", k_val=k_raw, s_val=s_raw)

        # Range validation with “chance to enter again” (i.e., show error and keep form)
        if not (3 <= k <= 6):
            return render_template_string(
                PAGE_KS,
                BASE_CSS=BASE_CSS,
                error="Error: The number of election cycles must be between 1 and 6.",
                k_val=k_raw,
                s_val=s_raw
            )

        if not (3 <= s <= 5):
            return render_template_string(
                PAGE_KS,
                BASE_CSS=BASE_CSS,
                error="Error: The number of states per combination must be between 3 and 6.",
                k_val=k_raw,
                s_val=s_raw
            )

        
        path = csv_path_for(k, s)
        print(f"DEBUG: k={k}, s={s}, loading file -> {path}")

        if not os.path.exists(path):
            return render_template_string(
                PAGE_KS,
                BASE_CSS=BASE_CSS,
                error=f"Error: No CSV found for k={k}, S={s}. Expected file: {os.path.basename(path)}",
                k_val=k,
                s_val=s
            )

        session["k"] = k
        session["s"] = s
        return redirect(url_for("states"))

    return render_template_string(PAGE_KS, BASE_CSS=BASE_CSS,error=None, k_val=None, s_val=None)

@app.route("/states", methods=["GET", "POST"])
def states():
    k = session.get("k")
    s = session.get("s")
    if not k or not s:
        return redirect(url_for("index"))

    path = csv_path_for(k, s)
    
    print(f"DEBUG: k={k}, s={s}, loading file -> {path}")

    filename = os.path.basename(path)

    # Load each request (fine for small files)
    df = pd.read_csv(path).iloc[:, 1:]  # drop first column
    df["_combo_norm"] = df[STATE_COLUMN].apply(lambda x: normalize_from_csv(x, expected_len=s))




    error = None
    table_html = None

    if request.method == "POST":
    # 1) Read S states dynamically
        states = []
        for i in range(1, s + 1):
            val = (request.form.get(f"state_{i}", "") or "").strip().upper()
            states.append(val)

    # 2) Validate abbreviations
        bad = [x for x in states if x not in VALID_STATES]
        if bad:
            error = f"Invalid state abbreviation(s): {', '.join(bad)}. Use 2-letter codes (50 states + DC)."

    # 3) Prevent duplicate states
        elif len(set(states)) != len(states):
            error = "Please enter unique states (no duplicates)."

    # 4) Lookup results
        else:
            result = get_transposed_matches(df, states)
            if result.empty:
                error = f"No rows found for states: ({', '.join(sorted(states))})"
            else:
                table_html = result.to_html(border=0,index=True)


    #return render_template_string(PAGE_STATES, filename=filename, error=error, table_html=table_html, s=s)
    return render_template_string(PAGE_STATES,BASE_CSS=BASE_CSS,filename=filename,error=error,table_html=table_html,s=s)



if __name__ == "__main__":
    app.run(debug=True)
