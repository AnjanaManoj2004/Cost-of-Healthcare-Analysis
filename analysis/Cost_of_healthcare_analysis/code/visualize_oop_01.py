# visualize_oop.py
import argparse
import os
import re
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt

# ---------- little helpers ----------

def find_one(patterns, columns, required=True, default=None):
    """
    Return the first column whose name matches any regex in `patterns` (case-insensitive).
    """
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in columns:
            if rx.search(c):
                return c
    if required:
        raise KeyError(f"Could not find any column matching: {patterns} in {list(columns)}")
    return default

def yearify(s):
    """
    Convert a string/number year column to int (e.g., '2003', '2003.0', 'FY 2003' -> 2003).
    Non-convertible rows will be dropped.
    """
    return (
        pd.to_numeric(
            pd.Series(s).astype(str).str.extract(r"(\d{4})", expand=False),
            errors="coerce"
        ).astype("Int64")
    )

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ---------- visual 1: SEIFA quintiles (Table 8) ----------

def plot_seifa_trend(path, outdir):
    print(f"[SEIFA] Reading: {path}")
    df = pd.read_csv(path)

    # likely columns
    year_col = find_one([r"^year$", r"service[_\s]*year", r"\bdate\b"], df.columns)
    state_col = find_one([r"^state$", r"jurisdiction"], df.columns)
    seifa_col = find_one([r"seifa.*quintile", r"quintile"], df.columns)
    # prefer actual over CPI-adjusted for default trend
    value_col = find_one(
        [r"actual.*(cost|price)", r"\bactual\b", r"oop.*actual", r"value|amount|price"],
        df.columns
    )

    # tidy up
    df[year_col] = yearify(df[year_col])
    df = df.dropna(subset=[year_col, seifa_col, value_col])
    df = df[df[seifa_col].astype(str).str.contains(r"quintile|Q? ?[1-5]", flags=re.I, na=False)]

    # wide by quintile across Australia only (if “Aus” present); otherwise use all states combined mean
    if "Aus" in df[state_col].astype(str).unique():
        dfa = df[df[state_col].astype(str).str.fullmatch(r"Aus", case=False, na=False)].copy()
    else:
        # fallback: average across states
        dfa = df.copy()

    # clean quintile labels to Q1..Q5
    qmap = {
        "quintile 1": "Q1",
        "quintile 2": "Q2",
        "quintile 3": "Q3",
        "quintile 4": "Q4",
        "quintile 5": "Q5",
    }
    def standardize_q(v):
        s = str(v).strip().lower()
        for k, lab in qmap.items():
            if k in s:
                return lab
        m = re.search(r"\b([1-5])\b", s)
        return f"Q{m.group(1)}" if m else v

    dfa["Quintile"] = dfa[seifa_col].map(standardize_q)
    grp = dfa.groupby([year_col, "Quintile"], as_index=False)[value_col].mean()

    # plot
    ensure_dir(outdir)
    plt.figure(figsize=(10, 6))
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        sub = grp[grp["Quintile"] == q]
        if not sub.empty:
            plt.plot(sub[year_col], sub[value_col], label=q)
    plt.title("Out-of-pocket cost per service by SEIFA quintile (actual prices)")
    plt.xlabel("Year")
    plt.ylabel("Cost ($)")
    plt.legend(title="SEIFA")
    plt.grid(True, alpha=0.3)
    fp = os.path.join(outdir, "seifa_quintiles_trend.png")
    plt.tight_layout()
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"[SEIFA] Saved: {fp}")

# ---------- visual 2: Remoteness (Table 9) ----------

def plot_seifa_gap(path, outdir):
    df = pd.read_csv(path)
    year_col = find_one([r"^year$", r"service[_\s]*year", r"\bdate\b"], df.columns)
    state_col = find_one([r"^state$", r"jurisdiction"], df.columns)
    seifa_col = find_one([r"quintile"], df.columns)
    value_col = find_one([r"actual"], df.columns)

    df[year_col] = yearify(df[year_col])
    dfa = df[df[state_col].str.fullmatch("Aus", case=False, na=False)]
    dfa["Quintile"] = dfa[seifa_col].str.extract(r"(\d)").astype(int)

    # Compute Q5 - Q1 gap
    gap = dfa.pivot_table(index=year_col, columns="Quintile", values=value_col, aggfunc="mean")
    gap["Gap_Q5_minus_Q1"] = gap[5] - gap[1]

    plt.figure(figsize=(8, 5))
    plt.plot(gap.index, gap["Gap_Q5_minus_Q1"], marker="o")
    plt.title("Gap in OOP costs between SEIFA Q5 and Q1")
    plt.xlabel("Year")
    plt.ylabel("Gap ($)")
    plt.grid(True, alpha=0.3)
    fp = os.path.join(outdir, "seifa_gap_q5_q1.png")
    plt.tight_layout()
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"[SEIFA GAP] Saved: {fp}")

def plot_remoteness_bar(path, outdir):
    df = pd.read_csv(path)
    year_col = find_one([r"^year$", r"service[_\s]*year", r"\bdate\b"], df.columns)
    area_col = area_col = find_one([r"remoteness", r"aria", r"ra\s*category", r"area"], df.columns)
    value_col = find_one([r"actual"], df.columns)

    df[year_col] = yearify(df[year_col])
    latest_year = int(df[year_col].dropna().max())
    latest = df[df[year_col] == latest_year].groupby(area_col)[value_col].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.barh(latest[area_col], latest[value_col])
    plt.title(f"OOP by remoteness area ({latest_year})")
    plt.xlabel("Cost ($)")
    plt.ylabel("Remoteness Area")
    fp = os.path.join(outdir, f"remoteness_bar_{latest_year}.png")
    plt.tight_layout()
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"[REMOTENESS BAR] Saved: {fp}")

def plot_remoteness_slope(path, outdir):
    df = pd.read_csv(path)
    year_col = find_one([r"^year$", r"service[_\s]*year", r"\bdate\b"], df.columns)
    area_col = find_one([r"remoteness"], df.columns)
    value_col = find_one([r"actual"], df.columns)

    df[year_col] = yearify(df[year_col])
    first, last = df[year_col].min(), df[year_col].max()
    subset = df[df[year_col].isin([first, last])]
    grp = subset.groupby([year_col, area_col])[value_col].mean().reset_index()

    plt.figure(figsize=(8, 5))
    for area in grp[area_col].unique():
        vals = grp[grp[area_col] == area]
        plt.plot(vals[year_col], vals[value_col], marker="o", label=area)
    plt.title(f"OOP by remoteness: {first} vs {last}")
    plt.xlabel("Year")
    plt.ylabel("Cost ($)")
    plt.legend()
    fp = os.path.join(outdir, "remoteness_slope.png")
    plt.tight_layout()
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"[REMOTENESS SLOPE] Saved: {fp}")

def plot_age_gap(path, outdir):
    df = pd.read_csv(path)
    date_col = find_one([r"year", r"date"], df.columns)
    df["Year"] = yearify(df[date_col])

    # Assume wide format with age columns
    age_cols = [c for c in df.columns if any(x in c for x in ["0", "16", "65"])]
    tidy = df.melt("Year", value_vars=age_cols, var_name="Age", value_name="Value")
    yearly = tidy.groupby(["Year", "Age"], as_index=False)["Value"].mean()

    gap = yearly.pivot(index="Year", columns="Age", values="Value")
    if "65+" in gap.columns and "16-64" in gap.columns:
        gap["Gap"] = gap["65+"] - gap["16-64"]

        plt.figure(figsize=(8, 5))
        plt.plot(gap.index, gap["Gap"], marker="o")
        plt.title("Gap in OOP between 65+ and 16–64")
        plt.xlabel("Year")
        plt.ylabel("Gap ($)")
        fp = os.path.join(outdir, "age_gap_trend.png")
        plt.tight_layout()
        plt.savefig(fp, dpi=150)
        plt.close()
        print(f"[AGE GAP] Saved: {fp}")

def plot_state_heatmap(path, outdir):
    import seaborn as sns
    df = pd.read_csv(path)
    year_col = find_one([r"^year$", r"service[_\s]*year", r"\bdate\b"], df.columns)
    region_col = find_one([r"region", r"state"], df.columns)
    value_cols = [c for c in df.columns if c.lower().startswith("actual")]

    df[year_col] = yearify(df[year_col])
    df["mean_actual"] = df[value_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    pivot = df.pivot(index=region_col, columns=year_col, values="mean_actual")

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="YlOrRd", annot=False, cbar_kws={"label": "Cost ($)"})
    plt.title("OOP costs by state and year (mean of quintiles)")
    plt.xlabel("Year")
    plt.ylabel("State/Territory")
    fp = os.path.join(outdir, "state_heatmap.png")
    plt.tight_layout()
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"[STATE HEATMAP] Saved: {fp}")


def plot_remoteness_trend(path, outdir):
    print(f"[REMOTENESS] Reading: {path}")
    df = pd.read_csv(path)

    year_col = find_one([r"^year$", r"service[_\s]*year", r"\bdate\b"], df.columns)
    area_col = find_one([r"remoteness", r"aria", r"ra\s*category", r"area"], df.columns)
    value_col = find_one(
        [r"actual.*(cost|price)", r"\bactual\b", r"oop.*actual", r"value|amount|price"],
        df.columns
    )

    df[year_col] = yearify(df[year_col])
    df = df.dropna(subset=[year_col, area_col, value_col])

    # standardize area labels a bit
    def norm_area(s):
        t = str(s).strip().lower()
        t = t.replace("veryremote", "very remote").replace("outerregional", "outer regional")
        t = t.replace("innerregional", "inner regional").replace("majorcities", "major cities")
        return t.title()

    df["Area"] = df[area_col].map(norm_area)
    grp = df.groupby([year_col, "Area"], as_index=False)[value_col].mean()

    ensure_dir(outdir)
    plt.figure(figsize=(10, 6))
    for a in ["Major Cities", "Inner Regional", "Outer Regional", "Remote", "Very Remote"]:
        sub = grp[grp["Area"] == a]
        if not sub.empty:
            plt.plot(sub[year_col], sub[value_col], label=a)
    plt.title("Out-of-pocket cost per service by remoteness (actual prices)")
    plt.xlabel("Year")
    plt.ylabel("Cost ($)")
    plt.legend(title="Area")
    plt.grid(True, alpha=0.3)
    fp = os.path.join(outdir, "remoteness_trend.png")
    plt.tight_layout()
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"[REMOTENESS] Saved: {fp}")

# ---------- visual 3: Age (monthly -> yearly mean) ----------

def plot_age_trend(path, outdir):
    print(f"[AGE] Reading: {path}")
    df = pd.read_csv(path)

    # try to detect a date column and age group/value columns
    date_col = find_one([r"^date$", r"month", r"year", r"period"], df.columns)
    # normalize to datetime, then year
    if re.search(r"year", date_col, re.I):
        df["Year"] = yearify(df[date_col])
    else:
        df["Year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year

    # find value columns for age groups (flexible)
    # common labels: 0–15, 16–64, 65+ or similar
    age_cols = []
    pats = [
        r"0.?15", r"0.?14", r"^0.*15", r"\b0[-–]15\b",
        r"16.?64", r"\b16[-–]64\b",
        r"65\+?", r"\b65\s*\+\b",
    ]
    for c in df.columns:
        if any(re.search(p, c, re.I) for p in pats):
            age_cols.append(c)

    # if the file is already long/tidy with columns AgeGroup & Value, handle that:
    if "AgeGroup" in df.columns and find_one([r"value|amount|price|cost|oop"], df.columns, required=False):
        val_col = find_one([r"value|amount|price|cost|oop"], df.columns)
        tidy = df.rename(columns={"AgeGroup": "Age", val_col: "Value"})
    else:
        # build tidy from wide age columns
        if not age_cols:
            # fallback: try three generic columns
            guess = [c for c in df.columns if re.search(r"0|15|16|64|65", c)]
            age_cols = guess[:3]
        keep = ["Year"] + age_cols
        tidy = df[keep].melt("Year", var_name="Age", value_name="Value")

    tidy = tidy.dropna(subset=["Year", "Value"])
    yearly = tidy.groupby(["Year", "Age"], as_index=False)["Value"].mean()

    ensure_dir(outdir)
    plt.figure(figsize=(10, 6))
    for age in sorted(yearly["Age"].unique(), key=lambda s: str(s)):
        sub = yearly[yearly["Age"] == age]
        plt.plot(sub["Year"], sub["Value"], label=str(age))
    plt.title("Out-of-pocket cost per service by age group (annual average from monthly)")
    plt.xlabel("Year")
    plt.ylabel("Cost ($)")
    plt.legend(title="Age group")
    plt.grid(True, alpha=0.3)
    fp = os.path.join(outdir, "age_groups_trend.png")
    plt.tight_layout()
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"[AGE] Saved: {fp}")

# ---------- visual 4: State comparison (latest year) ----------

def plot_state_latest_bar(path, outdir):
    print(f"[STATE BAR] Reading: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # keep original case, just trim

    # Try the tidy schema first: (Year, State, value column like "actual ...")
    try:
        year_col  = find_one([r"^year$", r"service\s*year", r"\bdate\b"], df.columns)
        state_col = find_one([r"^state$", r"jurisdiction", r"^region$"], df.columns, required=False)
        value_col = find_one(
            [r"actual.*(cost|price)", r"\bactual\b", r"oop.*actual", r"value|amount|price"],
            df.columns,
            required=False,
        )
    except KeyError:
        year_col = None
        state_col = None
        value_col = None

    # If the tidy schema isn’t present, fall back to the wide schema used in your file:
    # Columns like: Year, Region, actual1..actual5 (and maybe adjusted1..adjusted5)
    wide_actual_cols = [c for c in df.columns if re.fullmatch(r"actual[1-5]", c.strip().lower())]
    if (year_col is None or state_col is None or value_col is None) and wide_actual_cols:
        # Normalize to lower for matching, but keep originals for display
        lower_map = {c: c.lower() for c in df.columns}
        year_col   = find_one([r"^year$"], lower_map.values())
        # prefer Region as the state name if present
        region_candidate = find_one([r"^region$", r"^state$", r"jurisdiction"], lower_map.values())
        # map back to original column name casing
        # (find the original whose lower() == region_candidate)
        for orig, low in lower_map.items():
            if low == region_candidate:
                state_col = orig
            if low == "year":
                year_col = orig
        # Build a value as the mean across actual1..5
        df["_oop_actual_mean"] = df[wide_actual_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        value_col = "_oop_actual_mean"

    # Now proceed with plotting using the detected columns
    df[year_col] = yearify(df[year_col])
    df = df.dropna(subset=[year_col, state_col, value_col])

    latest_year = int(df[year_col].dropna().max())
    latest = df[df[year_col] == latest_year].copy()

    # If multiple rows per state/region, average them
    latest = latest.groupby(state_col, as_index=False)[value_col].mean()

    # Sort ASC for a clean horizontal bar
    latest = latest.sort_values(value_col, ascending=True)

    ensure_dir(outdir)
    plt.figure(figsize=(10, 6))
    plt.barh(latest[state_col].astype(str), latest[value_col])
    plt.title(f"Out-of-pocket cost per service by state/territory (actual prices, {latest_year})")
    plt.xlabel("Cost ($)")
    plt.ylabel("State/Territory")
    plt.grid(True, axis="x", alpha=0.3)
    fp = os.path.join(outdir, f"state_bar_{latest_year}.png")
    plt.tight_layout()
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"[STATE BAR] Saved: {fp}")


# ---------- main ----------

def glob_one(patterns):
    for pat in patterns:
        hits = sorted(glob(pat))
        if hits:
            return hits[0]
    return None

def main():
    ap = argparse.ArgumentParser(description="Create OOP visualizations")
    ap.add_argument("--table8", default=None, help="Path to Table 8 (SEIFA) CSV")
    ap.add_argument("--table9", default=None, help="Path to Table 9 (Remoteness) CSV")
    ap.add_argument("--age", default=None, help="Path to age (monthly) CSV")
    ap.add_argument("--states", default=None, help="Path to states & territories CSV")
    ap.add_argument("--outdir", default="figures", help="Output folder for PNGs")
    args = ap.parse_args()

    # If any path is missing, try to find a likely file
    if not args.table8:
        args.table8 = glob_one([
            "*Table8*YearFixed*.csv",
            "*Table8*Clean*.csv",
            "*SEIFA*.csv",
        ])
    if not args.table9:
        args.table9 = glob_one([
            "*Table9*Remoteness*Clean*.csv",
            "*Remoteness*.csv",
        ])
    if not args.age:
        args.age = glob_one([
            "*GP_Prices_by_Age_Monthly*.csv",
            "*Age*Monthly*.csv",
        ])
    if not args.states:
        args.states = glob_one([
            "*Out_of_pocket_costs_by_states*",
            "*states*territories*.csv",
        ])

    print("Resolved inputs:")
    print(f" - Table 8 (SEIFA): {args.table8}")
    print(f" - Table 9 (Remoteness): {args.table9}")
    print(f" - Age (monthly):       {args.age}")
    print(f" - States & territories:{args.states}")

    if args.table8:
        try:
            plot_seifa_trend(args.table8, args.outdir)
        except Exception as e:
            print(f"[SEIFA] Skipped due to error: {e}")

    if args.table9:
        try:
            plot_remoteness_trend(args.table9, args.outdir)
        except Exception as e:
            print(f"[REMOTENESS] Skipped due to error: {e}")

    if args.age:
        try:
            plot_age_trend(args.age, args.outdir)
        except Exception as e:
            print(f"[AGE] Skipped due to error: {e}")

    if args.states:
        try:
            plot_state_latest_bar(args.states, args.outdir)
        except Exception as e:
            print(f"[STATE BAR] Skipped due to error: {e}")

if __name__ == "__main__":
    main()
    print("Resolved inputs:")
    print(f" - Table 8 (SEIFA): {args.table8}")
    print(f" - Table 9 (Remoteness): {args.table9}")
    print(f" - Age (monthly):       {args.age}")
    print(f" - States & territories:{args.states}")

    # Make sure output dir exists once
    ensure_dir(args.outdir)

    # ---- TABLE 8 (SEIFA) ----
    if args.table8:
        try:
            plot_seifa_trend(args.table8, args.outdir)
        except Exception as e:
            print(f"[SEIFA trend] Skipped due to error: {e}")
        try:
            plot_seifa_gap(args.table8, args.outdir)
        except Exception as e:
            print(f"[SEIFA gap] Skipped due to error: {e}")

    # ---- TABLE 9 (REMOTENESS) ----
    if args.table9:
        try:
            plot_remoteness_trend(args.table9, args.outdir)
        except Exception as e:
            print(f"[REMOTENESS trend] Skipped due to error: {e}")
        try:
            plot_remoteness_bar(args.table9, args.outdir)
        except Exception as e:
            print(f"[REMOTENESS bar] Skipped due to error: {e}")
        try:
            plot_remoteness_slope(args.table9, args.outdir)
        except Exception as e:
            print(f"[REMOTENESS slope] Skipped due to error: {e}")

    # ---- AGE (MONTHLY -> YEARLY) ----
    if args.age:
        try:
            plot_age_trend(args.age, args.outdir)
        except Exception as e:
            print(f"[AGE trend] Skipped due to error: {e}")
        try:
            plot_age_gap(args.age, args.outdir)
        except Exception as e:
            print(f"[AGE gap] Skipped due to error: {e}")

    # ---- STATES & TERRITORIES ----
    if args.states:
        try:
            plot_state_latest_bar(args.states, args.outdir)
        except Exception as e:
            print(f"[STATE bar] Skipped due to error: {e}")
        try:
            plot_state_heatmap(args.states, args.outdir)
        except Exception as e:
            print(f"[STATE heatmap] Skipped due to error: {e}")
