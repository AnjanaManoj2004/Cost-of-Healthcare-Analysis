# test.py
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd

# -------- helpers --------
MONEY_RE = re.compile(r"[^0-9.\-]")  # strip $, commas, spaces

def find_header_start(df: pd.DataFrame, left0: str, left1: str) -> int:
    """
    Find the row index where the left-most two visible labels (after fill) are e.g.
    'Service year' and 'State'. We search a few top rows to be robust.
    """
    for i in range(0, min(40, len(df))):
        row = df.iloc[i].astype(str).str.strip()
        if row.iloc[0].lower().startswith(left0) and row.iloc[1].lower().startswith(left1):
            return i
    # fallback: try to find the row where col2 says 'SEIFA Quintile' or 'Remoteness area'
    for i in range(0, min(40, len(df))):
        c2 = str(df.iloc[i, 2]).strip().lower()
        if c2.startswith("seifa quintile") or c2.startswith("remoteness area"):
            return i
    raise RuntimeError(f"Could not find header start for {left0}/{left1}")

def flatten_columns(cols) -> list[str]:
    out = []
    for tup in cols:
        # tup is a tuple from MultiIndex (len 3 typically)
        parts = [str(x).strip() for x in tup if pd.notna(x) and str(x).strip() != "" and not str(x).startswith("Unnamed")]
        out.append(" | ".join(parts))
    return out

def clean_money(s):
    if pd.isna(s):
        return pd.NA
    s = str(s).strip()
    if s in {"", "—", "-", "na", "n/a", "None"}:
        return pd.NA
    return pd.to_numeric(MONEY_RE.sub("", s), errors="coerce")

def save_csv(df: pd.DataFrame, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False, encoding="utf-8")

# -------- table cleaners --------
def clean_table8(excel_path: Path, sheet_name: str = "Table 8") -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, dtype=str)
    hdr0 = find_header_start(raw, "service year", "state")
    # Read again with a 3-row header
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=[hdr0, hdr0+1, hdr0+2], dtype=str)

    # Drop rows above header (already handled by header=...); now drop fully empty rows
    df = df.dropna(how="all")
    # Ensure first two columns are the ID vars
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    flat_cols = flatten_columns(df.columns)
    df.columns = flat_cols

    # Find ID columns (first two should be Service year & State)
    # Some files might repeat the words in the flat header; match loosely
    id_map = {}
    for c in df.columns[:3]:  # first few
        lc = c.lower()
        if "service year" in lc and "state" not in id_map:
            id_map["Service year"] = c
        elif "state" in lc:
            id_map["State"] = c
    if "Service year" not in id_map or "State" not in id_map:
        # fallback to the first two columns
        id_map = {"Service year": df.columns[0], "State": df.columns[1]}

    # Value columns: must contain both 'SEIFA Quintile' and 'Out-of-pocket'
    value_cols = [c for c in df.columns if ("seifa quintile" in c.lower() and "out-of-pocket" in c.lower())]
    if not value_cols:
        print("[Table 8] Warning: no SEIFA columns parsed. Sample headers seen:")
        print(df.columns[:10].tolist())

    # Melt
    tidy = df.melt(
        id_vars=[id_map["Service year"], id_map["State"]],
        value_vars=value_cols if value_cols else [],
        var_name="combo",
        value_name="value"
    )

    # If value_cols was empty, create an empty scaffold with correct headers to avoid empty file surprises
    if tidy.empty and value_cols:
        # (shouldn’t happen, but safety)
        pass

    # Parse quintile and price type
    def parse_t8(x: str):
        xl = x.lower()
        # quintile text between 'seifa quintile |' and '| out-of-pocket'
        q = None
        m = None
        # find the segment that contains 'quintile'
        parts = [p.strip() for p in x.split("|")]
        for p in parts:
            pl = p.lower()
            if "quintile" in pl:
                q = p
            if "out-of-pocket" in pl:
                m = "actual" if "actual" in pl else ("inflation adjusted" if "inflation" in pl else p)
        return q, m

    parsed = tidy["combo"].apply(parse_t8)
    tidy["SEIFA quintile"] = [a for (a, b) in parsed]
    tidy["price_type"] = [b for (a, b) in parsed]

    # Rename id columns cleanly
    tidy = tidy.rename(columns={
        id_map["Service year"]: "Service year",
        id_map["State"]: "State"
    })

    # Clean money and drop empties
    tidy["Out-of-pocket per service"] = tidy["value"].apply(clean_money)
    tidy = tidy.drop(columns=["value", "combo"])
    # Keep only rows where we recognized quintile
    tidy = tidy.dropna(subset=["SEIFA quintile"])
    # Sort and return
    # Convert Service year to int when possible
    with pd.option_context("mode.chained_assignment", None):
        tidy["Service year"] = pd.to_numeric(tidy["Service year"], errors="coerce").astype("Int64")
    tidy = tidy.sort_values(["Service year", "State", "SEIFA quintile", "price_type"]).reset_index(drop=True)
    return tidy

def clean_table9(excel_path: Path, sheet_name: str = "Table 9") -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, dtype=str)
    hdr0 = find_header_start(raw, "service year", "state")
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=[hdr0, hdr0+1, hdr0+2], dtype=str)

    df = df.dropna(how="all")
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    flat_cols = flatten_columns(df.columns)
    df.columns = flat_cols

    # Map id columns
    id_map = {}
    for c in df.columns[:3]:
        lc = c.lower()
        if "service year" in lc and "state" not in id_map:
            id_map["Service year"] = c
        elif "state" in lc:
            id_map["State"] = c
    if "Service year" not in id_map or "State" not in id_map:
        id_map = {"Service year": df.columns[0], "State": df.columns[1]}

    # Value columns: contain 'Remoteness area' and 'Out-of-pocket'
    value_cols = [c for c in df.columns if ("remoteness area" in c.lower() and "out-of-pocket" in c.lower())]
    if not value_cols:
        print("[Table 9] Warning: no remoteness columns parsed. Sample headers seen:")
        print(df.columns[:10].tolist())

    tidy = df.melt(
        id_vars=[id_map["Service year"], id_map["State"]],
        value_vars=value_cols if value_cols else [],
        var_name="combo",
        value_name="value"
    )

    def parse_t9(x: str):
        # extract remoteness label and price_type
        ra = None
        pt = None
        parts = [p.strip() for p in x.split("|")]
        for p in parts:
            pl = p.lower()
            if "remoteness" in pl:
                # the next part likely is the specific area (e.g., Major cities, Inner regional)
                # often present as a separate part; otherwise p may already include it
                ra = p.replace("Remoteness area", "").strip(" -:")
            if "out-of-pocket" in pl:
                pt = "actual" if "actual" in pl else ("inflation adjusted" if "inflation" in pl else p)
        # If ra is still None, try to find the part that looks like a known area name
        if not ra:
            for p in parts:
                if any(k in p.lower() for k in ["major", "inner", "outer", "remote", "very remote"]):
                    ra = p.strip()
                    break
        return ra, pt

    parsed = tidy["combo"].apply(parse_t9)
    tidy["Remoteness area"] = [a for (a, b) in parsed]
    tidy["price_type"] = [b for (a, b) in parsed]

    tidy = tidy.rename(columns={
        id_map["Service year"]: "Service year",
        id_map["State"]: "State"
    })

    tidy["Out-of-pocket per service"] = tidy["value"].apply(clean_money)
    tidy = tidy.drop(columns=["value", "combo"])
    tidy = tidy.dropna(subset=["Remoteness area"])
    with pd.option_context("mode.chained_assignment", None):
        tidy["Service year"] = pd.to_numeric(tidy["Service year"], errors="coerce").astype("Int64")
    tidy = tidy.sort_values(["Service year", "State", "Remoteness area", "price_type"]).reset_index(drop=True)
    return tidy

# -------- main --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=str, required=False,
                        help="Path to AIHW Excel (e.g., AIHW-HWE-103-MBS-bulk-billing-summary-data-*.xlsx)")
    parser.add_argument("--table8-sheet", type=str, default="Table 8")
    parser.add_argument("--table9-sheet", type=str, default="Table 9")
    parser.add_argument("--outdir", type=str, default="clean")
    args = parser.parse_args()

    # Resolve Excel path
    if args.excel:
        excel_path = Path(args.excel)
    else:
        # try to autodiscover in current tree
        candidates = list(Path(".").glob("**/AIHW-HWE-103-MBS-bulk-billing-summary-data*.xlsx"))
        if not candidates:
            raise SystemExit("Excel not provided and not found. Use --excel PATH")
        excel_path = candidates[0]

    outdir = Path(args.outdir)

    t8 = clean_table8(excel_path, args.table8_sheet)
    t9 = clean_table9(excel_path, args.table9_sheet)

    save_csv(t8, outdir / "table8_oop_by_seifa_quintile.csv")
    save_csv(t9, outdir / "table9_oop_by_remoteness.csv")

    print("Saved:")
    print(" -", outdir / "table8_oop_by_seifa_quintile.csv")
    print(" -", outdir / "table9_oop_by_remoteness.csv")

if __name__ == "__main__":
    main()
