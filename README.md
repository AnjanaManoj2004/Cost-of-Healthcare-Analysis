# Cost of Healthcare Analysis — Treatment Efficiency

I focus on treatment efficiency for Redback Lachesis.  
I compare average cost and average outcome for each treatment.  
I report a cost-per-outcome score. Lower is better.

## How I run this work
1. 01_data_overview.ipynb — I inspect the raw file.
2. 02_cleaning.ipynb — I fix names and types, and save a clean file.
3. 03_efficiency_metrics_and_analysis.ipynb — I build the ranking table.
4. 04_visualisations_export.ipynb — I export PNG charts.

## Files I produce
- data/medicare_treatment_clean.csv
- data/treatment_efficiency.csv
- figures/cost_vs_outcome_scatter.png
- figures/efficiency_ranking_top20.png

## Notes on data handling
I keep only the fields I need.  
I drop rows with missing cost or outcome.  
I exclude zero or negative outcomes.
