# Checkout Funnel A/B Test

This repository is a **simulated A/B test project** built to practice experimentation methodology on realistic, synthetic e-commerce funnel data. It evaluates a proposed one-page checkout with a visible **Secure Payment** trust badge against an old multi-step checkout; it does not represent an experiment run at a real company.

## Business question

Does the simplified new checkout increase the rate at which users who start checkout complete a purchase, enough to justify shipping the change?

## Project structure

```text
checkout-ab-test/
├── README.md
├── requirements.txt
├── data_generation.py
├── analysis.py
├── data/funnel_events.csv
└── output/
    ├── funnel_summary.csv
    ├── ab_test_results.json
    ├── recommendation_memo.md
    └── charts/
        ├── funnel_comparison.png
        ├── conversion_lift.png
        └── conversion_by_device.png
```

## How to run

```bash
cd checkout-ab-test
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python data_generation.py
python analysis.py
```

The generator uses a fixed random seed, so the dataset and headline results are reproducible.

## Key results

The simulated treatment improved checkout-start-to-purchase conversion from
**63.2%** (control) to **68.7%** (treatment): a **5.6 percentage-point**
absolute lift, or **8.8%** relative lift. The two-sided two-proportion z-test
found the lift **statistically significant** at alpha = 0.05 (**p = 0.0009**;
95% CI: 2.3 to 8.8 percentage points), supporting a monitored simulated
shipping recommendation.

The primary metric is checkout-start-to-purchase conversion. Statistical significance is evaluated with a two-sided two-proportion z-test at alpha = 0.05. See [the recommendation memo](output/recommendation_memo.md) for the concise business interpretation and [the results file](output/ab_test_results.json) for complete statistical outputs.

## What the analysis includes

- Stage-by-stage and overall funnel conversion for control and treatment.
- Primary checkout-to-purchase comparison, including absolute/relative lift, z-statistic, p-value, and a 95% confidence interval for the difference.
- Device-level conversion cut and recruiter-friendly charts.
- An 80%-power minimum detectable effect (MDE) check and post-hoc power estimate.

## Tech stack

Python, pandas, NumPy, SciPy, statsmodels, and matplotlib.
