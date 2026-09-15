"""Generate a reproducible synthetic checkout-funnel A/B-test dataset.

This script intentionally models a simulated experiment. It does not represent
results from a real company or customer population.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 20260915
NUM_USERS = 20_000
STAGES = [
    "site_visit",
    "product_view",
    "add_to_cart",
    "checkout_start",
    "purchase",
]
DEVICE_PROBABILITIES = {"mobile": 0.62, "desktop": 0.32, "tablet": 0.06}


def clipped_probability(
    base: float, device_effect: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Add modest person-level variation while preserving plausible rates."""
    noise = rng.normal(loc=0, scale=0.018, size=NUM_USERS)
    return np.clip(base + device_effect + noise, 0.01, 0.99)


def generate_dataset() -> pd.DataFrame:
    """Simulate user-level funnel progression and return one row per user."""
    rng = np.random.default_rng(RANDOM_SEED)
    user_ids = [f"user_{number:05d}" for number in range(1, NUM_USERS + 1)]
    groups = rng.choice(["control", "treatment"], size=NUM_USERS, p=[0.5, 0.5])
    devices = rng.choice(
        list(DEVICE_PROBABILITIES), size=NUM_USERS, p=list(DEVICE_PROBABILITIES.values())
    )

    # Event timestamps are distributed over a simulated 45-day experiment window.
    experiment_start = pd.Timestamp("2026-07-01 00:00:00")
    offsets_minutes = rng.integers(0, 45 * 24 * 60, size=NUM_USERS)
    timestamps = experiment_start + pd.to_timedelta(offsets_minutes, unit="m")

    # Device effects make the traffic mix realistic without changing the treatment
    # effect outside the intended final checkout-to-purchase step.
    stage_device_effect = {
        "mobile": -0.015,
        "desktop": 0.015,
        "tablet": -0.005,
    }
    device_effect = np.array([stage_device_effect[device] for device in devices])

    reached_site_visit = np.ones(NUM_USERS, dtype=bool)
    reached_product_view = rng.random(NUM_USERS) < clipped_probability(0.60, device_effect, rng)
    reached_add_to_cart = reached_product_view & (
        rng.random(NUM_USERS) < clipped_probability(0.54, device_effect * 0.7, rng)
    )
    reached_checkout_start = reached_add_to_cart & (
        rng.random(NUM_USERS) < clipped_probability(0.51, device_effect * 0.5, rng)
    )

    # The intervention only affects the final step: simplified checkout + trust badge.
    # The 6 percentage-point underlying lift is deliberately modest for this sample.
    final_device_effect = np.array(
        [{"mobile": -0.010, "desktop": 0.010, "tablet": -0.004}[device] for device in devices]
    )
    base_purchase_probability = np.where(groups == "treatment", 0.695, 0.635)
    purchase_probability = np.clip(
        base_purchase_probability + final_device_effect + rng.normal(0, 0.012, NUM_USERS),
        0.01,
        0.99,
    )
    reached_purchase = reached_checkout_start & (rng.random(NUM_USERS) < purchase_probability)

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "group": groups,
            "timestamp": timestamps,
            "device_type": devices,
            "reached_site_visit": reached_site_visit,
            "reached_product_view": reached_product_view,
            "reached_add_to_cart": reached_add_to_cart,
            "reached_checkout_start": reached_checkout_start,
            "reached_purchase": reached_purchase,
        }
    ).sort_values("timestamp", kind="stable").reset_index(drop=True)


import argparse
import analysis


def print_stage_summary(data: pd.DataFrame) -> None:
    """Print stage counts and final-step rates for a quick generation check."""
    counts = data.groupby("group")[[f"reached_{stage}" for stage in STAGES]].sum().astype(int)
    counts.columns = STAGES
    final_rates = (counts["purchase"] / counts["checkout_start"]).rename("checkout_to_purchase_rate")
    print("\nSynthetic funnel stage counts by group")
    print(counts.to_string())
    print("\nFinal-step conversion rates")
    print(final_rates.map(lambda value: f"{value:.2%}").to_string())


def main() -> None:
    global NUM_USERS, RANDOM_SEED
    print("=== Checkout A/B Test Simulator ===")
    
    users_in = input("Enter the number of users to simulate (e.g. 15000) [Default: 20000]: ")
    try:
        NUM_USERS = int(users_in) if users_in.strip() else 20000
    except ValueError:
        print("Invalid input. Using default 20000.")
        NUM_USERS = 20000
        
    seed_in = input("Enter a random seed (e.g. 42) [Default: 20260915]: ")
    try:
        RANDOM_SEED = int(seed_in) if seed_in.strip() else 20260915
    except ValueError:
        print("Invalid input. Using default 20260915.")
        RANDOM_SEED = 20260915

    print(f"\nGenerating dataset for {NUM_USERS:,} users with seed {RANDOM_SEED}...")
    project_dir = Path(__file__).resolve().parent
    output_path = project_dir / "data" / "funnel_events.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_dataset()
    data.to_csv(output_path, index=False)
    print(f"Wrote {len(data):,} simulated user records to {output_path}")
    print_stage_summary(data)
    
    print("\n" + "="*40)
    print("Running Analysis Pipeline...")
    print("="*40 + "\n")
    analysis.main()


if __name__ == "__main__":
    main()
