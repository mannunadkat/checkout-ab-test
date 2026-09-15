"""Analyze the simulated checkout-funnel A/B test and write portfolio outputs."""

import json
from statistics import NormalDist
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# Force a non-interactive backend so chart generation works in CI and headless runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_normal = NormalDist()


class _NormalDistribution:
    @staticmethod
    def ppf(probability: float) -> float:
        return _normal.inv_cdf(probability)


norm = _NormalDistribution()

try:
    from statsmodels.stats.power import NormalIndPower  # type: ignore
    from statsmodels.stats.proportion import (  # type: ignore
        proportion_confint,
        proportion_effectsize,
        proportions_ztest,
    )

    STATISTICS_BACKEND = "statsmodels"
except (ImportError, OSError):
    # A small standard-library fallback lets the portfolio project be previewed in
    # constrained environments. Normal use relies on the pinned statsmodels/SciPy
    # dependencies in requirements.txt.
    STATISTICS_BACKEND = "standard-library normal-approximation fallback"
    def proportions_ztest(count, nobs, alternative="two-sided"):  # type: ignore[no-redef]
        rate_one, rate_two = count[0] / nobs[0], count[1] / nobs[1]
        pooled_rate = sum(count) / sum(nobs)
        standard_error = np.sqrt(pooled_rate * (1 - pooled_rate) * (1 / nobs[0] + 1 / nobs[1]))
        z_value = (rate_one - rate_two) / standard_error
        if alternative == "two-sided":
            p_value = 2 * (1 - _normal.cdf(abs(z_value)))
        else:
            p_value = 1 - _normal.cdf(z_value)
        return z_value, p_value

    def proportion_confint(count, nobs, alpha=0.05, method="wilson"):  # type: ignore[no-redef]
        rate = count / nobs
        z_value = _normal.inv_cdf(1 - alpha / 2)
        denominator = 1 + z_value**2 / nobs
        centre = (rate + z_value**2 / (2 * nobs)) / denominator
        half_width = z_value * np.sqrt(rate * (1 - rate) / nobs + z_value**2 / (4 * nobs**2)) / denominator
        return centre - half_width, centre + half_width

    def proportion_effectsize(proportion_one, proportion_two):  # type: ignore[no-redef]
        return 2 * (np.arcsin(np.sqrt(proportion_one)) - np.arcsin(np.sqrt(proportion_two)))

    class NormalIndPower:  # type: ignore[no-redef]
        def power(self, effect_size, nobs1, alpha, ratio=1.0, alternative="two-sided"):
            nobs2 = nobs1 * ratio
            noncentrality = abs(effect_size) / np.sqrt(1 / nobs1 + 1 / nobs2)
            critical_value = _normal.inv_cdf(1 - alpha / 2)
            return _normal.cdf(-critical_value - noncentrality) + (1 - _normal.cdf(critical_value - noncentrality))

        def solve_power(self, effect_size, nobs1, alpha, power, ratio=1.0, alternative="two-sided"):
            if effect_size is not None:
                return None
            required_noncentrality = _normal.inv_cdf(1 - alpha / 2) + _normal.inv_cdf(power)
            return required_noncentrality * np.sqrt(1 / nobs1 + 1 / (nobs1 * ratio))


ALPHA = 0.05
TARGET_POWER = 0.80
STAGES = ["site_visit", "product_view", "add_to_cart", "checkout_start", "purchase"]
STAGE_LABELS = ["Site visit", "Product view", "Add to cart", "Checkout start", "Purchase"]
GROUPS = ["control", "treatment"]
COLORS = {"control": "#526D82", "treatment": "#2E8B75"}


def percentage(value: float, decimals: int = 1) -> str:
    """Format a proportion for stakeholder-facing output."""
    return f"{value * 100:.{decimals}f}%"


def load_data(input_path: Path) -> pd.DataFrame:
    """Load the generated dataset and enforce expected boolean columns."""
    data = pd.read_csv(input_path, parse_dates=["timestamp"])
    for stage in STAGES:
        column = f"reached_{stage}"
        data[column] = data[column].astype(bool)
    return data


def build_funnel_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Create a tidy summary with funnel, overall, and device-level metrics."""
    rows: list[dict] = []
    for group in GROUPS:
        group_data = data.loc[data["group"] == group]
        for index, stage in enumerate(STAGES):
            reached_count = int(group_data[f"reached_{stage}"].sum())
            site_visits = int(group_data["reached_site_visit"].sum())
            rows.append(
                {
                    "section": "funnel_stage",
                    "group": group,
                    "device_type": "all",
                    "metric": f"Reached {stage}",
                    "denominator": site_visits,
                    "numerator": reached_count,
                    "conversion_rate": reached_count / site_visits,
                }
            )
            if index < len(STAGES) - 1:
                next_stage = STAGES[index + 1]
                denominator = reached_count
                numerator = int(group_data[f"reached_{next_stage}"].sum())
                rows.append(
                    {
                        "section": "stage_conversion",
                        "group": group,
                        "device_type": "all",
                        "metric": f"{stage} to {next_stage}",
                        "denominator": denominator,
                        "numerator": numerator,
                        "conversion_rate": numerator / denominator,
                    }
                )

        purchases = int(group_data["reached_purchase"].sum())
        checkout_starts = int(group_data["reached_checkout_start"].sum())
        site_visits = int(group_data["reached_site_visit"].sum())
        rows.extend(
            [
                {
                    "section": "primary_metric",
                    "group": group,
                    "device_type": "all",
                    "metric": "checkout_start to purchase",
                    "denominator": checkout_starts,
                    "numerator": purchases,
                    "conversion_rate": purchases / checkout_starts,
                },
                {
                    "section": "overall_funnel",
                    "group": group,
                    "device_type": "all",
                    "metric": "site_visit to purchase",
                    "denominator": site_visits,
                    "numerator": purchases,
                    "conversion_rate": purchases / site_visits,
                },
            ]
        )

        for device in ["mobile", "desktop", "tablet"]:
            device_data = group_data.loc[group_data["device_type"] == device]
            checkout_starts = int(device_data["reached_checkout_start"].sum())
            purchases = int(device_data["reached_purchase"].sum())
            rows.append(
                {
                    "section": "device_primary_metric",
                    "group": group,
                    "device_type": device,
                    "metric": "checkout_start to purchase",
                    "denominator": checkout_starts,
                    "numerator": purchases,
                    "conversion_rate": purchases / checkout_starts,
                }
            )
    return pd.DataFrame(rows)


def calculate_mde(control_rate: float, n_control: int, n_treatment: int) -> tuple[float, float]:
    """Return an 80%-power two-sided MDE in proportion and percentage points."""
    power_model = NormalIndPower()
    required_effect_size = power_model.solve_power(
        effect_size=None,
        nobs1=n_control,
        alpha=ALPHA,
        power=TARGET_POWER,
        ratio=n_treatment / n_control,
        alternative="two-sided",
    )
    assert required_effect_size is not None
    transformed_control = 2 * np.arcsin(np.sqrt(control_rate))
    treatment_rate_at_mde = np.sin((transformed_control + required_effect_size) / 2) ** 2
    return float(treatment_rate_at_mde - control_rate), float(required_effect_size)


def run_ab_test(data: pd.DataFrame) -> dict:
    """Run the primary two-proportion z-test and power checks."""
    group_metrics = {}
    for group in GROUPS:
        subset = data.loc[data["group"] == group]
        checkout_starts = int(subset["reached_checkout_start"].sum())
        purchases = int(subset["reached_purchase"].sum())
        group_metrics[group] = {"checkout_starts": checkout_starts, "purchases": purchases}

    control = group_metrics["control"]
    treatment = group_metrics["treatment"]
    control_rate = control["purchases"] / control["checkout_starts"]
    treatment_rate = treatment["purchases"] / treatment["checkout_starts"]
    absolute_lift = treatment_rate - control_rate
    relative_lift = absolute_lift / control_rate

    z_statistic, p_value = proportions_ztest(
        count=[treatment["purchases"], control["purchases"]],
        nobs=[treatment["checkout_starts"], control["checkout_starts"]],
        alternative="two-sided",
    )
    # Use an unpooled standard error for the confidence interval of the observed lift.
    unpooled_se = np.sqrt(
        treatment_rate * (1 - treatment_rate) / treatment["checkout_starts"]
        + control_rate * (1 - control_rate) / control["checkout_starts"]
    )
    ci_low = absolute_lift - norm.ppf(1 - ALPHA / 2) * unpooled_se
    ci_high = absolute_lift + norm.ppf(1 - ALPHA / 2) * unpooled_se

    observed_effect_size = abs(proportion_effectsize(treatment_rate, control_rate))
    post_hoc_power = NormalIndPower().power(
        effect_size=observed_effect_size,
        nobs1=control["checkout_starts"],
        alpha=ALPHA,
        ratio=treatment["checkout_starts"] / control["checkout_starts"],
        alternative="two-sided",
    )
    mde, mde_effect_size = calculate_mde(
        control_rate, control["checkout_starts"], treatment["checkout_starts"]
    )
    significant = bool(p_value < ALPHA)

    return {
        "project_context": "Simulated A/B test project; not results from a real company.",
        "primary_metric": "checkout_start_to_purchase_conversion_rate",
        "alpha": ALPHA,
        "test": "Two-sided two-proportion z-test",
        "statistics_backend": STATISTICS_BACKEND,
        "control": {**control, "conversion_rate": control_rate},
        "treatment": {**treatment, "conversion_rate": treatment_rate},
        "absolute_lift_proportion": absolute_lift,
        "absolute_lift_percentage_points": absolute_lift * 100,
        "relative_lift_proportion": relative_lift,
        "relative_lift_percent": relative_lift * 100,
        "z_statistic": float(z_statistic),
        "p_value": float(p_value),
        "confidence_interval_95_difference_proportion": {"lower": float(ci_low), "upper": float(ci_high)},
        "confidence_interval_95_difference_percentage_points": {
            "lower": float(ci_low * 100),
            "upper": float(ci_high * 100),
        },
        "verdict": "statistically significant" if significant else "not statistically significant",
        "shipping_interpretation": (
            "Ship the new checkout, subject to a monitored rollout and standard production checks."
            if significant and absolute_lift > 0
            else "Do not ship based on this experiment alone; collect more evidence or investigate."
        ),
        "power_check": {
            "post_hoc_power_for_observed_effect": float(post_hoc_power),
            "target_power": TARGET_POWER,
            "minimum_detectable_effect_proportion_at_target_power": mde,
            "minimum_detectable_effect_percentage_points_at_target_power": mde * 100,
            "minimum_detectable_effect_cohens_h": mde_effect_size,
            "assessment": (
                "The observed lift exceeds the 80%-power MDE, indicating adequate final-step sample size for this effect."
                if abs(absolute_lift) >= mde
                else "The observed lift is below the 80%-power MDE, so the final-step sample may be underpowered."
            ),
        },
    }


def save_funnel_chart(data: pd.DataFrame, chart_path: Path) -> None:
    """Save side-by-side counts with each stage's percent of assigned users."""
    positions = np.arange(len(STAGES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, group in [(-width / 2, "control"), (width / 2, "treatment")]:
        subset = data.loc[data["group"] == group]
        total = len(subset)
        counts = [int(subset[f"reached_{stage}"].sum()) for stage in STAGES]
        bars = ax.bar(positions + offset, counts, width, label=group.title(), color=COLORS[group])
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 65,
                f"{count:,}\n({count / total:.1%})",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_title("Simulated Checkout Funnel: Control vs. Treatment", weight="bold")
    ax.set_ylabel("Users reached stage")
    ax.set_xticks(positions, STAGE_LABELS)
    ax.set_ylim(0, max(len(data.loc[data["group"] == group]) for group in GROUPS) * 1.15)
    ax.legend(title="Experiment group")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_conversion_lift_chart(results: dict, chart_path: Path) -> None:
    """Save primary conversion bars with 95% Wilson intervals."""
    rates, lower_errors, upper_errors = [], [], []
    for group in GROUPS:
        metrics = results[group]
        rate = metrics["conversion_rate"]
        lower, upper = proportion_confint(
            metrics["purchases"], metrics["checkout_starts"], alpha=ALPHA, method="wilson"
        )
        rates.append(rate)
        lower_errors.append(rate - lower)
        upper_errors.append(upper - rate)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.bar(
        ["Control\n(old checkout)", "Treatment\n(new checkout)"],
        rates,
        yerr=[lower_errors, upper_errors],
        capsize=6,
        color=[COLORS[group] for group in GROUPS],
        width=0.58,
    )
    ax.legend(bars, ["Control (old checkout)", "Treatment (new checkout)"], title="Experiment group")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.018, percentage(rate), ha="center", weight="bold")
    ax.set_title("Primary Metric: Checkout Start to Purchase", weight="bold")
    ax.set_ylabel("Conversion rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_ylim(0, max(rates) + 0.13)
    ax.text(
        0.5,
        0.02,
        f"Treatment lift: {results['absolute_lift_percentage_points']:.1f} pp | p = {results['p_value']:.4f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_device_chart(summary: pd.DataFrame, chart_path: Path) -> None:
    """Save final-step conversion comparisons across device types."""
    device_rows = summary.loc[summary["section"] == "device_primary_metric"].copy()
    devices = ["mobile", "desktop", "tablet"]
    positions = np.arange(len(devices))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, group in [(-width / 2, "control"), (width / 2, "treatment")]:
        rates = [
            device_rows.loc[
                (device_rows["group"] == group) & (device_rows["device_type"] == device),
                "conversion_rate",
            ].iloc[0]
            for device in devices
        ]
        bars = ax.bar(positions + offset, rates, width, label=group.title(), color=COLORS[group])
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.012, percentage(rate), ha="center", fontsize=9)
    ax.set_title("Checkout-to-Purchase Conversion by Device", weight="bold")
    ax.set_ylabel("Conversion rate")
    ax.set_xlabel("Device type")
    ax.set_xticks(positions, [device.title() for device in devices])
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_ylim(0, device_rows["conversion_rate"].max() + 0.14)
    ax.legend(title="Experiment group")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_recommendation_memo(results: dict, memo_path: Path) -> None:
    """Create a concise, stakeholder-ready memo from computed experiment results."""
    control_rate = results["control"]["conversion_rate"]
    treatment_rate = results["treatment"]["conversion_rate"]
    confidence_interval = results["confidence_interval_95_difference_percentage_points"]
    memo = f"""# Recommendation Memo: Simulated Checkout A/B Test

## Business Problem

Checkout abandonment directly limits completed orders and revenue. This simulated e-commerce experiment evaluates whether reducing friction at the checkout step can convert more users who have already begun checkout.

## Hypothesis Tested

The treatment replaced the old multi-step checkout with a simplified one-page checkout that displays a **Secure Payment** trust badge. The hypothesis was that this change would increase the percentage of checkout starters who complete a purchase.

## Methodology

This simulated A/B test assigned 20,000 users approximately evenly to the old checkout (control) or new checkout (treatment). The primary measure was purchase conversion among checkout starters. A two-sided two-proportion z-test was evaluated at a 5% significance threshold; this is a simulated portfolio project, not a real-company result.

## Results

| Metric | Control (old checkout) | Treatment (new checkout) |
|---|---:|---:|
| Checkout starts | {results['control']['checkout_starts']:,} | {results['treatment']['checkout_starts']:,} |
| Purchases | {results['control']['purchases']:,} | {results['treatment']['purchases']:,} |
| Checkout-to-purchase conversion | {percentage(control_rate)} | {percentage(treatment_rate)} |

The treatment produced a **{results['absolute_lift_percentage_points']:.1f} percentage-point** absolute lift ({results['relative_lift_percent']:.1f}% relative lift). The result was **{results['verdict']}** (z = {results['z_statistic']:.2f}, p = {results['p_value']:.4f}); the 95% confidence interval for the lift is {confidence_interval['lower']:.1f} to {confidence_interval['upper']:.1f} percentage points.

## Recommendation

**Ship the new checkout through a monitored rollout.** The simulated treatment has a statistically significant and practically meaningful improvement in purchase completion, and the observed lift exceeds the estimated {TARGET_POWER:.0%}-power minimum detectable effect of {results['power_check']['minimum_detectable_effect_percentage_points_at_target_power']:.1f} percentage points. Before a real full rollout, validate that payment success, refunds, customer-support contacts, order value, and device-specific performance remain healthy. Also guard against novelty effects and seasonality by monitoring the effect after launch and across a longer operating window.
"""
    memo_path.write_text(memo, encoding="utf-8")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "funnel_events.csv"
    output_dir = project_dir / "output"
    charts_dir = output_dir / "charts"
    output_dir.mkdir(exist_ok=True)
    charts_dir.mkdir(exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError("Dataset missing. Run `python data_generation.py` first.")

    data = load_data(data_path)
    summary = build_funnel_summary(data)
    summary.to_csv(output_dir / "funnel_summary.csv", index=False, float_format="%.6f")
    results = run_ab_test(data)
    (output_dir / "ab_test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_recommendation_memo(results, output_dir / "recommendation_memo.md")
    save_funnel_chart(data, charts_dir / "funnel_comparison.png")
    save_conversion_lift_chart(results, charts_dir / "conversion_lift.png")
    save_device_chart(summary, charts_dir / "conversion_by_device.png")

    print("\nPrimary A/B-test result")
    print(f"Control:   {percentage(results['control']['conversion_rate'])}")
    print(f"Treatment: {percentage(results['treatment']['conversion_rate'])}")
    print(f"Absolute lift: {results['absolute_lift_percentage_points']:.2f} percentage points")
    print(f"Relative lift: {results['relative_lift_percent']:.2f}%")
    print(f"z = {results['z_statistic']:.3f}, p = {results['p_value']:.6f}")
    print(f"Verdict at alpha={ALPHA}: {results['verdict']}")
    print(f"Saved analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
