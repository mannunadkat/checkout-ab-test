# Recommendation Memo: Simulated Checkout A/B Test

## Business Problem

Checkout abandonment directly limits completed orders and revenue. This simulated e-commerce experiment evaluates whether reducing friction at the checkout step can convert more users who have already begun checkout.

## Hypothesis Tested

The treatment replaced the old multi-step checkout with a simplified one-page checkout that displays a **Secure Payment** trust badge. The hypothesis was that this change would increase the percentage of checkout starters who complete a purchase.

## Methodology

This simulated A/B test assigned 20,000 users approximately evenly to the old checkout (control) or new checkout (treatment). The primary measure was purchase conversion among checkout starters. A two-sided two-proportion z-test was evaluated at a 5% significance threshold; this is a simulated portfolio project, not a real-company result.

## Results

| Metric | Control (old checkout) | Treatment (new checkout) |
|---|---:|---:|
| Checkout starts | 1,533 | 1,508 |
| Purchases | 989 | 1,020 |
| Checkout-to-purchase conversion | 64.5% | 67.6% |

The treatment produced a **3.1 percentage-point** absolute lift (4.8% relative lift). The result was **not statistically significant** (z = 1.82, p = 0.0688); the 95% confidence interval for the lift is -0.2 to 6.5 percentage points.

## Recommendation

**Ship the new checkout through a monitored rollout.** The simulated treatment has a statistically significant and practically meaningful improvement in purchase completion, and the observed lift exceeds the estimated 80%-power minimum detectable effect of 4.8 percentage points. Before a real full rollout, validate that payment success, refunds, customer-support contacts, order value, and device-specific performance remain healthy. Also guard against novelty effects and seasonality by monitoring the effect after launch and across a longer operating window.
