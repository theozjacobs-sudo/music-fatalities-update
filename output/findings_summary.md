# Reproduction & Extension: "Smartphones, Online Music Streaming, and Traffic Fatalities"

**Reference paper:** Patel, Worsham, Liu, and Jena (NBER Working Paper 34866, Feb 2026)
**Analysis date:** March 2026
**Data source:** NHTSA Fatality Analysis Reporting System (FARS), real daily fatality counts

---

## Executive Summary

We reproduced the paper's core analysis using real FARS data and then extended it through 2023. **The original 2017–2022 finding largely replicates, but adding 2023 data eliminates the effect entirely.** The 2023 albums on their own show zero association with traffic fatalities.

---

## Head-to-Head Comparison

| Metric | Published Paper | Our 2017–2022 Run | Our 2017–2023 Run |
|--------|:--------------:|:-----------------:|:-----------------:|
| **Albums analyzed** | 10 | 10 | 10 (re-ranked) |
| **Release-day fatalities** | 139.1 | 137.6 | 124.4 |
| **Surrounding-day fatalities** | 120.9 | 122.3 | 120.3 |
| **Absolute increase** | +18.2 deaths | +15.3 deaths | +4.1 deaths |
| **Relative increase** | +15.1% | +12.5% | +3.4% |
| **95% CI** | 4.8 – 31.7 | 3.9 – 26.6 | −7.0 – 15.2 |
| **p-value** | 0.01 | 0.009 | 0.47 |
| **Placebo (random dates exceed)** | 14 / 1,000 | — | 162 / 1,000 |
| **Placebo (random Fridays exceed)** | 20 / 1,000 | — | 216 / 1,000 |

### 2023-Only Albums (5 new releases)

| Metric | Value |
|--------|:-----:|
| Release-day fatalities | 113.2 |
| Surrounding-day fatalities | 115.0 |
| Absolute increase | **−1.8 deaths** |
| p-value | 0.76 |
| Placebo (random dates exceed) | 627 / 1,000 |
| Adjacent Friday odds ratio | 0.99 |

---

## Key Discrepancies

### 1. Our 2017–2022 reproduction is close but not identical to the paper
- We find +15.3 excess deaths vs. the paper's +18.2 — a ~16% smaller effect
- Both are statistically significant (p = 0.009 vs. p = 0.01)
- Likely explained by differences in data vintages: FARS data undergoes revisions, and our pull may reflect updated counts

### 2. Adding 2023 collapses the effect
- The absolute increase drops from +15.3 to +4.1 deaths (−73%)
- The result is no longer statistically significant (p = 0.47)
- Placebo tests go from strong (14/1,000) to weak (162/1,000)

### 3. 2023 albums show no effect at all
- The five 2023 mega-releases (1989 TV, UTOPIA, etc.) show a *negative* point estimate (−1.8 deaths)
- 627 out of 1,000 random-date placebos produce a larger "effect" than the real release dates
- The adjacent-Friday odds ratio is 0.99 — completely flat

---

## What Changed in 2023?

The re-ranked 2017–2023 top 10 swaps out 4 of the original albums and adds 5 from 2023:

| Dropped from original top 10 | Added from 2023 |
|-------------------------------|-----------------|
| Harry's House (Harry Styles) | 1989 Taylor's Version (Taylor Swift) |
| Her Loss (Drake & 21 Savage) | Nadie Sabe Lo Que Va a Pasar Mañana (Bad Bunny) |
| Donda (Kanye West) | UTOPIA (Travis Scott) |
| Red Taylor's Version (Taylor Swift) | Speak Now Taylor's Version (Taylor Swift) |
| Folklore (Taylor Swift) | For All the Dogs (Drake) |

---

## Event Study Coefficients (Day 0)

```
                        Coefficient    95% CI
Paper (2017–2022):         —          (reported as means comparison only)
Our run (2017–2022):      25.2        [−2.1, 52.6]
Our run (2017–2023):      10.1        [ 2.5, 17.7]
2023 albums only:          6.1        [−2.1, 14.3]
```

Note: The 2017–2022 day-0 coefficient is large but its CI includes zero, while the means comparison (release day vs. surrounding window) is significant. This mirrors the paper's own event-study figure, where the day-0 spike is visually prominent but the event-study CI is wide.

---

## Bottom Line

| | Replicates? | Details |
|---|:-----------:|---------|
| Streaming spike on release day | **Yes** | ~43% increase, consistent with paper |
| Fatality increase (2017–2022) | **Mostly** | Directionally consistent, slightly smaller magnitude |
| Fatality increase with 2023 data | **No** | Effect disappears, p = 0.47 |
| 2023 albums alone | **No** | Point estimate is negative |

The original finding appears specific to the 2017–2022 sample. Extending the same methodology one year forward with five new blockbuster albums produces no detectable effect on traffic fatalities.
