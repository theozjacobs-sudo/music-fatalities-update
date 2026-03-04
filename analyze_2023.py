"""
Extension analysis: Applying the Jena et al. (2026) methodology to 2023 data.

Uses the top most-streamed albums released in 2023 and FARS fatality data
for 2023 to test whether the album-release/fatality relationship persists.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import shared utilities from the main script
sys.path.insert(0, BASE_DIR)
from reproduce_analysis import (
    get_federal_holidays, load_xlsx_data, check_xlsx_data,
    generate_synthetic_streaming_data, create_event_study_dataset,
)

# ============================================================================
# Top 10 Most Streamed Albums in a Single Day, 2023
# (Sources: Spotify Charts, Chart Data, Billboard)
# ============================================================================
ALBUM_RELEASES_2023 = pd.DataFrame([
    {"date": "2023-10-27", "album": "1989 (Taylor's Version)", "artist": "Taylor Swift",
     "tracks": 21, "first_day_streams": 176_000_000, "rank": 1},
    {"date": "2023-10-13", "album": "Nadie Sabe Lo Que Va a Pasar Mañana", "artist": "Bad Bunny",
     "tracks": 22, "first_day_streams": 145_900_000, "rank": 2},
    {"date": "2023-07-28", "album": "UTOPIA", "artist": "Travis Scott",
     "tracks": 19, "first_day_streams": 128_500_000, "rank": 3},
    {"date": "2023-07-07", "album": "Speak Now (Taylor's Version)", "artist": "Taylor Swift",
     "tracks": 22, "first_day_streams": 126_000_000, "rank": 4},
    {"date": "2023-10-06", "album": "For All the Dogs", "artist": "Drake",
     "tracks": 23, "first_day_streams": 109_000_000, "rank": 5},
    {"date": "2023-09-08", "album": "GUTS", "artist": "Olivia Rodrigo",
     "tracks": 12, "first_day_streams": 60_900_000, "rank": 6},
    {"date": "2023-03-03", "album": "One Thing at a Time", "artist": "Morgan Wallen",
     "tracks": 36, "first_day_streams": 52_300_000, "rank": 7},
    {"date": "2023-12-08", "album": "Pink Friday 2", "artist": "Nicki Minaj",
     "tracks": 22, "first_day_streams": 52_000_000, "rank": 8},
    {"date": "2023-06-23", "album": "Génesis", "artist": "Peso Pluma",
     "tracks": 26, "first_day_streams": 45_000_000, "rank": 9},
    {"date": "2023-03-24", "album": "Did You Know That There's a Tunnel Under Ocean Blvd",
     "artist": "Lana Del Rey", "tracks": 16, "first_day_streams": 43_000_000, "rank": 10},
])
ALBUM_RELEASES_2023["date"] = pd.to_datetime(ALBUM_RELEASES_2023["date"])


def create_event_study_dataset_2023(daily_data, album_releases, window=10):
    """Create album-day level dataset for 2023 event study."""
    release_dates = album_releases["date"].values
    holidays = get_federal_holidays(range(2023, 2024))

    records = []
    for i, rd in enumerate(release_dates):
        for day_offset in range(-window, window + 1):
            d = rd + pd.Timedelta(days=day_offset)
            d_ts = pd.Timestamp(d)

            match = daily_data[daily_data["date"] == d_ts]
            if len(match) == 0:
                continue

            records.append({
                "album_idx": i,
                "album": album_releases.iloc[i]["album"],
                "artist": album_releases.iloc[i]["artist"],
                "date": d_ts,
                "day_relative": day_offset,
                "fatalities": match.iloc[0]["fatalities"],
                "is_release_day": int(day_offset == 0),
                "dow": d_ts.dayofweek,
                "week_of_year": d_ts.isocalendar()[1],
                "year": d_ts.year,
                "is_holiday": int(d_ts in holidays),
            })

    return pd.DataFrame(records)


def run_event_study(event_df):
    """Run event study regression, return day-level coefficients."""
    event_df = event_df.copy()
    day_col_map = {}
    for d in range(-10, 11):
        col = f"day_m{abs(d)}" if d < 0 else f"day_p{d}"
        event_df[col] = (event_df["day_relative"] == d).astype(int)
        day_col_map[d] = col

    day_vars = [day_col_map[d] for d in range(-10, 11) if d != -1]
    formula = "fatalities ~ " + " + ".join(day_vars) + " + C(dow) + is_holiday"
    model = smf.ols(formula, data=event_df).fit()

    results = []
    for d in range(-10, 11):
        if d == -1:
            results.append({"day": d, "coef": 0, "ci_lower": 0, "ci_upper": 0})
        else:
            param = day_col_map[d]
            coef = model.params[param]
            ci = model.conf_int(alpha=0.05).loc[param]
            results.append({"day": d, "coef": coef, "ci_lower": ci[0], "ci_upper": ci[1]})

    return pd.DataFrame(results), model


def run_comparison(event_df):
    """Run release day vs surrounding days comparison."""
    formula = "fatalities ~ is_release_day + C(dow) + is_holiday"
    model = smf.ols(formula, data=event_df).fit()

    coef = model.params["is_release_day"]
    pval = model.pvalues["is_release_day"]
    ci = model.conf_int(alpha=0.05).loc["is_release_day"]
    base_pred = model.predict(event_df.assign(is_release_day=0)).mean()
    release_pred = model.predict(event_df.assign(is_release_day=1)).mean()

    return {
        "release_mean": float(release_pred),
        "surrounding_mean": float(base_pred),
        "absolute_increase": float(coef),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "relative_increase": float(coef / base_pred * 100),
        "p_value": float(pval),
    }


def run_placebo_test(daily_data, observed_effect, n_iterations=1000):
    """Run placebo falsification test for 2023."""
    np.random.seed(789)
    holidays = get_federal_holidays(range(2023, 2024))
    all_dates = daily_data["date"].values
    fridays = daily_data[daily_data["date"].dt.dayofweek == 4]["date"].values

    random_date_effects = []
    random_friday_effects = []

    for _ in range(n_iterations):
        for dates_pool, effects_list in [(all_dates, random_date_effects),
                                          (fridays, random_friday_effects)]:
            placebo = np.random.choice(dates_pool, size=10, replace=False)
            records = []
            for rd in placebo:
                rd = pd.Timestamp(rd)
                for offset in range(-10, 11):
                    d = rd + pd.Timedelta(days=offset)
                    match = daily_data[daily_data["date"] == d]
                    if len(match) == 0:
                        continue
                    records.append({
                        "date": d, "fatalities": match.iloc[0]["fatalities"],
                        "is_release_day": int(offset == 0),
                        "dow": d.dayofweek, "year": d.year,
                        "is_holiday": int(d in holidays),
                    })
            if len(records) < 20:
                effects_list.append(0)
                continue
            try:
                df = pd.DataFrame(records)
                model = smf.ols("fatalities ~ is_release_day + C(dow) + is_holiday",
                                data=df).fit()
                effects_list.append(model.params.get("is_release_day", 0))
            except Exception:
                effects_list.append(0)

    return np.array(random_date_effects), np.array(random_friday_effects)


def main():
    print("=" * 70)
    print("2023 EXTENSION ANALYSIS")
    print("Album Releases and Traffic Fatalities in 2023")
    print("=" * 70)

    # --- Table: 2023 Albums ---
    print("\n--- Top 10 Most Streamed Albums (2023) ---")
    print(ALBUM_RELEASES_2023[["rank", "date", "album", "artist", "tracks",
                                "first_day_streams"]].to_string(index=False))

    # --- Load 2023 data ---
    print("\n\n--- DATA ---")
    if not check_xlsx_data():
        print("ERROR: No fatalities.xlsx found. Cannot run 2023 analysis.")
        return

    daily_2023 = load_xlsx_data(years=range(2023, 2024))
    print(f"  Days: {len(daily_2023)}")
    print(f"  Date range: {daily_2023['date'].min().date()} to {daily_2023['date'].max().date()}")
    print(f"  Mean daily fatalities: {daily_2023['fatalities'].mean():.1f}")

    # --- Event study ---
    print("\n\n--- EVENT STUDY ---")
    event_df = create_event_study_dataset_2023(daily_2023, ALBUM_RELEASES_2023)
    print(f"  Observations: {len(event_df)} ({len(ALBUM_RELEASES_2023)} albums x 21 days)")

    event_results, event_model = run_event_study(event_df)
    day0 = event_results[event_results["day"] == 0].iloc[0]
    print(f"\n  Day 0 coefficient: {day0['coef']:.1f} (95% CI {day0['ci_lower']:.1f} to {day0['ci_upper']:.1f})")

    # --- Release vs surrounding ---
    comparison = run_comparison(event_df)
    print(f"\n  Release day fatalities:     {comparison['release_mean']:.1f}")
    print(f"  Surrounding day fatalities: {comparison['surrounding_mean']:.1f}")
    print(f"  Absolute increase: {comparison['absolute_increase']:.1f} "
          f"(95% CI {comparison['ci_lower']:.1f} to {comparison['ci_upper']:.1f})")
    print(f"  Relative increase: {comparison['relative_increase']:.1f}%")
    print(f"  p-value: {comparison['p_value']:.4f}")

    # --- Placebo test ---
    print("\n\n--- PLACEBO TEST ---")
    random_dates, random_fridays = run_placebo_test(daily_2023, comparison['absolute_increase'])
    exceed_dates = np.sum(random_dates >= comparison['absolute_increase'])
    exceed_fridays = np.sum(random_fridays >= comparison['absolute_increase'])
    print(f"  Random dates: {exceed_dates}/1000 exceeded observed effect")
    print(f"  Random Fridays: {exceed_fridays}/1000 exceeded observed effect")

    # --- Adjacent Friday test ---
    print("\n\n--- ADJACENT FRIDAY TEST ---")
    friday_releases = ALBUM_RELEASES_2023[ALBUM_RELEASES_2023["date"].dt.dayofweek == 4]
    release_fats = []
    control_fats = []
    for _, album in friday_releases.iterrows():
        rd = album["date"]
        match = daily_2023[daily_2023["date"] == rd]
        if len(match) > 0:
            release_fats.append(match.iloc[0]["fatalities"])
        for offset in [-7, 7]:
            cd = rd + pd.Timedelta(days=offset)
            match = daily_2023[daily_2023["date"] == cd]
            if len(match) > 0:
                control_fats.append(match.iloc[0]["fatalities"])

    if release_fats and control_fats:
        or_est = (np.mean(release_fats)) / (np.mean(control_fats))
        print(f"  Friday releases: {len(friday_releases)}")
        print(f"  Mean fatalities on release Fridays: {np.mean(release_fats):.1f}")
        print(f"  Mean fatalities on control Fridays: {np.mean(control_fats):.1f}")
        print(f"  Odds ratio: {or_est:.2f}")

    # --- Save results for comparison script ---
    results_2023 = {
        "period": "2023",
        "n_albums": len(ALBUM_RELEASES_2023),
        "comparison": comparison,
        "event_study": event_results.to_dict(orient="records"),
        "placebo_exceed_dates": int(exceed_dates),
        "placebo_exceed_fridays": int(exceed_fridays),
        "adjacent_friday_or": float(or_est) if release_fats and control_fats else None,
        "mean_daily_fatalities": float(daily_2023["fatalities"].mean()),
    }

    with open(os.path.join(OUTPUT_DIR, "results_2023.json"), "w") as f:
        json.dump(results_2023, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR}/results_2023.json")

    # --- Also save 2017-2022 results for comparison ---
    from reproduce_analysis import ALBUM_RELEASES
    daily_orig = load_xlsx_data(years=range(2017, 2023))
    event_orig = create_event_study_dataset(daily_orig)
    event_results_orig, _ = run_event_study(event_orig)
    comparison_orig = run_comparison(event_orig)

    results_orig = {
        "period": "2017-2022",
        "n_albums": len(ALBUM_RELEASES),
        "comparison": comparison_orig,
        "event_study": event_results_orig.to_dict(orient="records"),
        "mean_daily_fatalities": float(daily_orig["fatalities"].mean()),
    }

    with open(os.path.join(OUTPUT_DIR, "results_2017_2022.json"), "w") as f:
        json.dump(results_orig, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<40} {'2017-2022':>12} {'2023':>12}")
    print("-" * 64)
    print(f"{'Release day fatalities':<40} {comparison_orig['release_mean']:>12.1f} {comparison['release_mean']:>12.1f}")
    print(f"{'Surrounding day fatalities':<40} {comparison_orig['surrounding_mean']:>12.1f} {comparison['surrounding_mean']:>12.1f}")
    print(f"{'Absolute increase':<40} {comparison_orig['absolute_increase']:>12.1f} {comparison['absolute_increase']:>12.1f}")
    print(f"{'Relative increase (%)':<40} {comparison_orig['relative_increase']:>11.1f}% {comparison['relative_increase']:>11.1f}%")
    print(f"{'p-value':<40} {comparison_orig['p_value']:>12.4f} {comparison['p_value']:>12.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
