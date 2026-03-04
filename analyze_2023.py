"""
Extension analysis: Top 10 most-streamed albums across 2017-2023.

Re-ranks all albums by first-day Spotify streams across the full 2017-2023
period, then runs the same event study methodology as Jena et al.
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
    create_event_study_dataset, ALBUM_RELEASES,
)

# ============================================================================
# Top 10 Most Streamed Albums in a Single Day, 2017-2023
# Merges original 2017-2022 list with 2023 releases, re-ranked
# (Sources: Spotify Charts, Chart Data, Billboard)
# ============================================================================
ALL_ALBUMS = pd.DataFrame([
    # --- From original 2017-2022 ---
    {"date": "2022-10-21", "album": "Midnights", "artist": "Taylor Swift",
     "tracks": 20, "first_day_streams": 184_695_609},
    {"date": "2021-09-03", "album": "Certified Lover Boy", "artist": "Drake",
     "tracks": 21, "first_day_streams": 153_441_565},
    {"date": "2022-05-06", "album": "Un Verano Sin Ti", "artist": "Bad Bunny",
     "tracks": 23, "first_day_streams": 145_811_373},
    {"date": "2018-06-29", "album": "Scorpion", "artist": "Drake",
     "tracks": 25, "first_day_streams": 132_384_203},
    {"date": "2022-05-13", "album": "Mr. Morale & the Big Steppers", "artist": "Kendrick Lamar",
     "tracks": 18, "first_day_streams": 99_582_729},
    {"date": "2022-05-20", "album": "Harry's House", "artist": "Harry Styles",
     "tracks": 13, "first_day_streams": 97_621_794},
    {"date": "2022-11-04", "album": "Her Loss", "artist": "Drake and 21 Savage",
     "tracks": 16, "first_day_streams": 97_390_844},
    {"date": "2021-08-29", "album": "Donda", "artist": "Kanye West",
     "tracks": 25, "first_day_streams": 94_455_883},
    {"date": "2021-11-12", "album": "Red (Taylor's Version)", "artist": "Taylor Swift",
     "tracks": 30, "first_day_streams": 90_556_180},
    {"date": "2020-07-24", "album": "Folklore", "artist": "Taylor Swift",
     "tracks": 16, "first_day_streams": 79_443_136},
    # --- 2023 releases ---
    {"date": "2023-10-27", "album": "1989 (Taylor's Version)", "artist": "Taylor Swift",
     "tracks": 21, "first_day_streams": 176_000_000},
    {"date": "2023-10-13", "album": "Nadie Sabe Lo Que Va a Pasar Mañana", "artist": "Bad Bunny",
     "tracks": 22, "first_day_streams": 145_900_000},
    {"date": "2023-07-28", "album": "UTOPIA", "artist": "Travis Scott",
     "tracks": 19, "first_day_streams": 128_500_000},
    {"date": "2023-07-07", "album": "Speak Now (Taylor's Version)", "artist": "Taylor Swift",
     "tracks": 22, "first_day_streams": 126_000_000},
    {"date": "2023-10-06", "album": "For All the Dogs", "artist": "Drake",
     "tracks": 23, "first_day_streams": 109_000_000},
])
ALL_ALBUMS["date"] = pd.to_datetime(ALL_ALBUMS["date"])
ALL_ALBUMS = ALL_ALBUMS.sort_values("first_day_streams", ascending=False).reset_index(drop=True)
ALL_ALBUMS["rank"] = ALL_ALBUMS.index + 1

# Top 10 across 2017-2023
TOP10_2017_2023 = ALL_ALBUMS.head(10).copy()


def create_event_study_dataset_extended(daily_data, album_releases, window=10):
    """Create album-day level dataset for extended event study."""
    holidays = get_federal_holidays(range(2017, 2024))

    records = []
    for i, (_, album) in enumerate(album_releases.iterrows()):
        rd = album["date"]
        for day_offset in range(-window, window + 1):
            d = rd + pd.Timedelta(days=day_offset)
            d_ts = pd.Timestamp(d)

            match = daily_data[daily_data["date"] == d_ts]
            if len(match) == 0:
                continue

            records.append({
                "album_idx": i,
                "album": album["album"],
                "artist": album["artist"],
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


def run_placebo_test(daily_data, observed_effect, n_albums=10, n_iterations=1000):
    """Run placebo falsification test."""
    np.random.seed(789)
    years = sorted(daily_data["date"].dt.year.unique())
    holidays = get_federal_holidays(years)
    all_dates = daily_data["date"].values
    fridays = daily_data[daily_data["date"].dt.dayofweek == 4]["date"].values

    random_date_effects = []
    random_friday_effects = []

    for _ in range(n_iterations):
        for dates_pool, effects_list in [(all_dates, random_date_effects),
                                          (fridays, random_friday_effects)]:
            placebo = np.random.choice(dates_pool, size=n_albums, replace=False)
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
    print("EXTENDED ANALYSIS: Top 10 Albums Across 2017-2023")
    print("=" * 70)

    # --- Show all candidate albums ---
    print("\n--- All Candidate Albums (2017-2023) ---")
    print(ALL_ALBUMS[["rank", "date", "album", "artist",
                       "first_day_streams"]].to_string(index=False))

    print(f"\n--- Top 10 Selected ---")
    top10 = TOP10_2017_2023
    for _, row in top10.iterrows():
        marker = " [2023]" if row["date"].year == 2023 else ""
        print(f"  {row['rank']:>2}. {row['album']:<45} {row['artist']:<20} "
              f"{row['first_day_streams']/1e6:>6.1f}M  ({row['date'].date()}){marker}")

    n_from_2023 = (top10["date"].dt.year == 2023).sum()
    n_from_orig = len(top10) - n_from_2023
    print(f"\n  {n_from_orig} from 2017-2022, {n_from_2023} from 2023")

    # --- Load data ---
    print("\n\n--- DATA ---")
    if not check_xlsx_data():
        print("ERROR: No fatalities.xlsx found.")
        return

    daily_all = load_xlsx_data(years=range(2017, 2024))
    print(f"  Days: {len(daily_all)}")
    print(f"  Date range: {daily_all['date'].min().date()} to {daily_all['date'].max().date()}")
    print(f"  Mean daily fatalities: {daily_all['fatalities'].mean():.1f}")

    # --- Event study with all top 10 across 2017-2023 ---
    print("\n\n--- EVENT STUDY (Top 10 across 2017-2023) ---")
    event_df = create_event_study_dataset_extended(daily_all, top10)
    print(f"  Observations: {len(event_df)} ({len(top10)} albums x 21 days)")

    event_results, event_model = run_event_study(event_df)
    day0 = event_results[event_results["day"] == 0].iloc[0]
    print(f"\n  Day 0 coefficient: {day0['coef']:.1f} "
          f"(95% CI {day0['ci_lower']:.1f} to {day0['ci_upper']:.1f})")

    comparison = run_comparison(event_df)
    print(f"\n  Release day fatalities:     {comparison['release_mean']:.1f}")
    print(f"  Surrounding day fatalities: {comparison['surrounding_mean']:.1f}")
    print(f"  Absolute increase: {comparison['absolute_increase']:.1f} "
          f"(95% CI {comparison['ci_lower']:.1f} to {comparison['ci_upper']:.1f})")
    print(f"  Relative increase: {comparison['relative_increase']:.1f}%")
    print(f"  p-value: {comparison['p_value']:.4f}")

    # --- Also run original 2017-2022 for comparison ---
    print("\n\n--- ORIGINAL (Top 10 across 2017-2022, for comparison) ---")
    daily_orig = load_xlsx_data(years=range(2017, 2023))
    event_orig = create_event_study_dataset(daily_orig)
    event_results_orig, _ = run_event_study(event_orig)
    comparison_orig = run_comparison(event_orig)

    print(f"  Release day fatalities:     {comparison_orig['release_mean']:.1f}")
    print(f"  Surrounding day fatalities: {comparison_orig['surrounding_mean']:.1f}")
    print(f"  Absolute increase: {comparison_orig['absolute_increase']:.1f} "
          f"(95% CI {comparison_orig['ci_lower']:.1f} to {comparison_orig['ci_upper']:.1f})")
    print(f"  Relative increase: {comparison_orig['relative_increase']:.1f}%")
    print(f"  p-value: {comparison_orig['p_value']:.4f}")

    # --- Placebo test ---
    print("\n\n--- PLACEBO TEST (Top 10 across 2017-2023) ---")
    random_dates, random_fridays = run_placebo_test(
        daily_all, comparison['absolute_increase'], n_albums=len(top10))
    exceed_dates = np.sum(random_dates >= comparison['absolute_increase'])
    exceed_fridays = np.sum(random_fridays >= comparison['absolute_increase'])
    print(f"  Random dates: {exceed_dates}/1000 exceeded observed effect")
    print(f"  Random Fridays: {exceed_fridays}/1000 exceeded observed effect")

    # --- Adjacent Friday test ---
    print("\n\n--- ADJACENT FRIDAY TEST ---")
    friday_releases = top10[top10["date"].dt.dayofweek == 4]
    release_fats = []
    control_fats = []
    for _, album in friday_releases.iterrows():
        rd = album["date"]
        match = daily_all[daily_all["date"] == rd]
        if len(match) > 0:
            release_fats.append(match.iloc[0]["fatalities"])
        for offset in [-7, 7]:
            cd = rd + pd.Timedelta(days=offset)
            match = daily_all[daily_all["date"] == cd]
            if len(match) > 0:
                control_fats.append(match.iloc[0]["fatalities"])

    or_est = None
    if release_fats and control_fats:
        or_est = np.mean(release_fats) / np.mean(control_fats)
        print(f"  Friday releases: {len(friday_releases)}")
        print(f"  Mean fatalities on release Fridays: {np.mean(release_fats):.1f}")
        print(f"  Mean fatalities on control Fridays: {np.mean(control_fats):.1f}")
        print(f"  Ratio: {or_est:.2f}")

    # --- Save results ---
    results_extended = {
        "period": "2017-2023",
        "n_albums": len(top10),
        "n_albums_from_2023": int(n_from_2023),
        "albums": top10[["rank", "album", "artist", "first_day_streams"]].to_dict(orient="records"),
        "comparison": comparison,
        "event_study": event_results.to_dict(orient="records"),
        "placebo_exceed_dates": int(exceed_dates),
        "placebo_exceed_fridays": int(exceed_fridays),
        "adjacent_friday_or": float(or_est) if or_est else None,
        "mean_daily_fatalities": float(daily_all["fatalities"].mean()),
    }

    with open(os.path.join(OUTPUT_DIR, "results_2017_2023.json"), "w") as f:
        json.dump(results_extended, f, indent=2)

    results_orig_save = {
        "period": "2017-2022",
        "n_albums": len(ALBUM_RELEASES),
        "comparison": comparison_orig,
        "event_study": event_results_orig.to_dict(orient="records"),
        "mean_daily_fatalities": float(daily_orig["fatalities"].mean()),
    }

    with open(os.path.join(OUTPUT_DIR, "results_2017_2022.json"), "w") as f:
        json.dump(results_orig_save, f, indent=2)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<40} {'2017-2022':>15} {'2017-2023':>15}")
    print(f"{'':40} {'(original 10)':>15} {'(new top 10)':>15}")
    print("-" * 70)
    print(f"{'Albums analyzed':<40} {comparison_orig['release_mean']:>15.0f} {len(top10):>15}")
    print(f"{'Release day fatalities':<40} {comparison_orig['release_mean']:>15.1f} {comparison['release_mean']:>15.1f}")
    print(f"{'Surrounding day fatalities':<40} {comparison_orig['surrounding_mean']:>15.1f} {comparison['surrounding_mean']:>15.1f}")
    print(f"{'Absolute increase':<40} {comparison_orig['absolute_increase']:>15.1f} {comparison['absolute_increase']:>15.1f}")
    print(f"{'Relative increase (%)':<40} {comparison_orig['relative_increase']:>14.1f}% {comparison['relative_increase']:>14.1f}%")
    print(f"{'p-value':<40} {comparison_orig['p_value']:>15.4f} {comparison['p_value']:>15.4f}")
    print("=" * 70)

    print(f"\n  Results saved to {OUTPUT_DIR}/results_2017_2023.json")


if __name__ == "__main__":
    main()
