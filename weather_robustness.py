#!/usr/bin/env python3
"""
Weather robustness check: Was weather systematically worse on album release dates?

Creates a population-weighted national daily "bad weather index" from 20 major
US cities and compares album release dates vs. surrounding days. Then re-runs
the fatality regression with weather as a covariate.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ── Album release dates (2017-2022 run) ──
RELEASE_DATES = pd.to_datetime([
    "2022-10-21",  # Midnights
    "2021-09-03",  # Certified Lover Boy
    "2022-05-06",  # Un Verano Sin Ti
    "2018-06-29",  # Scorpion
    "2022-05-13",  # Mr. Morale & the Big Steppers
    "2022-05-20",  # Harry's House
    "2022-11-04",  # Her Loss
    "2021-08-29",  # Donda
    "2021-11-12",  # Red (Taylor's Version)
    "2020-07-24",  # Folklore
])

ALBUM_NAMES = [
    "Midnights", "Certified Lover Boy", "Un Verano Sin Ti", "Scorpion",
    "Mr. Morale & the Big Steppers", "Harry's House", "Her Loss",
    "Donda", "Red (Taylor's Version)", "Folklore"
]


def load_weather_data():
    """Load and combine weather data from both city groups."""
    east = pd.read_csv(os.path.join(OUTPUT_DIR, "weather_east_south.csv"))
    west = pd.read_csv(os.path.join(OUTPUT_DIR, "weather_midwest_west.csv"))
    df = pd.concat([east, west], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_national_weather_index(df):
    """
    Build a population-weighted national daily weather index.

    Bad weather score for each city-day:
      - Precipitation (mm): higher = worse
      - Wind speed max (km/h): higher = worse
      - Temperature extremes: penalize very hot (>35C) or very cold (<-5C)

    Then population-weight across cities for a national daily score.
    """
    df = df.copy()

    # Normalize each metric to 0-1 scale across the full dataset
    precip_99 = df["precipitation_sum"].quantile(0.99)
    wind_99 = df["windspeed_10m_max"].quantile(0.99)

    df["precip_score"] = (df["precipitation_sum"] / precip_99).clip(0, 1)
    df["wind_score"] = (df["windspeed_10m_max"] / wind_99).clip(0, 1)

    # Temperature extremes: distance from comfortable range (10-30C)
    df["temp_extreme_score"] = 0.0
    # Hot extreme
    hot_mask = df["temperature_2m_max"] > 35
    df.loc[hot_mask, "temp_extreme_score"] = (
        (df.loc[hot_mask, "temperature_2m_max"] - 35) / 15
    ).clip(0, 1)
    # Cold extreme
    cold_mask = df["temperature_2m_min"] < -5
    df.loc[cold_mask, "temp_extreme_score"] = (
        (-5 - df.loc[cold_mask, "temperature_2m_min"]) / 25
    ).clip(0, 1)

    # Composite: weight precipitation most heavily (most directly affects driving)
    df["bad_weather_score"] = (
        0.50 * df["precip_score"] +
        0.30 * df["wind_score"] +
        0.20 * df["temp_extreme_score"]
    )

    # Population-weighted national daily average
    total_pop = df.groupby("date")["population"].sum()
    df["weighted_score"] = df["bad_weather_score"] * df["population"]
    daily = df.groupby("date").agg(
        bad_weather_index=("weighted_score", "sum"),
        total_pop=("population", "sum"),
        mean_precip=("precipitation_sum", "mean"),
        mean_wind=("windspeed_10m_max", "mean"),
        mean_temp_max=("temperature_2m_max", "mean"),
    ).reset_index()
    daily["bad_weather_index"] = daily["bad_weather_index"] / daily["total_pop"]

    return daily


def compare_weather_on_release_dates(daily_weather, window=10):
    """Compare weather on album release dates vs surrounding days."""
    release_set = set(RELEASE_DATES)

    # Build event windows
    records = []
    for i, rd in enumerate(RELEASE_DATES):
        for offset in range(-window, window + 1):
            d = rd + pd.Timedelta(days=offset)
            match = daily_weather[daily_weather["date"] == d]
            if len(match) == 0:
                continue
            row = match.iloc[0]
            records.append({
                "album": ALBUM_NAMES[i],
                "date": d,
                "day_relative": offset,
                "is_release_day": int(offset == 0),
                "bad_weather_index": row["bad_weather_index"],
                "mean_precip": row["mean_precip"],
                "mean_wind": row["mean_wind"],
                "mean_temp_max": row["mean_temp_max"],
            })

    event_df = pd.DataFrame(records)

    # Summary
    release = event_df[event_df["is_release_day"] == 1]
    surrounding = event_df[event_df["is_release_day"] == 0]

    print("\n" + "=" * 60)
    print("WEATHER ON ALBUM RELEASE DATES vs SURROUNDING DAYS")
    print("=" * 60)

    print(f"\n{'Metric':<35} {'Release Days':>15} {'Surrounding':>15}")
    print("-" * 65)
    print(f"{'Bad weather index (0-1)':<35} {release['bad_weather_index'].mean():>15.4f} {surrounding['bad_weather_index'].mean():>15.4f}")
    print(f"{'Mean precipitation (mm)':<35} {release['mean_precip'].mean():>15.2f} {surrounding['mean_precip'].mean():>15.2f}")
    print(f"{'Mean max wind (km/h)':<35} {release['mean_wind'].mean():>15.2f} {surrounding['mean_wind'].mean():>15.2f}")
    print(f"{'Mean max temperature (C)':<35} {release['mean_temp_max'].mean():>15.1f} {surrounding['mean_temp_max'].mean():>15.1f}")

    # Per-album weather
    print(f"\n{'Album':<35} {'Weather Index':>15} {'Precip (mm)':>15}")
    print("-" * 65)
    for _, row in release.iterrows():
        print(f"{row['album']:<35} {row['bad_weather_index']:>15.4f} {row['mean_precip']:>15.2f}")

    # T-test
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(
        release["bad_weather_index"].values,
        surrounding["bad_weather_index"].values
    )
    print(f"\nT-test (release vs surrounding weather): t={t_stat:.3f}, p={p_val:.3f}")

    # Rank test (non-parametric)
    u_stat, p_rank = stats.mannwhitneyu(
        release["bad_weather_index"].values,
        surrounding["bad_weather_index"].values,
        alternative="two-sided"
    )
    print(f"Mann-Whitney U test: U={u_stat:.1f}, p={p_rank:.3f}")

    return event_df, {
        "release_weather_mean": float(release["bad_weather_index"].mean()),
        "surrounding_weather_mean": float(surrounding["bad_weather_index"].mean()),
        "release_precip_mean": float(release["mean_precip"].mean()),
        "surrounding_precip_mean": float(surrounding["mean_precip"].mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "mann_whitney_p": float(p_rank),
    }


def rerun_fatality_regression_with_weather(daily_weather, window=10):
    """
    Re-run the original fatality regression but add weather as a covariate.
    Compare the release-day coefficient with and without weather controls.
    """
    # Load fatality data
    xlsx_path = os.path.join(BASE_DIR, "fatalities.xlsx")
    if not os.path.exists(xlsx_path):
        xlsx_path = os.path.join(BASE_DIR, "CrashReport.xlsx")

    fat_df = pd.read_excel(xlsx_path)

    # Parse fatality data (same logic as reproduce_analysis.py)
    # The FARS xlsx has dates as rows and columns for different metrics
    # We need to extract daily fatality counts
    from reproduce_analysis import load_xlsx_data, get_federal_holidays, ALBUM_RELEASES

    daily_data = load_xlsx_data()
    holidays = get_federal_holidays(range(2017, 2023))

    # Merge weather into event study dataset
    records = []
    for i, rd in enumerate(ALBUM_RELEASES["date"].values):
        for offset in range(-window, window + 1):
            d = rd + pd.Timedelta(days=offset)
            d_ts = pd.Timestamp(d)

            fat_match = daily_data[daily_data["date"] == d_ts]
            wx_match = daily_weather[daily_weather["date"] == d_ts]
            if len(fat_match) == 0 or len(wx_match) == 0:
                continue

            records.append({
                "album_idx": i,
                "date": d_ts,
                "day_relative": offset,
                "fatalities": fat_match.iloc[0]["fatalities"],
                "is_release_day": int(offset == 0),
                "dow": d_ts.dayofweek,
                "week_of_year": d_ts.isocalendar()[1],
                "year": d_ts.year,
                "is_holiday": int(d_ts in holidays),
                "bad_weather_index": wx_match.iloc[0]["bad_weather_index"],
                "mean_precip": wx_match.iloc[0]["mean_precip"],
            })

    event_df = pd.DataFrame(records)

    # Model WITHOUT weather (baseline)
    formula_base = "fatalities ~ is_release_day + C(dow) + C(week_of_year) + C(year) + is_holiday"
    model_base = smf.ols(formula_base, data=event_df).fit()

    # Model WITH weather index
    formula_wx = "fatalities ~ is_release_day + bad_weather_index + C(dow) + C(week_of_year) + C(year) + is_holiday"
    model_wx = smf.ols(formula_wx, data=event_df).fit()

    # Model WITH precipitation directly
    formula_precip = "fatalities ~ is_release_day + mean_precip + C(dow) + C(week_of_year) + C(year) + is_holiday"
    model_precip = smf.ols(formula_precip, data=event_df).fit()

    print("\n" + "=" * 60)
    print("FATALITY REGRESSION: WITH vs WITHOUT WEATHER CONTROLS")
    print("=" * 60)

    results = {}
    for label, model in [("No weather controls", model_base),
                          ("+ Bad weather index", model_wx),
                          ("+ Mean precipitation", model_precip)]:
        coef = model.params["is_release_day"]
        pval = model.pvalues["is_release_day"]
        ci = model.conf_int(alpha=0.05).loc["is_release_day"]

        print(f"\n{label}:")
        print(f"  Release-day effect: {coef:+.1f} deaths (95% CI: {ci[0]:.1f} to {ci[1]:.1f})")
        print(f"  p-value: {pval:.4f}")

        if "weather" in label.lower():
            wx_coef = model.params["bad_weather_index"]
            wx_p = model.pvalues["bad_weather_index"]
            print(f"  Weather index coef: {wx_coef:+.1f} (p={wx_p:.3f})")
        elif "precip" in label.lower():
            pr_coef = model.params["mean_precip"]
            pr_p = model.pvalues["mean_precip"]
            print(f"  Precipitation coef: {pr_coef:+.2f} (p={pr_p:.3f})")

        results[label] = {
            "release_day_coef": float(coef),
            "release_day_p": float(pval),
            "release_day_ci_lower": float(ci[0]),
            "release_day_ci_upper": float(ci[1]),
        }

    return results


def plot_weather_comparison(weather_event_df, save_path=None):
    """Plot weather on release dates vs surrounding days."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    release = weather_event_df[weather_event_df["is_release_day"] == 1]
    surrounding = weather_event_df[weather_event_df["is_release_day"] == 0]

    # Panel A: Bad weather index by day relative to release
    daily_avg = weather_event_df.groupby("day_relative")["bad_weather_index"].mean()
    ax = axes[0]
    colors = ["#c0392b" if d == 0 else "#3498db" for d in daily_avg.index]
    ax.bar(daily_avg.index, daily_avg.values, color=colors, alpha=0.7)
    ax.axhline(y=surrounding["bad_weather_index"].mean(), color="gray",
               linestyle="--", alpha=0.5, label="Surrounding avg")
    ax.set_xlabel("Days Relative to Album Release")
    ax.set_ylabel("Bad Weather Index")
    ax.set_title("A. Weather Index by Day")
    ax.legend(fontsize=8)

    # Panel B: Precipitation comparison
    ax = axes[1]
    daily_precip = weather_event_df.groupby("day_relative")["mean_precip"].mean()
    colors = ["#c0392b" if d == 0 else "#2ecc71" for d in daily_precip.index]
    ax.bar(daily_precip.index, daily_precip.values, color=colors, alpha=0.7)
    ax.axhline(y=surrounding["mean_precip"].mean(), color="gray",
               linestyle="--", alpha=0.5, label="Surrounding avg")
    ax.set_xlabel("Days Relative to Album Release")
    ax.set_ylabel("Mean Precipitation (mm)")
    ax.set_title("B. Precipitation by Day")
    ax.legend(fontsize=8)

    # Panel C: Per-album weather index (dot plot)
    ax = axes[2]
    album_wx = release.sort_values("bad_weather_index")
    y_pos = range(len(album_wx))
    ax.barh(y_pos, album_wx["bad_weather_index"].values, color="#e67e22", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(album_wx["album"].values, fontsize=8)
    ax.axvline(x=surrounding["bad_weather_index"].mean(), color="gray",
               linestyle="--", alpha=0.5, label="Surrounding avg")
    ax.set_xlabel("Bad Weather Index")
    ax.set_title("C. Weather by Album")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure: {save_path}")
    plt.close()


def main():
    print("Loading weather data...")
    weather_raw = load_weather_data()
    print(f"  {len(weather_raw)} city-day records loaded")
    print(f"  {weather_raw['city'].nunique()} cities, {weather_raw['date'].nunique()} days")

    print("\nBuilding national bad-weather index...")
    daily_weather = build_national_weather_index(weather_raw)
    print(f"  {len(daily_weather)} daily national records")
    print(f"  Weather index range: {daily_weather['bad_weather_index'].min():.4f} – {daily_weather['bad_weather_index'].max():.4f}")
    print(f"  Weather index mean: {daily_weather['bad_weather_index'].mean():.4f}")

    # Compare weather on release dates
    weather_event_df, weather_stats = compare_weather_on_release_dates(daily_weather)

    # Plot
    plot_weather_comparison(
        weather_event_df,
        save_path=os.path.join(OUTPUT_DIR, "weather_robustness.png")
    )

    # Re-run fatality regression with weather
    try:
        regression_results = rerun_fatality_regression_with_weather(daily_weather)
    except Exception as e:
        print(f"\nCould not re-run fatality regression: {e}")
        regression_results = None

    # Save results
    output = {
        "weather_comparison": weather_stats,
        "regression_with_weather": regression_results,
    }
    out_path = os.path.join(OUTPUT_DIR, "weather_robustness.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
