"""
FitPulse report figure exporter
Run this script in the folder containing the supplied CSV, or pass the CSV path:
    python generate_report_figures.py "dataset for assignment 2-0426(1).csv"

Creates publication-ready PNGs in ./report_figures/
Dependencies: pandas, numpy, matplotlib, scikit-learn
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1]).expanduser()
    else:
        candidates = [
            Path("dataset for assignment 2-0426(1).csv"),
            Path("fitness_dataset.csv"),
            Path("data/fitness_dataset.csv"),
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            raise FileNotFoundError("CSV not found. Pass its path as an argument.")

    out = Path("report_figures")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    needed = ["User ID","Gender","Age","Activity Level","Location","App Sessions",
              "Distance Travelled (km)","Calories Burned"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    numeric = ["Age","App Sessions","Distance Travelled (km)","Calories Burned"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out / name)

    # Figure 1: target variable distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(
        axes, ["App Sessions","Distance Travelled (km)","Calories Burned"],
        ["App sessions", "Distance travelled (km)", "Calories burned"]
    ):
        vals = df[col].dropna()
        ax.hist(vals, bins=35)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Number of records")
    save(fig, "fig01_metric_distributions.png")

    # Figure 2: average measures by activity category
    grouped = df.groupby("Activity Level", dropna=False)[
        ["App Sessions","Distance Travelled (km)","Calories Burned"]
    ].mean()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(
        axes, grouped.columns,
        ["Mean app sessions by activity", "Mean distance by activity (km)", "Mean calories by activity"]
    ):
        grouped[col].plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Activity Level")
        ax.set_ylabel("Mean")
        ax.tick_params(axis="x", rotation=25)
    save(fig, "fig02_activity_group_means.png")

    # Figure 3: correlation matrix
    corr = df[numeric].corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr.index)), labels=corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
    ax.set_title("Pearson correlation matrix")
    fig.colorbar(im, ax=ax, label="Correlation")
    save(fig, "fig03_correlation_matrix.png")

    # Figure 4: sessions vs distance, with activity group colors
    plot = df.dropna(subset=["App Sessions","Distance Travelled (km)","Activity Level"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for category, g in plot.groupby("Activity Level"):
        ax.scatter(g["App Sessions"], g["Distance Travelled (km)"], s=18, alpha=.55, label=str(category))
    ax.set_xlabel("App Sessions")
    ax.set_ylabel("Distance travelled (km)")
    ax.set_title("Engagement and distance by activity category")
    ax.legend(title="Activity Level")
    save(fig, "fig04_sessions_distance_scatter.png")

    # Figure 5: gender and location mean comparisons
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    df.groupby("Gender")[["App Sessions","Distance Travelled (km)","Calories Burned"]].mean().plot(
        kind="bar", ax=axes[0])
    axes[0].set_title("Mean metrics by gender")
    axes[0].set_ylabel("Mean")
    axes[0].tick_params(axis="x", rotation=20)
    df.groupby("Location")[["App Sessions","Distance Travelled (km)","Calories Burned"]].mean().plot(
        kind="bar", ax=axes[1])
    axes[1].set_title("Mean metrics by location")
    axes[1].set_ylabel("Mean")
    axes[1].tick_params(axis="x", rotation=20)
    save(fig, "fig05_group_comparisons.png")

    # Figure 6: holdout model comparison for App Sessions, demographics-only
    target = "App Sessions"
    features = ["Age","Gender","Location"]
    model_df = df[features + [target]].dropna()
    X, y = model_df[features], model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    cats = X.select_dtypes(include=["object","string","category"]).columns.tolist()
    nums = [c for c in X.columns if c not in cats]
    pre = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), nums),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cats)
    ])
    estimators = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3,
                                               random_state=42, n_jobs=-1)
    }
    scores = []
    for name, est in estimators.items():
        model = Pipeline([("preprocess", pre), ("model", est)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores.append((name, mean_absolute_error(y_test,pred),
                       np.sqrt(mean_squared_error(y_test,pred)), r2_score(y_test,pred)))
    score_df = pd.DataFrame(scores, columns=["Model","MAE","RMSE","R2"])
    score_df.to_csv(out / "model_comparison_sessions.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(score_df["Model"], score_df["MAE"])
    ax.set_ylabel("Mean absolute error (sessions)")
    ax.set_title("Holdout model comparison: predicting App Sessions")
    ax.tick_params(axis="x", rotation=18)
    save(fig, "fig06_model_comparison_sessions.png")

    # Figure 7: sensitivity of distance summary to the extreme 920 km observation
    full = df["Distance Travelled (km)"].dropna()
    sensitivity = full[full != 920]
    stats = pd.DataFrame({
        "All records": [full.mean(), full.median(), full.std()],
        "Excluding 920 km": [sensitivity.mean(), sensitivity.median(), sensitivity.std()]
    }, index=["Mean", "Median", "Standard deviation"])
    stats.to_csv(out / "distance_sensitivity_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    stats.plot(kind="bar", ax=ax)
    ax.set_title("Distance summary sensitivity (920 km record)")
    ax.set_ylabel("Distance (km)")
    ax.tick_params(axis="x", rotation=0)
    save(fig, "fig07_distance_sensitivity.png")

    print(f"\nDone. Figure files and summary CSVs are in: {out.resolve()}")
    print("Please share the PNGs (and model_comparison_sessions.csv) back in the chat.")

if __name__ == "__main__":
    main()
