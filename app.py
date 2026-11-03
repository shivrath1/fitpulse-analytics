import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="FitPulse Analytics", page_icon="📈", layout="wide")
st.title("FitPulse Analytics")
st.caption("Fitness App Intelligence · Explore engagement, activity, and user-level patterns")
st.markdown("""
<style>
[data-testid="stMetric"] {background: rgba(120,120,120,.07); padding: 14px; border-radius: 12px;}
.block-container {padding-top: 1.5rem;}
</style>
""", unsafe_allow_html=True)

REQUIRED = ["User ID","Gender","Age","Activity Level","Location","App Sessions","Distance Travelled (km)","Calories Burned"]
NUMERIC = ["Age","App Sessions","Distance Travelled (km)","Calories Burned"]
DEFAULT_PATH = "data/fitness_dataset.csv"

@st.cache_data
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)

uploaded = st.sidebar.file_uploader("Import CSV data", type=["csv"])
try:
    raw = load_csv(uploaded) if uploaded else load_csv(DEFAULT_PATH)
except Exception as exc:
    st.error(f"Unable to read the dataset: {exc}")
    st.stop()

missing_cols = [c for c in REQUIRED if c not in raw.columns]
if missing_cols:
    st.error("Required columns missing: " + ", ".join(missing_cols))
    st.stop()

df = raw.copy()
for col in NUMERIC:
    df[col] = pd.to_numeric(df[col], errors="coerce")
for col in ["Gender","Activity Level","Location"]:
    df[col] = df[col].astype("string").fillna("Unknown")

# Use a reproducible, non-destructive sensitivity toggle.
exclude_extreme = st.sidebar.checkbox("Sensitivity view: omit the 920 km distance record", value=False)
analysis_df = df.copy()
if exclude_extreme and (analysis_df["Distance Travelled (km)"] == 920).any():
    analysis_df = analysis_df[analysis_df["Distance Travelled (km)"] != 920].copy()
    st.sidebar.caption("The source data remains unchanged; one record is omitted only from current analysis.")

pages = ["Overview", "Explore", "Model Lab", "Responsible Use", "Recommendations"]
page = st.sidebar.radio("Workspace", pages)
st.sidebar.caption("Decision support prototype · Not a medical or clinical tool")

if page == "Overview":
    st.subheader("At a glance")
    a,b,c,d = st.columns(4)
    a.metric("Records", f"{len(analysis_df):,}")
    b.metric("Average sessions", f"{analysis_df['App Sessions'].mean():.1f}")
    c.metric("Average distance", f"{analysis_df['Distance Travelled (km)'].mean():.2f} km")
    d.metric("Average calories", f"{analysis_df['Calories Burned'].mean():.1f}")
    left,right = st.columns(2)
    left.plotly_chart(px.histogram(analysis_df, x="App Sessions", nbins=35, title="App sessions distribution"), use_container_width=True)
    act = analysis_df.groupby("Activity Level", dropna=False, as_index=False)["App Sessions"].mean()
    right.plotly_chart(px.bar(act, x="Activity Level", y="App Sessions", title="Average sessions by activity category"), use_container_width=True)
    st.markdown("### What to keep in mind")
    st.info("These are descriptive summaries. Associations do not establish cause and effect, and category definitions or data-generation methods should be verified before operational use.")
    st.caption(f"Missing values across the dataset: {int(analysis_df.isna().sum().sum())}. Duplicate full rows: {int(analysis_df.duplicated().sum())}.")

elif page == "Explore":
    st.subheader("Explore patterns")
    nums = ["Age","App Sessions","Distance Travelled (km)","Calories Burned"]
    x = st.selectbox("Horizontal axis", nums, index=1)
    y = st.selectbox("Vertical axis", nums, index=2)
    color = st.selectbox("Segment", ["Activity Level","Gender","Location","No segment"])
    plot_df = analysis_df.dropna(subset=[x,y])
    fig = px.scatter(plot_df, x=x, y=y, color=None if color=="No segment" else color,
                     hover_data=["User ID"], title=f"{x} vs {y}")
    st.plotly_chart(fig, use_container_width=True)
    corr = analysis_df[nums].corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=".2f", title="Pearson correlation matrix", aspect="auto"), use_container_width=True)
    st.markdown("### Data quality")
    q1,q2,q3 = st.columns(3)
    q1.metric("Rows", f"{len(analysis_df):,}")
    q2.metric("Missing cells", f"{int(analysis_df.isna().sum().sum()):,}")
    q3.metric("Duplicate rows", f"{int(analysis_df.duplicated().sum()):,}")
    st.dataframe(pd.DataFrame({"Missing":analysis_df.isna().sum(),"Missing %":(analysis_df.isna().mean()*100).round(2)}), use_container_width=True)
    st.dataframe(analysis_df[nums].describe().T, use_container_width=True)

elif page == "Model Lab":
    st.subheader("Predictive model comparison")
    st.write("Compare a simple mean baseline with linear regression and a random forest on the same held-out test set.")
    target = st.selectbox("Prediction target", ["App Sessions","Calories Burned"])
    if target == "App Sessions":
        include_activity = st.checkbox("Include Activity Level (sensitivity only; may encode the target)", value=False)
        features = ["Age","Gender","Location"] + (["Activity Level"] if include_activity else [])
        if include_activity:
            st.warning("Activity Level may be constructed from engagement or app sessions. Results with this feature are a sensitivity test and may not represent valid prospective prediction.")
    else:
        features = ["Age","Gender","Location","Activity Level","App Sessions","Distance Travelled (km)"]
        st.warning("Sessions and distance are strongly associated with calories in this dataset. Verify timing and feature availability before interpreting this as a real-world forecast.")

    model_data = analysis_df[features + [target]].dropna()
    if len(model_data) < 30:
        st.error("Not enough complete rows for a stable holdout evaluation.")
        st.stop()
    X, y = model_data[features], model_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    cat_cols = X.select_dtypes(include=["object","string","category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), num_cols),
        ("categorical", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                                 ("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])
    candidates = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=250,min_samples_leaf=3,random_state=42,n_jobs=-1)
    }
    fitted, rows = {}, []
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocess",pre),("model",estimator)])
        pipeline.fit(X_train,y_train)
        pred = pipeline.predict(X_test)
        fitted[name] = (pipeline,pred)
        rows.append({"Model":name,"MAE":mean_absolute_error(y_test,pred),
                     "RMSE":np.sqrt(mean_squared_error(y_test,pred)),"R²":r2_score(y_test,pred)})
    scores = pd.DataFrame(rows).sort_values("MAE")
    st.dataframe(scores.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","R²":"{:.4f}"}), use_container_width=True)
    chosen = st.selectbox("Inspect model", list(candidates.keys()), index=0)
    pipe,pred = fitted[chosen]
    m1,m2,m3 = st.columns(3)
    m1.metric("Test MAE", f"{mean_absolute_error(y_test,pred):.2f}")
    m2.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test,pred)):.2f}")
    m3.metric("Test R²", f"{r2_score(y_test,pred):.3f}")
    st.plotly_chart(px.scatter(x=y_test,y=pred,labels={"x":"Observed","y":"Predicted"},
                              title=f"Observed vs predicted · {chosen}"), use_container_width=True)
    residuals = pd.DataFrame({"Observed":y_test.to_numpy(),"Predicted":pred,
                              "Residual":y_test.to_numpy()-pred,
                              "Gender":X_test["Gender"].astype(str).to_numpy(),
                              "Location":X_test["Location"].astype(str).to_numpy(),
                              "Age":X_test["Age"].to_numpy()})
    residuals["Age band"] = pd.cut(residuals["Age"], bins=[0,29,39,49,59,120],
                                   labels=["18–29","30–39","40–49","50–59","60+"])
    st.markdown("### Error review by group")
    groupby = st.selectbox("Group errors by", ["Gender","Location","Age band"])
    group_metrics = residuals.groupby(groupby, observed=False).agg(
        Count=("Residual","size"), MAE=("Residual",lambda s: float(np.mean(np.abs(s)))),
        Mean_error=("Residual","mean")).reset_index()
    st.dataframe(group_metrics.style.format({"MAE":"{:.2f}","Mean_error":"{:.2f}"}), use_container_width=True)
    st.caption("Group-level error comparisons are diagnostic only. Small samples, missing variables, and data quality can affect these values; they do not establish fairness or causality.")

elif page == "Responsible Use":
    st.subheader("Responsible analytics principles")
    st.markdown("""
**Privacy by design**
- Use only fields justified by a clear analytics purpose.
- Exclude direct identifiers such as User ID from model features.
- Restrict access, protect data in transit and at rest, and define retention/deletion practices for a deployed service.
- Explain data use and provide meaningful user choices.

**Fairness and inclusion**
- Inspect error patterns across available groups, while recognizing that demographic categories are incomplete proxies for individual needs.
- Do not infer culture, ethnicity, disability, health status, or personal goals from location or gender.
- Offer accessible interactions and let users specify preferences and goals themselves.

**Interpretation**
- Describe relationships as associations, not causes.
- Verify how Activity Level and other fields were defined and when they become available.
- Present model uncertainty and limitations; do not use outputs for clinical decisions.
""")
    risks = pd.DataFrame([
        ["Unnecessary data collection","Purpose review and data minimisation"],
        ["Exposure or unauthorised access","Least privilege, encryption, logging, retention controls"],
        ["Secondary use without transparency","Clear notice, purpose limitation, consent/appropriate legal basis"],
        ["Uneven prediction errors","Group-based error review and ongoing monitoring"],
        ["Misleading fitness interpretation","Contextual explanations; no medical claims"],
        ["Cultural or accessibility assumptions","User-led preferences, accessible design, avoid proxy inference"],
        ["Leakage or invalid forecast","Document feature timing and validate on unseen data"]
    ], columns=["Risk","Mitigation"])
    st.dataframe(risks, use_container_width=True)
    st.caption("This prototype demonstrates analytical review and documentation. Production safeguards such as access control, encryption, consent workflows, and deletion processes require implementation and verification.")

else:
    st.subheader("Actionable business considerations")
    st.markdown("""
1. **Understand engagement before acting:** compare activity segments, but verify category definitions and data collection context.
2. **Use models as decision support:** benchmark against a simple baseline and validate on future or independently collected data before deployment.
3. **Avoid demographic stereotyping:** group averages are not individual recommendations.
4. **Improve data quality:** review missing values and extreme records with source context before correcting or excluding them.
5. **Build user agency:** let people define fitness goals, accessibility preferences, and notification choices.
6. **Monitor responsibly:** track model error, drift, privacy concerns, and user feedback over time.
""")
    st.download_button("Export analyzed data (CSV)", analysis_df.to_csv(index=False).encode("utf-8"),
                       file_name="fitpulse_analytics_export.csv", mime="text/csv")

st.sidebar.divider()
st.sidebar.caption("FitPulse Analytics · Exploratory decision support · Not medical advice")
