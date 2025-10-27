import streamlit as st
import pandas as pd
import numpy as np
from feature_extractor import extract_from_iris
from predict_pipeline import predict_with_xgb, predict_with_ann
from config import FEATURE_LIST

st.set_page_config(page_title="PGA Predictor (P-wave -> PGA)", layout="wide")

st.title("PGA Predictor from P-wave Features")
st.markdown("Upload a CSV of the 17 P-wave features, or extract demo examples from IRIS.")

col1, col2 = st.columns([2,1])

with col1:
    uploaded = st.file_uploader("Upload CSV with columns: " + ", ".join(FEATURE_LIST), type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Loaded features (first rows):")
        st.dataframe(df.head())
    else:
        st.info("Or use the extractor in the right panel to fetch demo seismograms from IRIS.")
        df = None

with col2:
    st.header("Extractor (IRIS)")
    num = st.number_input("Number of samples to extract", min_value=1, max_value=50, value=5)
    if st.button("Extract from IRIS (demo)"):
        with st.spinner("Contacting IRIS and extracting features (may take some seconds)..."):
            df_ex = extract_from_iris(num_samples=int(num))
        if df_ex is None or df_ex.empty:
            st.error("No records extracted. Try increasing samples or different stations.")
        else:
            st.success(f"Extracted {len(df_ex)} records.")
            st.dataframe(df_ex.head())
            df = df_ex[FEATURE_LIST]

# if df available, allow prediction
if df is not None:
    st.sidebar.header("Prediction")
    model_choice = st.sidebar.selectbox("Model", ("XGBoost", "ANN", "Both"))
    if st.sidebar.button("Predict PGA"):
        # ensure features present
        missing = [c for c in FEATURE_LIST if c not in df.columns]
        if missing:
            st.error("Missing feature columns: " + ", ".join(missing))
        else:
            df_in = df[FEATURE_LIST].copy()
            if model_choice in ("XGBoost","Both"):
                try:
                    preds_xgb = predict_with_xgb(df_in)
                except Exception as e:
                    st.error("XGBoost prediction error: " + str(e))
                    preds_xgb = None
            else:
                preds_xgb = None
            if model_choice in ("ANN","Both"):
                try:
                    preds_ann = predict_with_ann(df_in)
                except Exception as e:
                    st.error("ANN prediction error: " + str(e))
                    preds_ann = None
            # show results
            out_df = pd.DataFrame(index=df_in.index)
            if preds_xgb is not None:
                out_df["PGA_xgb"] = preds_xgb
            if preds_ann is not None:
                out_df["PGA_ann"] = preds_ann
            st.subheader("Predictions")
            st.dataframe(out_df.head(20))
            st.download_button("Download predictions CSV", out_df.to_csv(index=False), file_name="predictions.csv")
            # simple scatter plot if both present
            if preds_xgb is not None and preds_ann is not None:
                st.subheader("XGBoost vs ANN predictions (scatter)")
                st.altair_chart(
                    (pd.DataFrame({"xgb":preds_xgb,"ann":preds_ann})
                     .reset_index().rename(columns={0:"idx"}))
                    .pipe(lambda d: st.vega_lite_chart(d, {
                        "mark": {"type":"point","tooltip":True},
                        "encoding": {
                            "x": {"field":"xgb","type":"quantitative"},
                            "y": {"field":"ann","type":"quantitative"},
                        }
                    })), use_container_width=True)
