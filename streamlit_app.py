"""
Fraud Detection — Interactive Demo.

Run with:
    streamlit run streamlit_app.py
"""

import warnings

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

st.set_page_config(page_title="Fraud Detection Demo",
                   page_icon="🔍", layout="centered")

MODEL_PATH = "models/final_model.joblib"
FEATURE_COLS_PATH = "models/feature_cols.joblib"
THRESHOLD_PATH = "models/threshold.joblib"


@st.cache_resource
def load_artifacts():
    # Record (rather than silently swallow) scikit-learn's pickle-version warnings so the
    # app can tell the user when the model was saved with a different scikit-learn version.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURE_COLS_PATH)
        threshold = float(joblib.load(THRESHOLD_PATH))
    version_warnings = [
        str(w.message) for w in caught
        if issubclass(w.category, InconsistentVersionWarning)
    ]
    return model, feature_cols, threshold, version_warnings


st.title("🔍 Fraud Detection — Live Transaction Scoring")
st.caption(
    "Enter a transaction's details below to get a live fraud-risk score from the "
    "trained model (PaySim1, TRANSFER / CASH_OUT transactions)."
)

try:
    model, feature_cols, threshold, version_warnings = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run `fraud-detection-system.ipynb` first — "
        "it saves `models/final_model.joblib`, `models/feature_cols.joblib` and "
        "`models/threshold.joblib` after the final-model-selection step."
    )
    st.stop()
except Exception as exc:  # e.g. artifacts pickled with an incompatible library version
    st.error(
        "Model artifacts could not be loaded — they were most likely saved with a different "
        "scikit-learn / XGBoost version than the one installed here. Reinstall the versions in "
        "`requirements.txt`, or re-run `fraud-detection-system.ipynb` to regenerate the artifacts. "
        f"({type(exc).__name__}: {exc})"
    )
    st.stop()

if version_warnings:
    st.warning(
        "The model was saved with a different scikit-learn version than the one running this app; "
        "predictions may be unreliable. " + version_warnings[0]
    )

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        txn_type = st.selectbox("Transaction type", ["CASH_OUT", "TRANSFER"])
        amount = st.number_input(
            "Amount", min_value=0.0, value=5000.0, step=100.0)
        old_balance_org = st.number_input(
            "Sender's balance before transaction", min_value=0.0, value=10000.0, step=100.0
        )

    with col2:
        new_balance_orig = st.number_input(
            "Sender's balance after transaction", min_value=0.0, value=5000.0, step=100.0
        )
        old_balance_dest = st.number_input(
            "Recipient's balance before transaction", min_value=0.0, value=0.0, step=100.0
        )
        new_balance_dest = st.number_input(
            "Recipient's balance after transaction", min_value=0.0, value=5000.0, step=100.0
        )

    submitted = st.form_submit_button("Score this transaction")

if submitted:
    # Re-derive exactly the same engineered features the notebook computes,
    # from the raw fields a user can plausibly supply for a single transaction.
    error_balance_orig = old_balance_org - amount - new_balance_orig
    error_balance_dest = old_balance_dest + amount - new_balance_dest
    amount_to_oldbalance_ratio = amount / \
        old_balance_org if old_balance_org > 0 else 0.0
    orig_balance_zeroed = int(new_balance_orig == 0)
    dest_balance_was_zero = int(old_balance_dest == 0)

    row = {
        "amount": amount,
        "errorBalanceOrig": error_balance_orig,
        "errorBalanceDest": error_balance_dest,
        "amount_to_oldbalanceOrg_ratio": amount_to_oldbalance_ratio,
        "orig_balance_zeroed": orig_balance_zeroed,
        "dest_balance_was_zero": dest_balance_was_zero,
        "type_CASH_OUT": int(txn_type == "CASH_OUT"),
        "type_TRANSFER": int(txn_type == "TRANSFER"),
        # Velocity features require the sender's transaction history, which a
        # single-transaction form can't supply — defaulted to 0 (first-seen
        # account). Known limitation of this demo vs. a real pipeline, which
        # would look these up from a live account-activity store.
        "orig_txn_count_so_far": 0,
        "orig_cum_amount_so_far": 0.0,
    }

    # Refuse to score if the model expects a feature this form doesn't compute,
    # instead of silently filling it with 0.
    missing_features = [col for col in feature_cols if col not in row]
    if missing_features:
        st.error(
            "The model expects features this demo does not compute: "
            f"{', '.join(missing_features)}. Update `streamlit_app.py` to match the notebook's "
            "feature engineering."
        )
        st.stop()

    # Build the row in the exact column order the model was trained on.
    input_df = pd.DataFrame([{col: row[col] for col in feature_cols}])

    fraud_probability = model.predict_proba(input_df)[0, 1]

    st.divider()
    st.metric("Fraud risk score", f"{fraud_probability:.2%}")
    st.progress(min(float(fraud_probability), 1.0))
    st.caption(
        f"Flagged as fraud when the score is at or above {threshold:.4f} — the operating "
        "threshold the notebook selected on the validation set."
    )

    if fraud_probability >= threshold:
        st.error("⚠️ This transaction is flagged as likely fraudulent.")
    else:
        st.success("✅ This transaction looks legitimate.")

    with st.expander("Engineered features sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Demo only — trained on synthetic PaySim1 data. Not validated against real "
    "transaction data. See the notebook's Act section for limitations."
)
