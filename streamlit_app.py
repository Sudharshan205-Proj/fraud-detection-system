"""
Fraud Detection — Interactive Demo
Optional deployment piece for the Codec Technologies internship project
(Part 4.1, Share phase). Loads the artifacts saved at the end of
Fraud_Detection_System_Final.ipynb and scores a single, user-entered transaction.

Run with:
    streamlit run streamlit_app.py

Requires models/final_model.joblib and models/feature_cols.joblib to
exist — run the notebook at least once first.
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection Demo",
                   page_icon="🔍", layout="centered")

MODEL_PATH = "models/final_model.joblib"
FEATURE_COLS_PATH = "models/feature_cols.joblib"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, feature_cols


st.title("🔍 Fraud Detection — Live Transaction Scoring")
st.caption(
    "Enter a transaction's details below to get a live fraud-risk score from the "
    "trained model (PaySim1, TRANSFER / CASH_OUT transactions)."
)

try:
    model, feature_cols = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run `Fraud_Detection_System_Final.ipynb` first — "
        "it saves `models/final_model.joblib` and `models/feature_cols.joblib` "
        "after the final-model-selection step."
    )
    st.stop()

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

    # Build the row in the exact column order the model was trained on,
    # filling anything the notebook's feature set has but this form doesn't.
    input_df = pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])

    fraud_probability = model.predict_proba(input_df)[0, 1]

    st.divider()
    st.metric("Fraud risk score", f"{fraud_probability:.2%}")
    st.progress(min(float(fraud_probability), 1.0))

    if fraud_probability >= 0.5:
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
