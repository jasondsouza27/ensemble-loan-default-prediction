"""
Streamlit Dashboard — Ensemble-Based Loan Default Risk Prediction
Based on Akinjole et al., MDPI Mathematics 2024, 12(21), 3423.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CreditGuard AI | Loan Default Risk Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
sys.path.append(str(BASE_DIR))

from src.config import STATE_TO_REGION
from src.data_preprocessing import LoanDataPreprocessor
from src.explainability import explain_single_loan


def highlight_max_cell(s):
    """
    Highlights the maximum value with a green badge where both
    background (#86EFAC) and text color (#022C22) are explicitly set.
    Guarantees full high-contrast readability in both Light and Dark themes.
    """
    return [
        "background-color: #86EFAC; color: #022C22; font-weight: 800;"
        if v == s.max()
        else ""
        for v in s
    ]


# ── Global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* ── typography ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

/* ── hide chrome ── */
#MainMenu, footer, header {visibility: hidden;}

/* ── hero ── */
.hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
    color: #fff;
    padding: 26px 30px 22px;
    border-radius: 12px;
    margin-bottom: 18px;
}
.hero h1 {
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    margin: 0 0 6px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hero p {
    font-size: 0.88rem;
    color: #CBD5E1;
    margin: 0;
    line-height: 1.45;
}
.hero-pills { margin-top: 10px; display: flex; gap: 6px; flex-wrap: wrap; }
.pill {
    display: inline-block;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 9999px;
    background: rgba(255,255,255,0.10);
    color: #E2E8F0;
    border: 1px solid rgba(255,255,255,0.16);
}

/* ── kpi row ── */
.kpi-row { display: flex; gap: 10px; margin-bottom: 18px; }
.kpi {
    flex: 1;
    background: #fff;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 13px 15px;
}
.kpi-lbl { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-bottom: 3px; }
.kpi-val { font-size: 1.35rem; font-weight: 700; color: #0F172A !important; }
.kpi-sub { font-size: 0.72rem; color: #334155 !important; font-weight: 500; }

/* ── card headers ── */
.card-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-color, inherit) !important;
    margin-bottom: 10px;
}
.card-title svg {
    stroke: #38BDF8 !important;
    flex-shrink: 0;
}

/* ── decision banners ── */
.verdict-safe {
    background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
    border-left: 5px solid #10B981;
    border-radius: 10px; padding: 18px 22px; margin-bottom: 14px;
    color: #064E3B !important;
}
.verdict-safe * { color: #064E3B !important; }
.verdict-safe .verdict-title { font-size: 1.15rem; font-weight: 800; color: #064E3B !important; margin: 0 0 4px; }
.verdict-safe .verdict-prob { font-size: 0.92rem; color: #047857 !important; margin: 0 0 3px; }
.verdict-safe .verdict-note { font-size: 0.82rem; color: #065F46 !important; font-weight: 500; margin: 0; }

.verdict-warn {
    background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
    border-left: 5px solid #F59E0B;
    border-radius: 10px; padding: 18px 22px; margin-bottom: 14px;
    color: #78350F !important;
}
.verdict-warn * { color: #78350F !important; }
.verdict-warn .verdict-title { font-size: 1.15rem; font-weight: 800; color: #78350F !important; margin: 0 0 4px; }
.verdict-warn .verdict-prob { font-size: 0.92rem; color: #92400E !important; margin: 0 0 3px; }
.verdict-warn .verdict-note { font-size: 0.82rem; color: #78350F !important; font-weight: 500; margin: 0; }

.verdict-deny {
    background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
    border-left: 5px solid #EF4444;
    border-radius: 10px; padding: 18px 22px; margin-bottom: 14px;
    color: #7F1D1D !important;
}
.verdict-deny * { color: #7F1D1D !important; }
.verdict-deny .verdict-title { font-size: 1.15rem; font-weight: 800; color: #7F1D1D !important; margin: 0 0 4px; }
.verdict-deny .verdict-prob { font-size: 0.92rem; color: #991B1B !important; margin: 0 0 3px; }
.verdict-deny .verdict-note { font-size: 0.82rem; color: #7F1D1D !important; font-weight: 500; margin: 0; }

/* ── section labels ── */
.sec-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #94A3B8; margin-bottom: 8px;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    font-weight: 600; font-size: 0.85rem;
    padding: 8px 18px; border-radius: 6px 6px 0 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load pipeline ────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    paths = {k: MODELS_DIR / f for k, f in {
        "bundle": "trained_models_bundle.joblib",
        "prep": "preprocessor.joblib",
        "feat": "selected_features.joblib",
        "shap": "shap_explainer.joblib",
    }.items()}
    return tuple(joblib.load(p) if p.exists() else None for p in paths.values())

bundle, preprocessor, selected_features, shap_explainer = load_pipeline()

# ── Hero header ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        </svg>
        Ensemble-Based Loan Default Risk Prediction
    </h1>
    <p>Replication & Production Architecture of Akinjole et al. (MDPI Mathematics 2024, 12, 3423) — Stacking Ensemble with SMOTE+ENN & SHAP Interpretability</p>
    <div class="hero-pills">
        <span class="pill">Stacking Classifier</span>
        <span class="pill">SMOTE + ENN Resampling</span>
        <span class="pill">RFECV Feature Selection</span>
        <span class="pill">XGBoost · Random Forest · MLP · SVM · DT · AdaBoost</span>
        <span class="pill">SHAP Explainability</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ────────────────────────────────────────────────
st.markdown("""
<div class="kpi-row">
    <div class="kpi">
        <div class="kpi-lbl">Champion Architecture</div>
        <div class="kpi-val">Stacking A</div>
        <div class="kpi-sub">LogReg Meta-Learner</div>
    </div>
    <div class="kpi">
        <div class="kpi-lbl">Accuracy</div>
        <div class="kpi-val">93.7 %</div>
        <div class="kpi-sub">↑ 13.7 pp vs baseline</div>
    </div>
    <div class="kpi">
        <div class="kpi-lbl">Precision</div>
        <div class="kpi-val">95.6 %</div>
        <div class="kpi-sub">Positive-class reliability</div>
    </div>
    <div class="kpi">
        <div class="kpi-lbl">Recall</div>
        <div class="kpi-val">95.5 %</div>
        <div class="kpi-sub">True defaulter capture</div>
    </div>
    <div class="kpi">
        <div class="kpi-lbl">ROC-AUC</div>
        <div class="kpi-val">0.978</div>
        <div class="kpi-sub">Top-tier discrimination</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Risk Assessment",
    "Research Benchmarks",
    "Methodology & Architecture",
    "Portfolio Scoring",
])

# ═════════════════════════════════════════════════════════════
# TAB 1 — Live Risk Assessment
# ═════════════════════════════════════════════════════════════
with tab1:

    # ── Quick-fill selector ──
    st.markdown('<div class="sec-label">Applicant Profile Template</div>', unsafe_allow_html=True)
    preset = st.radio(
        "Select a preset profile to auto-fill the form, or choose Custom:",
        ["Prime Borrower (Low Risk)", "Borderline Applicant (Moderate Risk)", "Subprime Borrower (High Risk)", "Custom Application"],
        horizontal=True,
        label_visibility="collapsed",
    )

    presets = {
        "Prime Borrower (Low Risk)":  dict(amt=12000, trm=" 36 months", rate=7.5, inc=95000, dti=11.5, fico=780, home="MORTGAGE", verif="Source Verified", dlq=0, bc75=10.0, bkr=0, purp="credit_card"),
        "Borderline Applicant (Moderate Risk)":      dict(amt=16000, trm=" 36 months", rate=14.2, inc=56000, dti=23.5, fico=685, home="RENT",     verif="Verified",        dlq=0, bc75=45.0, bkr=0, purp="debt_consolidation"),
        "Subprime Borrower (High Risk)":        dict(amt=28000, trm=" 60 months", rate=24.5, inc=34000, dti=39.5, fico=630, home="RENT",     verif="Not Verified",    dlq=2, bc75=85.0, bkr=1, purp="small_business"),
        "Custom Application":          dict(amt=15000, trm=" 36 months", rate=12.5, inc=65000, dti=16.5, fico=710, home="MORTGAGE", verif="Verified",        dlq=0, bc75=25.0, bkr=0, purp="debt_consolidation"),
    }
    P = presets[preset]

    # ── Input form ──
    st.markdown('<div class="sec-label">Loan Application Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        with st.container(border=True):
            st.markdown("""
            <div class="card-title">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.5"><rect width="20" height="14" x="2" y="5" rx="2"/><line x1="2" x2="22" y1="10" y2="10"/></svg>
                Loan Terms & Structure
            </div>
            """, unsafe_allow_html=True)
            loan_amnt = st.number_input("Loan amount (USD)", 1000, 50000, P["amt"], 500)
            term = st.selectbox("Term", [" 36 months", " 60 months"], 0 if "36" in P["trm"] else 1)
            int_rate = st.slider("Interest rate %", 5.0, 32.0, float(P["rate"]), 0.1)
            purposes = ["credit_card","debt_consolidation","home_improvement","small_business",
                         "major_purchase","medical","car","moving","vacation","house","other"]
            purpose = st.selectbox("Purpose", purposes, purposes.index(P["purp"]) if P["purp"] in purposes else 0)
            n_months = 36 if "36" in term else 60
            r_mo = int_rate / 100 / 12
            installment = round(loan_amnt * (r_mo * (1+r_mo)**n_months) / ((1+r_mo)**n_months - 1), 2)
            st.caption(f"Est. monthly payment: **${installment:,.2f}**")

    with c2:
        with st.container(border=True):
            st.markdown("""
            <div class="card-title">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.5"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
                Borrower & Financial Profile
            </div>
            """, unsafe_allow_html=True)
            annual_inc = st.number_input("Annual income (USD)", 10000, 1000000, P["inc"], 5000)
            dti = st.slider("DTI ratio %", 0.0, 60.0, float(P["dti"]), 0.5)
            homes = ["MORTGAGE","RENT","OWN","OTHER"]
            home_ownership = st.selectbox("Home ownership", homes, homes.index(P["home"]))
            verifs = ["Verified","Source Verified","Not Verified"]
            verification_status = st.selectbox("Income verification", verifs, verifs.index(P["verif"]))
            states = list(STATE_TO_REGION.keys())
            addr_state = st.selectbox("State", states, states.index("CA"))

    with c3:
        with st.container(border=True):
            st.markdown("""
            <div class="card-title">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                Credit Bureau History
            </div>
            """, unsafe_allow_html=True)
            fico_range_low = st.slider("FICO score", 620, 850, P["fico"], 5)
            delinq_2yrs = st.number_input("Delinquencies (2 yr)", 0, 10, P["dlq"])
            percent_bc_gt_75 = st.slider("Bank-cards > 75 % util.", 0.0, 100.0, float(P["bc75"]), 5.0)
            revol_bal = st.number_input("Revolving balance (USD)", 0, 200000, int(annual_inc * 0.18), 1000)
            pub_rec_bankruptcies = st.selectbox("Bankruptcies on record", [0, 1, 2], P["bkr"])

    # derived / secondary features
    open_acc, total_acc = 12, 24
    mort_acc = 2 if home_ownership == "MORTGAGE" else 0
    avg_cur_bal = annual_inc * 0.30
    raw = dict(
        loan_amnt=loan_amnt, term=term, int_rate=int_rate, installment=installment,
        annual_inc=annual_inc, home_ownership=home_ownership,
        verification_status=verification_status, purpose=purpose,
        addr_state=addr_state, dti=dti, delinq_2yrs=delinq_2yrs,
        fico_range_low=fico_range_low, open_acc=open_acc, pub_rec=pub_rec_bankruptcies,
        revol_bal=revol_bal, total_acc=total_acc,
        initial_list_status="w", application_type="Individual",
        tot_hi_cred_lim=avg_cur_bal*3.5 + loan_amnt, avg_cur_bal=avg_cur_bal,
        bc_open_to_buy=max(1000, 30000 - revol_bal), delinq_amnt=0,
        mo_sin_old_rev_tl_op=190, mo_sin_rcnt_rev_tl_op=10, mo_sin_rcnt_tl=5,
        mort_acc=mort_acc, mths_since_recent_bc=8,
        num_actv_bc_tl=3, num_actv_rev_tl=5, num_bc_sats=4,
        num_bc_tl=6, num_op_rev_tl=7,
        pct_tl_nvr_dlq=100 if delinq_2yrs == 0 else 80,
        percent_bc_gt_75=percent_bc_gt_75,
        pub_rec_bankruptcies=pub_rec_bankruptcies,
    )

    # ── Predict button ──
    st.write("")
    _, btn_c, _ = st.columns([1, 2, 1])
    with btn_c:
        go = st.button("Evaluate Default Risk", type="primary", use_container_width=True)

    if go or ("_preset" not in st.session_state) or st.session_state.get("_preset") != preset:
        st.session_state["_preset"] = preset
        if bundle and preprocessor and selected_features:
            X_in = preprocessor.transform(pd.DataFrame([raw]))[selected_features].values
            champ = bundle["models"].get("Stacking A", list(bundle["models"].values())[0])
            prob = champ.predict_proba(X_in)[0, 1] if hasattr(champ, "predict_proba") else float(champ.predict(X_in)[0])
            st.session_state["prob"] = prob
            st.session_state["X_in"] = X_in

    # ── Results ──
    if "prob" in st.session_state and bundle:
        prob = st.session_state["prob"]
        X_in = st.session_state["X_in"]
        mdls = bundle["models"]

        st.divider()
        st.markdown('<div class="sec-label">Assessment Findings</div>', unsafe_allow_html=True)

        left, right = st.columns([1, 1], gap="large")

        with left:
            # verdict banner
            if prob < 0.35:
                cls, title = "verdict-safe", "APPROVED — LOW DEFAULT RISK"
                note = "Meets prime institutional underwriting criteria. Automated approval recommended."
            elif prob < 0.60:
                cls, title = "verdict-warn", "CONDITIONAL — MANUAL REVIEW REQUIRED"
                note = "Borderline applicant. Request co-borrower or supplementary income verification."
            else:
                cls, title = "verdict-deny", "DECLINED — ELEVATED DEFAULT RISK"
                note = "Elevated charge-off likelihood. Risk metrics exceed institutional threshold."

            st.markdown(f"""
            <div class="{cls}">
                <div class="verdict-title">{title}</div>
                <div class="verdict-prob">Default probability: <b>{prob*100:.2f} %</b></div>
                <div class="verdict-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

            # consensus table
            with st.expander("Multi-Model Consensus Breakdown", expanded=True):
                rows = []
                for nm in ["XGBoost","Random Forest","MLP","ADABoost","Decision Tree","SVM",
                            "Voting A","Voting B","Stacking A","Stacking B"]:
                    if nm not in mdls:
                        continue
                    m = mdls[nm]
                    p = m.predict_proba(X_in)[0,1] if hasattr(m,"predict_proba") else float(m.predict(X_in)[0])
                    rows.append({
                        "Model": nm,
                        "P(Default)": f"{p*100:.2f} %",
                        "Verdict": "Charge-Off (1)" if p >= 0.5 else "Fully Paid (0)",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with right:
            with st.expander("Explainable AI: SHAP Attribution", expanded=True):
                st.caption("Decomposing applicant features pushing the model toward Default (+red) or Non-Default (-green).")
                if shap_explainer:
                    try:
                        df_s = pd.DataFrame(X_in, columns=selected_features)
                        cdf = explain_single_loan(shap_explainer, df_s, selected_features)
                        top = cdf.head(8).iloc[::-1]

                        fig, ax = plt.subplots(figsize=(5.5, 3.2))
                        colors = ["#EF4444" if v > 0 else "#10B981" for v in top["SHAP_Contribution"]]
                        ax.barh(top["Feature"], top["SHAP_Contribution"], color=colors, height=0.55, edgecolor="none")
                        ax.axvline(0, color="#CBD5E1", lw=1)
                        ax.set_xlabel("SHAP value", fontsize=8)
                        ax.tick_params(labelsize=7.5)
                        ax.spines[["top","right"]].set_visible(False)
                        ax.grid(axis="x", ls=":", alpha=0.4)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                        st.dataframe(
                            cdf.head(6)[["Feature","Applicant_Value","SHAP_Contribution","Impact"]].rename(
                                columns={"SHAP_Contribution": "SHAP Score"}
                            ),
                            use_container_width=True, hide_index=True,
                        )
                    except Exception as exc:
                        st.info(f"SHAP visualization: {exc}")
                else:
                    st.info("SHAP explainer not available — run the training pipeline first.")

# ═════════════════════════════════════════════════════════════
# TAB 2 — Research Benchmarks
# ═════════════════════════════════════════════════════════════
with tab2:

    st.markdown('<div class="sec-label">Paper Results — Akinjole et al. (2024)</div>', unsafe_allow_html=True)

    sub1, sub2, sub3 = st.tabs([
        "Models & Ensembles",
        "Resampling & Scaling",
        "Literature Comparison",
    ])

    # ── Models & Ensembles ──
    with sub1:
        lc, rc = st.columns(2, gap="medium")
        with lc:
            st.markdown("##### Table 6 — Individual classifiers")
            st.dataframe(pd.DataFrame({
                "Model": ["XGBoost","Random Forest","MLP (3 × 150)","AdaBoost","Decision Tree","SVM (RBF)"],
                "Accuracy": [.9156,.8987,.8775,.8458,.7778,.7318],
                "Precision": [.9478,.8996,.9008,.8548,.7743,.9476],
                "Recall": [.9330,.9656,.9305,.9439,.9713,.6601],
                "AUC": [.9726,.9589,.9229,.9305,.7256,.8824],
            }).style.format({c: "{:.2%}" for c in ["Accuracy","Precision","Recall","AUC"]})
              .apply(highlight_max_cell, subset=["Accuracy","AUC"]),
              use_container_width=True, hide_index=True)

        with rc:
            st.markdown("##### Table 7 — Ensemble architectures")
            st.dataframe(pd.DataFrame({
                "Ensemble": ["Stacking A [Champion]","Stacking B","Voting B","Voting A"],
                "Composition": ["All 6 + LogReg meta","Top 3 + LogReg meta","Top 3 soft vote","All 6 soft vote"],
                "Accuracy": [.9369,.9188,.9166,.9109],
                "Precision": [.9559,.9409,.9314,.9099],
                "Recall": [.9555,.9454,.9532,.9710],
                "AUC": [.9781,.9708,.9687,.9703],
            }).style.format({c: "{:.2%}" for c in ["Accuracy","Precision","Recall","AUC"]})
              .apply(highlight_max_cell, subset=["Accuracy","Precision","AUC"]),
              use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("##### Generated evaluation charts")
        ic1, ic2, ic3 = st.columns(3)
        for col, fname, cap in [
            (ic1, "roc_all_models.png",        "Figure 12: ROC Curves (All Models)"),
            (ic2, "auc_comparison_all.png",     "Figure 13: AUC Performance Comparison"),
            (ic3, "shap_feature_importance.png","Figure 9: Global Feature Importance (SHAP)"),
        ]:
            p = MODELS_DIR / fname
            with col:
                if p.exists():
                    st.image(str(p), caption=cap, use_container_width=True)

    # ── Resampling & Scaling ──
    with sub2:
        lc2, rc2 = st.columns(2, gap="medium")
        with lc2:
            st.markdown("##### Table 5 — Resampling benchmark (XGBoost evaluator)")
            st.dataframe(pd.DataFrame({
                "Method": ["SMOTE + ENN [Champion]","SMOTE-Tomek","SMOTE","ADASYN","ROS","Tomek-Links","None","RUS"],
                "Accuracy": [.9049,.8762,.8766,.8745,.6874,.7947,.8047,.6500],
                "Precision": [.9461,.9679,.9684,.9686,.6807,.5368,.5362,.6465],
                "Recall": [.9202,.7779,.7787,.7690,.7062,.1377,.1101,.6683],
                "AUC": [.9654,.9295,.9284,.9266,.7559,.7197,.7171,.7079],
            }).style.format({c: "{:.2%}" for c in ["Accuracy","Precision","Recall","AUC"]})
              .apply(highlight_max_cell, subset=["Accuracy","Recall","AUC"]),
              use_container_width=True, hide_index=True)

        with rc2:
            st.markdown("##### Table 3 — Outlier & normalization matrix")
            st.dataframe(pd.DataFrame({
                "Outlier": ["Winsorize [Selected]","Winsorize","Winsorize","IQR","IQR","Z-Score","Clip"],
                "Scaler": ["RobustScaler [Selected]","StandardScaler","MinMaxScaler","MinMaxScaler","RobustScaler","RobustScaler","RobustScaler"],
                "Accuracy": [.8045,.8044,.8040,.8275,.8274,.7963,.8038],
                "Recall": [.0582,.0584,.0567,.0035,.0031,.0449,.0550],
                "Precision": [.5664,.5625,.5544,.5882,.5294,.5497,.5516],
                "AUC": [.7039,.7038,.7032,.6407,.6410,.6972,.7050],
            }).style.format({c: "{:.4f}" for c in ["Accuracy","Recall","Precision","AUC"]}),
              use_container_width=True, hide_index=True)

        # show RFECV curve if available
        rfecv_p = MODELS_DIR / "rfecv_curve.png"
        if rfecv_p.exists():
            st.divider()
            st.markdown("##### Feature selection — RFECV curve (recall scorer)")
            st.image(str(rfecv_p), width=700)

    # ── Literature ──
    with sub3:
        st.markdown("##### Table 8 — Comparison with state-of-the-art")
        st.dataframe(pd.DataFrame({
            "Study": ["This study [Proposed]","Madaan et al. (2021)","Ma et al. (2018)","Chang et al. (2018)","Jumaa et al. (2023)"],
            "Resampling": ["SMOTE + ENN","None","None","Cluster US","SMOTE"],
            "Models": ["RF, DT, SVM, XGB, Ada, MLP","RF, DT","LGBM, XGB","LR, SVM, XGB","MLP, SVM, Ada"],
            "Ensemble": ["Stacking A + Voting","—","—","—","—"],
            "Best model": ["Stacking A","Random Forest","LightGBM","XGBoost","MLP"],
            "Accuracy": ["93.7 %","80.0 %","80.1 %","90.0 %","93.0 %"],
            "AUC": ["97.8 %","—","—","94.0 %","—"],
        }), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════
# TAB 3 — Methodology
# ═════════════════════════════════════════════════════════════
with tab3:

    st.markdown('<div class="sec-label">End-to-End Pipeline Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
```mermaid
flowchart LR
    A["Raw LendingClub\n2007 – 2018"] --> B["Clean & Filter\nr > 0.90 removal"]
    B --> C["Winsorize &\nRobustScaler"]
    C --> D["SMOTE + ENN\nResampling"]
    D --> E["RFECV\nRecall scorer"]
    E --> F["6 Base\nClassifiers"]
    F --> G["Stacking\nMeta-Learner"]
    G --> H["Decision\n& SHAP"]
```
    """)

    st.divider()

    lm, rm = st.columns(2, gap="large")

    with lm:
        with st.container(border=True):
            st.markdown("#### Winsorization & Robust Scaling")
            st.markdown(r"""
Financial features like `annual_inc` (max \$9.5 M) and `dti` (max 999) contain extreme
outliers that distort gradient-based learners.

**Winsorization** clips values to the 1st / 99th percentiles:

$$x_w = \text{clip}(x,\; Q_{0.01},\; Q_{0.99})$$

**RobustScaler** centres on the median and scales by IQR, so extreme tails
cannot compress the bulk of the distribution:

$$x_s = \frac{x - \tilde{x}}{Q_3 - Q_1}$$
            """)

    with rm:
        with st.container(border=True):
            st.markdown("#### SMOTE + ENN Hybrid Resampling")
            st.markdown(r"""
With an 80 : 20 class split, naïve models predict "Fully Paid" for every
applicant — high accuracy, zero recall.

**Step 1 — SMOTE** synthesises new minority samples along line segments
connecting $k$-nearest neighbours:

$$x_{\text{new}} = x_i + \lambda\,(x_{nn} - x_i),\quad \lambda \sim U(0,1)$$

**Step 2 — ENN** removes any sample whose $k$ nearest neighbours
disagree with its label, cleaning noisy decision boundaries created by
Step 1.
            """)

    st.divider()

    lm2, rm2 = st.columns(2, gap="large")

    with lm2:
        with st.container(border=True):
            st.markdown("#### Stacking Ensemble")
            st.markdown(r"""
Instead of simple majority vote, **Stacking** trains a meta-learner
(regularised Logistic Regression) on the *out-of-fold* probability
predictions of all six base classifiers:

$$P(\text{default}\mid x) = \sigma\!\Bigl(w_0 + \sum_{m=1}^{6} w_m\,\hat P_m(y\!=\!1\mid x)\Bigr)$$

Cross-validated folds prevent target leakage from base → meta stage.
            """)

    with rm2:
        with st.container(border=True):
            st.markdown("#### RFECV Feature Selection")
            st.markdown(r"""
Recursive Feature Elimination with Cross-Validation removes the
least-important feature at each step. The estimator (XGBoost) is
scored on **Recall** at each elimination round, and the optimal set
is the one that maximises recall on the held-out CV fold.

The paper identifies **48 optimal features** from an initial pool of
70+ engineered columns.
            """)

# ═════════════════════════════════════════════════════════════
# TAB 4 — Portfolio Scoring
# ═════════════════════════════════════════════════════════════
with tab4:

    st.markdown('<div class="sec-label">Batch Portfolio Simulation (1,000 Applications)</div>', unsafe_allow_html=True)
    st.write("Generate and score an institutional portfolio of **1,000 loan applications** using the champion Stacking Ensemble to assess aggregate risk exposure.")

    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        score_clicked = st.button("Score 1,000 Loans", type="primary", use_container_width=True)

    if score_clicked:
        if bundle and preprocessor and selected_features:
            from data.generate_sample_data import generate_lending_club_dataset

            with st.spinner("Scoring 1,000 loan applications through Stacking A ensemble..."):
                df_sim = generate_lending_club_dataset(n_samples=1000, random_state=77)
                X_b = preprocessor.transform(df_sim)[selected_features].values
                champ = bundle["models"].get("Stacking A")
                pr = champ.predict_proba(X_b)[:, 1] if hasattr(champ, "predict_proba") else champ.predict(X_b).astype(float)

                df_sim["P(Default) %"] = np.round(pr * 100, 2)
                df_sim["Verdict"] = np.where(pr < 0.35, "Approve", np.where(pr < 0.60, "Review", "Decline"))
                df_sim.index = range(1, len(df_sim) + 1)
                st.session_state["df_sim"] = df_sim
        else:
            st.info("Models not loaded. Run `python train_pipeline.py` first.")

    # Display portfolio results if scored
    if "df_sim" in st.session_state:
        df_sim = st.session_state["df_sim"]

        st.divider()

        # Summary KPIs across the 1,000 loans
        k1, k2, k3, k4 = st.columns(4)
        approve_n = (df_sim["Verdict"] == "Approve").sum()
        review_n  = (df_sim["Verdict"] == "Review").sum()
        decline_n = (df_sim["Verdict"] == "Decline").sum()

        k1.metric("Total Scored Applications", f"{len(df_sim):,}")
        k2.metric("Approved (Low Risk)", f"{approve_n}", f"{approve_n/10:.1f} %")
        k3.metric("Manual Review (Moderate)", f"{review_n}", f"{review_n/10:.1f} %")
        k4.metric("Declined (High Risk)", f"{decline_n}", f"{decline_n/10:.1f} %")

        st.write("")
        filter_col, export_col = st.columns([3, 1])

        with filter_col:
            v_filter = st.radio(
                "Filter portfolio view by decision:",
                ["All Applications (1,000)", f"Approved ({approve_n})", f"Manual Review ({review_n})", f"Declined ({decline_n})"],
                horizontal=True
            )

        with export_col:
            st.write("")
            csv_data = df_sim.to_csv(index=True, index_label="Application_ID").encode("utf-8")
            st.download_button(
                "Export Scored Portfolio (CSV)",
                data=csv_data,
                file_name="scored_loan_portfolio_1000.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Apply filter
        if "Approved" in v_filter:
            view_df = df_sim[df_sim["Verdict"] == "Approve"]
        elif "Manual Review" in v_filter:
            view_df = df_sim[df_sim["Verdict"] == "Review"]
        elif "Declined" in v_filter:
            view_df = df_sim[df_sim["Verdict"] == "Decline"]
        else:
            view_df = df_sim

        show_cols = ["loan_amnt", "term", "int_rate", "annual_inc", "dti", "fico_range_low", "P(Default) %", "Verdict"]
        
        st.caption(f"Showing **{len(view_df):,} of 1,000** applications (scroll vertically to view all rows):")
        st.dataframe(
            view_df[show_cols],
            use_container_width=True,
            height=520,
            hide_index=False
        )
    else:
        st.info("Click **Score 1,000 Loans** above to generate and evaluate the portfolio.")
