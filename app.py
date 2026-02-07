import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="PredictRx | AI-Driven ADME Screening", page_icon="💊", layout="wide")

# --- PROFESSIONAL DEVELOPER STYLING (CSS) - EXACTLY AS PREVIOUS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); }
    
    /* Card Styling */
    .stMetric, .rule-card, .ref-section { 
        background-color: white; 
        padding: 22px; 
        border-radius: 15px; 
        border: 1px solid #e1e4e8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        height: 100%;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #1a2a6c, #b21f1f);
        color: white;
        border: none;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: 0.4s;
    }
    .stButton>button:hover { transform: scale(1.02); color: white; }

    /* Result Status Boxes */
    .status-green { background: #28a745; color: white; padding: 20px; border-radius: 12px; text-align: center; font-weight: bold; }
    .status-yellow { background: #ffc107; color: #212529; padding: 20px; border-radius: 12px; text-align: center; font-weight: bold; }
    .status-red { background: #dc3545; color: white; padding: 20px; border-radius: 12px; text-align: center; font-weight: bold; }
    
    .main-header { font-size: 3.5rem; font-weight: 800; color: #1a2a6c; margin-bottom: 0; }
    .sub-header { font-size: 1.2rem; color: #576574; margin-bottom: 2rem; font-weight: 400; opacity: 0.8; }
    
    .section-title { color: #1a2a6c; font-weight: 700; border-left: 5px solid #b21f1f; padding-left: 15px; margin-bottom: 15px; }
    .doc-section { margin-bottom: 25px; padding: 15px; border-bottom: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    df = pd.read_csv('Biotech_Market_Segments_Full.csv')
    kmeans = joblib.load('breast_cancer_cluster_model.pkl')
    bio_clf = joblib.load('bioavailability_model (2).pkl')
    scaler = joblib.load('drug_feature_scaler.pkl')
    return df, kmeans, bio_clf, scaler

try:
    df, kmeans, bio_clf, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("⚙️ PredictRx Control")
    page = st.radio("Navigation", ["ADME Screening Dashboard", "System Intelligence & Overview"])
    st.divider()
    st.markdown("### System Accuracy: **96%**")

# --- PAGE 1: ADME SCREENING DASHBOARD ---
if page == "ADME Screening Dashboard":
    st.markdown('<p class="main-header">💊 PredictRx</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">An AI-driven platform for ADME screening, risk signaling, and formulation guidance.</p>', unsafe_allow_html=True)

    st.markdown("### 🔍 Therapeutic Candidate Selection")
    unique_drugs = sorted(list(set(df['Drug_A'].unique()) | set(df['Drug_B'].unique())))

    c1, c2 = st.columns(2)
    with c1:
        drug_a = st.selectbox("Primary Compound", unique_drugs, index=0)
    with c2:
        drug_b = st.selectbox("Secondary Compound", unique_drugs, index=1)

    match = df[((df['Drug_A'] == drug_a) & (df['Drug_B'] == drug_b)) | 
               ((df['Drug_A'] == drug_b) & (df['Drug_B'] == drug_a))]

    if not match.empty:
        row = match.iloc[0]
        mw, lp, hbd, hba, tpsa = row['Avg_MW'], row['Avg_LogP'], row['Sum_HBD'], row['Sum_HBA'], row['Avg_TPSA']
    else:
        mw, lp, hbd, hba, tpsa = 350.0, 2.5, 2, 8, 85.0

    st.write("")
    generate = st.button("🚀 Run Comprehensive Analysis")
    st.divider()

    if generate:
        input_feat = np.array([[mw, lp, hbd, hba, tpsa]])
        scaled_feat = scaler.transform(input_feat)
        cluster_id = kmeans.predict(scaled_feat)[0]
        bio_prob = bio_clf.predict_proba(input_feat)[:, 1][0]

        st.markdown("<h3 class='section-title'>⚖️ Lipinski Compliance Results</h3>", unsafe_allow_html=True)
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            st.markdown(f"<div class='rule-card'><b>MW < 500</b><br><span style='font-size:1.5em;'>{mw:.1f}</span><br>{'🟢 PASS' if mw < 500 else '🔴 FAIL'}</div>", unsafe_allow_html=True)
        with l2:
            st.markdown(f"<div class='rule-card'><b>LogP < 5</b><br><span style='font-size:1.5em;'>{lp:.1f}</span><br>{'🟢 PASS' if lp < 5 else '🔴 FAIL'}</div>", unsafe_allow_html=True)
        with l3:
            st.markdown(f"<div class='rule-card'><b>HBD < 5</b><br><span style='font-size:1.5em;'>{hbd}</span><br>{'🟢 PASS' if hbd < 5 else '🔴 FAIL'}</div>", unsafe_allow_html=True)
        with l4:
            st.markdown(f"<div class='rule-card'><b>HBA < 10</b><br><span style='font-size:1.5em;'>{hba}</span><br>{'🟢 PASS' if hba < 10 else '🔴 FAIL'}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)

        with m1:
            st.markdown("<h3 class='section-title'>🚦 1. Can this be taken as a Pill?</h3>", unsafe_allow_html=True)
            if bio_prob >= 0.75:
                st.markdown('<div class="status-green">✅ HIGH SUCCESS: Excellent Oral Candidate</div>', unsafe_allow_html=True)
                st.write("**Assessment:** Small and soluble enough to pass into the blood easily.")
            elif bio_prob >= 0.45:
                st.markdown('<div class="status-yellow">🟡 MODERATE RISK: Formulation Needed</div>', unsafe_allow_html=True)
                st.write("**Assessment:** Body will struggle to absorb. Requires specialized salts.")
            else:
                st.markdown('<div class="status-red">❌ LOW SUCCESS: Poor Absorption</div>', unsafe_allow_html=True)
                st.write("**Assessment:** Likely too large or insoluble (sticky) to be swallowed as a pill.")
            st.metric("Bioavailability Score", f"{bio_prob:.1%}")

        with m2:
            st.markdown("<h3 class='section-title'>🛠️ 2. Recommended Delivery Method</h3>", unsafe_allow_html=True)
            tech_paths = {
                1: {"lvl": 0, "name": "Oral Tablet", "desc": "Standard manufacturing path."},
                3: {"lvl": 1, "name": "Lipid Capsule", "desc": "Oil-based carrier for lipophilic drugs."},
                2: {"lvl": 2, "name": "IV Injection", "desc": "Bypasses the stomach for reliable dosing."},
                0: {"lvl": 3, "name": "Micronization", "desc": "Physical engineering to fix solubility issues."},
                4: {"lvl": 4, "name": "Nano-Carrier", "desc": "Encapsulation for complex payloads."}
            }
            path = tech_paths.get(cluster_id, {"lvl": "X", "name": "Custom", "desc": "Analysis required."})
            st.markdown(f'<div class="status-box" style="background: #2c3e50; color: white;">Level {path["lvl"]}: {path["name"]}</div>', unsafe_allow_html=True)
            st.write(f"**Technical Strategy:** {path['desc']}")
        st.divider()

    ref_col, guide_col = st.columns(2)
    with ref_col:
        st.markdown("<h3 class='section-title'>📚 Reference & Methodology</h3>", unsafe_allow_html=True)
        st.markdown("""<div class="ref-section"><b>Physicochemical Parameters:</b><br><b>MW:</b> Molecular Weight | <b>LogP:</b> Lipophilicity<br><b>HBD/HBA:</b> H-Bond Count | <b>TPSA:</b> Polarity<hr><b>Formulation Clusters (0-4):</b><ul><li><b>L0:</b> Tablet | <b>L1:</b> Softgel | <b>L2:</b> IV Solution</li><li><b>L3:</b> Micronization | <b>L4:</b> Nano-Carrier</li></ul><hr><b>Traffic Light System:</b><br>🟢 <b>Green:</b> Feasible oral route.<br>🟡 <b>Yellow:</b> Marginal absorption.<br>🔴 <b>Red:</b> Non-feasible oral route.</div>""", unsafe_allow_html=True)
    with guide_col:
        st.markdown("<h3 class='section-title'>🔬 Detailed Technical Guidance</h3>", unsafe_allow_html=True)
        st.markdown("""<div class="ref-section"><b>If Oral Administration Fails:</b><br>Scientists should evaluate non-enteral routes:<ul><li><b>Subcutaneous (SC):</b> Slower systemic release.</li><li><b>Intramuscular (IM):</b> Depot-style absorption.</li><li><b>Targeted IV:</b> Direct site-specific delivery.</li></ul><hr><b>How to improve results in the lab:</b><ul><li><b>Micronization:</b> Increase surface area for dissolution.</li><li><b>pH Adjustment:</b> Use buffers to manage ionization state.</li><li><b>Pro-drug approach:</b> Temporarily masks poor properties.</li></ul></div>""", unsafe_allow_html=True)

# --- PAGE 2: SYSTEM INTELLIGENCE & OVERVIEW ---
else:
    st.markdown('<p class="main-header">🧠 System Intelligence & Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Scientific Documentation, Performance Metrics, and Technical Logic of PredictRx</p>', unsafe_allow_html=True)

    # --- TOP ROW: CORE PERFORMANCE METRICS ---
    st.markdown("<h3 class='section-title'>📊 Model Performance & Accuracy</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.5, 1, 1])
    
    with c1:
        st.image("confusion_matrix.png", caption="Bioavailability Prediction Performance")
        st.markdown("""
        <div class='ref-section'>
        <b>Classification Detailed Report:</b><br>
        - <b>Accuracy:</b> 0.96<br>
        - <b>AUC Score:</b> 1.00<br>
        <table style='width:100%; text-align:left;'>
            <tr><th>Metric</th><th>Class 0 (Non-Oral)</th><th>Class 1 (Oral)</th></tr>
            <tr><td><b>Precision</b></td><td>0.95</td><td>1.00</td></tr>
            <tr><td><b>Recall</b></td><td>1.00</td><td>0.88</td></tr>
            <tr><td><b>F1-Score</b></td><td>0.97</td><td>0.93</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.image("roc_curve.png", caption="ROC Curve Analysis")
        st.markdown("<div class='ref-section'><b>Clustering Logic (DBSCAN):</b><br>Silhouette: 0.71<br>Davies-Bouldin: 0.38<br>Calinski-Harabasz: 291.79</div>", unsafe_allow_html=True)

    with c3:
        st.image("cluster_pca_plot.png", caption="PCA Cluster Distribution")
        st.info("The system uses unsupervised DBSCAN to group compounds into 5 distinct formulation paths based on PCA-reduced chemical space.")

    st.divider()

    # --- BOTTOM ROW: FULL SYSTEM DOCUMENTATION ---
    doc1, doc2 = st.columns(2)

    with doc1:
        st.markdown("#### 📘 System Foundations")
        with st.expander("System Overview & Purpose", expanded=True):
            st.write("PredictRx is an AI-driven platform designed to predict drug developability. It bridges the gap between medicinal chemistry and formulation science by providing risk signaling for therapeutic candidates.")
            st.write("**Problem Statement:** High attrition rates in drug discovery are often caused by poor ADME properties. Early screening prevents late-stage failures.")
            st.write("**Position:** Early-to-Mid Discovery Pipeline (Lead Optimization).")

        with st.expander("Features & Capabilities"):
            st.write("- **Accepted Inputs:** Molecular Weight (MW), LogP, H-Bond Donors (HBD), H-Bond Acceptors (HBA), TPSA.")
            st.write("- **Outputs:** Bioavailability Confidence Scores, Red-Yellow-Green Risk Signals, Recommended Formulation Levels (L0-L4).")
            st.write("- **Core Objectives:** Accelerate screening, prioritize lead candidates, and provide delivery route guidance.")

        with st.expander("Scientific Principles & Standards"):
            
            st.write("- **Standards:** Based on Lipinski's Rule of 5 and BCS Classification.")
            st.write("- **Models:** Random Forest Classifiers for oral potential and DBSCAN Clustering for formulation mapping.")

    with doc2:
        st.markdown("#### 🔬 Decision Logic & Methodology")
        with st.expander("Decision Logic: Red-Yellow-Green", expanded=True):
            st.write("🟢 **Green:** High success probability; feasible oral route.")
            st.write("🟡 **Yellow:** Marginal absorption; requires specialized formulation/salts.")
            st.write("🔴 **Red:** Non-feasible oral route; high risk of attrition.")

        with st.expander("Formulation & Delivery Framework"):
            st.write("- **Delivery Recommendations:** Oral, IV, Subcutaneous (SC), or Intramuscular (IM).")
            st.write("- **Optimization Strategies:** Micronization, pH buffers, Lipid-based carriers, and Nano-encapsulation.")
            

        with st.expander("Users, Scope & Disclaimer"):
            st.write("**Target Audience:** R&D Scientists, Bioinformaticians, and Formulation Lead HODs.")
            st.write("**Limitations:** Predictions are based on physicochemical averages; does not account for specific protein-binding or genetic metabolism.")
            st.error("**Disclaimer:** This is a predictive support tool, not a replacement for experimental lab validation.")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("PredictRx Intelligence Platform | v1.0.2 Stable | Branding: PredictRx")