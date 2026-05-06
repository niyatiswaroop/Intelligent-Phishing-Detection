import streamlit as st
import joblib
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tensorflow as tf

@st.cache_resource 
def build_fuzzy_system():
    spam_score = ctrl.Antecedent(np.arange(0, 101, 1), 'spam_score')
    domain_trust = ctrl.Antecedent(np.arange(0, 11, 1), 'domain_trust')
    threat_level = ctrl.Consequent(np.arange(0, 101, 1), 'threat_level')

    spam_score.automf(3, names=['low', 'medium', 'high'])
    domain_trust.automf(3, names=['untrusted', 'neutral', 'trusted'])
    threat_level.automf(3, names=['safe', 'warning', 'critical'])

    rule1 = ctrl.Rule(spam_score['low'], threat_level['safe'])
    rule2 = ctrl.Rule(spam_score['medium'] & domain_trust['trusted'], threat_level['safe'])
    rule3 = ctrl.Rule(spam_score['medium'] & domain_trust['untrusted'], threat_level['warning'])
    rule4 = ctrl.Rule(spam_score['high'] & domain_trust['trusted'], threat_level['warning'])
    rule5 = ctrl.Rule(spam_score['high'] & domain_trust['untrusted'], threat_level['critical'])
    rule6 = ctrl.Rule(spam_score['medium'] & domain_trust['neutral'], threat_level['warning'])
    rule7 = ctrl.Rule(spam_score['high'] & domain_trust['neutral'], threat_level['critical'])

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7]))

@st.cache_resource
def load_models():
    vec = joblib.load('vectorizer.pkl')
    nb = joblib.load('nb_model.pkl')
    mlp = tf.keras.models.load_model('mlp_model.keras')
    w = joblib.load('weights.pkl')
    return vec, nb, mlp, w

st.set_page_config(page_title="Threat Analyzer", layout="centered")
st.title("🛡️ Hybrid Phishing & Threat Analyzer")



try:
    vectorizer, nb_model, mlp_model, weights = load_models()
    fuzzy_sim = build_fuzzy_system()
except Exception as e:
    st.error(f"Error loading models. Run '2_train_models.py' first.")
    st.stop()

email_text = st.text_area("Paste the suspicious email text here:", height=150)
domain_trust = st.slider("Sender Domain Trust Score (0 = Shady/Unknown, 10 = Verified)", 0, 10, 5)

if st.button("Analyze Threat Level", type="primary"):
    if email_text.strip():
        with st.spinner("Analyzing text..."):
            email_vec = vectorizer.transform([email_text]).toarray()
            
            p_nb = nb_model.predict_proba(email_vec)[0][1]
            p_mlp = float(mlp_model.predict(email_vec, verbose=0)[0][0])
            
            final_prob = (weights['nb_weight'] * p_nb) + (weights['mlp_weight'] * p_mlp)
            spam_score_100 = final_prob * 100
            
            fuzzy_sim.input['spam_score'] = spam_score_100
            fuzzy_sim.input['domain_trust'] = domain_trust
            fuzzy_sim.compute()
            final_threat = fuzzy_sim.output['threat_level']
            
            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Naïve Bayes", f"{p_nb*100:.1f}%")
            col2.metric("Neural Net", f"{p_mlp*100:.1f}%")
            col3.metric("GA Ensemble", f"{spam_score_100:.1f}%")
            
            st.subheader(f"Final Fuzzy Threat Level: {final_threat:.1f} / 100")
            
            if final_threat > 75:
                st.error("🚨 VERY DANGEROUS - Locked Automatically")
            elif final_threat > 60:
                st.error("🔴 DANGEROUS - Do Not Open")
            elif final_threat > 40:
                st.warning("🟠 MODERATE - Be Careful")
            elif final_threat > 20:
                st.info("🟡 ALERT - Slight Warning")
            elif final_threat > 10:
                st.info("⚪ NEUTRAL - No Clear Risk")
            else:
                st.success("🟢 SAFE - Okay to Open")
    else:
        st.warning("Please enter some email text.")