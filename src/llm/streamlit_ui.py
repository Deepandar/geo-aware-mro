import sys
from pathlib import Path

# Path fix for module imports
file_path = Path(__file__).resolve()
root_path = str(file_path.parent.parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import streamlit as st  # noqa: E402
from src.llm.rag_indexer import FastRAGIndexer  # noqa: E402
from src.llm.llm_advisory import LLMAdvisoryEngine  # noqa: E402

st.set_page_config(page_title="MRO Strategic Advisory", layout="wide")
st.title("🤖 Geo-Aware MRO Control Tower")

@st.cache_resource
def get_engine():
    idx = FastRAGIndexer()
    idx.build() 
    return LLMAdvisoryEngine(idx)

engine = get_engine()

# --- Simulation & RL State ---
with st.sidebar:
    st.header("🌍 Global Supply Context")
    # Simulating a dynamic trade flow value
    trade_flow = st.slider("UN Comtrade Flow Index", 0.0, 1.0, 0.8)
    
    st.header("📦 Depot Status")
    inv = st.slider("Inventory Level", 0.0, 1.0, 0.3)
    health = st.slider("Equipment Health", 0.0, 1.0, 0.15)
    
    st.write("---")
    
    # Simple logic to show the "Agent's Mind"
    if health < 0.2:
        rec = "Immediate Resuscitation"
        color = "inverse"
    elif inv < 0.4 and trade_flow > 0.5:
        rec = "Restock Parts (Optimal)"
        color = "normal"
    elif inv < 0.4 and trade_flow <= 0.5:
        rec = "Strategic Holding (High Trade Risk)"
        color = "off"
    else:
        rec = "Status Quo / Idle"
        color = "normal"

    st.metric("Agent Recommendation", rec)
    st.info(f"Current Trade Fluidity: {int(trade_flow*100)}%")

# --- Advisory Chat ---
query = st.chat_input("Ask for a strategic briefing on this scenario...")

if query:
    with st.spinner("Analyzing code logic and trade data..."):
        res = engine.ask(query)
        st.subheader("Strategic Justification")
        st.success(res["answer"])
        
        with st.expander("Grounded Sources"):
            for s in res.get("sources", []):
                st.markdown(f"- `{s}` ")
