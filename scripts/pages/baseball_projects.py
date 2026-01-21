import streamlit as st

st.header("Baseball Projects",divider=True)
st.sidebar.markdown("# Baseball Projects")
st.sidebar.subheader("Pitch Solver (Count Only)")
st.sidebar.subheader("Pitch Solver (Expanded)")
st.subheader("Pitch Solver (Count Only)")
st.text("Okay, let's start with the problem at hand. A begin decision before each " \
"pitch is what pitch to throw. We aim to model this as a Markov Decision Process (MDP) " \
"with the states being the state of the game. As a starting point, we are only going " \
"to consider the count as the state of the at-bat. Future work will expand the state-space " \
"to include the runners and outs. The full state space consists of the current at-bat's " \
"count, termed *transient states*, and the results of these at-bats, termed *absorbing* " \
"states. Again, for simplicity, we break the results of each at-bat into one of seven " \
"outcomes (out, walk, hit by pitch, single, double, triple, home run). Later, these " \
"outcomes will be further expanded to include sac flys, double plays, etc. The full " \
"state-space of this simple problem is the union of the *transient* and *absorbing* " \
"state-spaces.")

