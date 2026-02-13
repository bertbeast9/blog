import streamlit as st

st.header("About Me",divider=True)
st.sidebar.markdown("# About Me")
st.text("In my free time, I am training for some race, listening to music, or enjoying a pint with friends.")
st.subheader("Socials")
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    st.link_button("LinkedIn","https://www.linkedin.com/in/sam-bertelson/")
with col2:
    st.link_button("GitHub","https://github.com/bertbeast9")
with col3:
    st.link_button("Google Scholar","https://scholar.google.com/citations?user=fhEiMLoAAAAJ&hl=en")
with col4:
    st.link_button("Strava","https://www.strava.com/athletes/63520124")




