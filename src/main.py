import streamlit as st

if __name__ == "__main__":
    # Define the pages
    main_page = st.Page("./pages/home_page.py", title="Home Page", icon="🏠")
    resume = st.Page("./pages/resume.py", title="Resume", icon="👨‍🎓")
    about_me = st.Page("./pages/about_me.py", title="About Me", icon="👨")
    # cv = st.Page("./pages/cv.py", title="CV", icon="📖")
    baseball_projects = st.Page("./pages/baseball_projects.py",title="Baseball Projects", icon="⚾")
    semiconductor_projects = st.Page("./pages/semiconductor_projects.py",title="Semiconductor Projects", icon="💻")

    # Set up navigation
    pg = st.navigation([main_page, baseball_projects, semiconductor_projects, resume, about_me])

    # Run the selected page
    pg.run()