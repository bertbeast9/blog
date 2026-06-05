import streamlit as st

if __name__ == "__main__":
    # Define the pages
    main_page = st.Page("./pages/home_page.py", title="Home Page", icon="🏠")
    about_me = st.Page("./pages/about_me.py", title="About Me", icon="👨")
    # cv = st.Page("./pages/cv.py", title="CV", icon="📖")
    pitch_solver = st.Page("./pages/pitch_solver.py",title="Pitch Solver", icon="⚾")
    # live_baseball_game_solver_personal = st.Page("./pages/live_baseball_game_solver_personal.py",title="Live Baseball Game Solver (My Model)", icon="⚾")
    live_baseball_game_solver_average = st.Page("./pages/live_baseball_game_solver_average.py",title="Live Baseball Average Game Solver", icon="⚾")
    # semiconductor_projects = st.Page("./pages/semiconductor_projects.py",title="Semiconductor Projects", icon="💻")

    # Set up navigation
    pg = st.navigation([pitch_solver, live_baseball_game_solver_average, about_me])#main_page, semiconductor_projects, , live_baseball_game_solver_personal

    # Run the selected page
    pg.run()