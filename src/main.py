import streamlit as st

if __name__ == "__main__":
    # Define the pages
    home_page = st.Page("./pages/home_page.py", title="Home Page", icon="🏠")
    about_me = st.Page("./pages/about_me.py", title="About Me", icon="👨")
    pitch_solver = st.Page("./pages/pitch_solver.py",title="Pitch Solver", icon="⚾")
    live_baseball_game_solver_average = st.Page("./pages/live_baseball_game_solver_average.py",title="Live Baseball Average Game Solver", icon="⚾")
    pitcher_biomech_velo = st.Page("./pages/pitcher_biomech_velo.py",title="Pitcher Biomechanics & Velocity Project", icon="⚾")
    
    # Set up navigation
    pg = st.navigation([home_page, pitch_solver, live_baseball_game_solver_average, pitcher_biomech_velo, about_me])

    # Run the selected page
    pg.run()