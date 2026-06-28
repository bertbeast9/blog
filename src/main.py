import streamlit as st
from st_files_connection import FilesConnection

if __name__ == "__main__":
    conn = st.connection('gcs', type=FilesConnection)
    df = conn.read("blog-data-storage/myfile.csv", input_format="csv", ttl=600)

    # Print results.
    
    for row in df.itertuples():
        st.write(f"{row.Owner} has a :{row.Pet}:")

    # Define the pages
    home_page = st.Page("./pages/home_page.py", title="Home Page", icon="🏠")
    about_me = st.Page("./pages/about_me.py", title="About Me", icon="👨")
    pitch_solver = st.Page("./pages/pitch_solver.py",title="Pitch Solver", icon="⚾")
    live_baseball_game_solver_average = st.Page("./pages/live_baseball_game_solver_average.py",title="Live Baseball Average Game Solver", icon="⚾")
    pitcher_biomech_velo = st.Page("./pages/pitcher_biomech_velo.py",title="Pitcher Biomechanics & Velocity Project", icon="⚾")
    pitch_predictor = st.Page("./pages/pitch_predictor.py", title="Pitch Predictor", icon="⚾")
    # Set up navigation
    pg = st.navigation([home_page, pitch_solver, live_baseball_game_solver_average, pitcher_biomech_velo, pitch_predictor, about_me])

    # Run the selected page
    pg.run()