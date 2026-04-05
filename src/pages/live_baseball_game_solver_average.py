import streamlit as st
from matplotlib import pyplot as plt
import time
import datetime
import os
import pandas as pd
import numpy as np
from zipfile import BadZipFile
from scipy.sparse import save_npz, load_npz
from collections import OrderedDict
import json
from bs4 import BeautifulSoup
import requests
import asyncio
import nest_asyncio


import streamlit as st

if 'button' not in st.session_state:
    st.session_state.button = False

def click_button():
    st.session_state.button = not st.session_state.button




############## SIMPLE
simple_state_space_cardinality = {"run_diff": 1 + 2 * 10, "inning": 12, "is_bot": 2, "away_batter_idx": 9, "home_batter_idx": 9, "outs": 3, "on_1b": 2, "on_2b": 2, "on_3b": 2}
simple_state_space_index_block_size = OrderedDict()
tmp = 1
simple_state_space_index_block_size["on_3b"] = tmp
last_key = "on_3b"
for key in ["on_2b", "on_1b", "outs", "home_batter_idx", "away_batter_idx", "is_bot", "inning", "run_diff"]:
    tmp *= simple_state_space_cardinality[last_key]
    simple_state_space_index_block_size[key] = tmp
    last_key = key
# Do this once outside the function
simple_state_code_weights = np.array([
    simple_state_space_index_block_size["run_diff"],
    simple_state_space_index_block_size["inning"],
    simple_state_space_index_block_size["is_bot"],
    simple_state_space_index_block_size["away_batter_idx"],
    simple_state_space_index_block_size["home_batter_idx"],
    simple_state_space_index_block_size["outs"],
    simple_state_space_index_block_size["on_1b"],
    simple_state_space_index_block_size["on_2b"],
    simple_state_space_index_block_size["on_3b"],
], dtype=np.int64)

def simple_state_code_2_idx(state_code):
    parts = np.array(state_code)
    return np.dot(parts, simple_state_code_weights).astype(int)


def get_avg_solve_game():
    game_solve_filename = f"{"\\".join(os.getcwd().split("\\")[:-1])}\\mlb\\data\\mdls\\avg\\2008_01_01_to_2026_03_23__21_12_9_9__000000_R__000000_R__000000_R_000000_R_000000_R_000000_R_000000_R_000000_R_000000_R_000000_R_000000_R__000000_R_000000_R_000000_R_000000_R_000000_R_000000_R_000000_R_000000_R_000000_R__0__0__simple_game_solve.npz"
    if os.path.isfile(game_solve_filename) and os.access(game_solve_filename, os.R_OK):
        opened = False
        while not opened:
            try:
                B = load_npz(game_solve_filename).todense()
                opened = True
            except (EOFError, BadZipFile) as exp:
                continue
    else:
        B = None
    return B


############### SESSION STATE VARIABLES ########################

if "curr_game_state" not in st.session_state:
    st.session_state["curr_game_state"] = {"away_team":"AWAY", "home_team":"HOME", "away_score":"0", "home_score":"0", "inning":"1", "topbot":"TOP", "balls":"0", "strikes":"0", "outs":"0", "on_1b":"0", "on_2b":"0", "on_3b":"0", "away_curr_batter_idx":"0", "home_curr_batter_idx":"0", "away_batting_order":[], "home_batting_order":[], "away_pitcher":"", "home_pitcher":""}

if "avg_game_solve" not in st.session_state:
    st.session_state["avg_game_solve"] = get_avg_solve_game()

############### SESSION STATE VARIABLES ########################


def get_todays_games():
    schedule_url = f"https://statsapi.mlb.com/api/v1/schedule/games/?sportId=1&date={datetime.datetime.now().strftime("%Y-%m-%d")}"
    content = json.loads(BeautifulSoup(requests.get(schedule_url, timeout=60).text, "html.parser").get_text())
    live_games = [x for x in content["dates"][0]["games"] if x["status"]["codedGameState"] == "I"]
    live_game_names = [f"{x["teams"]["away"]["team"]["name"]} vs {x["teams"]["home"]["team"]["name"]}" for x in live_games]
    live_game_pks = [x["gamePk"] for x in live_games]
    # breakpoint()
    return (live_games, live_game_names, live_game_pks)

def update_game(game_state, gamePk=None):
    if gamePk is not None:
        url_str = f"https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"
        watching = True
        while watching:
            try:
                content = json.loads(BeautifulSoup(requests.get(url_str, timeout=60).text, "html.parser").get_text())
                on_1b, on_2b, on_3b = int("postOnFirst" in content["liveData"]["plays"]["currentPlay"]["matchup"].keys()), int("postOnSecond" in content["liveData"]["plays"]["currentPlay"]["matchup"].keys()), int("postOnThird" in content["liveData"]["plays"]["currentPlay"]["matchup"].keys())
                away_score, home_score = content["liveData"]["plays"]["currentPlay"]["result"]["awayScore"], content["liveData"]["plays"]["currentPlay"]["result"]["homeScore"]
                if content["liveData"]["linescore"]["inningState"] in ["Top","Bottom"]:
                    balls, strikes, outs = content["liveData"]["plays"]["currentPlay"]["count"]["balls"], content["liveData"]["plays"]["currentPlay"]["count"]["strikes"], content["liveData"]["plays"]["currentPlay"]["count"]["outs"]
                    if balls == 4 or strikes == 3 or int(away_score) + int(home_score) + int(outs) + int(on_1b) + int(on_2b) + int(on_3b) != \
                                                    int(game_state["away_score"]) + int(game_state["home_score"]) + int(game_state["outs"]) + int(game_state["on_1b"]) + int(game_state["on_2b"]) + int(game_state["on_3b"]):
                        balls, strikes = 0, 0
                    all_plays_batter_ids = [x["matchup"]["batter"]["id"] for x in content["liveData"]["plays"]["allPlays"][:-1][::-1]]
                else:
                    balls, strikes, outs, on_1b, on_2b, on_3b = 0,0,0,0,0,0
                    all_plays_batter_ids = [x["matchup"]["batter"]["id"] for x in content["liveData"]["plays"]["allPlays"][::-1]]
                # while content["gameData"]["status"]["abstractGameState"] != "Final":
                current_batter_id = [id for id in content["liveData"]["boxscore"]["teams"]["away"]["players"].keys()  if content["liveData"]["boxscore"]["teams"]["away"]["players"][id]["gameStatus"]["isCurrentBatter"]] + [id for id in content["liveData"]["boxscore"]["teams"]["home"]["players"].keys()  if content["liveData"]["boxscore"]["teams"]["home"]["players"][id]["gameStatus"]["isCurrentBatter"]]
                current_pitcher_id = [id for id in content["liveData"]["boxscore"]["teams"]["away"]["players"].keys()  if content["liveData"]["boxscore"]["teams"]["away"]["players"][id]["gameStatus"]["isCurrentPitcher"]] + [id for id in content["liveData"]["boxscore"]["teams"]["home"]["players"].keys()  if content["liveData"]["boxscore"]["teams"]["home"]["players"][id]["gameStatus"]["isCurrentPitcher"]]
                current_batter_id = current_batter_id[0][2:]
                current_pitcher_id = current_pitcher_id[0][2:]
                away_batting_order = [str(x) for x in content["liveData"]["boxscore"]["teams"]["away"]["battingOrder"]]
                home_batting_order = [str(x) for x in content["liveData"]["boxscore"]["teams"]["home"]["battingOrder"]]

                away_pitcher = str(content["liveData"]["boxscore"]["teams"]["away"]["pitchers"][-1])
                home_pitcher = str(content["liveData"]["boxscore"]["teams"]["home"]["pitchers"][-1])
                last_away_batter_id = [str(x) for x in all_plays_batter_ids if str(x) in away_batting_order][0]
                last_home_batter_id = [str(x) for x in all_plays_batter_ids if str(x) in home_batting_order][0]
                away_curr_batter_idx = (away_batting_order.index(last_away_batter_id) + 1) % 9
                home_curr_batter_idx = (home_batting_order.index(last_home_batter_id) + 1) % 9
                away_pitcher = away_pitcher + "_" + content["gameData"]["players"][f"ID{away_pitcher}"]["pitchHand"]["code"]
                home_pitcher = home_pitcher + "_" + content["gameData"]["players"][f"ID{home_pitcher}"]["pitchHand"]["code"]
                away_batting_order = [f"{x}_{content["gameData"]["players"][f"ID{x}"]["batSide"]["code"]}" for x in away_batting_order]
                home_batting_order = [f"{x}_{content["gameData"]["players"][f"ID{x}"]["batSide"]["code"]}" for x in home_batting_order]
                
                # memory variables
                game_state["away_team"], game_state["home_team"] = content["liveData"]["boxscore"]["teams"]["away"]["team"]["name"], content["liveData"]["boxscore"]["teams"]["home"]["team"]["name"]
                game_state["inning"], game_state["topbot"] = content["liveData"]["linescore"]["currentInning"], content["liveData"]["linescore"]["inningState"]
                game_state["balls"], game_state["strikes"], game_state["outs"] = balls, strikes, outs
                game_state["on_1b"], game_state["on_2b"], game_state["on_3b"] = on_1b, on_2b, on_3b
                game_state["away_score"], game_state["home_score"] = away_score, home_score
                game_state["away_curr_batter_idx"], game_state["home_curr_batter_idx"] = away_curr_batter_idx, home_curr_batter_idx
                game_state["away_batting_order"], game_state["home_batting_order"] = away_batting_order, home_batting_order
                game_state["away_pitcher"] = away_pitcher
                game_state["home_pitcher"] = home_pitcher
                watching = False
            except (ConnectionError) as exp:
                continue
    return game_state

st.set_page_config(layout="wide")
st.header("Live Baseball Game Simulated Outcomes")
st.sidebar.markdown("# Live Baseball Game Simluated Outcomes")

(live_games, live_game_names, live_game_pks) = get_todays_games()

#### DISCLAIMER ####
st.write("DISCLAIMER: I am not liable for any gambling losses :)")

option = st.selectbox(
    "Which game do you want to watch?",
    live_game_names,
    placeholder="Select game to watch...",
)

curr_game_pk = live_game_pks[live_game_names.index(option)]
st.write(f"You are now watching the {option} game ({curr_game_pk})")



st.button('Update Game', on_click=click_button)

if st.session_state.button:
    st.session_state["curr_game_state"] = update_game(st.session_state["curr_game_state"], curr_game_pk)
    # st.write(st.session_state["curr_game_state"])
    
    curr_simple_state_code = (int(st.session_state["curr_game_state"]["away_score"]) - int(st.session_state["curr_game_state"]["home_score"]) + ((simple_state_space_cardinality["run_diff"] - 1) // 2), \
                              int(st.session_state["curr_game_state"]["inning"]), int(st.session_state["curr_game_state"]["topbot"] in ["Middle","Bottom"]), \
                                int(st.session_state["curr_game_state"]["away_curr_batter_idx"]), int(st.session_state["curr_game_state"]["home_curr_batter_idx"]), \
                                    int(st.session_state["curr_game_state"]["outs"]), int(st.session_state["curr_game_state"]["on_1b"]), \
                                        int(st.session_state["curr_game_state"]["on_2b"]), int(st.session_state["curr_game_state"]["on_3b"]))
    simple_state_idx = simple_state_code_2_idx(curr_simple_state_code)
    final_dist = st.session_state["avg_game_solve"][simple_state_idx,:].reshape(-1, )
    final_run_diffs = np.linspace(-(simple_state_space_cardinality["run_diff"] - 1 ) // 2, (simple_state_space_cardinality["run_diff"] - 1 ) // 2, simple_state_space_cardinality["run_diff"])
    spreads = np.linspace(-(simple_state_space_cardinality["run_diff"] - 1 ) // 2, (simple_state_space_cardinality["run_diff"] - 1 ) // 2, (simple_state_space_cardinality["run_diff"] - 1) * 2 + 1)
    spread = st.select_slider('Select a home spread', options=spreads, value=0.0)
    away_win_prob_sim = 100 * np.sum(final_dist[np.argwhere(final_run_diffs > 0.0)])
    home_win_prob_sim = 100 * np.sum(final_dist[np.argwhere(final_run_diffs < 0.0)])
    away_runline_prob_sim = 100 * np.sum(final_dist[final_run_diffs - (spread) >= 0.0])
    home_runline_prob_sim = 100 * np.sum(final_dist[final_run_diffs - (spread) <= 0.0])
    df = pd.DataFrame(np.concat([final_run_diffs.reshape(-1,1), 100 * final_dist.reshape(-1, 1)],axis=1), columns = ["run_diff","prob"])
    col1, col2, col3 = st.columns(3, border=True)
    with col1:
        st.title(st.session_state["curr_game_state"]["away_team"])
        st.write("AWAY")
        st.metric(label="score", value=f"{st.session_state["curr_game_state"]["away_score"]}")
        st.metric(label="Win Prob. [%]", value=np.round(away_win_prob_sim,2))
        st.metric(label="Cover Spread Prob. [%]", value=np.round(away_runline_prob_sim,2))
    with col2:
        st.metric(label="Inning", value=f"{st.session_state["curr_game_state"]["topbot"]} {st.session_state["curr_game_state"]["inning"]}")
        st.metric(label="Count", value=f"{st.session_state["curr_game_state"]["balls"]}-{st.session_state["curr_game_state"]["strikes"]}")
        st.metric(label="Outs", value=f"{st.session_state["curr_game_state"]["outs"]}")
        subcol1, subcol2, subcol3 = st.columns(3, border=True)
        with subcol1:
            st.metric(label="3B", value=f"{st.session_state["curr_game_state"]["on_3b"]}")
        with subcol2:
            st.metric(label="2B", value=f"{st.session_state["curr_game_state"]["on_2b"]}")
        with subcol3:
            st.metric(label="1B", value=f"{st.session_state["curr_game_state"]["on_1b"]}")

    with col3:
        st.title(st.session_state["curr_game_state"]["home_team"])
        st.write("HOME")
        st.metric(label="score", value=f"{st.session_state["curr_game_state"]["home_score"]}")
        st.metric(label="Win Prob. [%]", value=np.round(home_win_prob_sim,2))
        st.metric(label="Cover Spread Prob. [%]", value=np.round(home_runline_prob_sim,2))
    # tile.text(f"Win BOV: ({games[game_idx]["game_state"]["game_odds"]["away_win"]:0.4}/{games[game_idx]["game_state"]["game_odds"]["home_win"]:0.4})\nWin SIM: ({away_win_prob_sim:0.4}/{home_win_prob_sim:0.4})\nSpread BOV: ([{games[game_idx]["game_state"]["game_odds"]["away_spread_line"]:0.2}] {games[game_idx]["game_state"]["game_odds"]["away_spread"]:0.4}/[{games[game_idx]["game_state"]["game_odds"]["home_spread_line"]:0.2}] {games[game_idx]["game_state"]["game_odds"]["home_spread"]:0.4})\nSpread SIM: ([{games[game_idx]["game_state"]["game_odds"]["away_spread_line"]:0.2}] {away_runline_prob_sim:0.4}/[{games[game_idx]["game_state"]["game_odds"]["home_spread_line"]:0.2}] {home_runline_prob_sim:0.4})")
    st.bar_chart(df,x = "run_diff", y="prob", x_label = "run difference (away score - home score)", y_label = "prob. [%]")



# # # # # Monkey patch Streamlit's internal event loop
# # # # nest_asyncio.apply()

# # # # if st.button("Update Game"):
# # # #     
# # # #     # tile.bar_chart(dict([(x, y) for (x,y) in zip(final_run_diffs, final_dist)]))
# # # #     values = st.slider("Select a range of values", -10, 10, 0)
# # # #     st.write("Values:", values)
# # # #     away_win_prob_sim = 100 * np.sum(final_dist[np.argwhere(final_run_diffs > 0.0)])
# # # #     home_win_prob_sim = 100 * np.sum(final_dist[np.argwhere(final_run_diffs < 0.0)])
# # # #     # away_runline_prob_sim = 100 * np.sum(final_dist[final_run_diffs + (away_run_diff) > 0.0])
# # # #     # home_runline_prob_sim = 100 * np.sum(final_dist[final_run_diffs - (home_run_diff) < 0.0])
# # # #     df = pd.DataFrame(np.concat([final_run_diffs.reshape(-1,1), 100 * final_dist.reshape(-1, 1)],axis=1), columns = ["run_diff","prob"])
# # # #     # tile.text(f"Win BOV: ({games[game_idx]["game_state"]["game_odds"]["away_win"]:0.4}/{games[game_idx]["game_state"]["game_odds"]["home_win"]:0.4})\nWin SIM: ({away_win_prob_sim:0.4}/{home_win_prob_sim:0.4})\nSpread BOV: ([{games[game_idx]["game_state"]["game_odds"]["away_spread_line"]:0.2}] {games[game_idx]["game_state"]["game_odds"]["away_spread"]:0.4}/[{games[game_idx]["game_state"]["game_odds"]["home_spread_line"]:0.2}] {games[game_idx]["game_state"]["game_odds"]["home_spread"]:0.4})\nSpread SIM: ([{games[game_idx]["game_state"]["game_odds"]["away_spread_line"]:0.2}] {away_runline_prob_sim:0.4}/[{games[game_idx]["game_state"]["game_odds"]["home_spread_line"]:0.2}] {home_runline_prob_sim:0.4})")
# # # #     st.bar_chart(df,x = "run_diff", y="prob", x_label = "run difference (away score - home score)", y_label = "prob. [%]")
                
