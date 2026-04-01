import streamlit as st
import time
import datetime
import os
import pandas as pd
import numpy as np


def win_odds_2_implied_prob(odds_str):
    if odds_str == "EVEN":
        return 50.0
    elif odds_str == "---":
        return np.nan
    elif "-" in odds_str:
        num = float(odds_str[1:])
        return 100*num/(100.0 + num)
    elif "+" in odds_str:
        num = float(odds_str[1:])
        return 100*100/(100.0 + num)

def runline_odds_2_implied_prob(odds_str):
    if odds_str == "---":
        return (np.nan, np.nan)
    else:
        runline = float(odds_str.split(" ")[0])
        odds_str = odds_str.split(" ")[1].strip("()")
        if odds_str == "EVEN":
            return (runline, 50.0)
        elif "-" in odds_str:
            num = float(odds_str[1:])
            return (runline, 100*num/(100.0 + num))
        elif "+" in odds_str:
            num = float(odds_str[1:])
        return (runline, 100*100/(100.0 + num))

def read_last_line(filepath, lines_=0):
    """
    Reads the last line of a file efficiently.
    """
    with open(filepath, 'rb') as f:
        # Start at the end of the file
        f.seek(0, 2)
        # Get the current position (end of file)
        file_size = f.tell()
        ctr = 0
        # Iterate backwards byte by byte
        for i in range(2, file_size + 1):
            f.seek(-i, 2)  # Move cursor i bytes from the end
            char = f.read(1)
            if char == b'\n':
                if ctr == lines_:
                    # Found a newline, read the rest of the line
                    return f.readline().decode().strip()
                else:
                    ctr += 1
        
        # If no newline is found (e.g., single line file without trailing newline)
        f.seek(0)
        return f.read().decode().strip()


st.set_page_config(layout="wide")
st.header("Live Baseball Game Odds")
st.sidebar.markdown("# Live Baseball Game Odds")

# build out page with all of today's games
today = datetime.datetime.now()
log_dir = f"{"\\".join(os.getcwd().split("\\")[:-1])}\\mlb\\logs\\{today.strftime("%Y")}\\{today.strftime("%m")}\\{today.strftime("%d")}"
game_names = [f"{x.split("_GAME")[0]}" for x in os.listdir(log_dir) if "GAME_info" in x]
print(game_names)
game_files = [f"{log_dir}\\{x}_GAME_info.log" for x in game_names]
print(game_files)
games = []
for (game_name, game_file) in zip(game_names, game_files):
    games.append({"game_name": game_name, "game_file": game_file, "away_team":game_name.split("_vs_")[0], "home_team":game_name.split("_vs_")[1]})
game_cols = 5
game_rows = ((len(games) + (game_cols - 1)) // game_cols)


# @st.fragment(run_every=3)
def update_game_states():
    # Create empty element in each column and save to list
    rows = [st.columns(game_cols, width="stretch") for idx in range(game_rows)]
    empty_rows = []
    all_elements = [y for x in rows for y in x]
    for col_num in range(len(all_elements)):
        empty_rows.append(all_elements[col_num].empty())
    for game_idx, col in enumerate(empty_rows):
        tile = col.container(height = 250, width = 500)
        game_info = read_last_line(games[game_idx]["game_file"])
        if len(game_info) > 0:
            game_info = game_info.split(" INFO - ")[1].split("|")
            (away_team, home_team, inning_and_topbot, away_score, home_score, balls, strikes, outs, on1b, on2b, on3b, away_pitcher_id, home_pitcher_id, away_batter_idx, home_batter_idx), away_batting_ids, home_batting_ids = game_info[:15], game_info[15:24], game_info[24:33]
            games[game_idx]["game_state"] = {"inning": inning_and_topbot, "away_score": away_score, "home_score": home_score, "balls": balls, "strikes": strikes, "outs": outs, "on1b": on1b, "on2b": on2b, "on3b": on3b}#
            tile.text(f"{games[game_idx]["away_team"]} vs {games[game_idx]["home_team"]}\nINN: {games[game_idx]["game_state"]["inning"]} ({games[game_idx]["game_state"]["away_score"]} - {games[game_idx]["game_state"]["home_score"]})\nB: {games[game_idx]["game_state"]["balls"]}\tS: {games[game_idx]["game_state"]["strikes"]}\tO: {games[game_idx]["game_state"]["outs"]}\n1B: {games[game_idx]["game_state"]["on1b"]}\t2B: {games[game_idx]["game_state"]["on2b"]}\t3B: {games[game_idx]["game_state"]["on3b"]}")
    
        if "odds_file" not in games[game_idx].keys():
            odds_file = f"{log_dir}\\{games[game_idx]["game_name"]}_ODDS_info.log"
            if os.path.isfile(odds_file):
                games[game_idx]["odds_file"] = odds_file
        if "odds_file" in games[game_idx].keys():
            odds_info = read_last_line(games[game_idx]["odds_file"])
            away_team_str, home_team_str, _, _, _, away_runline, home_runline, away_win, home_win, total_1, total_2 = odds_info.split("INFO - ")[1].split("|")
            away_win_prob_odds = win_odds_2_implied_prob(away_win)
            home_win_prob_odds = win_odds_2_implied_prob(home_win)
            away_run_diff, away_run_diff_odds = runline_odds_2_implied_prob(away_runline)
            home_run_diff, home_run_diff_odds = runline_odds_2_implied_prob(home_runline)
            games[game_idx]["game_state"]["game_odds"] = {"away_win": away_win_prob_odds, "home_win": home_win_prob_odds, "away_spread_line": away_run_diff, "away_spread": away_run_diff_odds, "home_spread_line": home_run_diff, "home_spread": home_run_diff_odds}
            tile.text(f"Win [(BOV)/(SIM)]: ({games[game_idx]["game_state"]["game_odds"]["away_win"]:0.4}/{games[game_idx]["game_state"]["game_odds"]["home_win"]:0.4})")
            tile.text(f"Spread [(BOV)/(SIM)]: ([{games[game_idx]["game_state"]["game_odds"]["away_spread_line"]:0.2}] {games[game_idx]["game_state"]["game_odds"]["away_spread"]:0.4}/[{games[game_idx]["game_state"]["game_odds"]["home_spread_line"]:0.2}] {games[game_idx]["game_state"]["game_odds"]["home_spread"]:0.4})")
update_game_states()


