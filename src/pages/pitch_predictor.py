import streamlit as st
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib
import sys
import json
import datetime
import os
import pickle
from copy import copy
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, hstack, vstack
from collections import OrderedDict as ordered_dict
from mdls.pitch_predictor.models import AveragePitchPredictor, MarkovPitchPredictor, MarkovPitchPredictorHandedness


#### FUNCTIONS
def load_pitch_data():
    pitch_data = pd.read_csv("./src/data/Swish_Baseball_project/MLB/pitches.csv")#, nrows=100000
    print(pitch_data.head())
    return pitch_data

def order_pitches_global(df):
    df.sort_values(["date", "game_pk","at_bat_num","pcount_at_bat"], inplace=True)
    return df

def split_data_by_at_bats(pitch_data):
    pitch_data["date"] = pitch_data["date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    pitch_data = order_pitches_global(pitch_data)
    pitch_data["start_at_bat"] = pitch_data.apply(lambda x: 1 if x.strikes == 0 and x.balls == 0 else 0, axis = 1)
    pitch_data["end_at_bat"] = pitch_data["start_at_bat"].shift(-1)
    pitch_data["end_at_bat"].iat[-1] = 1
    start_ab_ind = pitch_data.index[pitch_data["start_at_bat"].apply(lambda x: bool(x))]
    end_ab_ind = pitch_data.index[pitch_data["end_at_bat"].apply(lambda x: bool(x))]
    at_bats = []
    progress_bar = st.progress(0, text="Splitting Data by At-Bat")
    N = len(start_ab_ind)
    for idx, (s_ind, e_ind) in enumerate(zip(start_ab_ind, end_ab_ind)):
        progress_bar.progress(idx/N, text="Splitting Data by At-Bat")
        at_bats.append(pitch_data.loc[s_ind:e_ind].copy())
        if not ((at_bats[-1]["pcount_at_bat"].diff()[1:] == 1).all() and at_bats[-1]["pcount_at_bat"].values[0] == 1):
            at_bats.pop()
    del pitch_data
    return at_bats

def build_markov_chain_model(at_bats, poss_states):
    N = len(poss_states)
    A = np.zeros([N,N])
    pi = np.zeros([N,N])
    for at_bat_idx, at_bat in enumerate(at_bats):
        for pitch_idx, pitch_data in at_bat.iterrows():
            if pitch_idx == at_bat.index[0] and at_bat_idx == 0:
                last_state = pitch_data.curr_state_int
                continue
            else:
                curr_state = pitch_data.curr_state_int
                A[last_state, curr_state] += 1
                last_state = curr_state
    A = A / np.sum(A, axis = 1).reshape(-1,1)
    (evals, evecs) = scipy.linalg.eig(A, left=True, right=False)
    tmp = list(zip(evals, np.array_split(evecs, N, axis=1)))
    tmp = sorted(tmp, key = lambda x: np.abs(x[0] - 1))
    pi = tmp[0][1]
    pi = pi / np.sum(pi)
    return(A, pi.reshape(1,-1))

def fit_and_score_avg_pitch_predictor(poss_states, at_bats, metric):
    avg_glob_mdl = AveragePitchPredictor(poss_states)
    avg_glob_mdl.fit(at_bats[:int(len(at_bats)/2)])
    avg_glob_mdl_metric = avg_glob_mdl.score(at_bats[int(len(at_bats)/2):], metric = metric)
    return (avg_glob_mdl, avg_glob_mdl_metric)

def fit_and_score_markov_glob_pitch_predictor(poss_states, at_bats, metric):
    markov_glob_mdl = MarkovPitchPredictor(poss_states)
    markov_glob_mdl.fit(at_bats[:int(len(at_bats)/2)])
    markov_glob_mdl_metric = markov_glob_mdl.score(at_bats[int(len(at_bats)/2):], metric = metric)
    return (markov_glob_mdl, markov_glob_mdl_metric)

def fit_and_score_markov_glob_split_pitch_predictor(poss_states, at_bats, metric):
    markov_glob_split_mdl = MarkovPitchPredictorHandedness(poss_states)
    markov_glob_split_mdl.fit(at_bats[:int(len(at_bats)/2)])
    markov_glob_split_mdl_metric = markov_glob_split_mdl.score(at_bats[int(len(at_bats)/2):], metric = metric)
    return (markov_glob_split_mdl, markov_glob_split_mdl_metric)

#### FUNCTIONS




#### VARIABLES
pitch_dict = {"FF":"4-Seam Fastball","SI":"Sinker (2-Seam)","FC":"Cutter",
          "CH":"Changeup","FS":"Split-finger","SC":"Screwball",
          "CU":"Curveball","KC":"Knuckle Curve","CS":"Slow Curve","SL":"Slider",
          "ST":"Sweeper","SV":"Slurve","KN":"Knuckleball","EP":"Eephus","FA":"Other",
          "IN":"Intentional Ball","PO":"Pitchout"}
metric = "avg_like"
#### VARIABLES

st.header("Pitch Predictor",divider=True)
st.sidebar.markdown("# Pitch Predictor")
st.subheader("Problem Overview")
st.markdown("The aim of this project is to build a model that can predict _with sufficient accuracy_ what the next pitch will be from a particular pitcher. As per usual, " \
"we will start with as simple a model possible and slowly make it more complex in order to improve predictive capabilities.")
st.subheader("Data Preprocessing", divider=True)
## build pitch classification model
pitch_data = load_pitch_data()

perc_nan = pitch_data.isna().sum(axis=0)/pitch_data.shape[0]
st.markdown("First of all, we need to clean of the dataset a bit since there are NaNs in the raw data. Since we are predicting the _pitch type_, I am going to drop any rows that " \
f"do not have a value for _pitch type_. Fortunately, _pitch type_ only has {100 * perc_nan["pitch_type"]:0.2f}% NaN values. Most of the other missing data are completely missing or " \
"the column is only missing a few elements. So, we will remove all columns with a percentage of NaNs greater than 50% and drop the rows that contain NaN values.")
fig = plt.figure()
plt.plot(100 * perc_nan)
plt.ylabel("NaN content [%]")
plt.xlabel("Variable Name")
plt.xticks(rotation=80, fontsize=4)
plt.title("NaN Content (before column removal/before filtering)")
st.pyplot(fig)
pitch_data = pitch_data[[x for x in pitch_data.columns if perc_nan[x] < 0.5]]
perc_nan = pitch_data.isna().sum(axis=0)/pitch_data.shape[0]
cols_with_nan = [x for x in pitch_data.columns if perc_nan[x] > 0.0]
perc_nan = pitch_data.isna().sum(axis=0)/pitch_data.shape[0]
fig = plt.figure()
plt.plot(100 * perc_nan)
plt.ylabel("NaN content [%]")
plt.xlabel("Variable Name")
plt.xticks(rotation=80, fontsize=6)
plt.title("NaN Content (after column removal/before filtering)")
st.pyplot(fig)

pitch_data.dropna(how="any",axis=0,inplace=True)
perc_nan = pitch_data.isna().sum(axis=0)/pitch_data.shape[0]
fig = plt.figure()
plt.plot(100 * perc_nan)
plt.ylabel("NaN content [%]")
plt.xlabel("Variable Name")
plt.xticks(rotation=80, fontsize=6)
plt.title("NaN Content (after column removal/after filtering)")
st.pyplot(fig)

st.markdown("For simplicity, I am also going to remove the pitches that do not occur often like (PO, AB, EP, etc.).")
pitch_data["states"] = pitch_data["pitch_type"]
pitch_data = pitch_data[pitch_data["states"].apply(lambda x: False if x in ["AB", "EP", "FA", "FO", "PO", "SC", "UN", "IN"] else True)]
pitch_hist = plt.figure()
poss_states, counts = np.unique(pitch_data["states"], return_counts=True)
percents = counts / np.sum(counts)
plt.bar(poss_states, percents)
plt.ylabel("pitch percent [%]")
plt.xlabel("pitch acronym")
plt.title("Pitch Type Histogram")
st.pyplot(pitch_hist)
print(poss_states)
pitch_data["curr_state_int"] = pitch_data["states"].apply(lambda x: list(poss_states).index(x))
st.markdown("Now, I need to order the data such that this methodology makes sense. I need to break up this data by game and by at-bat. Each new at-bat will be the start of a new Markov Chain. Also, " \
"I need to ensure that each pitch sequence is proper (pitch 1 -> pitch 2 -> pitch 3 etc.) and not (pitch 1 -> pitch 3).")

print(pitch_data["date"].iloc[0])
# st.dataframe(pitch_data)
at_bats = split_data_by_at_bats(pitch_data)

np.random.shuffle(at_bats)

# # st.subheader("Simplest Model", divider=True)
# # st.markdown("For this model, we will use a discrete-time Markov Chain to model the process of selecting the next pitch. The intuition behind this Markov Chain is that the next " \
# # "selected pitch only depends on the last thrown pitch. This should capture the _setup_ pitches in which a pitcher will setup a curveball by throwing a fastball first, for example.")

# # st.table(pitch_dict)
# # st.latex("x_k \\in S = \\text{all possible pitches from pitcher} = \\set{FF, SI, FC, \\dots}\\newline x_{k+1} = A x_{k}")
# # st.markdown("$A$ is the state transition matrix where $a_{i,j}$ is the probability of transitioning to state $i$ given one is in state $j$.")
# # st.latex("a_{i,j} = P(x_i | x_j) \\forall x_i, x_j \\in S")
# # st.markdown("The Markov Chain also must have an initial distribution, $\\pi_0$, on the probability of starting the Markov chain in each state. This initial distribution will " \
# # "be calculated as the average probability of a pitch over all time.")
# # st.markdown("This model is termed the _simplest_ model because we will use all available data to create a global, or average, pitcher. This ignores effects from splits (LHP vs LHH, etc.), " \
# # "counts (0-0, 3-2, etc.), or other factors.")


(avg_glob_mdl, avg_glob_mdl_metric) = fit_and_score_avg_pitch_predictor(poss_states, at_bats, metric)
(markov_glob_mdl, markov_glob_mdl_metric) = fit_and_score_markov_glob_pitch_predictor(poss_states, at_bats, metric)
P_mat = pd.DataFrame(markov_glob_mdl.A["NA"]["NA"], index = poss_states, columns = poss_states)
st.dataframe(P_mat)
pi_mat = pd.DataFrame(markov_glob_mdl.pi["NA"]["NA"].reshape(1,-1), index=["longterm_prob."], columns = poss_states)
st.dataframe(pi_mat)

(markov_glob_split_mdl, markov_glob_split_mdl_metric) = fit_and_score_markov_glob_split_pitch_predictor(poss_states, at_bats, metric)
# st.json(markov_glob_split_mdl.A)
# P_mat = pd.DataFrame(markov_glob_split_mdl.A["NA"]["RR"], index = poss_states, columns = poss_states)
# st.dataframe(P_mat)
# pi_mat = pd.DataFrame(markov_glob_split_mdl.pi["NA"]["RR"].reshape(1,-1), index=["longterm_prob."], columns = poss_states)
# st.dataframe(pi_mat)
st.markdown(f"For the model that predicts just the average pitch, the average probability of being correct is {100 * avg_glob_mdl_metric:0.2f}% " \
            f"while the Markov Chain model has an average probability of being correct of {100 * markov_glob_mdl_metric:0.2f}% and the Markov " \
            f"Chain considering split differences has an average probability of being correct of {100 * markov_glob_split_mdl_metric:0.2f}%.")
# print(len(at_bats))
# (A, pi) = build_markov_chain_model(at_bats, poss_states)
# print(A)
# print(pi)

# # curr_state = pi
# last_pitch_thrown = "CH"
# curr_state = np.zeros((1,len(poss_states)));curr_state[0,list(poss_states).index(last_pitch_thrown)] = 1
# next_state_pred_dist = curr_state @ A
# print(next_state_pred_dist)
# next_state_pred_max = np.argmax(next_state_pred_dist)
# print(f"Last pitch thrown: {last_pitch_thrown} Next predicted state is {poss_states[next_state_pred_max]} ({100 * next_state_pred_dist[0,next_state_pred_max]:0.2f}%) vs average {poss_states[np.argmax(pi)]} ({100 * np.max(pi):0.2f}%)")




