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
from mdls.pitch_predictor.models import AveragePitchPredictor, MarkovPitchPredictor, RFModelPitchPredictor
from st_files_connection import FilesConnection

conn = st.connection('gcs', type=FilesConnection)
# conn = st.connection('gcs', type="files")
#### FUNCTIONS
@st.cache_data(ttl=600)
def load_pitch_data():
    with conn.open("blog-data-storage/Swish_Baseball_project/pitches.csv", "r") as f:
        return pd.read_csv(f)

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
    
    return (avg_glob_mdl, avg_glob_mdl_metric)

def fit_and_score_markov_glob_pitch_predictor(poss_states, at_bats, metric):
    markov_glob_mdl = MarkovPitchPredictor(poss_states)
    markov_glob_mdl.fit(at_bats[:int(len(at_bats)/2)])
    markov_glob_mdl_metric = markov_glob_mdl.score(at_bats[int(len(at_bats)/2):], metric = metric)
    return (markov_glob_mdl, markov_glob_mdl_metric)


#### FUNCTIONS




#### VARIABLES
pitch_dict = {"FF":"4-Seam Fastball","SI":"Sinker (2-Seam)","FC":"Cutter",
          "CH":"Changeup","FS":"Split-finger","SC":"Screwball",
          "CU":"Curveball","KC":"Knuckle Curve","CS":"Slow Curve","SL":"Slider",
          "ST":"Sweeper","SV":"Slurve","KN":"Knuckleball","EP":"Eephus","FA":"Other",
          "IN":"Intentional Ball","PO":"Pitchout"}
metric = "cross-entropy"
features = ["balls","strikes","fouls","previous_pitch","previous_previous_pitch"]#
#### VARIABLES

st.header("Swish Pitch Predictor",divider=True)
st.sidebar.markdown("# Swish Pitch Predictor")
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
percents = 100 * counts / np.sum(counts)
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
## feature engineering
pitch_data["previous_pitch"] = pitch_data["pitch_type"].shift(1,fill_value="FF").apply(lambda x: list(poss_states).index(x))
pitch_data["previous_previous_pitch"] = pitch_data["pitch_type"].shift(2,fill_value="FF").apply(lambda x: list(poss_states).index(x))

# st.dataframe(pitch_data)
at_bats = split_data_by_at_bats(pitch_data)

np.random.shuffle(at_bats)

st.subheader("Evaluating Models", divider=True)
st.markdown("It is very important that we utilize a pertinent metric for this problem. This problem can be boiled down to a multi-class classification problem in which the class is the " \
"type of pitch thrown in that scenario. For each model and before each pitch, we can assign a probability of observing a particular class. We can assess each prediction by comparing it to the " \
"true distribution, $\\vec{q}_k$. For example, let's say that a model predicts the following distribution for the $k^{th}$ pitch while the true $k^{th}$ pitch was actually a CH.")
st.latex(rf"""
         \vec{{p}}_k \in \mathbb{{R}}^{len(poss_states)} = \left[ \begin{{matrix}} p_{{k,CH}} && p_{{k,CU}} && \dots && p_{{k,SL}}  \end{{matrix}} \right] \newline
         \vec{{q}}_k \in \mathbb{{R}}^{len(poss_states)} = \left[ \begin{{matrix}} 1.0 && 0.0 && \dots && 0.0  \end{{matrix}} \right]
""")
st.markdown("Then, we can compare the prediction and the true outcome with the cross-entropy metric [https://en.wikipedia.org/wiki/Cross-entropy].")

st.subheader("Baseline Model", divider=True)
st.markdown("As a baseline, one simplistic method for predicting what the next pitch will be is to predict the average distribution of pitches. This model will " \
"do exactly that. For each pitch, it will assign the average probability of throwing a fastball etc. in all scenarios. To 'train' this model, we will take the trainining " \
"data and calculate the average probability each particular pitch was thrown.")

st.subheader("Markov Models", divider=True)
st.markdown("For this model, we will use a discrete-time Markov Chain to model the process of selecting the next pitch. The intuition behind this Markov Chain is that the next " \
"selected pitch only depends on the last thrown pitch. This should capture the _setup_ pitches in which a pitcher will setup a curveball by throwing a fastball first, for example.")

st.table(pitch_dict)
st.latex("x_k \\in S = \\text{all possible pitches from pitcher} = \\set{FF, SI, FC, \\dots}\\newline x_{k+1} = A x_{k}")
st.markdown("$A$ is the state transition matrix where $a_{i,j}$ is the probability of transitioning to state $j$ given one is in state $i$ (A is a right or row stochastic matrix).")
st.latex("a_{i,j} = P(x_i | x_j) \\forall x_i, x_j \\in S")
st.markdown("The Markov Chain also must have an initial distribution, $\\vec{\\pi}_0$, on the probability of starting the Markov chain in each state. This initial distribution will " \
"be identified from the below formulation following [Modeling and Analysis of Stochastic Systems, Kulkarni].")
st.latex(rf"""
         \vec{{\pi}}_0 = \vec{{\pi}}_0 A
         """)
st.markdown("This model is termed the _simplest_ model because we will use all available data to create a global, or average, pitcher. This ignores effects from splits (LHP vs LHH, etc.), " \
"counts (0-0, 3-2, etc.), or other factors. We compare adding more information into the model by including split information into the global model, and then pitcher specific information, and " \
"finally pitcher specific split information.")
st.subheader("Random Forest Model", divider=True)
st.markdown("Additionally, we utilize a random forest model that is fed the following features: balls, strikes, fouls, last pitch, 2nd to last pitch. The model performs a grid search to identify " \
"the salient input features and the set of hyperparameters in the Random Forest model like the number of trees and the max depth of each tree.")

# average pitch model
avg_glob_mdl = AveragePitchPredictor(poss_states)
avg_glob_mdl.fit(at_bats[:int(len(at_bats)/2)])
avg_glob_mdl_metric = avg_glob_mdl.score(at_bats[int(len(at_bats)/2):], metric = metric)
# markov global pitch model
markov_glob_mdl = MarkovPitchPredictor(poss_states)
markov_glob_mdl.fit(at_bats[:int(len(at_bats)/2)])
markov_glob_mdl_metric = markov_glob_mdl.score(at_bats[int(len(at_bats)/2):], metric = metric, info = [""])
markov_glob_split_mdl_metric = markov_glob_mdl.score(at_bats[int(len(at_bats)/2):], metric = metric, info = ["split"])
markov_pitcher_mdl_metric = markov_glob_mdl.score(at_bats[int(len(at_bats)/2):], metric = metric, info = ["pitcher"])
markov_pitcher_split_mdl_metric = markov_glob_mdl.score(at_bats[int(len(at_bats)/2):], metric = metric, info = ["pitcher","split"])

# random forest pitch model
rf_mdl = RFModelPitchPredictor(poss_states)
rf_mdl.fit(at_bats[:int(len(at_bats)/2)], features)
rf_mdl_metric = rf_mdl.score(at_bats[int(len(at_bats)/2):], features)
results = pd.DataFrame([avg_glob_mdl_metric, markov_glob_mdl_metric, markov_glob_split_mdl_metric, markov_pitcher_mdl_metric, markov_pitcher_split_mdl_metric, rf_mdl_metric], index = ["Simplest Baseline Model","Global Markov Model","Global Split Markov Model", "Pitcher Markov Model", "Pitcher Split Markov Model", "Global Random Forest Model"], columns = ["exp(-cross-entropy) [%]"])
st.dataframe(results)

st.markdown("The metric ($e^{\\text{-cross-entropy}}$) can be seen as the average probability of predicting the correct pitch in an at-bat. The results show a large improvement from the baseline model. " \
"As with almost all models, the more complex the model is, the more data-hungry the model becomes. This can be seen in the final Markov Model in which it is likely that there was not enough data for each " \
"pitcher to have a proper representation of how they each pitch to right- and left-handed hitters. This can be mitigated using a conjugate prior for each pitcher. The conjugate prior would follow the heirarchy: " \
"Global Pitcher Model -> Pitcher Model -> Pitcher Split Model. The conjugate prior essentially assumes that (without any data) the distribution of pitches thrown by a new pitch will be similar to the global model, etc. " \
"I plan to come back and properly create these conjugate priors for better modeling of the pitch predictions.")
