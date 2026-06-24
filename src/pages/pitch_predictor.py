import streamlit as st
import numpy as np
import pandas as pd
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


#### FUNCTIONS
@st.cache_data
def load_pitch_data():
    pitch_data = pd.read_csv("./src/data/Swish_Baseball_project/MLB/pitches.csv")
    print(pitch_data.head())
    return pitch_data

@st.cache_data
def order_pitches_global(df):
    df.sort_values(["date"], inplace=True)
    return df
#### FUNCTIONS



#### VARIABLES
pitch_dict = {"FF":"4-Seam Fastball","SI":"Sinker (2-Seam)","FC":"Cutter",
          "CH":"Changeup","FS":"Split-finger","SC":"Screwball",
          "CU":"Curveball","KC":"Knuckle Curve","CS":"Slow Curve","SL":"Slider",
          "ST":"Sweeper","SV":"Slurve","KN":"Knuckleball","EP":"Eephus","FA":"Other",
          "IN":"Intentional Ball","PO":"Pitchout"}#
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
pitch_data = pitch_data[pitch_data["pitch_type"].apply(lambda x: False if x in ["AB", "EP", "FA", "FO", "PO", "SC", "UN"] else True)]
pitch_hist = plt.figure()
poss_pitches, counts = np.unique(pitch_data["pitch_type"], return_counts=True)
plt.bar(poss_pitches, counts)
plt.ylabel("pitch count")
plt.xlabel("pitch acronym")
plt.title("Pitch Type Histogram")
st.pyplot(pitch_hist)
print(poss_pitches)
pitch_data["curr_state_int"] = pitch_data["pitch_type"].apply(lambda x: list(poss_pitches).index(x))
st.markdown("Now, I need to order the data such that this methodology makes sense. I need to break up this data by game and by at-bat. Each new at-bat will be the start of a new Markov Chain. Also, " \
"I need to ensure that each pitch sequence is proper (pitch 1 -> pitch 2 -> pitch 3 etc.) and not (pitch 1 -> pitch 3).")

print(pitch_data["date"].iloc[0])
pitch_data["date"] = pitch_data["date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
print(pitch_data["date"].dtype)
pitch_data = order_pitches_global(pitch_data)
st.dataframe(pitch_data)


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




