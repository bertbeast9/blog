import numpy as np
import scipy
import streamlit as st
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
class AveragePitchPredictor():

    def __init__(self, state_names):
        self.state_names = state_names
        self.n_states = len(self.state_names)


    def fit(self, at_bats):
        pi = np.zeros([1,self.n_states])
        progress_bar = st.progress(0, text="Fitting Average Model")
        for at_bat_idx, at_bat in enumerate(at_bats):
            progress_bar.progress(at_bat_idx/len(at_bats), text="Fitting Average Model")
            for pitch_idx, pitch_data in at_bat.iterrows():
                pi[0,pitch_data.curr_state_int] += 1
        self.pi = {"NA": {"NA":pi / np.sum(pi)}}
        return
    
    def predict(self, curr_state, split = "NA", pitcher_id = "NA"):
        """
        Predict the distribution of next state
        """
        return self.pi[pitcher_id][split]
    
    def score(self, at_bats, metric = "cross-entropy"):
        """
        Scores the model based on the data from at_bats
        """
        if metric == "cross-entropy":
            ctr = 0
            y_true = []
            y_pred = []
            progress_bar = st.progress(0, text="Scoring Model")
            for at_bat_idx, at_bat in enumerate(at_bats):
                progress_bar.progress(at_bat_idx/len(at_bats), text="Scoring Model")
                for pitch_idx, pitch_data in at_bat.iterrows():
                    if pitch_idx == at_bat.index[0]:
                        pitcher_id = "NA"
                        pitch_throws = "N"
                        bat_stand = "A"
                        curr_state_dist = self.pi[f"{pitcher_id}"][f"{pitch_throws}{bat_stand}"]
                    next_state_dist = self.predict(curr_state_dist, split=f"{pitch_throws}{bat_stand}", pitcher_id=f"{pitcher_id}")
                    y_pred.append(next_state_dist)
                    last_state = curr_state_dist
                    curr_state_dist = np.zeros([1,self.n_states])
                    curr_state_dist[0,pitch_data.curr_state_int] = 1
                    y_true.append(curr_state_dist)
                    ctr += 1
            y_true = np.concat(y_true, axis=0)
            y_pred = np.concat(y_pred, axis=0)
            metric_val = np.exp(-log_loss(y_true, y_pred))
        return metric_val

class MarkovPitchPredictor(AveragePitchPredictor):

    def fit(self, at_bats):
        A = 1e-0 * np.zeros([self.n_states,self.n_states])
        pi = np.zeros([1,self.n_states])
        progress_bar = st.progress(0, text="Fitting Markov Model")
        for at_bat_idx, at_bat in enumerate(at_bats):
            progress_bar.progress(at_bat_idx/len(at_bats), text="Fitting Markov Model")
            for pitch_idx, pitch_data in at_bat.iterrows():
                if pitch_idx == at_bat.index[0]:
                    last_state = pitch_data.curr_state_int
                    continue
                else:
                    curr_state = pitch_data.curr_state_int
                    A[last_state, curr_state] += 1
                    last_state = curr_state
        A = A / np.sum(A, axis = 1).reshape(-1,1)
        (evals, evecs) = scipy.linalg.eig(A, left=True, right=False)
        tmp = list(zip(evals, np.array_split(evecs, self.n_states, axis=1)))
        tmp = sorted(tmp, key = lambda x: np.abs(x[0] - 1))
        pi = tmp[0][1]
        pi = pi / np.sum(pi)
        self.A = {"NA":{"NA":A}}
        self.pi = {"NA":{"NA":np.real(pi).reshape(1,-1)}}
        return
    
    def predict(self, curr_state, pitcher_id="NA", split="NA"):
        next_state_dist = curr_state @ self.A[pitcher_id][split]
        return next_state_dist
    

class MarkovPitchPredictorHandedness(MarkovPitchPredictor):
    def fit(self, at_bats):
        self.A = {"NA":{"NA": 1e-0 * np.zeros([self.n_states, self.n_states]), 
                        "LL": 1e-0 * np.zeros([self.n_states, self.n_states]), 
                        "LR": 1e-0 * np.zeros([self.n_states, self.n_states]), 
                        "RL": 1e-0 * np.zeros([self.n_states, self.n_states]), 
                        "RR": 1e-0 * np.zeros([self.n_states, self.n_states])}}
        self.pi = {"NA":{"NA": 1e-0 * np.zeros([1, self.n_states]), 
                        "LL": 1e-0 * np.zeros([1, self.n_states]), 
                        "LR": 1e-0 * np.zeros([1, self.n_states]), 
                        "RL": 1e-0 * np.zeros([1, self.n_states]), 
                        "RR": 1e-0 * np.zeros([1, self.n_states])}}
        progress_bar = st.progress(0, text="Fitting Markov Model")
        for at_bat_idx, at_bat in enumerate(at_bats):
            progress_bar.progress(at_bat_idx/len(at_bats), text="Fitting Markov Model")
            for pitch_idx, pitch_data in at_bat.iterrows():
                if pitch_idx == at_bat.index[0]:
                    if f"{pitch_data.pitcher_id}" not in self.A.keys():
                        self.A[f"{pitch_data.pitcher_id}"] = {"NA": 1e-0 * np.zeros([self.n_states, self.n_states]), 
                        "LL": 1e-0 * np.zeros([self.n_states, self.n_states]), 
                        "LR": 1e-0 * np.zeros([self.n_states, self.n_states]), 
                        "RL": 1e-0 * np.zeros([self.n_states, self.n_states]), 
                        "RR": 1e-0 * np.zeros([self.n_states, self.n_states])}
                        self.pi[f"{pitch_data.pitcher_id}"] = {"NA": 1e-0 * np.zeros([1, self.n_states]), 
                        "LL": 1e-0 * np.zeros([1, self.n_states]), 
                        "LR": 1e-0 * np.zeros([1, self.n_states]), 
                        "RL": 1e-0 * np.zeros([1, self.n_states]), 
                        "RR": 1e-0 * np.zeros([1, self.n_states])}
                    pitcher_id = pitch_data.pitcher_id
                    last_state = pitch_data.curr_state_int
                    bat_stand = pitch_data.stand
                    pitch_throws = pitch_data.p_throws
                    continue
                else:
                    curr_state = pitch_data.curr_state_int
                    self.A[f"{pitcher_id}"][f"{pitch_throws}{bat_stand}"][last_state, curr_state] += 1
                    self.A[f"{pitcher_id}"]["NA"][last_state, curr_state] += 1
                    self.A["NA"][f"{pitch_throws}{bat_stand}"][last_state, curr_state] += 1
                    self.A["NA"]["NA"][last_state, curr_state] += 1
                    last_state = curr_state
        for key0, val0 in self.A.items():
            for key1, val1 in val0.items():
                if np.sum(self.A[key0][key1]) > 0:
                    self.A[key0][key1] = self.A[key0][key1] / np.maximum(1,np.sum(self.A[key0][key1], axis=1).reshape(-1,1))
                    (evals, evecs) = scipy.linalg.eig(self.A[key0][key1], left=True, right=False)
                    tmp = list(zip(evals, np.array_split(evecs, self.n_states, axis=1)))
                    tmp = sorted(tmp, key = lambda x: np.abs(x[0] - 1))
                    pi = tmp[0][1]
                    pi = pi / np.sum(pi)
                    self.pi[key0][key1] = np.real(pi).reshape(1,-1)
        return
    
    def predict(self, curr_state, pitcher_id="NA", split="NA"):
        next_state_dist = curr_state @ self.A[pitcher_id][split]
        return next_state_dist
    
    def score(self, at_bats, metric = "cross-entropy"):
        """
        Scores the model based on the data from at_bats
        """
        if metric == "cross-entropy":
            ctr = 0
            y_true = []
            y_pred = []
            progress_bar = st.progress(0, text="Scoring Model")
            for at_bat_idx, at_bat in enumerate(at_bats):
                progress_bar.progress(at_bat_idx/len(at_bats), text="Scoring Model")
                for pitch_idx, pitch_data in at_bat.iterrows():
                    if pitch_idx == at_bat.index[0]:
                        bat_stand = pitch_data.stand
                        pitch_throws = pitch_data.p_throws
                        pitcher_id = pitch_data.pitcher_id
                        if pitcher_id not in self.pi.keys():
                            pitcher_id = "NA"
                        curr_state_dist = self.pi[f"{pitcher_id}"][f"{pitch_throws}{bat_stand}"]
                    next_state_dist = self.predict(curr_state_dist, split=f"{pitch_throws}{bat_stand}", pitcher_id=f"{pitcher_id}")
                    y_pred.append(next_state_dist)
                    last_state = curr_state_dist
                    curr_state_dist = np.zeros([1,self.n_states])
                    curr_state_dist[0,pitch_data.curr_state_int] = 1
                    y_true.append(curr_state_dist)
                    ctr += 1
            y_true = np.concat(y_true, axis=0)
            y_pred = np.concat(y_pred, axis=0)
            metric_val = np.exp(-log_loss(y_true, y_pred))
        return metric_val
    

class RFModelPitchPredictor():

    def __init__(self, state_names):
        self.state_names = state_names
        self.n_states = len(self.state_names)
        return
    
    def fit(self, at_bats, features):
        progress_bar = st.progress(0, text="Fitting Random Forest Model")
        X, y = [], []
        for at_bat_idx, at_bat in enumerate(at_bats):
            progress_bar.progress(at_bat_idx/len(at_bats), text="Fitting Random Forest Model")
            for pitch_idx, pitch_data in at_bat.iterrows():
                X.append(pitch_data[features].values.reshape(1,-1))
                # tmp = np.zeros([1, self.n_states])
                # tmp[0,list(self.state_names).index(pitch_data["pitch_type"])] = 1
                tmp = [list(self.state_names).index(pitch_data["pitch_type"])]
                y.append(tmp)
        X = np.concat(X, axis=0)
        y = np.concat(y, axis=0)
        self.rf_mdl = RandomForestClassifier()
        self.rf_mdl.fit(X,y)
        return
    
    def score(self, at_bats, features, metric="cross-entropy"):
        """
        Scores the model based on the data from at_bats
        """
        if metric == "cross-entropy":
            progress_bar = st.progress(0, text="Scoring Random Forest Model")
            
            X, y = [], []
            for at_bat_idx, at_bat in enumerate(at_bats):
                progress_bar.progress(at_bat_idx/len(at_bats), text="Scoring Random Forest Model")
                for pitch_idx, pitch_data in at_bat.iterrows():
                    X.append(pitch_data[features].values.reshape(1,-1))
                    # tmp = np.zeros([1, self.n_states])
                    # tmp[0,list(self.state_names).index(pitch_data["pitch_type"])] = 1
                    tmp = [list(self.state_names).index(pitch_data["pitch_type"])]
                    y.append(tmp)
            X = np.concat(X, axis=0)
            y_true = np.concat(y, axis=0)
            y_pred = self.rf_mdl.predict_proba(X)
            metric_val = np.exp(-log_loss(y_true, y_pred))

        return metric_val