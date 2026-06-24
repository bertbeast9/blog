import numpy as np
import scipy

class AveragePitchPredictor():

    def __init__(self, state_names):
        self.state_names = state_names
        self.n_states = len(self.state_names)


    def fit(self, at_bats):
        pi = np.zeros([1,self.n_states])
        for at_bat_idx, at_bat in enumerate(at_bats):
            for pitch_idx, pitch_data in at_bat.iterrows():
                pi[0,pitch_data.curr_state_int] += 1
        self.pi = {"NA": {"NA":pi / np.sum(pi)}}
        return
    
    def predict(self, curr_state):
        """
        Predict the distribution of next state
        """
        return self.pi["NA"]["NA"]
    
    def score(self, at_bats, metric = "nll"):
        """
        Scores the model based on the data from at_bats
        """
        if metric == "nll" or metric == "avg_like":
            metric_val = 0
            ctr = 0
            for at_bat_idx, at_bat in enumerate(at_bats):
                curr_state = self.pi["NA"]["NA"]
                for pitch_idx, pitch_data in at_bat.iterrows():
                    next_state_dist = self.predict(curr_state)
                    last_state = curr_state
                    metric_val -= np.log(next_state_dist[0,pitch_data.curr_state_int])
                    curr_state = np.zeros([1,self.n_states])
                    curr_state[0,pitch_data.curr_state_int] = 1
                    ctr += 1
            if metric == "avg_like":
                metric_val = np.exp(-(metric_val/ctr))
        return metric_val

class MarkovPitchPredictor(AveragePitchPredictor):

    def fit(self, at_bats):
        A = np.ones([self.n_states,self.n_states])
        pi = np.zeros([1,self.n_states])
        for at_bat_idx, at_bat in enumerate(at_bats):
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
    
    def predict(self, curr_state, name="NA", split="NA"):
        next_state_dist = curr_state @ self.A[name][split]
        return next_state_dist
    

class MarkovPitchPredictorHandedness(MarkovPitchPredictor):
    def fit(self, at_bats):
        self.A = {"NA":{"NA": np.ones([self.n_states, self.n_states]), 
                        "LL": np.ones([self.n_states, self.n_states]), 
                        "LR": np.ones([self.n_states, self.n_states]), 
                        "RL": np.ones([self.n_states, self.n_states]), 
                        "RR": np.ones([self.n_states, self.n_states])}}
        self.pi = {"NA":{"NA": np.zeros([1, self.n_states]), 
                        "LL": np.zeros([1, self.n_states]), 
                        "LR": np.zeros([1, self.n_states]), 
                        "RL": np.zeros([1, self.n_states]), 
                        "RR": np.zeros([1, self.n_states])}}
        for at_bat_idx, at_bat in enumerate(at_bats):
            for pitch_idx, pitch_data in at_bat.iterrows():
                if pitch_idx == at_bat.index[0]:
                    last_state = pitch_data.curr_state_int
                    bat_stand = pitch_data.stand
                    pitch_throws = pitch_data.p_throws
                    continue
                else:
                    curr_state = pitch_data.curr_state_int
                    self.A["NA"][f"{pitch_throws}{bat_stand}"][last_state, curr_state] += 1
                    self.A["NA"]["NA"][last_state, curr_state] += 1
                    last_state = curr_state
        for key0, val0 in self.A.items():
            for key1, val1 in val0.items():
                self.A[key0][key1] = self.A[key0][key1] / np.sum(self.A[key0][key1], axis=1).reshape(-1,1)
                (evals, evecs) = scipy.linalg.eig(self.A[key0][key1], left=True, right=False)
                tmp = list(zip(evals, np.array_split(evecs, self.n_states, axis=1)))
                tmp = sorted(tmp, key = lambda x: np.abs(x[0] - 1))
                pi = tmp[0][1]
                pi = pi / np.sum(pi)
                self.pi[key0][key1] = np.real(pi).reshape(1,-1)
        return
    
    def predict(self, curr_state, name="NA", split="NA"):
        next_state_dist = curr_state @ self.A[name][split]
        return next_state_dist
    
    def score(self, at_bats, metric = "nll"):
        """
        Scores the model based on the data from at_bats
        """
        if metric == "nll" or metric == "avg_like":
            metric_val = 0
            ctr = 0
            for at_bat_idx, at_bat in enumerate(at_bats):
                for pitch_idx, pitch_data in at_bat.iterrows():
                    if pitch_idx == at_bat.index[0]:
                        bat_stand = pitch_data.stand
                        pitch_throws = pitch_data.p_throws
                        curr_state = self.pi["NA"][f"{pitch_throws}{bat_stand}"]
                    next_state_dist = self.predict(curr_state, split=f"{pitch_throws}{bat_stand}")
                    last_state = curr_state
                    metric_val -= np.log(next_state_dist[0,pitch_data.curr_state_int])
                    curr_state = np.zeros([1,self.n_states])
                    curr_state[0,pitch_data.curr_state_int] = 1
                    ctr += 1
            if metric == "avg_like":
                metric_val = np.exp(-(metric_val/ctr))
        return metric_val