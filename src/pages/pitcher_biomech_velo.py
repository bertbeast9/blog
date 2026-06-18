
import pandas as pd
import numpy as np
import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing, impute, linear_model, metrics, feature_selection
import matplotlib.pyplot as plt
import scipy

streamlit_on = False
if streamlit_on:
    import streamlit as st


### FUNCTION SECTION

def split_data_by_pitch(df, feature_cols, target_col = "velocity", k_folds = 5):
    X_train, X_test, y_train, y_test = None, None, None, None
    pitch_ids = np.arange(0,df.shape[0])
    np.random.shuffle(pitch_ids)
    stepsize = int(len(pitch_ids) / k_folds)
    folds_data = []
    for k in range(k_folds):
        pitch_ids = np.roll(pitch_ids, stepsize)
        train_pitch_ids = pitch_ids[:stepsize * (k_folds - 1)]
        test_pitch_ids = pitch_ids[stepsize * (k_folds - 1):]
        X_train = df.iloc[train_pitch_ids][feature_cols].values
        X_test = df.iloc[test_pitch_ids][feature_cols].values
        y_train = df.iloc[train_pitch_ids][target_col].values
        y_test = df.iloc[test_pitch_ids][target_col].values
        # standardize target
        mu_y = np.mean(y_train)
        std_y = np.std(y_train)
        y_train = (y_train - mu_y) / std_y
        y_test = (y_test - mu_y) / std_y
        folds_data.append(((X_train, y_train), (X_test, y_test)))
    return folds_data

def split_data_by_pitcher(df, feature_cols, target_col = "velocity", k_folds = 5):
    X_train, X_test, y_train, y_test = None, None, None, None
    pitch_ids = df["pitcher_id"].value_counts().index.to_numpy()
    np.random.shuffle(pitch_ids)
    stepsize = int(len(pitch_ids) / k_folds)
    folds_data = []
    for k in range(k_folds):
        pitch_ids = np.roll(pitch_ids, stepsize)
        train_pitch_ids = pitch_ids[:stepsize * (k_folds - 1)]
        test_pitch_ids = pitch_ids[stepsize * (k_folds - 1):]
        X_train = df[df["pitcher_id"].apply(lambda x: True if x in train_pitch_ids else False)][feature_cols].values
        X_test = df[df["pitcher_id"].apply(lambda x: True if x in test_pitch_ids else False)][feature_cols].values
        y_train = df[df["pitcher_id"].apply(lambda x: True if x in train_pitch_ids else False)][target_col].values
        y_test = df[df["pitcher_id"].apply(lambda x: True if x in test_pitch_ids else False)][target_col].values
        # standardize target
        mu_y = np.mean(y_train)
        std_y = np.std(y_train)
        y_train = (y_train - mu_y) / std_y
        y_test = (y_test - mu_y) / std_y
        folds_data.append(((X_train, y_train), (X_test, y_test)))
    return folds_data
### FUNCTION SECTION
if streamlit_on:
    st.header ("Pitcher Biomechanics & Velocity Project", divider=True)
    st.sidebar.markdown("# Pitcher Biomechanics & Velocity Project")
    st.subheader("Problem Overview",divider=True)
    st.markdown("The aim of this small project is to essentially create a mapping between certain biomechanical features "
    "and an estimate of the pitcher's fastball velocity. However, the true question is '_which pitcher do you believe will " \
    "throw the hardest in five years?'_ In order to answer this question better, more data on pitcher's age and their velocities " \
    "over time would be helpful. Yet, we only have the data that we were provided. We have been provided two distinct datasets. " \
    "The first dataset, " \
    "termed the _training_ dataset consists of the player's unique ID, a few physical traits (height and handedness), and a " \
    "lot of biomechanical features extracted from Hawkeye. The _training_ dataset also contains the pitcher's fastball velocity for " \
    "that specific pitch. The _testing_ dataset only has the biomechanic features upon which we are supposed to assign fastball velocities.")
    st.subheader("Testing Framework", divider=True)
    st.markdown("Ultimately, the powers-that-be have the true fastball velocities for the _testing_ dataset. So, I aim to validate " \
    "and compare each of these models using some k-fold cross-validation of the _training_ dataset. To make this testing more realistic, " \
    "I am not going to shuffle across pitches but shuffle pitchers. I want the model to see brand new pitchers to assess " \
    "the model's generalizability.")
    st.subheader("Dataset Breakdown", divider=True)

train_data = pd.read_csv("./src/data/Phillies_AB_project/velo_and_mechanics.csv")
train_data["date"] = train_data["date"].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%y"))
# train_data = train_data[train_data["pitcher_handedness"] == "l"]
print(train_data["date"])
train_data.sort_values(by=["date","pitcher_id"], inplace = True)
feature_cols = [x for x in train_data.columns if x not in ["date","pitcher_id","velocity"]]
# for feature_col in feature_cols:
#     plt.scatter(train_data[feature_col], train_data["velocity"])
#     plt.title(feature_col)
#     plt.show()
uniq_pitcher_and_n_pitches = train_data["pitcher_id"].value_counts()
n_pitchers = len(uniq_pitcher_and_n_pitches)
if streamlit_on:
    st.dataframe(train_data)
    st.markdown(f"The data ranges from {train_data["date"].min().strftime("%m/%d/%y")} to {train_data["date"].max().strftime("%m/%d/%y")}. " \
    f"There are {n_pitchers} unique pitchers in the dataset with the minimum and maximum number of pitches from a particular pitcher are " \
    f"{uniq_pitcher_and_n_pitches.iloc[-1]} (ID: {uniq_pitcher_and_n_pitches.index[-1]}) and {uniq_pitcher_and_n_pitches.iloc[0]} " \
    f"(ID: {uniq_pitcher_and_n_pitches.index[0]}), respectively. The average number of pitches from any given pitcher is {uniq_pitcher_and_n_pitches.mean():0.2f} pitches. " \
    f"There are {len(feature_cols)} features and {train_data.shape[0]} data points in the _training_ dataset." \
    "Let's go ahead visualize the target distribution.")

target_hist = plt.figure(0)
ax = target_hist.add_subplot(111)
ax.hist(train_data["velocity"], 100)
ax.set_xlabel("velocity [mph]")
ax.set_ylabel("count")
ax.set_title("Training data: velocity histogram")
stat_norm_result = scipy.stats.normaltest(train_data["velocity"])
if not streamlit_on:
    target_hist.show()
if streamlit_on:
    st.pyplot(fig=target_hist)
    st.markdown("The target distribution looks pretty close to normally distributed. We can verify this via SciPy's normaltest (D’Agostino, R. B. (1971). An omnibus test of normality for moderate and large sample size. Biometrika, 58, 341-348)")
    st.markdown(f"The statistics p-value is {stat_norm_result.pvalue} which confirms the visual inspection of normality. ")
    st.markdown(f"There is missing values in the dataset, so we will need to perform some imputation on each training dataset.")

    st.subheader("Simplest Model", divider=True)
    st.markdown("It's best to start with the simplest model, and use this as the benchmark to beat with more complex models.")
feature_cols.pop(feature_cols.index("pitcher_handedness"))
folds_data = split_data_by_pitch(train_data, feature_cols, k_folds = 5)
# folds_data = split_data_by_pitcher(train_data, feature_cols, k_folds = 5)
for k, ((X_train, y_train), (X_test, y_test)) in enumerate(folds_data):
    # impute data
    imputer = impute.IterativeImputer()
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    # standardize data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled_train = scaler.transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    # X_scaled_train = np.clip(X_scaled_train, -2, 2)
    # X_scaled_test = np.clip(X_scaled_test, -2, 2)
    # # include intercept term
    # X_scaled_train = np.concat([np.ones([X_scaled_train.shape[0],1]), X_scaled_train], axis = 1)
    # X_scaled_test = np.concat([np.ones([X_scaled_test.shape[0],1]), X_scaled_test], axis = 1)
    # # select features
    base_est = linear_model.LinearRegression(fit_intercept=True)
    # base_est = linear_model.HuberRegressor(epsilon = 1.1, fit_intercept=True)
    # base_est = linear_model.RANSACRegressor()
    # base_est = linear_model.TheilSenRegressor()
    sfs = feature_selection.SequentialFeatureSelector(base_est, tol = 0.1)
    # sfs = feature_selection.SequentialFeatureSelector(base_est, n_features_to_select=1)
    sfs.fit(X_scaled_train, y_train)
    X_scaled_train = sfs.transform(X_scaled_train)
    X_scaled_test = sfs.transform(X_scaled_test)
    sel_col_idx = sfs.get_support()
    selected_features = [x for (x,y) in zip(feature_cols, sel_col_idx) if y]
    print(selected_features)
    # for idx in range(X_scaled_test.shape[1]):
    #     plt.hist(X_scaled_test[:,idx],100)
    #     plt.show()
    
    base_est.fit(X_scaled_train, y_train)
    print(base_est.coef_)

    # # train data
    # y_train_pred = base_est.predict(X_scaled_train)
    # fit_plot = plt.figure(k+1)
    # ax = fit_plot.add_subplot(111)
    # ax.scatter(y_train, y_train_pred)
    # ax.set_xlabel("true velocity [mph]")
    # ax.set_ylabel("predicted velocity [mph]")
    # ax.set_title(f"Training data (fold {k}): LASSO model prediction $R^2$: {metrics.r2_score(y_train, y_train_pred) * 100: 0.2f}")
    # st.pyplot(fig = fit_plot)
    # test data
    y_test_pred = base_est.predict(X_scaled_test)
    r2_score = metrics.r2_score(y_test, y_test_pred) * 100
    print(f"R2 score for fold {k}: {r2_score: 0.2f}")
    fit_plot = plt.figure(k+1)
    ax = fit_plot.add_subplot(111)
    ax.scatter(y_test, y_test_pred)
    ax.set_xlabel("true velocity [mph]")
    ax.set_ylabel("predicted velocity [mph]")
    ax.set_title(f"Testing data (fold {k}): LASSO model prediction $R^2$: {r2_score: 0.2f}")
    # # ax.hist(X_scaled_test, 100)
    if streamlit_on:
        st.pyplot(fig = fit_plot)
    else:
        plt.show()
    

        

