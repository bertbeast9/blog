
import pandas as pd
import numpy as np
import os
import pickle
import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing, impute, linear_model, metrics, feature_selection, model_selection, ensemble, neighbors, gaussian_process
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import scipy
from sklearn import set_config
set_config(enable_metadata_routing=True)

streamlit_on = True
sfs_tol = 0.01
k_folds = 5


if streamlit_on:
    import streamlit as st


### FUNCTION SECTION

def split_data_by_pitcher_id(pitcher_ids, k_folds=5, indices_only=False):
    uniq_pitcher_ids, counts = np.unique(pitcher_ids, return_counts=True)
    # tmp = zip(uniq_pitcher_ids, counts)
    # tmp = sorted(tmp, key = lambda x: x[1])
    # uniq_pitcher_ids = [x[0] for x in tmp]
    # counts = [x[1] for x in tmp]
    np.random.shuffle(uniq_pitcher_ids)
    stepsize = int(len(uniq_pitcher_ids) / k_folds)
    cv_obj = []
    for i in range(k_folds):
        uniq_pitcher_ids = np.roll(uniq_pitcher_ids, stepsize)
        train_pitch_ids = uniq_pitcher_ids[:stepsize * (k_folds - 1)]
        test_pitch_ids = uniq_pitcher_ids[stepsize * (k_folds - 1):]
        # test_pitch_ids = uniq_pitcher_ids[i::k_folds]
        # train_pitch_ids = [x for x in uniq_pitcher_ids if x not in test_pitch_ids]
        train_indices, train_pitcher_ids, test_indices, test_pitcher_ids = [], [], [], []
        for k, pitcher_id in enumerate(pitcher_ids):
            if pitcher_id in train_pitch_ids:
                train_indices.append(k)
                train_pitcher_ids.append(pitcher_id)
            elif pitcher_id in test_pitch_ids:
                test_indices.append(k)
                test_pitcher_ids.append(pitcher_id)
        if indices_only:
            cv_obj.append((train_indices, test_indices))
        else:
            cv_obj.append(((train_indices, train_pitcher_ids), (test_indices, test_pitcher_ids)))
    return cv_obj

def split_data_by_pitch_id(pitch_ids, k_folds=5, indices_only=False):
    uniq_pitch_ids, counts = np.unique(pitch_ids, return_counts=True)
    np.random.shuffle(uniq_pitch_ids)
    stepsize = int(len(uniq_pitch_ids) / k_folds)
    cv_obj = []
    for i in range(k_folds):
        uniq_pitch_ids = np.roll(uniq_pitch_ids, stepsize)
        train_pitch_ids = uniq_pitch_ids[:stepsize * (k_folds - 1)]
        test_pitch_ids = uniq_pitch_ids[stepsize * (k_folds - 1):]
        # test_pitch_ids = uniq_pitch_ids[i::k_folds]
        # train_pitch_ids = [x for x in uniq_pitch_ids if x not in test_pitch_ids]
        train_indices, train_pitcher_ids, test_indices, test_pitcher_ids = [], [], [], []
        for k, pitcher_id in enumerate(pitch_ids):
            if pitcher_id in train_pitch_ids:
                train_indices.append(k)
                train_pitcher_ids.append(pitcher_id)
            elif pitcher_id in test_pitch_ids:
                test_indices.append(k)
                test_pitcher_ids.append(pitcher_id)
        if indices_only:
            cv_obj.append((train_indices, test_indices))
        else:
            cv_obj.append(((train_indices, train_pitcher_ids), (test_indices, test_pitcher_ids)))
    return cv_obj

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
    "I am not only going to shuffle across pitches but shuffle across pitchers as well. I want the model to see brand new pitchers to assess " \
    "the model's generalizability.")
    st.subheader("Dataset Breakdown", divider=True)


train_df = pd.read_csv("./src/data/Phillies_AB_project/velo_and_mechanics.csv")
test_df = pd.read_csv("./src/data/Phillies_AB_project/fcl_mechanics_2025.csv")
train_df["date"] = train_df["date"].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%y"))
test_df["date"] = test_df["date"].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%y"))
train_df.sort_values(by=["date","pitcher_id"], inplace = True)
test_df.sort_values(by=["date","pitcher_id"], inplace = True)
feature_cols = [x for x in train_df.columns if x not in ["date","pitcher_id","velocity"]]
uniq_pitcher_and_n_pitches = train_df["pitcher_id"].value_counts()
n_pitchers = len(uniq_pitcher_and_n_pitches)
if streamlit_on:
    st.dataframe(train_df)
    st.markdown(f"The data ranges from {train_df["date"].min().strftime("%m/%d/%y")} to {train_df["date"].max().strftime("%m/%d/%y")}. " \
    f"There are {n_pitchers} unique pitchers in the dataset with the minimum and maximum number of pitches from a particular pitcher are " \
    f"{uniq_pitcher_and_n_pitches.iloc[-1]} (ID: {uniq_pitcher_and_n_pitches.index[-1]}) and {uniq_pitcher_and_n_pitches.iloc[0]} " \
    f"(ID: {uniq_pitcher_and_n_pitches.index[0]}), respectively. The average number of pitches from any given pitcher is {uniq_pitcher_and_n_pitches.mean():0.2f} pitches. " \
    f"There are {len(feature_cols)} features and {train_df.shape[0]} data points in the _training_ dataset. " \
    "Let's go ahead visualize the target distribution.")

target_hist = plt.figure(0)
ax = target_hist.add_subplot(111)
ax.hist(train_df["velocity"], 100)
ax.set_xlabel("velocity [mph]")
ax.set_ylabel("count")
ax.set_title("Training data: velocity histogram")
stat_norm_result = scipy.stats.normaltest(train_df["velocity"])
if not streamlit_on:
    target_hist.show()
if streamlit_on:
    st.pyplot(fig=target_hist)
    st.markdown("The target distribution looks pretty close to normally distributed. We can verify this via SciPy's normaltest (D’Agostino, R. B. (1971). An omnibus test of normality for moderate and large sample size. Biometrika, 58, 341-348)")
    st.markdown(f"The statistics p-value is {stat_norm_result.pvalue} which confirms the visual inspection of normality. ")
    st.markdown(f"There are missing values in the dataset, so we will need to perform some imputation on each training dataset.")

    st.subheader("Pitch Cross Validation", divider=True)
    st.markdown("For this cross validation, we will cross-validate purely on shuffled pitch data across all pitchers in the training dataset. This is likely not to be the best " \
    "test for replicating seeing a brand new pitcher, but it is worth exploring.")
    st.subheader("Pitcher Cross Validation", divider=True)
    st.markdown("For this cross validation, we will separate the data by pitcher such that the testing data is brand new to the trained model (i.e. Pitcher 3 does not " \
    "have a pitch in the _training_ dataset, only in the _testing_ dataset). I believe this is a more realistic test for the model since this is what will happen in real-life with " \
    "a new player that needs to be evaluated.")
    
    st.subheader("Model Pipeline", divider=True)
    st.markdown("For each fold, we will fit the pipeline to the training dataset and then test this fit pipeline to the testing dataset (depending on the cross validation type). Each model pipeline will " \
    "start with data imputation for the missing NaNs in the _training_ and _testing_ datasets. This will be achieved via SciKit Learn's IterativeImputer function (https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html). " \
    "Then, the input data will be normalized via the RobustScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) AKA IQR. Finally, we will perform feature selection using " \
    f"the SequentialFeatureSelector (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) with a tolerance of {sfs_tol} change in the estimator's score ($R^2$). " \
    "We will use a few different model's for the base estimator: OLS, Ridge, LASSO, and Random Forest Regression.")



feature_cols.pop(feature_cols.index("pitcher_handedness"))
X = train_df[feature_cols].values
y = train_df["velocity"].values
pitcher_ids = train_df["pitcher_id"].values
pitch_cv_results_file_name = f"./src/data/Phillies_AB_project/pitch_cv_results.pkl"
if not os.path.isfile(pitch_cv_results_file_name):
    ### Pitch Cross Validation
    pitch_cv_results = {}
    cv_obj = split_data_by_pitch_id(train_df.index.to_numpy(), k_folds = k_folds)
    for kth_fold, (train_data, test_data) in enumerate(cv_obj):
        X_train, y_train, X_test, y_test = X[train_data[0],:], y[train_data[0]], X[test_data[0],:], y[test_data[0]]
        imputer = impute.IterativeImputer()
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)
        # scale X
        scaler = preprocessing.RobustScaler(unit_variance=True)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = np.clip(X_train, -3, 3)
        X_test = np.clip(X_test, -3, 3)
        # scale y
        scaler = preprocessing.RobustScaler(unit_variance=True)
        scaler.fit(y_train.reshape(-1,1))
        y_train = scaler.transform(y_train.reshape(-1,1)).reshape(-1)
        y_test = scaler.transform(y_test.reshape(-1,1)).reshape(-1)
        y_train = np.clip(y_train, -3, 3)
        y_test = np.clip(y_test, -3, 3)
        pitch_cv_base_ests = [
            ("OLS", linear_model.LinearRegression(fit_intercept=True).set_score_request(sample_weight=True).set_fit_request(sample_weight=True)),
            ("Ridge", linear_model.RidgeCV(alphas = np.logspace(-3,3,10)).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)),
            ("LASSO", linear_model.LassoCV(alphas = np.logspace(-3,3,10)).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)),
            ("RF", ensemble.RandomForestRegressor(n_estimators=25, n_jobs=-1)),
        ]
        fold_results =  []
        for base_est_name, base_est in pitch_cv_base_ests:
            
            sfs = feature_selection.SequentialFeatureSelector(
                                                    estimator=base_est,
                                                    # n_features_to_select=1,
                                                    tol=sfs_tol,
                                                    direction="forward",
                                                    # cv=split_data_by_pitcher_id(train_data[1], indices_only=True),
                                                    cv=split_data_by_pitch_id(train_data[1], indices_only=True),
                                                    n_jobs=1,
                                                )
            
            sfs.fit(X_train, y_train)
            sel_col_idx = sfs.get_support()
            selected_features = [x for (x,y) in zip(feature_cols, sel_col_idx) if y]
            print(base_est_name)
            print(selected_features)
            X_train_sel = sfs.transform(X_train)
            X_test_sel = sfs.transform(X_test)
            base_est.fit(X_train_sel, y_train)
            if kth_fold == 0:
                pitch_cv_results[base_est_name] = dict()
                pitch_cv_results[base_est_name]["selected_features"] = []
                pitch_cv_results[base_est_name]["training"] = []
                pitch_cv_results[base_est_name]["testing"] = []
            pitch_cv_results[base_est_name]["selected_features"].append(selected_features)
            pitch_cv_results[base_est_name]["training"].append(100 * base_est.score(X_train_sel, y_train))
            pitch_cv_results[base_est_name]["testing"].append(100 * base_est.score(X_test_sel, y_test))

            # print(pitch_cv_results)
    pickle.dump((pitch_cv_results, pitch_cv_base_ests), open(pitch_cv_results_file_name,"wb"))
else:
    (pitch_cv_results, pitch_cv_base_ests) = pickle.load(open(pitch_cv_results_file_name,"rb"))
for base_est_name, base_est in pitch_cv_base_ests:
    print(f"{base_est_name}\nTraining $R^2$: {np.mean(pitch_cv_results[base_est_name]["training"])}% (+/-) {np.std(pitch_cv_results[base_est_name]["training"])}" \
            f"\nTesting $R^2$: {np.mean(pitch_cv_results[base_est_name]["testing"])}% (+/-) {np.std(pitch_cv_results[base_est_name]["testing"])}")

if streamlit_on:
    st.subheader("Pitch Cross Validation Raw Results",divider=True)
    st.json(pitch_cv_results)

X = train_df[feature_cols].values
y = train_df["velocity"].values
pitcher_ids = train_df["pitcher_id"].values
pitcher_cv_results_file_name = f"./src/data/Phillies_AB_project/pitcher_cv_results.pkl"
if not os.path.isfile(pitcher_cv_results_file_name):
    ### Pitcher Cross Validation
    pitcher_cv_results = {}
    cv_obj = split_data_by_pitcher_id(pitcher_ids, k_folds = k_folds)
    # cv_obj = split_data_by_pitch_id(train_df.index.to_numpy(), k_folds = k_folds)
    for kth_fold, (train_data, test_data) in enumerate(cv_obj):
        X_train, y_train, X_test, y_test = X[train_data[0],:], y[train_data[0]], X[test_data[0],:], y[test_data[0]]
        imputer = impute.IterativeImputer()
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)
        # scale X
        scaler = preprocessing.RobustScaler(unit_variance=True)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = np.clip(X_train, -3, 3)
        X_test = np.clip(X_test, -3, 3)
        # scale y
        scaler = preprocessing.RobustScaler(unit_variance=True)
        scaler.fit(y_train.reshape(-1,1))
        y_train = scaler.transform(y_train.reshape(-1,1)).reshape(-1)
        y_test = scaler.transform(y_test.reshape(-1,1)).reshape(-1)
        y_train = np.clip(y_train, -3, 3)
        y_test = np.clip(y_test, -3, 3)
        pitcher_cv_base_ests = [
            ("OLS", linear_model.LinearRegression(fit_intercept=True).set_score_request(sample_weight=True).set_fit_request(sample_weight=True)),
            ("Ridge", linear_model.RidgeCV(alphas = np.logspace(-3,3,10)).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)),
            ("LASSO", linear_model.LassoCV(alphas = np.logspace(-3,3,10)).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)),
            ("RF", ensemble.RandomForestRegressor(n_estimators=25, n_jobs=-1)),
        ]
        fold_results =  []
        for base_est_name, base_est in pitcher_cv_base_ests:
            sfs = feature_selection.SequentialFeatureSelector(
                                                    estimator=base_est,
                                                    tol=sfs_tol,
                                                    direction="forward",
                                                    cv=split_data_by_pitcher_id(train_data[1], indices_only=True),
                                                    n_jobs=1,
                                                )
            
            sfs.fit(X_train, y_train)
            sel_col_idx = sfs.get_support()
            selected_features = [x for (x,y) in zip(feature_cols, sel_col_idx) if y]
            print(base_est_name)
            print(selected_features)
            X_train_sel = sfs.transform(X_train)
            X_test_sel = sfs.transform(X_test)
            base_est.fit(X_train_sel, y_train)
            if kth_fold == 0:
                pitcher_cv_results[base_est_name] = dict()
                pitcher_cv_results[base_est_name]["selected_features"] = []
                pitcher_cv_results[base_est_name]["training"] = []
                pitcher_cv_results[base_est_name]["testing"] = []
            pitcher_cv_results[base_est_name]["selected_features"].append(selected_features)
            pitcher_cv_results[base_est_name]["training"].append(100 * base_est.score(X_train_sel, y_train))
            pitcher_cv_results[base_est_name]["testing"].append(100 * base_est.score(X_test_sel, y_test))
            # print(pitcher_cv_results)
    pickle.dump((pitcher_cv_results, pitcher_cv_base_ests), open(pitcher_cv_results_file_name,"wb"))
else:
    (pitcher_cv_results, pitcher_cv_base_ests) = pickle.load(open(pitcher_cv_results_file_name,"rb"))
    
for base_est_name, base_est in pitcher_cv_base_ests:
    print(f"{base_est_name}\nTraining $R^2$: {np.mean(pitcher_cv_results[base_est_name]["training"])}% (+/-) {np.std(pitcher_cv_results[base_est_name]["training"])}" \
            f"\nTesting $R^2$: {np.mean(pitcher_cv_results[base_est_name]["testing"])}% (+/-) {np.std(pitcher_cv_results[base_est_name]["testing"])}")


if streamlit_on:
    st.subheader("Pitcher Cross Validation Raw Results",divider=True)
    st.json(pitcher_cv_results)
    st.subheader("Results Analysis", divider=True)
    st.markdown("It is quite clear that the results from the pitch cross validation show strong predictive performance. However, the player cross validation does not " \
    "show strong predictive performance. Given more time, I would explore sample weights for the dataset to balance the effects of pitchers' with plenty of data.")

prediction_results_file_name = f"./src/data/Phillies_AB_project/prediction_results.pkl"
if not os.path.isfile(prediction_results_file_name):
    # PREDICTIONS
    X_pred = test_df[feature_cols].values
    pitcher_ids = test_df["pitcher_id"].values
    # PITCH CV MODELS
    imputer = impute.IterativeImputer()
    imputer.fit(X)
    X = imputer.transform(X)
    X_pred = imputer.transform(X_pred)
    # scale X
    scaler = preprocessing.RobustScaler(unit_variance=True)
    scaler.fit(X)
    X = scaler.transform(X)
    X_pred = scaler.transform(X_pred)
    X = np.clip(X, -3, 3)
    X_pred = np.clip(X_pred, -3, 3)
    # scale y
    scaler = preprocessing.RobustScaler(unit_variance=True)
    scaler.fit(y.reshape(-1,1))
    y = scaler.transform(y.reshape(-1,1)).reshape(-1)
    pitch_cv_predictions = {}
    df_cv_predictions = pd.DataFrame()
    cv_obj = split_data_by_pitch_id(train_df.index.to_numpy(), k_folds = k_folds, indices_only=True)
    pred_cv_base_ests = [
            ("OLS", linear_model.LinearRegression(fit_intercept=True).set_score_request(sample_weight=True).set_fit_request(sample_weight=True)),
            ("Ridge", linear_model.RidgeCV(alphas = np.logspace(-3,3,10)).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)),
            ("LASSO", linear_model.LassoCV(alphas = np.logspace(-3,3,10)).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)),
            ("RF", ensemble.RandomForestRegressor(n_estimators=25, n_jobs=-1)),
        ]
    for base_est_name, base_est in pred_cv_base_ests:
        sfs = feature_selection.SequentialFeatureSelector(
                                                    estimator=base_est,
                                                    tol=sfs_tol,
                                                    direction="forward",
                                                    cv=cv_obj,
                                                    n_jobs=1,
                                                )
            
        sfs.fit(X, y)
        sel_col_idx = sfs.get_support()
        selected_features = [x for (x,y) in zip(feature_cols, sel_col_idx) if y]
        print(base_est_name)
        print(selected_features)
        X_sel = sfs.transform(X)
        X_pred_sel = sfs.transform(X_pred)
        base_est.fit(X_sel, y)
        tmp = scaler.inverse_transform(base_est.predict(X_pred_sel).reshape(-1,1)).reshape(-1,)
        pitch_cv_predictions[f"{base_est_name} predicted velocities"] = sorted([(x,np.mean(tmp[pitcher_ids == x])) for x in np.unique(pitcher_ids)], key = lambda x: x[1], reverse = True)
        tmp = pd.Series([x[1] for x in pitch_cv_predictions[f"{base_est_name} predicted velocities"]], index = [x[0] for x in pitch_cv_predictions[f"{base_est_name} predicted velocities"]])
        df_cv_predictions[f"{base_est_name} predicted velocities"] = tmp
    pickle.dump((df_cv_predictions, pred_cv_base_ests), open(prediction_results_file_name, "wb"))
else:
    (df_cv_predictions, pred_cv_base_ests) = pickle.load(open(prediction_results_file_name, "rb"))

df_cv_predictions.index.name = "pitcher_id"

if streamlit_on:
    st.subheader("2025 FCL Predicted Velocities", divider=True)
    st.dataframe(df_cv_predictions)


