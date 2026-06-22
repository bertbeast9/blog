
import pandas as pd
import numpy as np
import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing, impute, linear_model, metrics, feature_selection, model_selection, ensemble, neighbors, gaussian_process
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import scipy
from sklearn import set_config
set_config(enable_metadata_routing=True)

streamlit_on = False
sfs_tol = 0.01
k_folds = 5
gpr_data_pts = 50


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

# def split_data_by_pitch(df, feature_cols, target_col = "velocity", k_folds = 5):
#     X_train, X_test, y_train, y_test = None, None, None, None
#     pitch_ids = np.arange(0,df.shape[0])
#     df["sample_weights"] = df["pitcher_id"].apply(lambda x: 1.0)
#     np.random.shuffle(pitch_ids)
#     stepsize = int(len(pitch_ids) / k_folds)
#     folds_data = []
#     for k in range(k_folds):
#         pitch_ids = np.roll(pitch_ids, stepsize)
#         train_pitch_ids = pitch_ids[:stepsize * (k_folds - 1)]
#         test_pitch_ids = pitch_ids[stepsize * (k_folds - 1):]
#         X_train = df.iloc[train_pitch_ids][feature_cols].values
#         X_test = df.iloc[test_pitch_ids][feature_cols].values
#         y_train = df.iloc[train_pitch_ids][target_col].values
#         y_test = df.iloc[test_pitch_ids][target_col].values
#         train_weights = df.loc[train_pitch_ids]["sample_weights"].values
#         test_weights = df.loc[test_pitch_ids]["sample_weights"].values
#         # # standardize target
#         # mu_y = np.mean(y_train)
#         # std_y = np.std(y_train)
#         # y_train = (y_train - mu_y) / std_y
#         # y_test = (y_test - mu_y) / std_y
#         folds_data.append(((X_train, y_train, train_weights), (X_test, y_test, test_weights)))
#     return folds_data

def split_data_by_pitcher_avg_pitch(df, feature_cols, target_col = "velocity", k_folds = 5):
    X_train, X_test, y_train, y_test = None, None, None, None
    pitch_ids = df["pitcher_id"].value_counts().index.to_numpy()
    # fig_ = plt.figure(1000)
    # plt.hist(pitch_ids,100)
    # plt.show()
    # breakpoint()
    df = df[["pitcher_id"] + [target_col] + feature_cols].groupby("pitcher_id").agg("mean")
    pitch_ids = df.index.to_numpy()
    df["sample_weights"] = df["pitcher_id"].apply(lambda x: 1)
    stepsize = int(len(pitch_ids) / k_folds)
    folds_data = []
    for k in range(k_folds):
        pitch_ids = np.roll(pitch_ids, stepsize)
        train_pitch_ids = pitch_ids[:stepsize * (k_folds - 1)]
        test_pitch_ids = pitch_ids[stepsize * (k_folds - 1):]
        X_train = df.loc[train_pitch_ids][feature_cols].values
        X_test = df.loc[test_pitch_ids][feature_cols].values
        y_train = df.loc[train_pitch_ids][target_col].values
        y_test = df.loc[test_pitch_ids][target_col].values
        train_weights = df.loc[train_pitch_ids]["sample_weights"].values
        test_weights = df.loc[test_pitch_ids]["sample_weights"].values
        # # standardize target
        # mu_y = np.mean(y_train)
        # std_y = np.std(y_train)
        # y_train = (y_train - mu_y) / std_y
        # y_test = (y_test - mu_y) / std_y
        folds_data.append(((X_train, y_train, train_weights), (X_test, y_test, test_weights)))
    return folds_data


def split_data_by_pitcher_weighted_pitch(df, feature_cols, target_col = "velocity", k_folds = 5):
    X_train, X_test, y_train, y_test = None, None, None, None
    pitch_ids = df["pitcher_id"].value_counts().index.to_numpy()
    pitch_counts = df["pitcher_id"].value_counts()
    df["sample_weights"] = df["pitcher_id"].apply(lambda x: 1 / pitch_counts[x])
    # df["sample_weights"] = df["pitcher_id"].apply(lambda x: pitch_counts[x])
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
        train_weights = df[df["pitcher_id"].apply(lambda x: True if x in train_pitch_ids else False)]["sample_weights"].values
        test_weights = df[df["pitcher_id"].apply(lambda x: True if x in test_pitch_ids else False)]["sample_weights"].values
        # # standardize target
        # mu_y = np.mean(y_train)
        # std_y = np.std(y_train)
        # y_train = (y_train - mu_y) / std_y
        # y_test = (y_test - mu_y) / std_y
        folds_data.append(((X_train, y_train, train_weights), (X_test, y_test, test_weights)))
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


train_df = pd.read_csv("./src/data/Phillies_AB_project/velo_and_mechanics.csv")
train_df["date"] = train_df["date"].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%y"))
# train_df = train_df[train_df["pitcher_handedness"] == "l"]
print(train_df["date"])
train_df.sort_values(by=["date","pitcher_id"], inplace = True)
feature_cols = [x for x in train_df.columns if x not in ["date","pitcher_id","velocity"]]
# for feature_col in feature_cols:
#     plt.scatter(train_df[feature_col], train_df["velocity"])
#     plt.title(feature_col)
#     plt.show()
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
    st.subheader("Player Cross Validation", divider=True)
    st.markdown("For this cross validation, we will separate the data by pitcher such that the testing data is brand new to the trained model (i.e. Pitcher 3 does not " \
    "have a pitch in the _training_ dataset, only in the _testing_ dataset). I believe this is a more realistic test for the model since this is what will happen in real-life with " \
    "a new player that needs to be evaluated.")
    
    st.subheader("Model Pipeline", divider=True)
    st.markdown("For each fold, we will fit the pipeline to the training dataset and then test this fit pipeline to the testing dataset (depending on the cross validation type). Each model pipeline will " \
    "start with data imputation for the missing NaNs in the _training_ and _testing_ datasets. This will be achieved via SciKit Learn's IterativeImputer function (https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html). " \
    "Then, the input data will be normalized via the RobustScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) AKA IQR. Finally, we will perform feature selection using " \
    f"the SequentialFeatureSelector (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) with a tolerance of {sfs_tol} change in the estimator's score ($R^2$). " \
    "We will use a few different model's for the base estimator: OLS, Ridge, LASSO, and Gaussian Process Regression (GPR).")



feature_cols.pop(feature_cols.index("pitcher_handedness"))
X = train_df[feature_cols].values
y = train_df["velocity"].values

pitcher_ids = train_df["pitcher_id"].values
### Pitcher Cross Validation
pitcher_cv_results = {}
# cv_obj = split_data_by_pitcher_id(pitcher_ids, k_folds = k_folds)
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
    base_ests = [
        # ("OLS", linear_model.LinearRegression(fit_intercept=True).set_score_request(sample_weight=True).set_fit_request(sample_weight=True)),
        # ("Ridge", linear_model.RidgeCV(alphas = np.logspace(-3,3,10)).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)),
        # ("LASSO", linear_model.LassoCV(alphas = np.logspace(-3,3,10)).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)),
        ("RF", ensemble.RandomForestRegressor(n_estimators=10)),
        # ("GPR",gaussian_process.GaussianProcessRegressor(kernel = gaussian_process.kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)), n_restarts_optimizer=1)),#, normalize_y=True
    ]
    fold_results =  []
    for base_est_name, base_est in base_ests:
        if base_est_name == "GPR":
            sfs = feature_selection.SequentialFeatureSelector(
                                                    # estimator=linear_model.LinearRegression(fit_intercept=True).set_score_request(sample_weight=True).set_fit_request(sample_weight=True),
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
            print(selected_features)
            X_train = sfs.transform(X_train)
            X_test = sfs.transform(X_test)
        else:
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
            print(selected_features)
            X_train = sfs.transform(X_train)
            X_test = sfs.transform(X_test)
        if base_est_name == "GPR":
            # breakpoint()
            # need to select data points
            data_pt_idx = []
            pot_data_pt_idx = np.arange(y_train.shape[0])
            for nth_data_pt in range(gpr_data_pts):
                if len(data_pt_idx) > 0:
                    base_est.fit(X_train[data_pt_idx,:], y_train[data_pt_idx])
                y_train_pred, y_train_pred_std = base_est.predict(X_train, return_std = True)
                # max variance sampling
                new_idx = -1
                while new_idx == -1:
                    new_idx = np.argmax(y_train_pred_std)
                    # print(new_idx)
                    if new_idx not in data_pt_idx:
                        data_pt_idx.append(new_idx)
                    else:
                        y_train_pred_std = np.delete(y_train_pred_std, new_idx)
                        new_idx = -1
                    
                # plt.scatter(X_train[data_pt_idx,0],y_train[data_pt_idx])
                # plt.show()
                # plt.plot(y_train)
                # plt.plot(y_train_pred)
                # plt.show()
                # print(np.max(y_train_pred_std))
                # print(data_pt_idx)

                
        else:
            base_est.fit(X_train, y_train)
        if kth_fold == 0:
            pitcher_cv_results[base_est_name] = dict()
            pitcher_cv_results[base_est_name]["features"] = []
            pitcher_cv_results[base_est_name]["training"] = []
            pitcher_cv_results[base_est_name]["testing"] = []
        pitcher_cv_results[base_est_name]["features"].append(selected_features)
        pitcher_cv_results[base_est_name]["training"].append(100 * base_est.score(X_train, y_train))
        pitcher_cv_results[base_est_name]["testing"].append(100 * base_est.score(X_test, y_test))
        print(pitcher_cv_results)
    
for base_est_name, base_est in base_ests:
    print(f"{base_est_name}\nTraining $R^2$: {np.mean(pitcher_cv_results[base_est_name]["training"])}% (+/-) {np.std(pitcher_cv_results[base_est_name]["training"])}" \
            f"\nTesting $R^2$: {np.mean(pitcher_cv_results[base_est_name]["testing"])}% (+/-) {np.std(pitcher_cv_results[base_est_name]["testing"])}")




# # breakpoint()

# cv_obj = split_data_by_pitcher_id(pitcher_ids, k_folds = k_folds)
# for kth_fold, (train_data, test_data) in enumerate(cv_obj):
#     # scaler = preprocessing.StandardScaler().set_fit_request(sample_weight=True)
#     scaler = preprocessing.RobustScaler()
#     base_est = linear_model.LinearRegression(fit_intercept=True).set_score_request(sample_weight=True).set_fit_request(sample_weight=True)
#     # base_est = linear_model.HuberRegressor(fit_intercept=True).set_score_request(sample_weight=True).set_fit_request(sample_weight=True)
#     # base_est = linear_model.TheilSenRegressor(fit_intercept=True, n_jobs=-1, verbose=True)#.set_score_request(sample_weight=True)
#     # base_est = linear_model.RidgeCV(alphas = np.logspace(-1,1,10)).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)
#     # base_est = linear_model.RANSACRegressor(linear_model.LinearRegression(fit_intercept=True).set_score_request(sample_weight=True).set_fit_request(sample_weight=True))
#     # base_est = ensemble.RandomForestRegressor(n_jobs=-1, n_estimators=10).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)
#     # param_grid = {"n_neighbors":[10,50,100], "weights":["uniform","distance"]};base_est = model_selection.GridSearchCV(estimator=neighbors.KNeighborsRegressor(), param_grid=param_grid)
#     pipeline = Pipeline([
#                     #  ("imputer", impute.IterativeImputer()),
#                     #  ("scaler", scaler),
#                      ("feature_selector", feature_selection.SequentialFeatureSelector(
#                                                 estimator=base_est,
#                                                 # n_features_to_select=1,
#                                                 tol=0.01,
#                                                 direction="forward",
#                                                 cv=split_data_by_pitcher_id(train_data[1], indices_only=True),
#                                                 n_jobs=1,
#                                             )),
#                      ("estimator", base_est),
#                      ])
#     # X_test = pipeline.fit(X[train_data[0],:], y[train_data[0]]).transform(X[test_data[0],:])
#     # for i in range(X_test.shape[1]):
#     #     plt.hist(X_test[:,i],100)
#     #     plt.title(f"{i}th selected feature")
#     #     plt.show()
#     values, counts = np.unique(train_data[1], return_counts=True)
#     value_count_dict = dict(zip(values,counts))
#     train_sample_weights = np.array([1 / value_count_dict[x] for x in train_data[1]])
#     # train_sample_weights = np.array([np.log(value_count_dict[x]) for x in train_data[1]])
#     # train_sample_weights = np.array([1.0 for x in train_data[1]])
#     values, counts = np.unique(test_data[1], return_counts=True)
#     value_count_dict = dict(zip(values,counts))
#     test_sample_weights = np.array([1 / value_count_dict[x] for x in test_data[1]])
#     # test_sample_weights = np.array([np.log(value_count_dict[x]) for x in test_data[1]])
#     # test_sample_weights = np.array([1.0 for x in test_data[1]])
#     pipeline.fit(X[train_data[0],:], y[train_data[0]], sample_weight = train_sample_weights)#
#     sel_col_idx = pipeline.named_steps["feature_selector"].get_support()
#     selected_features = [x for (x,y) in zip(feature_cols, sel_col_idx) if y]
#     print(selected_features)
#     train_fold_score = pipeline.score(X[train_data[0],:], y[train_data[0]], sample_weight=train_sample_weights)#
#     test_fold_score = pipeline.score(X[test_data[0],:], y[test_data[0]], sample_weight=test_sample_weights)#
#     print(f"{kth_fold}-th train fold score: {train_fold_score * 100:0.2f}\n"
#           f"{kth_fold}-th test fold score: {test_fold_score * 100:0.2f}")
#     y_test_pred = pipeline.predict(X[test_data[0],:])
#     # plt.plot(y[test_data[0]])
#     # plt.plot(y_test_pred)
#     # plt.show()
#     train_r2_scores.append(100 * train_fold_score)
#     test_r2_scores.append(100 * test_fold_score)
# print(f"mean train r2 score: {np.mean(train_r2_scores):0.2f} (+/-) {np.std(train_r2_scores):0.2f}")
# print(f"mean test r2 score: {np.mean(test_r2_scores):0.2f} (+/-) {np.std(test_r2_scores):0.2f}")
# # cv_scores = model_selection.cross_val_score(
# #     estimator=pipeline,
# #     X=X,
# #     y=y,
# #     cv = cv_obj,
# #     scoring = "r2",
# #     n_jobs = -1
# # )
# # print(cv_scores)
# # breakpoint()
# # # impute data
# # imputer = impute.IterativeImputer()
# # imputer.fit(X)
# # X = imputer.transform(X)


# # # base_est = linear_model.LinearRegression(fit_intercept=True).set_score_request(sample_weight=True)
# # # base_est.set_fit_request(sample_weight=True)
# # base_est = linear_model.LinearRegression(fit_intercept=True)
# # sfs = feature_selection.SequentialFeatureSelector(
# #     estimator=base_est,
# #     n_features_to_select=1,
# #     direction="forward",
# #     cv=cv_obj,
# #     scoring="r2",
# #     n_jobs=-1
# # )
# # sfs.fit(X, y)
# # X = sfs.transform(X)
# # sel_col_idx = sfs.get_support()
# # selected_features = [x for (x,y) in zip(feature_cols, sel_col_idx) if y]
# # print(selected_features)
# # # base_est.fit(X_scaled_train, y_train, sample_weight=train_weights)#
# # base_est.fit(X, y)
# # print(base_est.coef_)
# # print("intercept")
# # print(base_est.intercept_)

# # breakpoint()




# # # folds_data = split_data_by_pitch(train_data, feature_cols, k_folds = 5)
# # # folds_data = split_data_by_pitcher_avg_pitch(train_data, feature_cols, k_folds = 5)
# # folds_data = split_data_by_pitcher_weighted_pitch(train_data, feature_cols, k_folds = 5)
# # for k, ((X_train, y_train, train_weights), (X_test, y_test, test_weights)) in enumerate(folds_data):
# #     # impute data
# #     imputer = impute.IterativeImputer()
# #     imputer.fit(X_train)
# #     X_train = imputer.transform(X_train)
# #     X_test = imputer.transform(X_test)
# #     # # # standardize data
# #     # # scaler = preprocessing.StandardScaler().fit(X_train, sample_weight=train_weights)
# #     # # X_scaled_train = scaler.transform(X_train)
# #     # # X_scaled_test = scaler.transform(X_test)
# #     X_scaled_train = X_train
# #     X_scaled_test = X_test
# #     # X_scaled_train = np.clip(X_scaled_train, -2, 2)
# #     # X_scaled_test = np.clip(X_scaled_test, -2, 2)
# #     # # include intercept term
# #     # X_scaled_train = np.concat([np.ones([X_scaled_train.shape[0],1]), X_scaled_train], axis = 1)
# #     # X_scaled_test = np.concat([np.ones([X_scaled_test.shape[0],1]), X_scaled_test], axis = 1)
# #     # # select features
# #     base_est = linear_model.LinearRegression(fit_intercept=True).set_score_request(sample_weight=True)
# #     base_est.set_fit_request(sample_weight=True)
# #     # base_est = linear_model.HuberRegressor(epsilon = 1.1, fit_intercept=True)
# #     # base_est = linear_model.RANSACRegressor()
# #     # base_est = linear_model.TheilSenRegressor()
# #     sfs = feature_selection.SequentialFeatureSelector(base_est, n_features_to_select=1)#
# #     # sfs = feature_selection.SequentialFeatureSelector(base_est, n_features_to_select=1)
# #     sfs.fit(X_scaled_train, y_train, sample_weight=train_weights)
# #     X_scaled_train = sfs.transform(X_scaled_train)
# #     X_scaled_test = sfs.transform(X_scaled_test)
# #     # breakpoint()
# #     sel_col_idx = sfs.get_support()
# #     selected_features = [x for (x,y) in zip(feature_cols, sel_col_idx) if y]
# #     print(selected_features)
# #     # for idx in range(X_scaled_test.shape[1]):
# #     #     plt.hist(X_scaled_test[:,idx],100)
# #     #     plt.show()
# #     base_est.fit(X_scaled_train, y_train, sample_weight=train_weights)#
# #     print(base_est.coef_)
# #     print("intercept")
# #     print(base_est.intercept_)

# #     fit_plot, ax = plt.subplots(2,1)
# #     # # train data
# #     y_train_pred = base_est.predict(X_scaled_train)
# #     train_r2_score = base_est.score(X_scaled_train, y_train, sample_weight=train_weights) * 100
# #     # ax[0].scatter(y_train, y_train_pred)
# #     ax[0].plot(y_train)
# #     ax[0].plot(y_train_pred)
# #     ax[0].set_xlabel("true velocity [mph]")
# #     ax[0].set_ylabel("predicted velocity [mph]")
# #     ax[0].set_title(f"Training data (fold {k}): LASSO model prediction $R^2$: {train_r2_score: 0.2f}")
# #     # test data
# #     y_test_pred = base_est.predict(X_scaled_test)
# #     test_r2_score = base_est.score(X_scaled_test, y_test, sample_weight=test_weights) * 100
# #     # ax[1].scatter(y_test, y_test_pred)
# #     ax[1].plot(y_test)
# #     ax[1].plot(y_test_pred)
# #     ax[1].set_xlabel("true velocity [mph]")
# #     ax[1].set_ylabel("predicted velocity [mph]")
# #     ax[1].set_title(f"Testing data (fold {k}): LASSO model prediction $R^2$: {test_r2_score: 0.2f}")
# #     if streamlit_on:
# #         st.pyplot(fig = fit_plot)
# #     else:
# #         plt.show()
    

        

