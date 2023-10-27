import pandas as pd
from pandas.core.dtypes.common import is_string_dtype
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, skew, shapiro
from sklearn import mixture
import math
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold  # cross-validating against overfitting
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from xgboost import plot_importance
from matplotlib import pyplot

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("heart_2020_cleaned.csv")


def get_q1_q3(df, feature_name):
    q1 = df[feature_name].quantile(0.25)
    q3 = df[feature_name].quantile(0.75)
    return q1, q3


def get_outliers(df, feature_name, tail='both'):
    q1, q3 = get_q1_q3(df, feature_name)
    iqr = q3 - q1
    if tail == 'both':
        outliers = df[(df[feature_name] <= q1 - 1.5 * iqr) | (df[feature_name] >= q3 + 1.5 * iqr)]
    elif tail == 'left':
        outliers = df[(df[feature_name] <= q1 - 1.5 * iqr)]
    elif tail == 'right':
        outliers = df[(df[feature_name] >= q3 + 1.5 * iqr)]
    else:
        raise ValueError('tail must be "both", "left" or "right"')
    return outliers


def fit_gmm(x, num_components):
    gmm = mixture.GaussianMixture(n_components=num_components, covariance_type='full', random_state=0)
    gmm.fit(x)
    return gmm


def calc_AIC_BIC(models, df, feature_name):
    bics = []
    aics = []
    for curModel in models:
        X = df[feature_name].values.reshape(-1, 1)
        bics.append(curModel.bic(X))
        aics.append(curModel.aic(X))
    return np.array(bics).argmin(), np.array(aics).argmin()


def f_test(group1, group2):
    # Calculate the sample variances
    variance1 = np.var(group1, ddof=1)
    variance2 = np.var(group2, ddof=1)
    # Calculate the F-statistic
    f_value = variance1 / variance2
    # Calculate the degrees of freedom
    df1 = len(group1) - 1
    df2 = len(group2) - 1
    # Calculate the p-value
    p_value = stats.f.cdf(f_value, df1, df2)
    return p_value


def check_if_duplicates_differ(all_data, duplicate_rows, feature_name, alpha):
    num_samples = 100
    dataset_models = [fit_gmm(all_data[feature_name].values.reshape(-1, 1), 2),
                      fit_gmm(all_data[feature_name].values.reshape(-1, 1), 3),
                      fit_gmm(all_data[feature_name].values.reshape(-1, 1), 4),
                      fit_gmm(all_data[feature_name].values.reshape(-1, 1), 5),
                      fit_gmm(all_data[feature_name].values.reshape(-1, 1), 6),
                      fit_gmm(all_data[feature_name].values.reshape(-1, 1), 7)]
    bic_data_model_id, aic_data_model_id = calc_AIC_BIC(dataset_models, all_data, feature_name)

    duplicate_models = [fit_gmm(duplicate_rows[feature_name].values.reshape(-1, 1), 2),
                        fit_gmm(duplicate_rows[feature_name].values.reshape(-1, 1), 3),
                        fit_gmm(duplicate_rows[feature_name].values.reshape(-1, 1), 4),
                        fit_gmm(duplicate_rows[feature_name].values.reshape(-1, 1), 5),
                        fit_gmm(duplicate_rows[feature_name].values.reshape(-1, 1), 6),
                        fit_gmm(duplicate_rows[feature_name].values.reshape(-1, 1), 7)]
    bic_dup_model_id, aic_dup_model_id = calc_AIC_BIC(duplicate_models, duplicate_rows, feature_name)

    # if there are more components found in the duplicates then we cut off the maximum up to the original components cnt
    if aic_dup_model_id > aic_data_model_id:
        aic_dup_model_id = aic_data_model_id

    statistically_insignificant_differences = 0
    for dataset_mu_id in sorted(range(len(dataset_models[aic_data_model_id].means_)),
                                key=dataset_models[aic_data_model_id].means_.__getitem__):
        for duplicate_mu_id in sorted(range(len(duplicate_models[aic_dup_model_id].means_)),
                                      key=duplicate_models[aic_dup_model_id].means_.__getitem__):
            dataset = np.random.normal(dataset_models[aic_data_model_id].means_[dataset_mu_id],
                                       math.sqrt(dataset_models[aic_data_model_id].covariances_[dataset_mu_id]),
                                       num_samples)
            duplicates = np.random.normal(duplicate_models[aic_dup_model_id].means_[duplicate_mu_id],
                                          math.sqrt(duplicate_models[aic_dup_model_id].covariances_[duplicate_mu_id]),
                                          num_samples)
            res = stats.ttest_ind(duplicates, dataset, equal_var=True)
            print(res)
            if res.pvalue > alpha:  # second check that vars are the same
                p_value_variance = f_test(duplicates, dataset)
                if p_value_variance > alpha:
                    # if both differences are statistically insignificant then we have a match in our components
                    statistically_insignificant_differences = statistically_insignificant_differences + 1

    return False if statistically_insignificant_differences > 0 else True

#=================================================================

# DATASET CLEANUP ==========================================
print("DATASET CLEANUP:")
# 1. Data types
print("1. DATA TYPES:")
print(df.describe())

# 2. Check for missing values
print("2. MISSING VALUES:")
missing_values = df.isnull().sum()
print(missing_values)

# 3. Check for duplicate rows
print("3. DUPLICATE ROWS:")
if __name__ == '__main__':
    all_data = pd.read_csv("heart_2020_cleaned.csv")
    # print(all_data.head())
    print(all_data.info())
    print(all_data.count())

    # first we check for normality per each of the numerical feature using Shapiro-Wilk test
    print(shapiro(all_data["BMI"]))
    print(shapiro(all_data["SleepTime"]))
    print(shapiro(all_data["PhysicalHealth"]))
    print(shapiro(all_data["MentalHealth"]))

    print(skew(all_data["BMI"]))

    duplicate_rows = all_data[all_data.duplicated()]
    # test if duplicates come from the same distribution as the original one: for that end, we fit a mixture of Gaussians
    # for each numerical feature and identify the number of components utilizing AIC (Akaike information criterion)
    # subsequently we do the same for the duplicates only per feature assuming that they should come from the modes
    # of the original datasets. If the number of fit mixture components is greater for duplicatest than for the original
    # dataset then we simply take the number of components of the original one. Finally, we run t-tests for all of the
    # possible pairs of the components between the original dataset and the duplicates and check if the means are
    # statistically signicantly different. In case of any p_value > alpha for any of the possible pairing we conclude
    # that the duplicates are not statistically different from the original distribution. Moreover, we do the major voting
    # to check if there is a dominant result for all separate numerical features. Both tests without removing and with
    # removing duplicates indicate that there is no statistically significant difference between the duplicates and the
    # original dataset which leads us to keep them as is for the subsequent analysis.
    alpha = 0.05
    plot = False
    bmi_ttest = check_if_duplicates_differ(all_data, duplicate_rows, "BMI", alpha)
    sleep_ttest = check_if_duplicates_differ(all_data, duplicate_rows, "SleepTime", alpha)
    physical_ttest = check_if_duplicates_differ(all_data, duplicate_rows, "PhysicalHealth", alpha)
    mental_ttest = check_if_duplicates_differ(all_data, duplicate_rows, "MentalHealth", alpha)
    # majority vote
    are_duplicates_differ = bool(
        stats.mode(np.array([int(bmi_ttest), int(sleep_ttest), int(physical_ttest), int(mental_ttest)]))[0])
    # test also if there is a statistically significant difference between the means of the components of the duplicates
    # and the original dataset without duplicates (i.e., with removed duplicates from the original data)
    bmi_ttest = check_if_duplicates_differ(all_data.drop_duplicates(), duplicate_rows, "BMI", alpha)
    sleep_ttest = check_if_duplicates_differ(all_data.drop_duplicates(), duplicate_rows, "SleepTime", alpha)
    physical_ttest = check_if_duplicates_differ(all_data.drop_duplicates(), duplicate_rows, "PhysicalHealth", alpha)
    mental_ttest = check_if_duplicates_differ(all_data.drop_duplicates(), duplicate_rows, "MentalHealth", alpha)
    # majority vote
    are_duplicates_differ = bool(
        stats.mode(np.array([int(bmi_ttest), int(sleep_ttest), int(physical_ttest), int(mental_ttest)]))[0])

# 4. Detect outliers
print("4. OUTLIERS:")
# visualize outliers as boxplot:
df.plot.box(column=["BMI"])
df.plot.box(column=["SleepTime"])
df.plot.box(column=["PhysicalHealth", "MentalHealth"])
plt.show()
outliers_BMI_mean = get_outliers(df, 'BMI', tail='right')['BMI'].mean()
outliers_SleepTime_mean = get_outliers(df, 'SleepTime', tail='right')['SleepTime'].mean()
outliers_PhysicalHealth_mean = get_outliers(df, 'PhysicalHealth', tail='right')['PhysicalHealth'].mean()
outliers_MentalHealth_mean = get_outliers(df, 'MentalHealth', tail='right')['MentalHealth'].mean()
print(df.loc[df['BMI'] > outliers_BMI_mean])
print(df.loc[df['SleepTime'] > outliers_SleepTime_mean])
print(df.loc[df['PhysicalHealth'] > outliers_PhysicalHealth_mean])
print(df.loc[df['MentalHealth'] > outliers_MentalHealth_mean])

# 5. Check for inconsistent categorical values & other issues:
print("5. OTHER ISSUES:")
for col in df.columns:
    if is_string_dtype(df[col]):
        print(col)
        print(df[col].value_counts())
        print(df[col].unique())
        print(df[col].nunique())
        print(df[col].dtype)
        print(df[col].isnull().sum())
        print(df[col].isna().sum())

# CORRELATIONS ==========================================
print("CORRELATIONS:")
# 1. Correlations for numerical features:
print("1. CORRELATIONS FOR NUMERICAL FEATURES:")
correlations = df.corr(method='pearson', min_periods=1, numeric_only=True)
print(correlations)
# plot correlation matrix:
plt.figure(figsize=(12, 8))
sn.heatmap(correlations, annot=True)
plt.subplots_adjust(left=0.25)
plt.show()

# 2. Correlations for categorical features:
print("2. CORRELATIONS FOR CATEGORICAL FEATURES:")
# calculate chi-square for all categorical features:
categorical_data = df[
    ["HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic",
     "PhysicalActivity", "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]]
factors_paired = []
p_values = []
chi2 = []

for i in range(len(categorical_data.columns.values) - 1):
    for j in range(i + 1, len(categorical_data.columns.values)):
        factors_paired.append((categorical_data.columns.values[i], categorical_data.columns.values[j]))
print(factors_paired)

for f in factors_paired:
    chitest = chi2_contingency(pd.crosstab(df[f[0]], df[f[1]]))
    chi2.append(chitest[0])
    p_values.append(chitest[1])
print("Chi-square:")
print(chi2)
print("P-values:")
print(p_values)

for i in range(len(factors_paired)):
    if p_values[i] < 0.05:
        print(factors_paired[i])
        print(p_values[i])
        print(chi2[i])

# Select the columns of interest from the DataFrame
categorical_data = df[
    ["HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic",
     "PhysicalActivity", "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]]

# Initialize empty DataFrames to store p-values and chi-square statistics
p_values_df = pd.DataFrame(columns=categorical_data.columns, index=categorical_data.columns)

# Calculate chi-square p-values for pairs of categorical variables
for col1 in categorical_data.columns:
    for col2 in categorical_data.columns:
        if col1 != col2:
            crosstab = pd.crosstab(df[col1], df[col2])
            chi2, p, _, _ = chi2_contingency(crosstab)
            p_values_df.loc[col1, col2] = p

# Create a heatmap of p-values
plt.figure(figsize=(12, 8))
sn.heatmap(p_values_df.astype(float), annot=True, fmt=".4f", cmap="coolwarm")
plt.subplots_adjust(bottom=0.35)
plt.title("Chi-Square Test P-Values Heatmap")
plt.savefig("chi_square_p_values_heatmap.png")  # Save the heatmap as an image file
plt.show()

# =========================================================

# DATA MODELLING ==========================================
print("DATA MODELLING:")
if __name__ == '__main__':
    all_data = pd.read_csv("heart_2020_cleaned.csv")
    print(all_data.info())
    print(all_data.count())
    X = all_data[["HeartDisease", "Smoking"]]
    all_data["HeartDisease"] = all_data["HeartDisease"].str.lower().map({"yes": 1, "no": 0})
    all_data["AgeCategory"] = all_data["AgeCategory"].str.lower().map({"18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3, "40-44": 4, "45-49": 5, "50-54": 6, "55-59": 7, "60-64": 8, "65-69": 9, "70-74": 10, "75-79": 11, "80 or older": 12})

    all_data_healthy_undersampled = all_data[all_data["HeartDisease"] == 0]
    all_data_unhealthy = all_data[all_data["HeartDisease"] == 1]
    all_data = pd.concat([all_data_healthy_undersampled, all_data_unhealthy], ignore_index=False, axis=0)
    print(all_data.info())
    numerical_data = all_data[["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]]
    # numerical_data = (numerical_data - numerical_data.mean()) / numerical_data.std()
    categorical_data = all_data[["HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "Sex", "Race",
         "Diabetic", "PhysicalActivity", "GenHealth", "Asthma", "KidneyDisease", "SkinCancer", "AgeCategory"]]

    all_data = pd.concat([numerical_data, categorical_data], axis="columns")

    all_data.loc[all_data["Diabetic"] == "No, borderline diabetes", "Diabetic"] = "Borderline"
    all_data.loc[all_data["Diabetic"] == "Yes (during pregnancy)", "Diabetic"] = "PregnancyDiabetes"
    all_data.loc[all_data["Race"] == "American Indian/Alaskan Native", "Race"] = "AmericanNative"
    all_data.loc[all_data["GenHealth"] == "Very good", "GenHealth"] = "VeryGood"
    BMI_outliers = get_outliers(all_data, "BMI")

    print(all_data.info())

    all_data = pd.get_dummies(all_data)

    X = all_data.drop(["HeartDisease"], axis = 1)

    print(all_data.info())

    y = all_data['HeartDisease']
# oversampling
    oversample = SMOTE()
    X_smote, y_smote = oversample.fit_resample(X, y)

# split dataset into test & train (otherwise model overfitting)
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.33, random_state=42)

    X_train_unbalanced, X_test_unbalanced, y_train_unbalanced, y_test_unbalanced = train_test_split(X, y, test_size=0.33, random_state=42)
    xgbc = XGBClassifier()
    xgbc.fit(X_train, y_train)
    scores = cross_val_score(xgbc, X_train, y_train, cv=5)
    print("Mean cross-validation score: %.2f" % scores.mean())
    kfold = KFold(n_splits=10, shuffle=True)
    kf_cv_scores = cross_val_score(xgbc, X_train, y_train, cv=kfold)
    print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

    plot_importance(xgbc)
    pyplot.show()

    ypred = xgbc.predict(X_test)
    cm = confusion_matrix(y_test, ypred)
    print(cm)
    print(classification_report(y_test, ypred))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Sick"])
    disp.plot()
    plt.show()
    plt.figure()

    y_pred_proba = xgbc.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    # create ROC curve
    plt.plot(fpr, tpr, label="AUC=" + str(round(auc, 2)))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
    plt.figure()

# sensitivity, specificity, accuracy
    sensitivity1 = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Specificity : ', specificity1)

    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    print('Accuracy : ', accuracy )

# regularization
    logreg = LogisticRegression(solver='liblinear', penalty='l1')
    logreg.fit(X_train, y_train)

    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.axhline(y=-0.5, color='r', linestyle='--')
    for i in range(len(logreg.coef_.flatten())):
        if i % 2 == 0:
            plt.axvline(x=i, color='b', linestyle='--')
        else:
            plt.axvline(x=i, color='g', linestyle='--')
    my_xticks = X_train.columns
    plt.xticks(range(len(logreg.coef_.flatten())), my_xticks, rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.plot(logreg.coef_.flatten())

    print(logreg.coef_)
    print(X_train.columns)

    prediction = logreg.predict(X_test)

    cm = confusion_matrix(y_test, prediction)
    print(cm)
    print(classification_report(y_test, prediction))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
    disp.plot()
    plt.show()
    plt.figure()

    y_pred_proba = logreg.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    # create ROC curve
    plt.plot(fpr, tpr, label="AUC=" + str(round(auc, 2)))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
    plt.figure()

# sensitivity, specificity, accuracy
    sensitivity1 = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Specificity : ', specificity1)

    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    print('Accuracy : ', accuracy )
