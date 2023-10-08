# CA2

Steps taken for CA2 
#1 Analysis, Creation of Database to work with and Feature Engineering. 
I had gone over both datasets and the feature descriptions to understand what each feature meant. I merged the two datasets based off of the column. ‘Unique-id’ and then went on to work on that merged dataset to form the following features: 
1. Avg_Time_Per_Action: Average time taken per action can be indicative of a student's engagement level. Students who are off task may exhibit longer average times between actions as they might be distracted or disengaged.
2. Help_Ratio_rolling: The ratio of help requests to total actions, especially in a rolling time window, can provide insights into a student's behavior. Higher help request ratios may indicate that a student is struggling or frequently seeking assistance, which could be a sign of being off task.
3. Time_Diff: Sequential time differences between actions can reveal patterns in a student's behavior. Sudden long time differences between actions may indicate breaks or distractions, which are potential signs of being off task.
4. Percentage_Correct:A low percentage of correct actions may suggest that a student is making many mistakes or providing incorrect responses, which could be related to being off task or not fully engaged.
5. Cumulative_Knowledge_Change: Cumulative knowledge change reflects the overall learning progress. Sudden changes or lack of progress may be indicative of off-task behavior, as engaged students are expected to show continuous improvement.
6. Total_Actions:Total actions can provide context regarding a student's level of interaction with the educational content. Extremely low or high total actions might indicate off-task behavior, either due to disengagement or excessive interaction not related to the learning content.
7. Total_Correct_Actions:  The total number of correct actions can indicate a student's level of mastery of the material. A lack of correct actions may suggest being off task or experiencing difficulties with the content.
8. Help_Request_Frequency: Frequent help requests could signify that a student is facing challenges or needing assistance, which could be associated with being off task.
9. Help_Request_Ratio: Similar to help request frequency, the ratio of help requests to total actions can indicate how often a student seeks assistance relative to their overall activity, providing insights into their engagement and potential off-task behavior.
10. Avg_Time_Between_Actions:The average time between actions can reveal the pacing of a student's interactions. Longer average times might suggest periods of inactivity or distraction.
#2 Adding new features to old dataset 
Once the features were set and were added to the merged dataset that I had been working on, it was time for me to add these new features to the old dataset. I looked at the columns that I was using from the new dataset to build the new features. I took the 10 feature columns and the 3 other columns that I needed from the new dataset for the features and added that to the old dataset to create a new csv folder. (This was basically the updated version of the ca1-dataset with the same number of rows) 

#3 Creating a function that evaluated the new dataset using the three classifier models to see what produces the best kappa, f1, auc-roc and accuracy score. 

I created a function called classifier_competition that looked like this: 

def classifier_competition(data, classifier):
    # Convert 'OffTask' to binary labels (1 for 'Y', 0 for 'N')
    data['OffTask'] = data['OffTask'].replace({'Y': 1, 'N': 0})
    # Define your features and labels
    X = data.drop(columns=['OffTask', 'Unique-id', 'namea'], axis=1)
    y = data['OffTask']
    # Initialize a 10 Group K-Fold cross-validator
    gkf = GroupKFold(n_splits=10)
    # Initialize lists to store evaluation metrics
    kappa_values = []
    f1_values = []
    roc_auc_values = []
    acc_values = []
    # Standardize features (optional, but can help with some algorithms)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Initialize your classifier
    clf = classifier
    
# Performing Group K-Fold Cross-Validation
    for train_idx, test_idx in gkf.split(X, y, groups=data['namea']):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # CheckING if both classes are present in the current split
        if len(np.unique(y_test)) == 1:
            continue  # Skip this split
        # TrainING classifier
        clf.fit(X_train, y_train)
        # Make predictions
        y_pred = clf.predict(X_test)
        # Calculate evaluation metrics
        kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # Handle ROC AUC calculation when both classes are present
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        else:
            roc_auc = np.nan
        acc = accuracy_score(y_test, y_pred)
        # Append metrics to lists
        kappa_values.append(kappa)
        f1_values.append(f1)
        roc_auc_values.append(roc_auc)
        acc_values.append(acc)
    # Calculate mean metrics
    mean_kappa = round(np.nanmean(kappa_values),2)
    mean_f1 = round(np.mean(f1_values),2)
    mean_roc_auc = round(np.nanmean(roc_auc_values),2)
    mean_acc = round(np.mean(acc_values),2)
    # Print the mean metrics
    print(f'Mean Kappa: {mean_kappa}')
    print(f'Mean F1 Score: {mean_f1}')
    print(f'Mean ROC AUC: {mean_roc_auc}')
    print(f'Mean Accuracy: {mean_acc}')


It took in the dataset and the classifier as arguments and did the following: 
It separates the dataset into features (X) and the target variable (y), excluding the 'OffTask' column as well as 'Unique-id' and 'namea' columns. 
The function initializes a 10-fold Group K-Fold cross-validator (gkf) to perform cross-validation while considering groups (in this case, groups of students with the same 'namea').
It standardizes the features using StandardScaler.
The function enters a loop where it performs 10-fold cross-validation.
#4 Feeding the dataset into the model to get information on which classifier gives the best scores
Here are the outputs from each classifier: 
Gaussian Naive Bayes (GNB) Classifier:
Mean Kappa: 0.14
Mean F1 Score: 0.17
Mean ROC AUC: 0.83
Mean Accuracy: 0.84
RandomForestClassifier:
Mean Kappa: 0.22
Mean F1 Score: 0.22
Mean ROC AUC: 0.89
Mean Accuracy: 0.98
XGBClassifier: 
Mean Kappa: 0.27
Mean F1 Score: 0.28
Mean ROC AUC: 0.91
Mean Accuracy: 0.98
Insights:
Accuracy: Both RandomForestClassifier and XGBClassifier have high mean accuracy scores of around 0.98, indicating that they correctly classify the majority of instances.
Kappa Score: Kappa is a measure of agreement beyond chance. XGBClassifier has the highest mean Kappa score (0.27), followed by RandomForestClassifier (0.22), while GNB has the lowest (0.14). This suggests that XGBClassifier and RandomForestClassifier have better agreement with the ground truth than GNB.
F1 Score: The F1 Score is a balance between precision and recall. XGBClassifier has the highest mean F1 score (0.28), followed by RandomForestClassifier (0.22), and GNB has the lowest (0.17). This indicates that XGBClassifier has a better balance between precision and recall.
ROC AUC: ROC AUC measures the area under the receiver operating characteristic curve. XGBClassifier has the highest mean ROC AUC (0.91), indicating that it has a better ability to distinguish between classes. RandomForestClassifier also performs well with a mean ROC AUC of 0.89. 
Overall, based on these metrics:
XGBClassifier performs the best, having the highest Kappa, F1 Score, ROC AUC, and Accuracy among the three classifiers.
RandomForestClassifier also performs well, especially in terms of Accuracy and ROC AUC.
Gaussian Naive Bayes (GNB) Classifier has the lowest scores in all metrics, indicating that it might not be the best choice for this particular classification task
This code prevents overfitting by doing the following: 
Cross-Validation: The code uses k-fold cross-validation (specifically, Group K-Fold cross-validation) to split the dataset into multiple training and testing sets. This helps in assessing the model's performance on different subsets of the data, reducing the risk of overfitting to a single split.
Group-Based Splits: It considers groups of students (defined by the 'namea' column) when splitting the data. This is especially important if there are multiple records for the same students. By ensuring that students from the same group are not present in both the training and testing sets, it helps prevent the model from overfitting to specific students or groups.
Checking for Both Classes: Before evaluating the model's performance for each fold, the code checks if both classes ('Y' and 'N') are present in the testing set. If only one class is present, it skips the fold. This step prevents the model from being evaluated on imbalanced or incomplete data, which could lead to inaccurate performance metrics.
Standardization: The code optionally standardizes the features using `StandardScaler`. Standardization can help prevent overfitting by scaling features to have a mean of 0 and a standard deviation of 1. This ensures that features with different scales do not dominate the learning process.
Mean Metrics: After evaluating the model's performance across all folds, the code calculates and reports the mean evaluation metrics (e.g., mean Kappa, mean F1 Score). This provides a more stable estimate of the model's performance across different subsets of the data, reducing the impact of variability that can lead to overfitting.
Used Early Stopping
Upon research, I found out that early stopping is a regularization technique used during the training of machine learning models, including gradient-boosting-based models like XGBoost. Its primary purpose is to prevent overfitting and improve the generalization ability of the model. So I used it to implement XGBoost. 

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score, accuracy_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

data = pd.read_csv('ca1_df_newfeatures.csv')
data['OffTask'] = data['OffTask'].replace({'Y': 1, 'N': 0})

# Define features and labels
X = data.drop(columns=['OffTask', 'Unique-id', 'namea'], axis=1)
y = data['OffTask']

# Apply oversampling to address class imbalance
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Standardize features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Initialize GroupKFold cross-validator
gkf = GroupKFold(n_splits=10)

# Initialize lists to store evaluation metrics
kappa_values = []
f1_values = []
roc_auc_values = []
acc_values = []

# Initialize your XGBoost classifier with early stopping
clf = XGBClassifier(n_estimators=1000, eval_metric="logloss", verbose=False, early_stopping_rounds=10)

# Get the 'namea' values after oversampling
namea_resampled = data['namea'].iloc[oversampler.sample_indices_]

# Perform GroupKFold Cross-Validation
for train_idx, test_idx in gkf.split(X_resampled, y_resampled, groups=namea_resampled):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

    # Check if both classes are present in the current split
    if len(np.unique(y_test)) == 1:
        continue  # Skip this split

    # Split data into training, validation, and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.7, random_state=42)

    # Train classifier with early stopping using validation dataset
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate evaluation metrics
    kappa = cohen_kappa_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    acc = accuracy_score(y_test, y_pred)

    # Append metrics to lists
    kappa_values.append(kappa)
    f1_values.append(f1)
    roc_auc_values.append(roc_auc)
    acc_values.append(acc)

# Calculate mean metrics
mean_kappa = round(np.nanmean(kappa_values), 2)
mean_f1 = round(np.mean(f1_values), 2)
mean_roc_auc = round(np.nanmean(roc_auc_values), 2)
mean_acc = round(np.mean(acc_values), 2)

# Print the mean metrics
print(f'Mean Kappa: {mean_kappa}')
print(f'Mean F1 Score: {mean_f1}')
print(f'Mean ROC AUC: {mean_roc_auc}')
print(f'Mean Accuracy: {mean_acc}')




This is my output: 
Mean Kappa: 0.47
Mean F1 Score: 0.64
Mean ROC AUC: 0.88
Mean Accuracy: 0.74
This seems better but I’m not exactly sure if the approach I am taking is right. 

By using these techniques, the code aims to ensure that the evaluation of each classifier is not overly influenced by specific data splits or individual students, helping to prevent overfitting and providing a more reliable assessment of each model's generalisation performance.

