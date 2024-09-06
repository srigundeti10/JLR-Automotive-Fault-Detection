import os
import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from matplotlib import  pyplot as plt
from scipy.stats import randint,skew,kurtosis

def extract_mfcc_features(signal, sr):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

    std_mfccs = np.std(mfccs, axis=1)

    return list(std_mfccs)

def help(main_file):
    feature_data = []
    for file_id in main_file['FileID']:
        file_path = os.path.join('C:/Users/Dhruv Agarwal/jaguar-land-rover-problem-statement-1/Data/Data/', file_id)
        vibration_data = pd.read_csv(file_path)

        # Feature extraction 
        std_value = np.std(vibration_data['data'])
        signal=vibration_data['data']
        sr = 5100
        mfcc_features = extract_mfcc_features(signal.values, sr)


        # Adding features to the list
        feature_data.append([std_value]+mfcc_features)

    return pd.DataFrame(feature_data)


def model(feature_df):
    x = feature_df.drop('label', axis=1)
    y = feature_df['label']

    # Standardizing features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Training a Random Forest classifier

    param_dist_rf = {
        'n_estimators': randint(100, 1000),
        'max_depth': list(range(10, 101)),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42)

    # Create a RandomizedSearchCV object with cross-validation
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist_rf, n_iter=100,
                                       cv=StratifiedKFold(n_splits=2), scoring='f1_micro', random_state=42)

    # Fit the random search to your data
    random_search.fit(x_scaled, y)

    modl = random_search.estimator
    modl.fit(x_scaled, y)
    return modl


# Load the main CSV file
train_df = pd.read_csv('C:/Users/Dhruv Agarwal/jaguar-land-rover-problem-statement-1/ps1/train.csv')

feature_df = help(train_df)
feature_df['label'] = train_df['Label']

m = model(feature_df)


test_df = pd.read_csv('C:/Users/Dhruv Agarwal/jaguar-land-rover-problem-statement-1/ps1/test.csv')

feature_df_test = help(test_df)

scaler = StandardScaler()
x_test_scaled = scaler.fit_transform(feature_df_test)

label_pred = m.predict(x_test_scaled)



test_df['Label'] = label_pred

print(test_df)
test_df.to_csv('submission.csv',index=False)

# Creating a classification report
report = classification_report(test_df['Label'], label_pred, target_names=['Class 0', 'Class 1'])

# Printing the classification report
print(report)

# Plot of feature importance
feature_importance = m.feature_importances_
features = feature_df.columns[:-1]

plt.figure(figsize=(8, 6))
plt.bar(features, feature_importance, color=['blue', 'green'])  # Using 'coolwarm' colormap
plt.title('Feature Importance Scores')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.show()

# Plot of confusion matrix
confusion_mat = confusion_matrix(test_df['Label'], label_pred)
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mat, interpolation='nearest', cmap='coolwarm')  # Using 'coolwarm' colormap
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(['Class 0', 'Class 1']))
plt.xticks(tick_marks, ['Class 0', 'Class 1'], rotation=45)
plt.yticks(tick_marks, ['Class 0', 'Class 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Annotate the confusion matrix cells with the count
for i in range(len(['Class 0', 'Class 1'])):
    for j in range(len(['Class 0', 'Class 1'])):
        plt.text(j, i, str(confusion_mat[i, j]), horizontalalignment='center', verticalalignment='center', color='black', fontsize=12)

plt.show()