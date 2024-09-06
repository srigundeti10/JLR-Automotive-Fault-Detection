import os
import librosa
import pandas as pd
import numpy as np
import pywt
import scipy.signal as signal
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from scipy.signal import find_peaks
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score,f1_score
from scipy.signal import welch  # For PSD estimation
from scipy.integrate import simps  # For power calculation
from matplotlib import  pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint,skew,kurtosis

def extract_mfcc_features(signal, sr):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mean_mfccs = np.mean(mfccs, axis=1)
    std_mfccs = np.std(mfccs, axis=1)
    # Calculate skewness of MFCCs
    skewness_mfccs = skew(mfccs, axis=1)

    # Calculate kurtosis of MFCCs
    kurtosis_mfccs = kurtosis(mfccs, axis=1)

    return list(mean_mfccs) + list(std_mfccs)+list(kurtosis_mfccs)+list(skewness_mfccs)


def extract_wavelet_features(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    max_coeff_length = max(len(coeff) for coeff in coeffs)
    padded_coeffs = [np.pad(coeff, (0, max_coeff_length - len(coeff))) for coeff in coeffs]
    wavelet_features = []
    for coeff in padded_coeffs:
        mean_value = np.mean(coeff)
        std_value = np.std(coeff)
        skewness = np.mean((coeff - mean_value) ** 3) / (std_value ** 3)
        kurtosis = np.mean((coeff - mean_value) ** 4) / (std_value ** 4)
        rms = np.sqrt(np.mean(coeff ** 2))
        wavelet_features.extend([mean_value, std_value, skewness, kurtosis, rms])
    return wavelet_features


def extract_other_features(signal):
    mean_value = np.mean(signal)
    std_value = np.std(signal)
    skewness = np.mean((signal - mean_value) ** 3) / (std_value ** 3)
    kurtosis = np.mean((signal - mean_value) ** 4) / (std_value ** 4)
    rms = np.sqrt(np.mean(signal** 2))

    # Frequency-domain features using FFT
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal))
    magnitude_spectrum = np.abs(fft_result)

    # Find dominant frequency (index of the peak in the magnitude spectrum)
    peaks, _ = find_peaks(magnitude_spectrum)
    dominant_frequency_index = peaks[np.argmax(magnitude_spectrum[peaks])]
    dominant_frequency = frequencies[dominant_frequency_index]

    psd = np.abs(fft_result) ** 2

    # Normalize the PSD
    psd /= psd.sum()

    # Calculate spectral entropy
    spectral_entropy = -np.sum(psd * np.log2(psd))

    peak_frequency_index = np.argmax(psd)
    peak_frequency = frequencies[peak_frequency_index]

    spectral_energy = np.sum(np.abs(fft_result) ** 2)

    spectral_centroid = np.sum(frequencies * psd)
    # Calculate spectral bandwidth
    spectral_bandwidth = np.sqrt(np.sum((frequencies - spectral_centroid) ** 2 * psd))
    # Calculate spectral flatness
    spectral_flatness = np.exp(np.mean(np.log(psd + 1e-12)))  # Adding a small constant to avoid log(0)




    # Add more feature extraction code as needed
    return [std_value,skewness,dominant_frequency,spectral_entropy,spectral_flatness, kurtosis, rms, spectral_energy, spectral_bandwidth]

    # return [mean_value, skewness, kurtosis]  # Return the extracted features as a list


def get_featuredata(main_file):
    feature_data = []
    for file_id in main_file['FileID']:
        file_path = os.path.join('C:/Users/Dhruv Agarwal/jaguar-land-rover-problem-statement-1/Data/Data/', file_id)
        vibration_data = pd.read_csv(file_path)

        signal = vibration_data['data']

        wavelet_features = extract_wavelet_features(signal, wavelet='db4', level=5)
        other_features = extract_other_features(signal)
        # Extract MFCC features
        sr = 44100  # Adjust this based on your audio sampling rate
        mfcc_features = extract_mfcc_features(signal.values, sr)

        all_features = wavelet_features + other_features + mfcc_features

        feature_data.append(all_features)

    return pd.DataFrame(feature_data)


def stacking_classifier(x_resampled, y_resampled):
    param_dist_rf = {
        'n_estimators': randint(100, 1000),
        'max_depth': list(range(10, 101)),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    param_dist_gb = {
        'n_estimators': randint(100, 1000),
        'max_depth': list(range(3, 20)),  # Adjust the range as needed
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3]
    }

    param_dist_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf']
    }

    param_dist_logreg = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }

    # Create a Random Forest classifier
    rf = RandomForestClassifier(random_state=42)

    # Create a RandomizedSearchCV object with cross-validation
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist_rf, n_iter=100,
                                       cv=StratifiedKFold(n_splits=5), scoring='f1_micro', random_state=42)

    # Fit the random search to your data
    random_search.fit(x_resampled, y_resampled)

    # GradientBoostingClassifier hyperparameter tuning (similar process)
    gr = GradientBoostingClassifier(random_state=42)
    Gradient_search = RandomizedSearchCV(estimator=gr, param_distributions=param_dist_gb, n_iter=100,
                                         cv=StratifiedKFold(n_splits=5), scoring='f1_micro', random_state=42)

    # Fit the random search to your data
    Gradient_search.fit(x_resampled, y_resampled)

    # SVM hyperparameter tuning (similar process)
    svc = SVC(random_state=42)
    svm_search = RandomizedSearchCV(estimator=svc, param_distributions=param_dist_svm, n_iter=100,
                                    cv=StratifiedKFold(n_splits=5), scoring='f1_micro', random_state=42)

    # Fit the random search to your data
    svm_search.fit(x_resampled, y_resampled)

    # Logistic Regression hyperparameter tuning (similar process)
    lg = LogisticRegression(random_state=42)
    lg_search = RandomizedSearchCV(estimator=lg, param_distributions=param_dist_logreg, n_iter=100,
                                   cv=StratifiedKFold(n_splits=5), scoring='f1_micro', random_state=42)
    # Fit the random search to your data
    lg_search.fit(x_resampled, y_resampled)

    # Define your base models
    base_models = [
        ('rf', random_search.best_estimator_),
        ('gb', Gradient_search.estimator),
        ('svm', svm_search.estimator),
    ]

    # Define your meta-model
    meta_model = lg_search.estimator

    # Create the stacking classifier
    stacking_classifier = StackingClassifier(estimators=base_models, final_estimator=meta_model)

    # Train the stacking classifier
    stacking_classifier.fit(x_resampled, y_resampled)

    return stacking_classifier


def model(feature_df):
    x = feature_df.drop('label', axis=1)
    y = feature_df['label']

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Instantiate the RandomUnderSampler
    undersampler = RandomUnderSampler(random_state=42)

    # Fit and transform the training data

    x_resampled, y_resampled = undersampler.fit_resample(x_scaled, y)


    model = RandomForestClassifier()

    # Create a range of values for n_features_to_select to test
    num_features_to_select_values = [1, 2, 3, 4, 5, 6, 7, 8]  # Adjust as needed

    # Initialize a list to store cross-validation scores
    cv_scores = []

    # Perform cross-validation for each value of n_features_to_select
    for num_features_to_select in num_features_to_select_values:
        rfe = RFE(model, n_features_to_select=num_features_to_select, step=1)
        # Use StratifiedKFold for classification tasks
        kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        scores = cross_val_score(rfe, x_resampled, y_resampled, cv=kfold, scoring='f1_micro')  # Adjust scoring as needed
        cv_scores.append(np.mean(scores))

    # Find the optimal n_features_to_select
    optimal_num_features = num_features_to_select_values[np.argmax(cv_scores)]
    print(f"Optimal n_features_to_select: {optimal_num_features}")

    model = RandomForestClassifier()

    rfe = RFE(model, n_features_to_select=8, step=1)


    # Get the selected features
    selected_feature_indices = rfe.support_
    selected_features = x.columns[selected_feature_indices]
    print(selected_features)

    return stacking_classifier(x_resampled,y_resampled)

# Load the main CSV file
train_df = pd.read_csv('C:/Users/Dhruv Agarwal/jaguar-land-rover-problem-statement-1/ps3/train.csv')

feature_df = get_featuredata(train_df)
feature_df['label'] = train_df['Label']

m = model(feature_df)

test_df = pd.read_csv('C:/Users/Dhruv Agarwal/jaguar-land-rover-problem-statement-1/ps3/test.csv')

feature_df_test = help(test_df)

scaler = StandardScaler()
x_test_scaled = scaler.fit_transform(feature_df_test)

label_pred = m.predict(x_test_scaled)

print(label_pred)
test_df['Label'] = label_pred
print(test_df)
test_df.to_csv('submission3.csv',index=False)

