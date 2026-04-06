import argparse
import csv
import os
from collections import Counter

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def parse_filename(filename):
    filename = filename.strip()
    if filename.lower().endswith('.wav'):
        filename = filename[:-4]

    if filename.startswith('EGFxSet_'):
        name = filename[len('EGFxSet_'):]
        parts = name.split('_')
        effect_type = parts[0] if parts else ''
        if len(parts) == 1:
            return effect_type
        if len(parts) == 2:
            return effect_type
        return effect_type

    for prefix in ['handmade_test_', 'pedalboard_test_', 'test_', 'handmade_', 'pedalboard_']:
        if filename.startswith(prefix):
            filename = filename[len(prefix):]
            break

    parts = filename.split('_')
    return parts[0] if parts else ''


def load_csv_features(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'CSV file not found: {csv_path}')

    features = []
    labels = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'filename' not in row:
                raise ValueError('CSV must contain a filename column')

            label = parse_filename(row['filename'])
            if label == '':
                continue

            row_features = []
            for key, value in row.items():
                if key == 'filename':
                    continue
                try:
                    row_features.append(float(value))
                except ValueError:
                    row_features.append(float('nan'))

            if any(map(lambda x: x != x, row_features)):
                continue

            features.append(row_features)
            labels.append(label)

    if not features:
        raise ValueError(f'No valid rows found in {csv_path}')

    return features, labels


def prepare_datasets(egfx_path, handmade_path, pedalboard_path, test_size=0.3, random_state=42):
    egfx_features, egfx_labels = load_csv_features(egfx_path)
    handmade_features, handmade_labels = load_csv_features(handmade_path)
    pedalboard_features, pedalboard_labels = load_csv_features(pedalboard_path)

    X_train, X_egfx_test, y_train, y_egfx_test = train_test_split(
        egfx_features,
        egfx_labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=egfx_labels if len(set(egfx_labels)) > 1 else None,
    )

    X_other_test = handmade_features + pedalboard_features
    y_other_test = handmade_labels + pedalboard_labels

    X_test = X_egfx_test + X_other_test
    y_test = y_egfx_test + y_other_test

    return X_train, y_train, X_test, y_test, X_egfx_test, y_egfx_test, X_other_test, y_other_test


def train_decision_tree(X_train, y_train, max_depth=None, random_state=42):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred, digits=4, zero_division=0)
    labels = sorted(set(y_test))
    matrix = metrics.confusion_matrix(y_test, y_pred, labels=labels)
    return accuracy, report, matrix


def main():
    parser = argparse.ArgumentParser(description='Train a Decision Tree model on audio feature CSV files.')
    parser.add_argument('--egfx', default='audio_features_EGFxSet.csv', help='EGFxSet CSV file path')
    parser.add_argument('--handmade', default='audio_features_handmade.csv', help='handmade CSV file path')
    parser.add_argument('--pedalboard', default='audio_features_pedalboard.csv', help='pedalboard CSV file path')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test split fraction for EGFxSet data')
    parser.add_argument('--max-depth', type=int, default=None, help='Maximum depth of the decision tree')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for split and training')
    args = parser.parse_args()

    print('Loading datasets...')
    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_egfx_test,
        y_egfx_test,
        X_other_test,
        y_other_test,
    ) = prepare_datasets(
        args.egfx,
        args.handmade,
        args.pedalboard,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f'Training samples: {len(X_train)}')
    print(f'Overall test samples: {len(X_test)}')
    print(f'EGFxSet split test samples: {len(X_egfx_test)}')
    print(f'Handmade + Pedalboard test samples: {len(X_other_test)}')
    print('Label distribution in training set:', Counter(y_train))
    print('Label distribution in overall test set:', Counter(y_test))

    print('Training Decision Tree classifier...')
    model = train_decision_tree(X_train, y_train, max_depth=args.max_depth, random_state=args.random_state)

    print('Evaluating model...')
    accuracy_all, report_all, matrix_all = evaluate_model(model, X_test, y_test)
    accuracy_egfx, _, _ = evaluate_model(model, X_egfx_test, y_egfx_test)
    accuracy_other, _, _ = evaluate_model(model, X_other_test, y_other_test)

    print(f'Overall accuracy: {accuracy_all:.4f}')
    print(f'EGFxSet 30% split accuracy: {accuracy_egfx:.4f}')
    print(f'Handmade + Pedalboard accuracy: {accuracy_other:.4f}')
    print('\nOverall classification report:\n')
    print(report_all)
    print('Overall confusion matrix:')
    print(matrix_all)
    print('Classes:', list(model.classes_))


if __name__ == '__main__':
    main()
