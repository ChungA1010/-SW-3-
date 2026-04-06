import argparse
import csv
import os
from collections import Counter

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def parse_filename(filename):
    filename = filename.strip()
    if filename.lower().endswith('.wav'):
        filename = filename[:-4]

    if filename.startswith('EGFxSet_'):
        name = filename[len('EGFxSet_'):]
        parts = name.split('_')
        return parts[0] if parts else ''

    for prefix in ['handmade_test_', 'pedalboard_test_', 'test_', 'handmade_', 'pedalboard_']:
        if filename.startswith(prefix):
            filename = filename[len(prefix):]
            break

    parts = filename.split('_')
    return parts[0] if parts else ''


def map_to_group(effect_type):
    effect_type = effect_type.strip()
    if not effect_type:
        return None

    lower = effect_type.lower()
    if any(token in lower for token in ['drive', 'dist', 'overdrive', 'fuzz', 'crunch', 'boost', 'bluesdriver', 'rats', 'rat', 'tubescreamer']):
        return 'Drive'
    if any(token in lower for token in ['delay', 'reverb', 'echo', 'hall', 'space']):
        return 'Space'
    if any(token in lower for token in ['chorus', 'phaser', 'phase', 'flanger', 'vibrato']):
        return 'Phase'

    # If multiple groups are present, prefer Drive > Space > Phase by order above
    tokens = effect_type.replace('+', ' ').replace('-', ' ').split()
    groups = set()
    for token in tokens:
        token_lower = token.lower()
        if token_lower in ['drive', 'dist', 'overdrive', 'fuzz', 'crunch', 'boost', 'bluesdriver', 'rat', 'tubescreamer']:
            groups.add('Drive')
        elif token_lower in ['delay', 'reverb', 'echo', 'hall', 'space']:
            groups.add('Space')
        elif token_lower in ['chorus', 'phaser', 'phase', 'flanger', 'vibrato']:
            groups.add('Phase')

    if 'Drive' in groups:
        return 'Drive'
    if 'Space' in groups:
        return 'Space'
    if 'Phase' in groups:
        return 'Phase'
    return None


def load_csv_features(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'CSV file not found: {csv_path}')

    features = []
    labels = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'filename' not in reader.fieldnames:
            raise ValueError('CSV must contain a filename column')

        for row in reader:
            label_raw = parse_filename(row['filename'])
            label = map_to_group(label_raw)
            if label is None:
                continue

            row_features = []
            for key, value in row.items():
                if key == 'filename':
                    continue
                try:
                    row_features.append(float(value))
                except ValueError:
                    row_features.append(float('nan'))

            if any(x != x for x in row_features):
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


def train_neural_network(X_train, y_train, max_iter=1000, random_state=42):
    model = MLPClassifier(
        hidden_layer_sizes=(),
        activation='logistic',
        solver='adam',
        max_iter=max_iter,
        random_state=random_state,
        verbose=False,
    )
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
    parser = argparse.ArgumentParser(description='Train a 3-class neural network classifier on audio feature CSV data.')
    parser.add_argument('--egfx', default='audio_features_EGFxSet.csv', help='EGFxSet CSV file path')
    parser.add_argument('--handmade', default='audio_features_handmade.csv', help='handmade CSV file path')
    parser.add_argument('--pedalboard', default='audio_features_pedalboard.csv', help='pedalboard CSV file path')
    parser.add_argument('--test-size', type=float, default=0.3, help='Fraction of EGFxSet rows held out for EGFxSet test set')
    parser.add_argument('--max-iter', type=int, default=1000, help='Maximum training iterations for the neural network')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for splitting and training')
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
    print('Training classes:', sorted(set(y_train)))
    print('Test classes:', sorted(set(y_test)))
    print('Label distribution in training set:', Counter(y_train))
    print('Label distribution in test set:', Counter(y_test))

    print('Training neural network classifier...')
    model = train_neural_network(X_train, y_train, max_iter=args.max_iter, random_state=args.random_state)

    print('Evaluating model...')
    accuracy_all, report_all, matrix_all = evaluate_model(model, X_test, y_test)
    accuracy_egfx, report_egfx, matrix_egfx = evaluate_model(model, X_egfx_test, y_egfx_test)
    accuracy_other, report_other, matrix_other = evaluate_model(model, X_other_test, y_other_test)

    print(f'Overall accuracy: {accuracy_all:.4f}')
    print(f'EGFxSet 30% split accuracy: {accuracy_egfx:.4f}')
    print(f'Handmade + Pedalboard accuracy: {accuracy_other:.4f}')

    print('\nOverall classification report:\n')
    print(report_all)
    print('Overall confusion matrix:')
    print(matrix_all)
    print('Overall classes:', sorted(set(y_test)))

    print('\nEGFxSet subset classification report:\n')
    print(report_egfx)
    print('EGFxSet confusion matrix:')
    print(matrix_egfx)

    print('\nHandmade + Pedalboard subset classification report:\n')
    print(report_other)
    print('Handmade + Pedalboard confusion matrix:')
    print(matrix_other)


if __name__ == '__main__':
    main()
