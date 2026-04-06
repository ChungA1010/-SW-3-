import argparse
import csv
import os
from collections import Counter

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ==============================
# filename parsing
# ==============================
def parse_filename(filename):  # Extract effect tokens from filename
    filename = filename.strip()
    if filename.lower().endswith('.wav'):
        filename = filename[:-4]

    if filename.startswith('EGFxSet_'):
        filename = filename[len('EGFxSet_'):]

    for prefix in ['handmade_test_', 'pedalboard_test_', 'test_', 'handmade_', 'pedalboard_']:
        if filename.startswith(prefix):
            filename = filename[len(prefix):]
            break

    tokens = filename.replace('+', '_').replace('-', '_').split('_')
    return tokens


# ==============================
# multi-label mapping
# ==============================
def map_to_groups(tokens):  # Drive / Space / Phase multi-label mapping
    groups = {'Drive': 0, 'Space': 0, 'Phase': 0}

    for token in tokens:
        t = token.lower()

        if any(x in t for x in ['drive', 'dist', 'overdrive', 'fuzz', 'crunch', 'boost', 'bluesdriver', 'rat', 'tubescreamer']):
            groups['Drive'] = 1
        if any(x in t for x in ['delay', 'reverb', 'echo', 'hall', 'space']):
            groups['Space'] = 1
        if any(x in t for x in ['chorus', 'phaser', 'phase', 'flanger', 'vibrato']):
            groups['Phase'] = 1

    return groups


# ==============================
# CSV load
# ==============================
def load_csv_features(csv_path):  # CSV → feature + multi-label
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'CSV file not found: {csv_path}')

    features = []
    labels = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'filename' not in reader.fieldnames:
            raise ValueError('CSV must contain a filename column')

        for row in reader:
            tokens = parse_filename(row['filename'])
            label_dict = map_to_groups(tokens)

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
            labels.append([label_dict['Drive'], label_dict['Space'], label_dict['Phase']])

    if not features:
        raise ValueError(f'No valid rows found in {csv_path}')

    return features, labels


# ==============================
# dataset split
# ==============================
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
    )

    X_other_test = handmade_features + pedalboard_features
    y_other_test = handmade_labels + pedalboard_labels

    X_test = X_egfx_test + X_other_test
    y_test = y_egfx_test + y_other_test

    return X_train, y_train, X_test, y_test, X_egfx_test, y_egfx_test, X_other_test, y_other_test


# ==============================
# train model (multi-output)
# ==============================
def train_neural_network(X_train, y_train, max_iter=1000, random_state=42):
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


# ==============================
# evaluation
# ==============================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # 각 label accuracy
    acc_drive = metrics.accuracy_score([y[0] for y in y_test], [y[0] for y in y_pred])
    acc_space = metrics.accuracy_score([y[1] for y in y_test], [y[1] for y in y_pred])
    acc_phase = metrics.accuracy_score([y[2] for y in y_test], [y[2] for y in y_pred])

    # exact match
    exact_match = sum(all(p == t for p, t in zip(pred, true)) for pred, true in zip(y_pred, y_test)) / len(y_test)

    return acc_drive, acc_space, acc_phase, exact_match


# ==============================
# main
# ==============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-label neural network classifier for audio effects')
    parser.add_argument('--egfx', default='audio_features_EGFxSet.csv')
    parser.add_argument('--handmade', default='audio_features_handmade.csv')
    parser.add_argument('--pedalboard', default='audio_features_pedalboard.csv')
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--random-state', type=int, default=42)
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

    # =========================
    # Normalize data
    # =========================
    print('Normalizing data...')
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_egfx_test = scaler.transform(X_egfx_test)
    X_other_test = scaler.transform(X_other_test)

    print('Training neural network...')
    model = train_neural_network(X_train, y_train, max_iter=args.max_iter, random_state=args.random_state)

    print('\n===== Overall Test =====')
    d, s, p, exact = evaluate_model(model, X_test, y_test)
    print(f'Drive acc: {d:.4f}, Space acc: {s:.4f}, Phase acc: {p:.4f}')
    print(f'Exact match acc: {exact:.4f}')

    print('\n===== EGFxSet (30%) =====')
    d, s, p, exact = evaluate_model(model, X_egfx_test, y_egfx_test)
    print(f'Drive acc: {d:.4f}, Space acc: {s:.4f}, Phase acc: {p:.4f}')
    print(f'Exact match acc: {exact:.4f}')

    print('\n===== Real Playing =====')
    d, s, p, exact = evaluate_model(model, X_other_test, y_other_test)
    print(f'Drive acc: {d:.4f}, Space acc: {s:.4f}, Phase acc: {p:.4f}')
    print(f'Exact match acc: {exact:.4f}')