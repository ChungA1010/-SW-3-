import argparse
import csv
import os
from collections import Counter

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ==============================
# filename → effect tokens
# ==============================
def parse_filename(filename):  # 파일명에서 이펙터 토큰 리스트 추출
    filename = filename.strip()
    if filename.lower().endswith('.wav'):
        filename = filename[:-4]

    if filename.startswith('EGFxSet_'):
        filename = filename[len('EGFxSet_'):]

    for prefix in ['handmade_test_', 'pedalboard_test_', 'test_', 'handmade_', 'pedalboard_']:
        if filename.startswith(prefix):
            filename = filename[len(prefix):]
            break

    # +, -, _ 모두 분리
    tokens = filename.replace('+', '_').replace('-', '_').split('_')
    return tokens


# ==============================
# tokens → multi-label
# ==============================
def map_to_groups(tokens):  # 여러 이펙터를 Drive/Space/Phase multi-label로 변환
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
        for row in reader:
            if 'filename' not in row:
                raise ValueError('CSV must contain a filename column')

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

            if any(map(lambda x: x != x, row_features)):
                continue

            features.append(row_features)
            labels.append(label_dict)

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

    # EGFxSet split
    X_train, X_egfx_test, y_train, y_egfx_test = train_test_split(
        egfx_features,
        egfx_labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # 실제 연주 데이터
    X_other_test = handmade_features + pedalboard_features
    y_other_test = handmade_labels + pedalboard_labels

    return (
        X_train, y_train,
        X_egfx_test, y_egfx_test,
        X_other_test, y_other_test
    )


# ==============================
# train 3 models
# ==============================
def train_models(X_train, y_train, max_depth=None, random_state=42):  
    models = {
        'Drive': DecisionTreeClassifier(max_depth=max_depth, random_state=random_state),
        'Space': DecisionTreeClassifier(max_depth=max_depth, random_state=random_state),
        'Phase': DecisionTreeClassifier(max_depth=max_depth, random_state=random_state),
    }

    for key in models:
        y = [label[key] for label in y_train]
        models[key].fit(X_train, y)

    return models


# ==============================
# evaluation
# ==============================
def evaluate_models(models, X_test, y_test):
    results = {}

    for key in models:
        y_true = [label[key] for label in y_test]
        y_pred = models[key].predict(X_test)

        acc = metrics.accuracy_score(y_true, y_pred)
        report = metrics.classification_report(y_true, y_pred, zero_division=0)

        results[key] = (acc, report)

    # exact match
    exact_match = 0
    for i in range(len(X_test)):
        pred = {k: models[k].predict([X_test[i]])[0] for k in models}
        if all(pred[k] == y_test[i][k] for k in models):
            exact_match += 1

    exact_match_acc = exact_match / len(X_test)

    return results, exact_match_acc


# ==============================
# main
# ==============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-label Decision Tree for audio effects')
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    csv_dir = os.path.join(project_root, 'csv')

    parser.add_argument('--egfx', default=os.path.join(csv_dir, 'audio_features_EGFxSet.csv'))
    parser.add_argument('--handmade', default=os.path.join(csv_dir, 'audio_features_handmade.csv'))
    parser.add_argument('--pedalboard', default=os.path.join(csv_dir, 'audio_features_pedalboard.csv'))
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    print('Loading datasets...')
    (
        X_train, y_train,
        X_egfx_test, y_egfx_test,
        X_other_test, y_other_test
    ) = prepare_datasets(
        args.egfx,
        args.handmade,
        args.pedalboard,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f'Train samples (EGFx 70%): {len(X_train)}')
    print(f'EGFx test samples (30%): {len(X_egfx_test)}')
    print(f'Other test samples (real playing): {len(X_other_test)}')

    print('Training models...')
    models = train_models(X_train, y_train, max_depth=args.max_depth, random_state=args.random_state)

    # =========================
    # EGFxSet 평가 (in-domain)
    # =========================
    print('\n===== EGFxSet TEST (30%) =====')
    results_egfx, exact_egfx = evaluate_models(models, X_egfx_test, y_egfx_test)

    for key in results_egfx:
        acc, report = results_egfx[key]
        print(f'\n[{key}] Accuracy: {acc:.4f}')
        print(report)

    print(f'Exact match accuracy: {exact_egfx:.4f}')

    # =========================
    # 실제 연주 평가 (out-of-domain)
    # =========================
    print('\n===== REAL PLAYING TEST (handmade + pedalboard) =====')
    results_other, exact_other = evaluate_models(models, X_other_test, y_other_test)

    for key in results_other:
        acc, report = results_other[key]
        print(f'\n[{key}] Accuracy: {acc:.4f}')
        print(report)

    print(f'Exact match accuracy: {exact_other:.4f}')