import csv
import matplotlib.pyplot as plt

# Effect type to group mapping
# Add specific EGFxSet names here when you want them classified into the existing groups.
effect_to_group = {
    'clean': 'clean',
    'drive': 'drive',
    'dist': 'drive',
    'delay': 'space',
    'reverb': 'space',
    'chorus': 'phase',
    'phaser': 'phase',
    'BluesDriver': 'drive',
    # Add more EGFxSet-specific names as needed
}

# Group to color mapping (RGB tuples for blending)
group_to_color = {
    'clean': (0.5, 0.5, 0.5),  # gray
    'drive': (1.0, 0.0, 0.0),  # red
    'space': (0.0, 0.0, 1.0),  # blue
    'phase': (1.0, 1.0, 0.0),  # yellow
    'unknown': (0.2, 0.2, 0.2)
}

# Group to marker mapping
group_to_marker = {
    'clean': 'o',
    'drive': 's',
    'space': '^',
    'phase': 'D',
    'unknown': 'x'
}


def get_effect_group(effect_type):
    if not effect_type:
        return 'unknown'

    if effect_type in effect_to_group:
        return effect_to_group[effect_type]

    lower = effect_type.lower()
    if 'clean' in lower:
        return 'clean'
    if any(token in lower for token in ['drive', 'dist', 'overdrive', 'fuzz', 'crunch', 'boost']):
        return 'drive'
    if any(token in lower for token in ['delay', 'reverb', 'echo', 'space']):
        return 'space'
    if any(token in lower for token in ['chorus', 'phaser', 'phase', 'flanger', 'vibrato']):
        return 'phase'
    return 'unknown'


def get_color_for_effect_group(effect_group):
    return group_to_color.get(effect_group, group_to_color['unknown'])


def get_marker_for_effect_group(effect_group):
    return group_to_marker.get(effect_group, group_to_marker['unknown'])


def parse_filename(filename):
    """
    Parse filename to extract the raw effect type, play type, and numeric suffix.
    Supports test_, handmade_, pedalboard_, and EGFxSet filename conventions.
    """
    name = filename.strip()
    if name.lower().endswith('.wav'):
        name = name[:-4]

    if name.startswith('EGFxSet_'):
        name = name[len('EGFxSet_'):]
        parts = name.split('_')
        effect_type = parts[0]
        if len(parts) == 1:
            return effect_type, '', ''
        if len(parts) == 2:
            return effect_type, parts[1], ''
        return effect_type, '_'.join(parts[1:-1]), parts[-1]

    prefixes = ['handmade_test_', 'pedalboard_test_', 'test_', 'handmade_', 'pedalboard_']
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    parts = name.split('_')
    if len(parts) == 1:
        return parts[0], '', ''
    if len(parts) == 2:
        return parts[0], parts[1], ''
    return parts[0], '_'.join(parts[1:-1]), parts[-1]


def visualize_features(csv_file):
    """
    Visualize audio features from CSV file, grouped by inferred effect categories.
    """
    data = []
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    if not data:
        raise ValueError(f'CSV file {csv_file} contains no rows.')

    for row in data:
        effect_type, play_type, number = parse_filename(row['filename'])
        row['effect_type'] = effect_type
        row['effect_group'] = get_effect_group(effect_type)
        row['play_type'] = play_type
        row['number'] = number
        if play_type and number:
            row['play_group'] = f'{play_type}_{number}'
        elif play_type:
            row['play_group'] = play_type
        elif number:
            row['play_group'] = number
        else:
            row['play_group'] = effect_type

    all_keys = list(data[0].keys())
    feature_columns = [col for col in all_keys if col not in ['filename', 'effect_type', 'effect_group', 'play_type', 'number', 'play_group']]

    effect_groups = sorted(set(row['effect_group'] for row in data))
    play_groups = sorted(set(row['play_group'] for row in data))
    x_map = {pg: i for i, pg in enumerate(play_groups)}

    color_map = {group: get_color_for_effect_group(group) for group in effect_groups}
    marker_map = {group: get_marker_for_effect_group(group) for group in effect_groups}

    n_features = len(feature_columns)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, feature in enumerate(feature_columns):
        ax = axes[i]
        for effect_group in effect_groups:
            subset = [row for row in data if row['effect_group'] == effect_group]
            if not subset:
                continue
            x_vals = [x_map[row['play_group']] for row in subset]
            y_vals = [float(row[feature]) for row in subset]
            ax.scatter(
                x_vals,
                y_vals,
                color=color_map[effect_group],
                marker=marker_map[effect_group],
                label=effect_group,
                alpha=0.7,
                s=50,
            )

        ax.set_title(feature.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Value', fontsize=8)
        ax.set_xticks(range(len(play_groups)))
        ax.set_xticklabels(play_groups, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[n_features:]:
        ax.axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.suptitle('Audio Feature Visualization by Effect Group', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()


if __name__ == '__main__':
    csv_file = 'audio_features_EGFxSet.csv'
    visualize_features(csv_file)
