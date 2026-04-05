import csv
import matplotlib.pyplot as plt
import numpy as np

# Effect type to group mapping
effect_to_group = {
    'clean': 'clean',
    'drive': 'drive',
    'dist': 'drive',
    'delay': 'space',
    'reverb': 'space',
    'chorus': 'phase',
    'phaser': 'phase',
    # Add more as needed
}

# Group to color mapping (RGB tuples for blending)
group_to_color = {
    'clean': (0.5, 0.5, 0.5),  # gray
    'drive': (1.0, 0.0, 0.0),  # red
    'space': (0.0, 0.0, 1.0),  # blue
    'phase': (1.0, 1.0, 0.0)   # yellow
}

# Group to marker mapping
group_to_marker = {
    'clean': 'o',
    'drive': 's',
    'space': '^',
    'phase': 'D'
}

def get_color_for_effect_type(effect_type):
    sub_effects = effect_type.split('+')
    groups = [effect_to_group.get(sub, 'clean') for sub in sub_effects]
    colors = [group_to_color[group] for group in groups]
    if len(colors) == 1:
        return colors[0]
    else:
        # Average RGB values
        avg_r = sum(c[0] for c in colors) / len(colors)
        avg_g = sum(c[1] for c in colors) / len(colors)
        avg_b = sum(c[2] for c in colors) / len(colors)
        return (avg_r, avg_g, avg_b)

def get_marker_for_effect_type(effect_type):
    sub_effects = effect_type.split('+')
    groups = [effect_to_group.get(sub, 'clean') for sub in sub_effects]
    # Use the first group's marker
    return group_to_marker.get(groups[0], 'o')

def parse_filename(filename):
    """
    Parse filename to extract effect_type, play_type, number.
    
    Format: test_{effect_type}_{play_type}_{number}.wav
    """
    name = filename.replace('.wav', '').replace('test_', '')
    parts = name.split('_')
    
    if len(parts) == 1:
        effect_type = parts[0]
        play_type = ''
        number = ''
    elif len(parts) == 2:
        effect_type = parts[0]
        play_type = parts[1]
        number = ''
    else:
        effect_type = parts[0]
        play_type = '_'.join(parts[1:-1])
        number = parts[-1]
    
    return effect_type, play_type, number

def visualize_features(csv_file):
    """
    Visualize audio features from CSV file, grouped by effect type.
    """
    # Load CSV
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    # Convert to list of dicts
    df = data  # list of dicts
    
    # Parse filenames and add parsed columns
    for row in df:
        effect_type, play_type, number = parse_filename(row['filename'])
        row['effect_type'] = effect_type
        row['play_type'] = play_type
        row['number'] = number
        row['play_group'] = f"{play_type}_{number}" if number else play_type
    
    # Feature columns (exclude filename and parsed columns)
    all_keys = list(df[0].keys())
    feature_columns = [col for col in all_keys if col not in ['filename', 'effect_type', 'play_type', 'number', 'play_group']]
    
    # Unique effect types and play groups
    effect_types = sorted(set(row['effect_type'] for row in df))
    play_groups = sorted(set(row['play_group'] for row in df))
    x_map = {pg: i for i, pg in enumerate(play_groups)}
    
    # Color and marker map for effect types
    color_map = {}
    marker_map = {}
    for effect_type in effect_types:
        color_map[effect_type] = get_color_for_effect_type(effect_type)
        marker_map[effect_type] = get_marker_for_effect_type(effect_type)
    
    # Create subplots: 5x4 for 20 features
    fig, axes = plt.subplots(5, 4, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_columns):
        ax = axes[i]
        
        for effect_type in effect_types:
            subset = [row for row in df if row['effect_type'] == effect_type]
            if subset:
                x_vals = [x_map[row['play_group']] for row in subset]
                y_vals = [float(row[feature]) for row in subset]
                ax.scatter(x_vals, y_vals, 
                          color=color_map[effect_type], 
                          marker=marker_map[effect_type],
                          label=effect_type, 
                          alpha=0.7, 
                          s=50)
        
        ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=10)
        ax.set_ylabel('Value', fontsize=8)
        ax.set_xticks(range(len(play_groups)))
        ax.set_xticklabels(play_groups, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Add legend to the last subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.suptitle('Audio Feature Visualization by Effect Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Leave space for legend
    plt.show()

if __name__ == "__main__":
    csv_file = "audio_features.csv"
    visualize_features(csv_file)
