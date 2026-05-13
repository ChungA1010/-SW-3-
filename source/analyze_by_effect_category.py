import os
import re
import pandas as pd
import numpy as np
from scipy import stats


def extract_intensity_num(row):
    s1 = str(row.get('intensity', ''))
    s2 = str(row.get('play_type_and_number', ''))
    m = re.search(r"(\d+)", s1)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", s2)
    if m:
        return int(m.group(1))
    return np.nan


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_std


def main(csv_path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    csv_dir = os.path.join(project_root, 'csv')
    analysis_dir = os.path.join(csv_dir, 'feature_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    if csv_path is None:
        csv_path = os.path.join(csv_dir, 'features_pedalboarded_handmade.csv')

    df = pd.read_csv(csv_path)

    meta_cols = ['top_category', 'effect_kind', 'intensity', 'play_type_and_number']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    df['intensity_num'] = df.apply(extract_intensity_num, axis=1)
    df['is_clean'] = df['top_category'].str.lower() == 'clean'

    clean_df = df[df['is_clean'] == True]
    
    categories = ['drive', 'space', 'phase']
    
    print("=" * 80)
    print("ANALYSIS BY EFFECT CATEGORY")
    print("=" * 80)

    all_results = []

    for category in categories:
        cat_df = df[df['top_category'].str.lower() == category]
        
        if cat_df.empty:
            print(f"\nNo data for {category.upper()}")
            continue

        print(f"\n{'=' * 80}")
        print(f"{category.upper()} (n={len(cat_df)} samples)")
        print(f"{'=' * 80}")

        # 1. Clean vs This Category
        results_clean_vs_cat = []
        for feat in feature_cols:
            a = clean_df[feat].dropna()
            b = cat_df[feat].dropna()
            mean_clean = a.mean() if len(a) else np.nan
            mean_cat = b.mean() if len(b) else np.nan
            d = cohen_d(a.values, b.values) if len(a) and len(b) else np.nan
            tstat, pval = (np.nan, np.nan)
            if len(a) and len(b):
                try:
                    tstat, pval = stats.ttest_ind(a, b, equal_var=False)
                except Exception:
                    tstat, pval = (np.nan, np.nan)
            results_clean_vs_cat.append({
                'category': category,
                'analysis': 'clean_vs_category',
                'feature': feat,
                'mean_clean': mean_clean,
                'mean_category': mean_cat,
                'cohen_d': d,
                'tstat': tstat,
                'pval': pval
            })

        df_clean_cat = pd.DataFrame(results_clean_vs_cat)
        df_clean_cat['abs_cohen'] = df_clean_cat['cohen_d'].abs()
        df_clean_cat = df_clean_cat.sort_values('abs_cohen', ascending=False)
        
        all_results.extend(df_clean_cat.to_dict('records'))

        print(f"\nClean vs {category.upper()} - Top 10 features by effect size:")
        print(df_clean_cat[['feature', 'mean_clean', 'mean_category', 'cohen_d', 'pval']].head(10).to_string(index=False))

        # 2. Intensity correlation within this category
        cat_with_intensity = cat_df.dropna(subset=['intensity_num']).copy()
        corr_results = []
        if not cat_with_intensity.empty:
            for feat in feature_cols:
                x = cat_with_intensity['intensity_num'].values
                y = cat_with_intensity[feat].values
                try:
                    rho, p = stats.spearmanr(x, y)
                except Exception:
                    rho, p = (np.nan, np.nan)
                corr_results.append({
                    'category': category,
                    'analysis': 'intensity_correlation',
                    'feature': feat,
                    'spearman_rho': rho,
                    'pval': p,
                    'n_samples': len(cat_with_intensity)
                })

        df_corr = pd.DataFrame(corr_results).sort_values('spearman_rho', key=lambda s: s.abs(), ascending=False)
        all_results.extend(df_corr.to_dict('records'))

        print(f"\n{category.upper()} - Intensity Correlation - Top 10 features:")
        print(df_corr[['feature', 'spearman_rho', 'pval']].head(10).to_string(index=False))

        # 3. Weak vs Strong within this category
        cat_weak = cat_with_intensity[cat_with_intensity['intensity_num'] <= 25]
        cat_strong = cat_with_intensity[cat_with_intensity['intensity_num'] >= 75]
        results_ws = []
        for feat in feature_cols:
            a = cat_weak[feat].dropna()
            b = cat_strong[feat].dropna()
            if len(a) and len(b):
                tstat, pval = stats.ttest_ind(a, b, equal_var=False)
                d = cohen_d(a.values, b.values)
            else:
                tstat, pval, d = (np.nan, np.nan, np.nan)
            results_ws.append({
                'category': category,
                'analysis': 'weak_vs_strong',
                'feature': feat,
                'n_weak': len(a),
                'n_strong': len(b),
                'cohen_d': d,
                'tstat': tstat,
                'pval': pval
            })

        df_ws = pd.DataFrame(results_ws)
        df_ws['abs_cohen'] = df_ws['cohen_d'].abs()
        df_ws = df_ws.sort_values('abs_cohen', ascending=False)
        all_results.extend(df_ws.to_dict('records'))

        print(f"\n{category.upper()} - Weak(25%) vs Strong(100%) - Top 10 features:")
        print(df_ws[['feature', 'n_weak', 'n_strong', 'cohen_d', 'pval']].head(10).to_string(index=False))

    # Save comprehensive results
    print("\n" + "=" * 80)
    print("SAVING RESULTS TO CSV")
    print("=" * 80)

    # Convert list of dicts to DataFrame and save
    df_all_results = pd.DataFrame(all_results)
    
    # Save by analysis type
    for analysis in ['clean_vs_category', 'intensity_correlation', 'weak_vs_strong']:
        mask = df_all_results['analysis'] == analysis
        df_subset = df_all_results[mask].drop('analysis', axis=1)
        
        # Save overall
        output_file = os.path.join(analysis_dir, f'analysis_by_category_{analysis}.csv')
        df_subset.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
        
        # Save per category
        for category in categories:
            cat_mask = (df_all_results['analysis'] == analysis) & (df_all_results['category'] == category)
            cat_subset = df_all_results[cat_mask].drop(['analysis', 'category'], axis=1)
            if not cat_subset.empty:
                output_file = os.path.join(analysis_dir, f'analysis_{category}_{analysis}.csv')
                cat_subset.to_csv(output_file, index=False)
                print(f"Saved: {output_file}")


if __name__ == '__main__':
    main()
