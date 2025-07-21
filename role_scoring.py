import pandas as pd
import numpy as np
from scipy.stats import gmean

# Position configuration dictionary
POSITION_CONFIGS = {
    'CM': {
        'aoi_mapping': {
            'Duel Volume': [
                'EFx Aerial Duels',
                'EFx Ground Duels'
            ],
            'Ball Winning': [
                'Interceptions per 90',
                'Successful defensive actions per 90',
                'Total Duel %'
            ],
            'Mobility': [
                'Progressive runs per 90',
                'Accelerations per 90',
                'Dribbles per 90'
            ],
            'Deep Pen': [
                'EFx Prog. Pass',
                'Passes to final third per 90',
                'P2P'
            ],
            'Adv Pen': [
                'Passes to penalty area per 90',
                'Passes to final third per 90',
                'PPA Share',
                'Deep completions per 90'
            ],
            'Creation': [
                'xA per 90',
                'Key passes per 90',
            ],
            'Ball Security': [
                'Accurate passes, %',
                'Successful dribbles, %'
            ],
            'Box-Crashing': [
                'Touches in box per 90',
                'Successful attacking actions per 90',
                'xG per 90',
                'xG per Shot'
            ]
        },
        'roles': {
            'Ball-Winner': {
                'Duel Volume': 0.3,
                'Ball Winning': 0.3,
                'Ball Security': 0.2,
                'Mobility': 0.1,
                'Deep Pen': 0.1,
                'Creation': 0.0,
                'Box-Crashing': 0.0,
                'Adv Pen': 0.0
            },
            'Conductor': {
                'Duel Volume': 0.1,
                'Ball Winning': 0.1,
                'Ball Security': 0.25,
                'Mobility': 0.1,
                'Deep Pen': 0.35,
                'Creation': 0.0,
                'Box-Crashing': 0.0,
                'Adv Pen': 0.1
            },
            'Attacking Box-Crasher': {
                'Duel Volume': 0.1,
                'Ball Winning': 0.1,
                'Ball Security': 0.1,
                'Mobility': 0.1,
                'Deep Pen': 0.0,
                'Creation': 0.1,
                'Box-Crashing': 0.4,
                'Adv Pen': 0.1
            },
            'Attacking Creator/#10': {
                'Duel Volume': 0.00,
                'Ball Winning': 0.0,
                'Ball Security': 0.1,
                'Mobility': 0.15,
                'Deep Pen': 0.1,
                'Creation': 0.35,
                'Box-Crashing': 0.0,
                'Adv Pen': 0.3
            }
        }
    },
    'FB': {
        'aoi_mapping': {
            'Lockdown': [
                'Defensive duels won, %',
                'EFx Ground Duels'
            ],
            'Defending': [
                'Interceptions per 90',
                'Successful defensive actions per 90',
                'Aerial duels won, %',
            ],
            'Running Pen': [
                'Progressive runs per 90',
                'Accelerations per 90',
                'Dribbles per 90'
            ],
            'Pass Pen': [
                'Progressive passes per 90',
                'Passes to final third per 90',
                'Passes to penalty area per 90'
            ],
            'Creation': [
                'xA per 90',
                'Key passes per 90',
                'Crosses per 90'
            ],
            'Ball Security': [
                'Accurate passes, %',
                'Successful dribbles, %'
            ]
        },
        'roles': {
            'Penetrator': {
                'Lockdown': 0.1,
                'Defending': 0.15,
                'Running Pen': 0.0,
                'Pass Pen': 0.4,
                'Creation': 0.2,
                'Ball Security': 0.15,
            },
            'Runner': {
                'Lockdown': 0.1,
                'Defending': 0.15,
                'Running Pen': 0.4,
                'Pass Pen': 0.0,
                'Creation': 0.2,
                'Ball Security': 0.15,
            },
            'Padlock': {
                'Lockdown': 0.4,
                'Defending': 0.35,
                'Running Pen': 0.05,
                'Pass Pen': 0.05,
                'Creation': 0.0,
                'Ball Security': 0.15,
            },
            'Creator': {
                'Lockdown': 0.1,
                'Defending': 0.1,
                'Running Pen': 0.1,
                'Pass Pen': 0.2,
                'Creation': 0.35,
                'Ball Security': 0.15,
            }
        }
    },
    'CB': {
        'aoi_mapping': {
            'Wide Def': [
                'EFx Ground Duels',
                'Defensive duels won, %'
            ],
            'Aggression': [
                'Defensive duels per 90',
                'Sliding tackles per 90',
                'Successful defensive actions per 90'
            ],
            'Deeper Def': [
                'EFx Aerial Duels',
                'Shots blocked per 90',
                'Interceptions per 90'
            ],
            'Carrying': [
                'Progressive runs per 90',
                'Dribbles per 90',
            ],
            'Pass Pen': [
                'Progressive passes per 90',
                'Passes to final third per 90',
                'Accurate forward passes, %',
                'P2P'
            ],
            'Ball Security': [
                'Accurate passes, %',
                'Successful dribbles, %'
            ],
        },
        'roles': {
            'Ball Player': {
                'Aggression': 0.1,
                'Wide Def': 0.1,
                'Deeper Def': 0.05,
                'Pass Pen': 0.3,
                'Ball Security': 0.25,
                'Carrying': 0.2
            },
            'Aggressor': {
                'Aggression': 0.65,
                'Wide Def': 0.2,
                'Deeper Def': 0.2,
                'Pass Pen': 0.0,
                'Ball Security': 0.0,
                'Carrying': 0.0,
            },
            'Protector': {
                'Aggression': 0.2,
                'Wide Def': 0.15,
                'Deeper Def': 0.65,
                'Pass Pen': 0.0,
                'Ball Security': 0.0,
                'Carrying': 0.0,
            }
        }
    },
    'CF': {
        'aoi_mapping': {
            'Goal Generation': [
                'xG per 90',
                'Shots per 90',
            ],
            'Box-Dominance': [
                'Touches in box per 90'
            ],
            'Hold-up': [
                'Received long passes per 90',
                'EFx Aerial Duels'
            ],
            'Link Up': [
                'Received passes per 90',
                'Accurate short / medium passes, %',
                'Smart passes per 90'
            ],
            'Creativity': [
                'xA per 90',
                'Key passes per 90',
                'Passes to penalty area per 90'
            ]
        },
        'roles': {
            'Target Forward': {
                'Goal Generation': 0.2,
                'Hold-up': 0.35,
                'Link Up': 0.15,
                'Creativity': 0.1,
                'Box-Dominance': 0.2
            },
            'Roaming Enabler': {
                'Goal Generation': 0.2,
                'Hold-up': 0.1,
                'Link Up': 0.25,
                'Creativity': 0.35,
                'Box-Dominance': 0.1
            },
            'Box-Dominator': {
                'Goal Generation': 0.35,
                'Hold-up': 0.1,
                'Link Up': 0.1,
                'Creativity': 0.1,
                'Box-Dominance': 0.35
            }
        }
    },
    'WM': {
        'aoi_mapping': {
            'Goal Focus': [
                'xG per 90',
                'Shots per 90',
                'Touches in box per 90'
            ],
            'Takeon': [
                'Dribbles per 90',
                'Successful dribbles, %'
            ],
            'Creativity': [
                'xA per 90',
                'Key passes per 90',
                'Passes to penalty area per 90'
            ],
            'Outlet': [
                'Received long passes per 90',
                'Progressive runs per 90',
                'Accelerations per 90',
                'Fouls suffered per 90',
                'OutletMarker1'
            ],
            'Ball Security': [
                'Accurate passes, %',
            ],
        },
        'roles': {
            'Outlet': {
                'Goal Focus': 0.1,
                'Takeon': 0.1,
                'Creativity': 0.15,
                'Outlet': 0.45,
                'Ball Security': 0.2
            },
            'Creator': {
                'Goal Focus': 0.1,
                'Takeon': 0.1,
                'Creativity': 0.4,
                'Outlet': 0.1,
                'Ball Security': 0.3
            },
            'Goal-Driven': {
                'Goal Focus': 0.4,
                'Takeon': 0.15,
                'Creativity': 0.1,
                'Outlet': 0.2,
                'Ball Security': 0.15
            },
            'One on One': {
                'Goal Focus': 0.0,
                'Takeon': 0.5,
                'Creativity': 0.2,
                'Outlet': 0.2,
                'Ball Security': 0.1
            },
        }
    }
}

def add_custom_metrics(df):
    #Add custom calculated metrics to the dataframe, in-place.
    # EFx Aerial Duels
    df['EFx Aerial Duels'] = (df['Aerial duels per 90'] * df['Aerial duels won, %']) / 100
    # EFx Ground Duels
    df['EFx Ground Duels'] = (df['Defensive duels per 90'] * df['Defensive duels won, %']) / 100
    # EFx Duels - Essentially total duels won.
    df['EFx Duels'] = (df['EFx Ground Duels'] + df['EFx Aerial Duels'])
    # Total Duel %
    df['Total Duel %'] = (df['Aerial duels won, %'] + df['Defensive duels won, %']) / 2
    #Total Duels per 90
    df['Duels Contested'] = df['Aerial duels per 90'] + df['Defensive duels per 90']
    # EFx Prog. Pass
    df['EFx Prog. Pass'] = (df['Progressive passes per 90'] * df['Accurate progressive passes, %']) / 100
    # P2P
    df['P2P'] = df['Progressive passes per 90'] / df['Passes per 90']
    # PPA Share
    df['PPA Share'] = df['Passes to penalty area per 90'] / df['Passes per 90']
    # OutletMarker1
    df['OutletMarker1'] = df['Received passes per 90'] - df['Passes per 90']
    #xG per Shot
    df['xG per Shot'] = df['xG per 90'] / df['Shots per 90']
    return df

def calculate_role_scores(df, position, min_minutes=1000, output_filename=None, role=None):
    """
    Calculate role scores for any position using POSITION_CONFIGS
    
    Parameters:
    df: Main DataFrame with all player statistics
    position: Position key from POSITION_CONFIGS (e.g., 'CM', 'CB', 'FB')
    min_minutes: Minimum minutes played threshold (default: 450)
    output_filename: Optional CSV output filename
    role: Optional role name to filter output columns
    
    Returns:
    DataFrame with role scores and rankings for filtered position, or just for a specific role if requested
    """
    
    # Get configuration for this position
    if position not in POSITION_CONFIGS:
        raise ValueError(f"Position '{position}' not found in POSITION_CONFIGS. Available: {list(POSITION_CONFIGS.keys())}")
    
    config = POSITION_CONFIGS[position]
    aoi_mapping = config['aoi_mapping']
    role_weights = config['roles']
    
    # Use all players from the same position as reference population
    reference_population = df[df['Position'] == position]
    
    # Filter dataframe for specific position
    filtered_df = df[df['Position'] == position]
    
    if len(filtered_df) == 0:
        return None
    
    # Filter by minimum minutes played
    filtered_df = filtered_df[filtered_df['Minutes played'] >= min_minutes].copy()
    
    if len(filtered_df) == 0:
        return None
    
    # Flatten metric list for later use (remove duplicates)
    all_metrics = []
    for metrics_list in aoi_mapping.values():
        all_metrics.extend(metrics_list)
    all_metrics = list(set(all_metrics))  # Remove duplicates
        
    # Use list to extract just a copy of JUST required data
    metrics_df = filtered_df[['Player', 'Team within selected timeframe', 'Minutes played'] + all_metrics].copy()

    # Convert metrics to z-scores AGAINST REFERENCE POPULATION
    for metric in all_metrics:
        ref_mean = reference_population[metric].mean()
        ref_std = reference_population[metric].std()
        metrics_df[f'{metric}_z'] = (metrics_df[metric] - ref_mean) / ref_std
        metrics_df[f'{metric}_z'] = metrics_df[f'{metric}_z'].clip(-3, 3)

    # Calculating AOI scores by accumulating z-scores (sum and geometric mean)
    for aoi, metrics_list in aoi_mapping.items():
        z_metrics = [f'{metric}_z' for metric in metrics_list]
        # Sum (current method)
        metrics_df[f'{aoi}_Z_Sum'] = metrics_df[z_metrics].sum(axis=1)
        # Geometric mean (shifted by +4)
        shifted = metrics_df[z_metrics] + 4
        metrics_df[f'{aoi}_GeoMean'] = shifted.apply(lambda row: gmean(row), axis=1)

    # Calculate role scores by applying weights to AOI z-score sums (sum and geometric mean)
    for role_name, weights in role_weights.items():
        # Sum-based
        metrics_df[f'{role_name}_Z_Score'] = 0
        # Geometric mean-based
        metrics_df[f'{role_name}_GeoMean_Z_Score'] = 0
        for aoi, weight in weights.items():
            metrics_df[f'{role_name}_Z_Score'] += metrics_df[f'{aoi}_Z_Sum'] * weight
            metrics_df[f'{role_name}_GeoMean_Z_Score'] += metrics_df[f'{aoi}_GeoMean'] * weight

    # Calculate role scores for REFERENCE POPULATION to get proper min/max (for both sum and geometric mean)
    ref_metrics_df = reference_population[['Player', 'Team within selected timeframe', 'Minutes played'] + all_metrics].copy()
    for metric in all_metrics:
        ref_mean = reference_population[metric].mean()
        ref_std = reference_population[metric].std()
        ref_metrics_df[f'{metric}_z'] = (ref_metrics_df[metric] - ref_mean) / ref_std
        ref_metrics_df[f'{metric}_z'] = ref_metrics_df[f'{metric}_z'].clip(-3, 3)
    for aoi, metrics_list in aoi_mapping.items():
        z_metrics = [f'{metric}_z' for metric in metrics_list]
        ref_metrics_df[f'{aoi}_Z_Sum'] = ref_metrics_df[z_metrics].sum(axis=1)
        shifted = ref_metrics_df[z_metrics] + 4
        ref_metrics_df[f'{aoi}_GeoMean'] = shifted.apply(lambda row: gmean(row), axis=1)
    for role_name, weights in role_weights.items():
        ref_metrics_df[f'{role_name}_Z_Score'] = 0
        ref_metrics_df[f'{role_name}_GeoMean_Z_Score'] = 0
        for aoi, weight in weights.items():
            ref_metrics_df[f'{role_name}_Z_Score'] += ref_metrics_df[f'{aoi}_Z_Sum'] * weight
            ref_metrics_df[f'{role_name}_GeoMean_Z_Score'] += ref_metrics_df[f'{aoi}_GeoMean'] * weight
    for role_name in role_weights.keys():
        # Sum-based fit
        ref_min_z = ref_metrics_df[f'{role_name}_Z_Score'].min()
        ref_max_z = ref_metrics_df[f'{role_name}_Z_Score'].max()
        metrics_df[f'{role_name}_Fit'] = 100 * (metrics_df[f'{role_name}_Z_Score'] - ref_min_z) / (ref_max_z - ref_min_z)
        # GeoMean-based fit
        ref_min_g = ref_metrics_df[f'{role_name}_GeoMean_Z_Score'].min()
        ref_max_g = ref_metrics_df[f'{role_name}_GeoMean_Z_Score'].max()
        metrics_df[f'{role_name}_GeoMean_Fit'] = 100 * (metrics_df[f'{role_name}_GeoMean_Z_Score'] - ref_min_g) / (ref_max_g - ref_min_g)
    roles = list(role_weights.keys())
    for role_name in roles:
        metrics_df[f'{role_name}_Rank'] = metrics_df[f'{role_name}_Fit'].rank(ascending=False)
        metrics_df[f'{role_name}_GeoMean_Rank'] = metrics_df[f'{role_name}_GeoMean_Fit'].rank(ascending=False)
    metrics_df['Best_Role_Fit'] = metrics_df[[f'{role_name}_Fit' for role_name in roles]].idxmax(axis=1)
    metrics_df['Best_Role_Fit'] = metrics_df['Best_Role_Fit'].str.replace('_Fit', '')
    metrics_df['Best_Role_GeoMean_Fit'] = metrics_df[[f'{role_name}_GeoMean_Fit' for role_name in roles]].idxmax(axis=1)
    metrics_df['Best_Role_GeoMean_Fit'] = metrics_df['Best_Role_GeoMean_Fit'].str.replace('_GeoMean_Fit', '').str.replace('_Fit', '')

    # If a specific role is requested, filter columns
    if role is not None:
        if role not in role_weights:
            raise ValueError(f"Role '{role}' not found for position '{position}'. Available: {list(role_weights.keys())}")
        base_cols = ['Player', 'Team within selected timeframe', 'Minutes played']
        # Always include Best_Role_Fit and Best_Role_GeoMean_Fit if present
        extra_cols = []
        for col in ['Best_Role_Fit', 'Best_Role_GeoMean_Fit']:
            if col in metrics_df.columns:
                extra_cols.append(col)
        role_cols = [
            f'{role}_Z_Score', f'{role}_Fit', f'{role}_Rank',
            f'{role}_GeoMean_Z_Score', f'{role}_GeoMean_Fit', f'{role}_GeoMean_Rank'
        ]
        aoi_cols = []
        for aoi in role_weights[role].keys():
            aoi_cols += [f'{aoi}_Z_Sum', f'{aoi}_GeoMean']
        cols_to_return = base_cols + extra_cols + aoi_cols + role_cols
        cols_to_return = [col for col in cols_to_return if col in metrics_df.columns]
        return metrics_df[cols_to_return]
    return metrics_df 