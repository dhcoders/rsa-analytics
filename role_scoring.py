import pandas as pd
import numpy as np
from scipy.stats import gmean

# Position configuration dictionary
POSITION_CONFIGS = {
    'CM': {
        'aoi_mapping': {
            'Duel Volume': {
                'EFx Aerial Duels': 0.5,
                'EFx Ground Duels': 0.5
            },
            'Ball Winning': {
                'Interceptions per 90': 0.4,
                'Successful defensive actions per 90': 0.4,
                'Total Duel %': 0.2
            },
            'Mobility': {
                'Progressive runs per 90': 0.4,
                'Accelerations per 90': 0.3,
                'Dribbles per 90': 0.3
            },
            'Deep Pen': {
                'EFx Prog. Pass': 0.7,
                'Passes to final third per 90': 0.3
            },
            'Prog. Tendency': {
                'P2P': 0.7, #Rate of passes per progressive pass made.
                'PPA Share': 0.3 #Rate of passes to PA compared to total passes.
            },
            'Adv Pen': {
                'Passes to penalty area per 90': 1
            },
            'Creation': {
                'xA per 90': 0.6,
                'Key passes per 90': 0.4
            },
            'Ball Security': {
                'Accurate passes, %': 0.8,
                'Successful dribbles, %': 0.2
            },
            'Box-Crashing': {
                'BoxTouchPerc': 0.6,
                'xG per Shot': 0.4
            }
        },
        'roles': {
            'Ball-Winner': {
                'Duel Volume': 0.4,
                'Ball Winning': 0.4,
                'Ball Security': 0.2,
                'Mobility': 0.0,
                'Deep Pen': 0.0,
                'Creation': 0.0,
                'Box-Crashing': 0.0,
                'Adv Pen': 0.0,
                'Prog. Tendency': 0.0
            },
            'Midfield Conductor': { #Player who is safe with the ball and distributes forwards - they conduct their teams play.
                'Duel Volume': 0.0,
                'Ball Winning': 0.0,
                'Ball Security': 0.6,
                'Mobility': 0.0,
                'Deep Pen': 0.3,
                'Creation': 0.0,
                'Box-Crashing': 0.0,
                'Adv Pen': 0.0,
                'Prog. Tendency': 0.1
            },
            'Box-Crasher': { #This is a player who is good at getting into the box and scoring.
                'Duel Volume': 0.2,
                'Ball Winning': 0.2,
                'Ball Security': 0.0,
                'Mobility': 0.0,
                'Deep Pen': 0.0,
                'Creation': 0.0,
                'Box-Crashing': 0.6,
                'Adv Pen': 0.0,
                'Prog. Tendency': 0.0
            },
            'Creative #10': {
                'Duel Volume': 0.00,
                'Ball Winning': 0.0,
                'Ball Security': 0.0,
                'Mobility': 0.1,
                'Deep Pen': 0.1,
                'Creation': 0.5,
                'Box-Crashing': 0.0,
                'Adv Pen': 0.3,
                'Prog. Tendency': 0.0
            },
            'All-out Progressor': {
                'Duel Volume': 0.0,
                'Ball Winning': 0.0,
                'Ball Security': 0.0,
                'Mobility': 0.0,
                'Deep Pen': 0.4,
                'Creation': 0.0,
                'Box-Crashing': 0.0,
                'Adv Pen': 0.2,
                'Prog. Tendency': 0.4
            }
        }
    },
    'FB': {
        'aoi_mapping': {
            'Winger Lockdown': {
                'Defensive duels won, %': 0.7,
                'EFx Ground Duels': 0.3
            },
            'Defending': {
                'Interceptions per 90': 0.33,
                'Successful defensive actions per 90': 0.33,
                'Aerial duels won, %': 0.34
            },
            'Running Pen': {
                'Progressive runs per 90': 0.33,
                'Accelerations per 90': 0.33,
                'Dribbles per 90': 0.34
            },
            'Pass Pen': {
                'Progressive passes per 90': 0.55,
                'Passes to final third per 90': 0.45,
            },
            'Creation': {
                'xA per 90': 0.5,
                'Key passes per 90': 0.4,
                'Passes to penalty area per 90': 0.1
            },
            'Ball Security': {
                'Accurate passes, %': 0.5,
                'Successful dribbles, %': 0.5
            },
            'Crossing': {
                'Acc. Crosses': 0.3,
                'Deep completed crosses per 90': 0.1,
                'Cross Tendency': 0.6
            }
        },
        'roles': {
            'Penetrator': {
                'Winger Lockdown': 0.1,
                'Defending': 0.1,
                'Running Pen': 0.0,
                'Pass Pen': 0.5,
                'Creation': 0.2,
                'Ball Security': 0.1,
                'Crossing': 0.0,
            },
            'Runner': {
                'Winger Lockdown': 0.1,
                'Defending': 0.1,
                'Running Pen': 0.5,
                'Pass Pen': 0.0,
                'Creation': 0.2,
                'Ball Security': 0.1,
                'Crossing': 0.0,
            },
            'Padlock': {
                'Winger Lockdown': 0.5,
                'Defending': 0.4,
                'Running Pen': 0.0,
                'Pass Pen': 0.0,
                'Creation': 0.0,
                'Ball Security': 0.1,
                'Crossing': 0.0,
            },
            'General Creator': {
                'Winger Lockdown': 0.0,
                'Defending': 0.0,
                'Running Pen': 0.0,
                'Pass Pen': 0.3,
                'Creation': 0.6,
                'Ball Security': 0.0,
                'Crossing': 0.1,
            },
            'Cross Whipper': {
                'Winger Lockdown': 0.0,
                'Defending': 0.0,
                'Running Pen': 0.0,
                'Pass Pen': 0.0,
                'Creation': 0.2,
                'Ball Security': 0.0,
                'Crossing': 0.8,
            },
            'Reliable Retainer': {
                'Winger Lockdown': 0.1,
                'Defending': 0.2,
                'Running Pen': 0.0,
                'Pass Pen': 0.0,
                'Creation': 0.0,
                'Ball Security': 0.7,
                'Crossing': 0.0,
            }
        }
    },
    'IP: CB': {
        'aoi_mapping': {
            'Dribble Security': {
                'Successful dribbles, %': 1
            },
            'Carrying': {
                'Progressive runs per 90': 0.5,
                'PrgCarries_ToRec': 0.5 #More accurate measure of tendency to carry the ball forward - possession independent.
            },
            'Pass Pen': {
                'EFx Prog. Pass': 0.35,
                'P2P': 0.65 #More accurate measure of the tendency to progress ball via pass - possession independent.
            },
            'Pass Security': {
                'Accurate passes, %': 1
            }
        },
        'roles': {
            'Penetrative Ball Player': {
                'Pass Pen': 0.5,
                'Carrying': 0.5,
                'Dribble Security': 0,
                'Pass Security': 0.0
            },
            'Fwd Carrier': {
                'Pass Pen': 0.0,
                'Carrying': 0.9,
                'Dribble Security': 0.1,
                'Pass Security': 0.0
            },
            'Fwd Passer': {
                'Pass Pen': 0.9,
                'Carrying': 0.0,
                'Dribble Security': 0.0,
                'Pass Security': 0.1
            },
            'Safe Ball Player': {
                'Pass Pen': 0.0,
                'Carrying': 0.0,
                'Dribble Security': 0.5,
                'Pass Security': 0.5
            }
        }
    },
    'OOP: CB': {
        'aoi_mapping': {
            'Gr. Duel Vol.': {
                'EFx Ground Duels': 1
            },
            'Aerial Duel Vol.': {
                'EFx Aerial Duels': 1
            },
            'Gr. Duel Success': {
                'Defensive duels won, %': 1
            },
            'Aerial Duel Success': {
                'Aerial duels won, %': 1
            },
            'Aggression': {
                'Sliding tackles per 90': 0.33,
                'Successful defensive actions per 90': 0.34,
                'Fouls per 90': 0.33
            },
            'Box Def': {
                'Shots blocked per 90': 0.5,
                'Interceptions per 90': 0.5
            }
        },
        'roles': {
            'Aggressive Dueler': { #Prioritises volume of duels, not success. Flies into lots of duels.
                'Aggression': 0.2,
                'Box Def': 0.0,
                'Gr. Duel Vol.': 0.4,
                'Aerial Duel Vol.': 0.4,
                'Gr. Duel Success': 0.0,
                'Aerial Duel Success': 0.0,
            },
            'Box Protector': { #Prioritises metrics focused on deeper defensive action volume.
                'Aggression': 0.0,
                'Box Def': 0.34,
                'Gr. Duel Vol.': 0.0,
                'Aerial Duel Vol.': 0.33,
                'Gr. Duel Success': 0.0,
                'Aerial Duel Success': 0.33,
            },
            'Passive Dueler': { #Prioritises success of duels, not volume. Wins duels, but doesn't fly into them.
                'Aggression': 0.0,
                'Box Def': 0.0,
                'Gr. Duel Vol.': 0.0,
                'Aerial Duel Vol.': 0.0,
                'Gr. Duel Success': 0.5,
                'Aerial Duel Success': 0.5,
            },
            'Wide Defender': { #Prioritises ground duelling success and volume, key for WCBs.
                'Aggression': 0.0,
                'Box Def': 0.0,
                'Gr. Duel Vol.': 0.5,
                'Aerial Duel Vol.': 0.0,
                'Gr. Duel Success': 0.5,
                'Aerial Duel Success': 0.0,
            },
            'Duel Dominator': { #Combines high volume and successful duelling.
                'Aggression': 0.0,
                'Box Def': 0.0,
                'Gr. Duel Vol.': 0.3,
                'Aerial Duel Vol.': 0.3,
                'Gr. Duel Success': 0.2,
                'Aerial Duel Success': 0.2,
            }
        }
    },
    'CF': {
        'aoi_mapping': {
            'Goal Generation': {
                'xG per 90': 0.5, #This is our proxy for the quality of chance the player is involved in
                'Shots per 90': 0.5 #This is our proxy for the quantity of chances the player is able to generate.
            },
            'Box-Dominance': { 
                'BoxTouchPerc': 0.34,
                'xG per Shot': 0.33, #Idea is that shots inside the box carry more xG than shots outside the box, so higher avg xG per shot is better.
                'Head goals per 90': 0.33 # You can't header outside the box
            },
            'Hold-up': { #weighted to favour players who get into more aerial duels - we are thinking of this as a "target forward"-focused AOI
                'Received long passes per 90': 0.3,
                'EFx Aerial Duels': 0.5,
                'Offensive duels per 90': 0.2 #Off. duels are dribbles, but also just when players hold off other players whilst in poss.
            },
            'Link Up': {
                'Received passes per 90': 0.33, #Not ideal as a measure, but more rec. passes = dropping in and linking up more.
                'Accurate short / medium passes, %': 0.33, #indicates reliability of passes.
                'Smart passes per 90': 0.34 #defined as a creative and penetrative pass that aims to gain significant advantage, seems ideal for link-up.
            },
            'Creativity': {
                'xA per 90': 0.34,
                'Key passes per 90': 0.33,
                'Passes to penalty area per 90': 0.33
            },
            'Ball Carrier': {
                'Progressive runs per 90': 0.7, #how often do they carry the ball forward?
                'Dribbles per 90': 0.3 #is the player someone who likes to carry?
            }
        },
        'roles': {
            'Target Forward': {
                'Goal Generation': 0.3,
                'Hold-up': 0.4,
                'Link Up': 0.1,
                'Creativity': 0.0,
                'Box-Dominance': 0.2,
                'Ball Carrier': 0.0
            },
            'Roaming Enabler': {
                'Goal Generation': 0.0,
                'Hold-up': 0.0,
                'Link Up': 0.35,
                'Creativity': 0.45,
                'Box-Dominance': 0.0, #Best work done outside the box.
                'Ball Carrier': 0.2 #we want to avoid players who are too static - they need to ROAM.
            },
            'Box Dominator': {
                'Goal Generation': 0.5,
                'Hold-up': 0,
                'Link Up': 0,
                'Creativity': 0,
                'Box-Dominance': 0.5,
                'Ball Carrier': 0.0
            },
            'Channel Runner': {
                'Goal Generation': 0,
                'Hold-up': 0.1,
                'Link Up': 0.3,
                'Creativity': 0.1,
                'Box-Dominance': 0.0,
                'Ball Carrier': 0.5 #big focus on ball carrying. 
            }
        }
    },
    'WM': {
        'aoi_mapping': {
            'Goal Focus': {
                'xG per 90': 0.33,
                'Shots per 90': 0.34,
                'Touches in box per 90': 0.33
            },
            'Takeon': {
                'Dribbles per 90': 0.7,
                'Successful dribbles, %': 0.3
            },
            'Creativity': {
                'xA per 90': 0.34,
                'Key passes per 90': 0.33,
                'Passes to penalty area per 90': 0.33
            },
            'Outlet': { #Winger who gains the team yards.
                'Received long passes per 90': 0.1, #How often are they getting high to receive?
                'Progressive runs per 90': 0.6, #Biggest indicator of yards gained via an outlet
                'Fouls suffered per 90': 0.1, #Part of gaining yards is winnning fouls
                'OutletMarker1': 0.2 #Idea is that they receive more passes than they make.
            },
            'Ball Security': {
                'Accurate passes, %': 1.0
            },
            'Crossing': {
                'Acc. Crosses': 0.5, #How often is the player finding a teammate with a cross?
                'Cross Tendency': 0.5 #How often is the player trying to cross proportionally to the number of passes they make?
            }
        },
        'roles': {
            'Outlet': {
                'Goal Focus': 0.0,
                'Takeon': 0.2,
                'Creativity': 0.0,
                'Outlet': 0.6,
                'Ball Security': 0.2,
                'Crossing': 0.0
            },
            'Creator': {
                'Goal Focus': 0.0,
                'Takeon': 0.0,
                'Creativity': 0.7,
                'Outlet': 0.0,
                'Ball Security': 0.0,
                'Crossing': 0.3
            },
            'Goal-Driven': {
                'Goal Focus': 0.8,
                'Takeon': 0.1,
                'Creativity': 0.0,
                'Outlet': 0.1,
                'Ball Security': 0.0,
                'Crossing': 0.0
            },
            'One on One': {
                'Goal Focus': 0.0,
                'Takeon': 0.8,
                'Creativity': 0.1,
                'Outlet': 0.1,
                'Ball Security': 0.0,
                'Crossing': 0.0
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
    # Passes to Progressions
    df['P2P'] = df['Progressive passes per 90'] / df['Passes per 90']
    # Passes to Pen Area Penetrations
    df['PPA Share'] = df['Passes to penalty area per 90'] / df['Passes per 90']
    #Prog Carries to Passes Received
    df['PrgCarries_ToRec'] = df['Progressive runs per 90'] / df['Received passes per 90']
    # OutletMarker1
    df['OutletMarker1'] = df['Received passes per 90'] - df['Passes per 90']
    #xG per Shot
    df['xG per Shot'] = df['xG per 90'] / df['Shots per 90']
    #Accurate crosses per 90
    df['Acc. Crosses'] = df['Crosses per 90'] * df['Accurate crosses, %'] / 100
    #BoxCrashingMetric
    df['BoxTouchPerc'] = df['Touches in box per 90'] / df['Received passes per 90']
    #Cross Tendency
    df['Cross Tendency'] = df['Crosses per 90'] / df['Passes per 90']

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
    
    # Handle CB variants - both map to CB data
    data_position = 'CB' if position in ['IP: CB', 'OOP: CB'] else position
    
    # Use all players from the same position as reference population
    reference_population = df[df['Position'] == data_position]
    
    # Filter dataframe for specific position
    filtered_df = df[df['Position'] == data_position]
    
    if len(filtered_df) == 0:
        return None
    
    # Filter by minimum minutes played
    filtered_df = filtered_df[filtered_df['Minutes played'] >= min_minutes].copy()
    
    if len(filtered_df) == 0:
        return None
    
    # Flatten metric list for later use (remove duplicates)
    all_metrics = []
    for metrics_config in aoi_mapping.values():
        if isinstance(metrics_config, dict):
            all_metrics.extend(metrics_config.keys())
        else:
            # Backward compatibility for lists
            all_metrics.extend(metrics_config)
    all_metrics = list(set(all_metrics))  # Remove duplicates
        
    # Use list to extract just a copy of JUST required data
    metrics_df = filtered_df[['Player', 'Team within selected timeframe', 'Minutes played'] + all_metrics].copy()

    # Convert metrics to z-scores AGAINST REFERENCE POPULATION
    for metric in all_metrics:
        ref_mean = reference_population[metric].mean()
        ref_std = reference_population[metric].std()
        metrics_df[f'{metric}_z'] = (metrics_df[metric] - ref_mean) / ref_std
        metrics_df[f'{metric}_z'] = metrics_df[f'{metric}_z'].clip(-3, 3)

    # Calculating AOI scores by accumulating z-scores
    for aoi, metrics_config in aoi_mapping.items():
        # Calculate weighted sum
        weighted_sum = 0
        
        for metric, weight in metrics_config.items():
            metric_z = f'{metric}_z'
            if metric_z in metrics_df.columns:
                # Weighted sum
                weighted_sum += metrics_df[metric_z] * weight
        
        metrics_df[f'{aoi}_Z_Sum'] = weighted_sum

    # Calculate role scores by applying weights to AOI z-score sums
    for role_name, weights in role_weights.items():
        metrics_df[f'{role_name}_Z_Score'] = 0
        for aoi, weight in weights.items():
            metrics_df[f'{role_name}_Z_Score'] += metrics_df[f'{aoi}_Z_Sum'] * weight

    # Calculate role scores for REFERENCE POPULATION to get proper min/max
    ref_metrics_df = reference_population[['Player', 'Team within selected timeframe', 'Minutes played'] + all_metrics].copy()
    for metric in all_metrics:
        ref_mean = reference_population[metric].mean()
        ref_std = reference_population[metric].std()
        ref_metrics_df[f'{metric}_z'] = (ref_metrics_df[metric] - ref_mean) / ref_std
        ref_metrics_df[f'{metric}_z'] = ref_metrics_df[f'{metric}_z'].clip(-3, 3)
    
    for aoi, metrics_config in aoi_mapping.items():
        # Calculate weighted sum
        weighted_sum = 0
        
        for metric, weight in metrics_config.items():
            metric_z = f'{metric}_z'
            if metric_z in ref_metrics_df.columns:
                # Weighted sum
                weighted_sum += ref_metrics_df[metric_z] * weight
        
        ref_metrics_df[f'{aoi}_Z_Sum'] = weighted_sum
    
    for role_name, weights in role_weights.items():
        ref_metrics_df[f'{role_name}_Z_Score'] = 0
        for aoi, weight in weights.items():
            ref_metrics_df[f'{role_name}_Z_Score'] += ref_metrics_df[f'{aoi}_Z_Sum'] * weight
    for role_name in role_weights.keys():
        # Calculate role fit scores (0-100 scale)
        ref_min_z = ref_metrics_df[f'{role_name}_Z_Score'].min()
        ref_max_z = ref_metrics_df[f'{role_name}_Z_Score'].max()
        metrics_df[f'{role_name}_Fit'] = 100 * (metrics_df[f'{role_name}_Z_Score'] - ref_min_z) / (ref_max_z - ref_min_z)
    
    roles = list(role_weights.keys())
    for role_name in roles:
        metrics_df[f'{role_name}_Rank'] = metrics_df[f'{role_name}_Fit'].rank(ascending=False)
    metrics_df['Best_Role_Fit'] = metrics_df[[f'{role_name}_Fit' for role_name in roles]].idxmax(axis=1)
    metrics_df['Best_Role_Fit'] = metrics_df['Best_Role_Fit'].str.replace('_Fit', '')

    # If a specific role is requested, filter columns
    if role is not None:
        if role not in role_weights:
            raise ValueError(f"Role '{role}' not found for position '{position}'. Available: {list(role_weights.keys())}")
        base_cols = ['Player', 'Team within selected timeframe', 'Minutes played']
        # Always include Best_Role_Fit if present
        extra_cols = []
        for col in ['Best_Role_Fit']:
            if col in metrics_df.columns:
                extra_cols.append(col)
        role_cols = [
            f'{role}_Z_Score', f'{role}_Fit', f'{role}_Rank'
        ]
        aoi_cols = []
        for aoi in role_weights[role].keys():
            aoi_cols += [f'{aoi}_Z_Sum']
        cols_to_return = base_cols + extra_cols + aoi_cols + role_cols
        cols_to_return = [col for col in cols_to_return if col in metrics_df.columns]
        return metrics_df[cols_to_return]
    return metrics_df 