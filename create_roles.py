import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import plotly.express as px
from role_scoring import POSITION_CONFIGS, calculate_role_scores, add_custom_metrics

# Set page config
st.set_page_config(page_title="Role Score Analysis", layout="wide")

# Title and description
st.title("Adv. Role Analysis & Custom Role Builder")
st.markdown("Upload your own data to analyze with predefined roles OR create completely custom roles for your analysis.")

# Color coding legend
with st.expander("🎨 Score Color Guide", expanded=False):
    st.markdown("""
    **Role scores are color-coded to help you quickly assess performance levels:**
    
    🟢 **Excellent (80-100)**: Outstanding performance in this role
    🟢 **Very Good (70-79)**: Strong performance, clearly above average
    🟢 **Good (60-69)**: Solid performance, better than most
    🟡 **Average (50-59)**: League average performance
    🟠 **Below Average (40-49)**: Room for improvement
    🔴 **Poor (30-39)**: Significant weaknesses in this role
    🔴 **Very Poor (0-29)**: Critical deficiencies, major improvement needed
    
    *Note: Scores are relative to the dataset being analyzed and normalized to a 0-100 scale.*
    """)

def style_role_scores(df, score_columns):
    """
    Apply color styling to role score columns based on score ranges.
    
    Parameters:
    df: DataFrame containing the data
    score_columns: List of column names that contain role scores (should be 0-100 scale)
    
    Returns:
    Styled DataFrame for display
    """
    def color_score(val):
        """Apply color based on score value (0-100 scale)"""
        if pd.isna(val):
            return ''
        
        # Convert to numeric if possible, otherwise return no styling
        try:
            val = float(val)
        except (ValueError, TypeError):
            return ''
        
        # Define color thresholds
        if val >= 80:
            # Excellent (dark green)
            return 'background-color: #1e7e34; color: white; font-weight: bold'
        elif val >= 70:
            # Very good (medium green)
            return 'background-color: #28a745; color: white; font-weight: bold'
        elif val >= 60:
            # Good (light green)
            return 'background-color: #6f9a37; color: white'
        elif val >= 50:
            # Average (yellow)
            return 'background-color: #ffc107; color: black'
        elif val >= 40:
            # Below average (orange)
            return 'background-color: #fd7e14; color: white'
        elif val >= 30:
            # Poor (light red)
            return 'background-color: #e74c3c; color: white'
        else:
            # Very poor (dark red)
            return 'background-color: #c82333; color: white; font-weight: bold'
    
    # Create a styler object
    styler = df.style
    
    # Apply styling to score columns only - filter to numeric columns
    for col in score_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            styler = styler.applymap(color_score, subset=[col])
    
    # Format score columns to 1 decimal place - only numeric columns
    format_dict = {col: '{:.1f}' for col in score_columns 
                   if col in df.columns and pd.api.types.is_numeric_dtype(df[col])}
    if format_dict:
        styler = styler.format(format_dict)
    
    return styler

# Universal metric groupings for custom roles
METRIC_GROUPS = {
    'Ball Carrying': [
        'Progressive runs per 90', 'Dribbles per 90', 'Accelerations per 90', 'Touches in box per 90', 'OutletMarker1',
        'Received passes per 90', 'Received long passes per 90', 'Fouls suffered per 90', 'Successful dribbles, %',
        'Ball losses per 90', 'Dispossessed per 90', 'Turnovers per 90', 'Ball recoveries per 90', 'Ball Security',
    ],
    'Passing': [
        'Passes per 90', 'Accurate passes, %', 'Forward passes per 90', 'Accurate forward passes, %',
        'Back passes per 90', 'Accurate back passes, %', 'Lateral passes per 90', 'Accurate lateral passes, %',
        'Short / medium passes per 90', 'Accurate short / medium passes, %', 'Long passes per 90', 'Accurate long passes, %',
        'Average pass length, m', 'Average long pass length, m', 'Progressive passes per 90', 'Accurate progressive passes, %',
        'Passes to final third per 90', 'Accurate passes to final third, %', 'Passes to penalty area per 90', 'Accurate passes to penalty area, %',
        'Through passes per 90', 'Accurate through passes, %', 'Deep completions per 90', 'Deep completed crosses per 90',
        'Smart passes per 90', 'Accurate smart passes, %', 'Key passes per 90', 'P2P', 'PPA Share', 'EFx Prog. Pass',
    ],
    'Duelling & Defending': [
        'Defensive duels per 90', 'Defensive duels won, %', 'Aerial duels per 90', 'Aerial duels won, %',
        'Offensive duels per 90', 'Offensive duels won, %', 'EFx Aerial Duels', 'EFx Ground Duels', 'EFx Duels', 'Total Duel %', 'Duels Contested',
        'Interceptions per 90', 'Successful defensive actions per 90', 'Shots blocked per 90', 'Sliding tackles per 90',
        'PAdj Sliding tackles', 'PAdj Interceptions',
    ],
    'Creating': [
        'xA per 90', 'Second assists per 90', 'Third assists per 90', 'Key passes per 90', 'Smart passes per 90',
        'Crosses per 90', 'Accurate crosses, %', 'Crosses from left flank per 90', 'Accurate crosses from left flank, %',
        'Crosses from right flank per 90', 'Accurate crosses from right flank, %', 'Crosses to goalie box per 90',
        'Deep completions per 90', 'Deep completed crosses per 90',
    ],
    'Shooting': [
        'Goals', 'Goals per 90', 'Non-penalty goals', 'Non-penalty goals per 90', 'xG', 'xG per 90', 'Shots', 'Shots per 90',
        'Shots on target, %', 'Goal conversion, %', 'Head goals', 'Head goals per 90', 'Penalties taken', 'Penalty conversion, %',
        'Successful attacking actions per 90',
    ],
}

# Custom role management functions
def get_role_dir(position):
    dir_path = os.path.join('saved_roles', position)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def list_saved_roles(position):
    dir_path = get_role_dir(position)
    if not os.path.exists(dir_path):
        return []
    return [f[:-5] for f in os.listdir(dir_path) if f.endswith('.json')]

def save_role(position, role_name, aspects, aspect_weights):
    dir_path = get_role_dir(position)
    data = {
        'position': position,
        'aspects': aspects,
        'aspect_weights': aspect_weights
    }
    with open(os.path.join(dir_path, f'{role_name}.json'), 'w') as f:
        json.dump(data, f)

def load_role(position, role_name):
    dir_path = get_role_dir(position)
    with open(os.path.join(dir_path, f'{role_name}.json'), 'r') as f:
        data = json.load(f)
    return data['aspects'], data['aspect_weights']

def calculate_custom_role_scores(df, position, aspects, aspect_weights):
    """Calculate custom role scores based on user-defined aspects and weights"""
    position_df = df[df['Position'] == position].copy()
    
    if len(position_df) == 0:
        return None
        
    metrics_df = position_df[['Player', 'Team within selected timeframe', 'Minutes played']].copy()
    aspect_scores = {}
    
    for asp in aspects:
        z_cols = []
        for metric in asp['metrics']:
            if metric in position_df.columns:
                mean = position_df[metric].mean()
                std = position_df[metric].std()
                if std > 0:
                    z = ((position_df[metric] - mean) / std).clip(-3, 3)
                else:
                    z = pd.Series([0] * len(position_df), index=position_df.index)
                metrics_df[f"{asp['name']}_z_{metric}"] = z
                z_cols.append(f"{asp['name']}_z_{metric}")
        
        if z_cols:
            metrics_df[asp['name']] = metrics_df[z_cols].sum(axis=1)
            aspect_scores[asp['name']] = aspect_weights[asp['name']] / 100
    
    # Calculate overall custom role score
    if aspect_scores:
        metrics_df['Custom_Role_Score'] = sum(metrics_df[asp] * aspect_scores[asp] for asp in aspect_scores)
        
        # Min-max normalization to 0-100
        min_score = metrics_df['Custom_Role_Score'].min()
        max_score = metrics_df['Custom_Role_Score'].max()
        if max_score > min_score:
            metrics_df['Custom_Role_Fit'] = 100 * (metrics_df['Custom_Role_Score'] - min_score) / (max_score - min_score)
        else:
            metrics_df['Custom_Role_Fit'] = 50
    
    return metrics_df

# Data processing function
def process_wyscout_data(df, minimum_minutes=1000):
    """Process raw Wyscout data with position mapping and data cleaning"""
    
    # Initial data cleaning
    df['Position'] = df['Position'].str.split(',').str[0].str.strip()
    
    # Position mapping
    position_mapping = {
        'GK': ['GK'],
        'CB': ['CB', 'LCB', 'RCB'],  # Will be split into IP and OOP versions below
        'FB': ['LB', 'RB', 'RWB', 'LWB'],
        'CM': ['DMF', 'LCMF', 'RCMF', 'LDMF', 'RDMF', 'AMF'],
        'WM': ['LW', 'RW', 'RWF', 'LWF', 'RAMF', 'LAMF'],
        'CF': ['CF']
    }
    
    # Create reverse mapping
    reverse_mapping = {}
    for key, values in position_mapping.items():
        for value in values:
            reverse_mapping[value] = key
    
    # Apply mapping
    df['Position'] = df['Position'].apply(lambda x: reverse_mapping.get(x, x) if isinstance(x, str) else x)
    
    # Filter for minimum minutes
    df = df[df['Minutes played'] >= minimum_minutes]
    
    # Fill null values
    df = df.fillna(0)
    
    return df

# Load and preprocess data
@st.cache_data(show_spinner="Loading data...")
def load_data(minimum_minutes=1000):
    df = pd.read_excel('Data/PL Data Wyscout.xlsx')
    return process_wyscout_data(df, minimum_minutes)

# Possession adjustment functions
@st.cache_data
def load_possession_data():
    """Load possession data from FBRef CSV"""
    try:
        possession_df = pd.read_csv('FBREF_avgposs - Sheet1.csv')
        # Create a lookup dictionary for PL teams
        pl_possession = possession_df[possession_df['comp'] == 'PL'].set_index('squad')['avgpossessionperc'].to_dict()
        return pl_possession
    except FileNotFoundError:
        st.error("Possession data file not found. Possession adjustment disabled.")
        return {}
    except Exception as e:
        st.error(f"Error loading possession data: {str(e)}")
        return {}

def apply_possession_adjustment(df, possession_data):
    """Apply possession adjustment to defensive counting metrics"""
    if not possession_data:
        return df
    
    # Define defensive metrics that should be possession adjusted (counting stats, not percentages)
    defensive_metrics = [
        'Interceptions per 90',
        'Defensive duels per 90', 
        'Aerial duels per 90',
        'Successful defensive actions per 90',
        'Shots blocked per 90',
        'Sliding tackles per 90',
        'EFx Aerial Duels',  # Custom metric
        'EFx Ground Duels',  # Custom metric  
        'EFx Duels',  # Custom metric
        'Duels Contested',  # Custom metric
        'PAdj Sliding tackles',  # Already PAdj, skip
        'PAdj Interceptions'  # Already PAdj, skip
    ]
    
    # Filter to only metrics that exist in the data and aren't already PAdj
    available_defensive_metrics = [
        metric for metric in defensive_metrics 
        if metric in df.columns and not metric.startswith('PAdj')
    ]
    
    # Show which metrics are being adjusted
    if available_defensive_metrics:
        st.write(f"📊 **Normalizing {len(available_defensive_metrics)} defensive metrics** to 50% possession baseline")
        with st.expander("📋 View adjusted metrics", expanded=False):
            st.write(", ".join(available_defensive_metrics))
    
    df_adjusted = df.copy()
    
    # Apply possession adjustment team by team and show examples
    adjustment_examples = []
    for team in df['Team within selected timeframe'].unique():
        if team in possession_data:
            team_possession = possession_data[team]
            defending_time = 100 - team_possession  # Time spent defending
            adjustment_factor = 50 / defending_time  # Normalize to 50% defending time
            
            # Store example for display
            if len(adjustment_examples) < 3:  # Show first 3 teams as examples
                adjustment_examples.append({
                    'Team': team,
                    'Possession': f"{team_possession}%",
                    'Defending Time': f"{defending_time:.1f}%",
                    'Adjustment': f"×{adjustment_factor:.2f}",
                    'Effect': "↗️ Boosted" if adjustment_factor > 1 else "↘️ Reduced" if adjustment_factor < 1 else "→ Neutral"
                })
            
            # Apply adjustment to all defensive metrics for this team
            team_mask = df_adjusted['Team within selected timeframe'] == team
            for metric in available_defensive_metrics:
                df_adjusted.loc[team_mask, metric] = df_adjusted.loc[team_mask, metric] * adjustment_factor
    
    # Show adjustment examples
    if adjustment_examples:
        with st.expander("🔧 Possession adjustment examples", expanded=False):
            st.markdown("**Logic:** Normalize all teams as if they defended 50% of the time")
            example_df = pd.DataFrame(adjustment_examples)
            st.dataframe(example_df, use_container_width=True, hide_index=True)
    
    return df_adjusted

# Data import section
st.header("📁 Data Import")
st.markdown("Upload your Wyscout export (Excel or CSV format)")

# Add minimum minutes filter control
st.subheader("⚙️ Data Filters")
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    min_minutes = st.number_input(
        "Minimum Minutes Played",
        min_value=0,
        max_value=5000,
        value=1000,
        step=50,
        help="Players with fewer than this many minutes will be excluded from analysis"
    )
with col2:
    possession_adjusted = st.checkbox(
        "Apply Defensive Possession Adjustments",
        value=False,
        help="Adjust defensive stats based on team possession to enable more accurate comparisons"
    )
with col3:
    st.write("")  # Add spacing
    if st.button("Clear Cache", help="Clear cached data to force refresh"):
        st.cache_data.clear()
        st.success("Cache cleared!")
        st.rerun()

uploaded_file = st.file_uploader(
    "Choose a file",
    type=['xlsx', 'xls', 'csv'],
    help="Upload a Wyscout export containing player statistics"
)

# Initialize data variable
df = None

if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Found {len(df)} players.")
        
        # Store original count before filtering
        original_count = len(df)
        
        # Process the uploaded data
        df = process_wyscout_data(df, min_minutes)
        
        # Apply possession adjustment if enabled
        if possession_adjusted:
            possession_data = load_possession_data()
            if possession_data:
                df = apply_possession_adjustment(df, possession_data)
                st.info(f"✅ Applied possession adjustment to defensive metrics using FBRef data")
            else:
                st.warning("⚠️ Possession adjustment requested but data not available")
        
        df = add_custom_metrics(df)
        
        # Show filtering results
        filtered_count = len(df)
        excluded_count = original_count - filtered_count
        
        if excluded_count > 0:
            st.info(f"ℹ️ Filtered out {excluded_count} players with <{min_minutes} minutes. {filtered_count} players remaining.")
        
        # Show data summary
        with st.expander("📊 Data Summary"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Players (≥{} min)".format(min_minutes), len(df))
            with col2:
                st.metric("Positions Found", len(df['Position'].unique()))
            with col3:
                st.metric("Teams", len(df['Team within selected timeframe'].unique()))
            
            st.subheader("Position Distribution")
            position_counts = df['Position'].value_counts()
            st.bar_chart(position_counts)
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.stop()

else:
    # Option to use default PL data
    use_default = st.checkbox("Use PL 24/25 data (no upload required)")
    if use_default:
        df = load_data(min_minutes)
        
        # Apply possession adjustment if enabled
        if possession_adjusted:
            possession_data = load_possession_data()
            if possession_data:
                df = apply_possession_adjustment(df, possession_data)
                st.info(f"✅ Applied possession adjustment to defensive metrics using FBRef data")
            else:
                st.warning("⚠️ Possession adjustment requested but data not available")
        
        df = add_custom_metrics(df)
        st.info(f"Using default Premier League data ({len(df)} players with ≥{min_minutes} minutes)")

if df is not None:
    # Analysis mode selection
    st.header("⚙️ Analysis Mode")
    analysis_mode = st.radio(
        "Choose your analysis approach:",
        ["🎯 Predefined Roles", "🛠️ Custom Role Builder", "🧑🏼‍🔬 Squad Gap Analysis"],
        help="Use predefined roles for individual analysis, create custom roles, or analyze team strengths and weaknesses"
    )
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Position selection (common to both modes)
    if analysis_mode == "🎯 Predefined Roles":
        data_positions = set(df['Position'].unique())
        config_positions = list(POSITION_CONFIGS.keys())
        
        available_positions = []
        for config_pos in config_positions:
            # Handle CB variants - both map to CB data
            if config_pos in ['IP: CB', 'OOP: CB']:
                if 'CB' in data_positions:
                    available_positions.append(config_pos)
            else:
                # Regular position matching
                if config_pos in data_positions:
                    available_positions.append(config_pos)
        
        available_positions = sorted(available_positions)
        
        if not available_positions:
            st.warning("No supported positions found in the data. Supported positions: " + 
                      ", ".join(POSITION_CONFIGS.keys()))
            st.stop()
    else:
        available_positions = sorted(df['Position'].unique())
    
    position = st.sidebar.selectbox(
        "Select Position",
        options=available_positions
    )

    # Initialize session state for custom roles
    if 'aspects' not in st.session_state:
        st.session_state['aspects'] = []
    if 'aspect_weights' not in st.session_state:
        st.session_state['aspect_weights'] = {}

    # Mode-specific content
    if analysis_mode == "🎯 Predefined Roles":
        # Original predefined role analysis
        # Add role selection
        role_options = list(POSITION_CONFIGS[position]['roles'].keys())
        role_options_display = ['All'] + role_options
        selected_role = st.sidebar.selectbox("Select Role", role_options_display)

        # Main content
        st.header(f"🎯 Role Fit Analysis for {position} Position")

        # Calculate role scores for the selected position and role
        if selected_role == 'All':
            role_scores_df = calculate_role_scores(df, position, min_minutes=min_minutes)
        else:
            role_scores_df = calculate_role_scores(df, position, min_minutes=min_minutes, role=selected_role)

        if role_scores_df is not None:
            base_cols = ['Player', 'Team within selected timeframe', 'Minutes played', 'Best_Role_Fit']
            if selected_role == 'All':
                fit_cols = [col for col in role_scores_df.columns if (col.endswith('_Fit') and col != 'Best_Role_Fit')]
                gmean_fit_cols = [col for col in role_scores_df.columns if col.endswith('_GeoMean_Fit')]
                display_cols = list(dict.fromkeys(base_cols + fit_cols + gmean_fit_cols))
            else:
                fit_cols = [f'{selected_role}_Fit', f'{selected_role}_GeoMean_Fit']
                # Always include Best_Role_Fit if present
                display_cols = ['Player', 'Team within selected timeframe', 'Minutes played']
                if 'Best_Role_Fit' in role_scores_df.columns:
                    display_cols.append('Best_Role_Fit')
                display_cols += [col for col in fit_cols if col in role_scores_df.columns]
            
            # Apply color coding to score columns (exclude Best_Role_Fit as it contains role names, not scores)
            score_cols = [col for col in display_cols if col.endswith('_Fit') and col != 'Best_Role_Fit']
            styled_df = style_role_scores(role_scores_df[display_cols], score_cols)
            st.dataframe(styled_df, use_container_width=True)

            # League averages section
            st.header("📊 League Average Role Scores")
            st.markdown(f"Average role fit scores across all {len(role_scores_df)} {position} players in the league")
            
            # Calculate league averages for role fit scores
            league_avg_cols = [col for col in role_scores_df.columns if col.endswith('_Fit') or col.endswith('_GeoMean_Fit')]
            numeric_league_cols = []
            for col in league_avg_cols:
                if pd.api.types.is_numeric_dtype(role_scores_df[col]):
                    numeric_league_cols.append(col)
            
            if len(numeric_league_cols) > 0:
                league_averages = {}
                for col in numeric_league_cols:
                    avg_score = role_scores_df[col].mean()
                    league_averages[col.replace('_Fit', '').replace('_GeoMean', ' (GeoMean)')] = round(avg_score, 1)
                
                # Display league averages in columns
                if len(league_averages) <= 4:
                    cols = st.columns(len(league_averages))
                    for i, (role_name, avg_score) in enumerate(league_averages.items()):
                        with cols[i]:
                            st.metric(f"{role_name}", f"{avg_score}")
                else:
                    # For many roles, use a more compact display
                    avg_df = pd.DataFrame([
                        {"Role": role_name, "League Average": avg_score} 
                        for role_name, avg_score in league_averages.items()
                    ])
                    st.dataframe(avg_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No numeric role fit columns found for league averages.")

            # CSV download button (outputs everything for these players)
            csv = role_scores_df.to_csv(index=False)
            st.download_button(
                label="📥 Download full data as CSV",
                data=csv,
                file_name=f"{position}_{selected_role}_role_scores.csv",
                mime='text/csv'
            )
            
            # Team-based summary statistics
            st.header("🏆 Team-Based Role Summary")
            st.markdown("Average role fit scores by team for this position")
            
            # Calculate team averages for role fit scores
            team_summary_cols = ['Team within selected timeframe']
            role_fit_cols = [col for col in role_scores_df.columns if col.endswith('_Fit') or col.endswith('_GeoMean_Fit')]
            
            # Filter to only numeric columns to avoid aggregation errors
            numeric_role_fit_cols = []
            for col in role_fit_cols:
                if pd.api.types.is_numeric_dtype(role_scores_df[col]):
                    numeric_role_fit_cols.append(col)
            
            if len(numeric_role_fit_cols) == 0:
                st.warning("No numeric role fit columns found for team summary.")
            else:
                team_summary = role_scores_df.groupby('Team within selected timeframe')[numeric_role_fit_cols].agg(['mean', 'count']).round(1)
                
                # Flatten column names for better display
                team_summary.columns = [f'{col[0]}_{col[1]}' if col[1] == 'count' else f'{col[0]}_avg' for col in team_summary.columns]
                
                # Add player count column (using any fit score count since they should all be the same)
                first_fit_col = [col for col in team_summary.columns if col.endswith('_count')][0]
                team_summary['Players'] = team_summary[first_fit_col]
                
                # Remove individual count columns and keep only averages and total player count
                avg_cols = [col for col in team_summary.columns if col.endswith('_avg')]
                team_summary_display = team_summary[avg_cols + ['Players']].copy()
                
                # Calculate overall average level across all roles for each team
                # Only use the main role fit scores (exclude GeoMean for cleaner calculation)
                main_role_avg_cols = [col for col in avg_cols if not 'GeoMean' in col]
                if len(main_role_avg_cols) > 0:
                    team_summary_display['Average_Level'] = team_summary_display[main_role_avg_cols].mean(axis=1).round(1)
                
                # Rename columns for better display
                team_summary_display.columns = [col.replace('_avg', '') for col in team_summary_display.columns]
                
                # Sort by the Average Level (descending) if available, otherwise by first role fit score
                if 'Average_Level' in team_summary_display.columns:
                    team_summary_display = team_summary_display.sort_values('Average_Level', ascending=False)
                elif len(avg_cols) > 0:
                    first_avg_col = avg_cols[0].replace('_avg', '')
                    team_summary_display = team_summary_display.sort_values(first_avg_col, ascending=False)
                
                # Apply color coding to team summary table
                score_cols = [col for col in team_summary_display.columns 
                             if col not in ['Players'] and pd.api.types.is_numeric_dtype(team_summary_display[col])]
                styled_team_display = style_role_scores(team_summary_display, score_cols)
                st.dataframe(styled_team_display, use_container_width=True)
                
                # Create visualizations for team comparisons
                if len(team_summary_display) > 1:  # Only show charts if we have multiple teams
                    st.subheader("📊 Team Role Comparison Charts")
                    
                    # Get role fit columns for visualization (exclude GeoMean for cleaner charts)
                    viz_cols = [col for col in team_summary_display.columns if not col.endswith('_GeoMean_Fit') and col != 'Players' and col != 'Average_Level']
                    
                    if len(viz_cols) > 0:
                        # Create tabs for different visualizations
                        tab1, tab2 = st.tabs(["Bar Chart", "Heatmap"])
                        
                        with tab1:
                            # Bar chart for each role
                            selected_role_viz = st.selectbox(
                                "Select role for detailed comparison:",
                                options=viz_cols + ['Average_Level'],
                                key="team_viz_role"
                            )
                            
                            # Prepare data for bar chart
                            chart_data = team_summary_display.reset_index()
                            
                            fig_bar = px.bar(
                                chart_data,
                                x='Team within selected timeframe',
                                y=selected_role_viz,
                                title=f'Team Average: {selected_role_viz}',
                                color=selected_role_viz,
                                color_continuous_scale='viridis'
                            )
                            fig_bar.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        with tab2:
                            # Heatmap showing all roles for all teams
                            if len(viz_cols) > 1:
                                heatmap_data = team_summary_display[viz_cols]
                                
                                fig_heatmap = px.imshow(
                                    heatmap_data.T,  # Transpose so roles are on y-axis
                                    x=heatmap_data.index,
                                    y=heatmap_data.columns,
                                    color_continuous_scale='viridis',
                                    title='Team Role Fit Heatmap (All Roles)',
                                    aspect='auto'
                                )
                                fig_heatmap.update_xaxes(tickangle=45)
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                            else:
                                st.info("Heatmap requires multiple roles. Select 'All' roles to see the full comparison.")
                
                # Download button for team summary
                team_csv = team_summary_display.to_csv()
                st.download_button(
                    label="📥 Download team summary as CSV",
                    data=team_csv,
                    file_name=f"{position}_{selected_role}_team_summary.csv",
                    mime='text/csv'
                )

        else:
            st.warning(f"No data available for position: {position}")

    elif analysis_mode == "🛠️ Custom Role Builder":
        st.header(f"🛠️ Custom Role Builder for {position} Position")
        
        # Load/Save role interface
        st.markdown("### Load/Save Custom Roles")
        saved_roles = list_saved_roles(position)
        load_col, save_col = st.columns([2, 2])
        
        with load_col:
            load_role_name = st.selectbox("Load Saved Role", options=["(None)"] + saved_roles, key="load_role_select")
            if st.button("Load Role"):
                if load_role_name and load_role_name != "(None)":
                    loaded_aspects, loaded_weights = load_role(position, load_role_name)
                    st.session_state['aspects'] = loaded_aspects
                    st.session_state['aspect_weights'] = loaded_weights
                    st.success(f"Loaded role '{load_role_name}' for {position}.")
                    st.rerun()
        
        with save_col:
            save_role_name = st.text_input("Role Name to Save", value="", key="save_role_input")
            if st.button("Save Role"):
                if save_role_name and st.session_state['aspects']:
                    save_role(position, save_role_name, st.session_state['aspects'], st.session_state['aspect_weights'])
                    st.success(f"Saved role '{save_role_name}' for {position}.")
                    st.rerun()
                elif not save_role_name:
                    st.error("Please enter a role name to save.")
                elif not st.session_state['aspects']:
                    st.error("Please add at least one aspect before saving.")

        # Dynamic aspect builder with tabbed metric selection
        st.markdown("### Build Custom Aspect")
        
        position_df = df[df['Position'] == position]
        exclude_cols = ['Player', 'Team within selected timeframe', 'Minutes played', 'Position']
        metric_options = [col for col in position_df.columns if col not in exclude_cols and np.issubdtype(position_df[col].dtype, np.number)]

        # Aspect name input
        aspect_name = st.text_input("Aspect Name", key="aspect_name_input", help="Give your custom aspect a descriptive name")
        
        # Initialize session state for metric selections
        if 'metric_selections' not in st.session_state:
            st.session_state['metric_selections'] = {}
        
        st.markdown("#### Select Metrics by Category")
        st.markdown("Choose metrics from different categories to build your custom aspect:")
        
        # Create tabs for each metric group
        tabs = st.tabs([f"🏃 {group}" if group == "Ball Carrying" 
                       else f"⚽ {group}" if group == "Passing"
                       else f"🥊 {group}" if group == "Duelling" 
                       else f"🛡️ {group}" if group == "Defending"
                       else f"🎯 {group}" if group == "Creating"
                       else f"🥅 {group}" for group in METRIC_GROUPS.keys()])
        
        # Track selected metrics across all tabs
        all_selected_metrics = []
        
        for i, (group_name, tab) in enumerate(zip(METRIC_GROUPS.keys(), tabs)):
            with tab:
                st.markdown(f"**{group_name} Metrics**")
                
                # Get available metrics for this group
                group_metrics = [m for m in METRIC_GROUPS[group_name] if m in metric_options]
                
                if group_metrics:
                    # Show metrics with checkboxes in columns for better layout
                    cols = st.columns(2)
                    for idx, metric in enumerate(group_metrics):
                        with cols[idx % 2]:
                            # Create unique key for each metric
                            checkbox_key = f"metric_checkbox_{metric}_{group_name}"
                            
                            # Get current state (default to False if clearing)
                            current_value = st.session_state['metric_selections'].get(checkbox_key, False)
                            
                            # Show checkbox
                            selected = st.checkbox(
                                metric, 
                                value=current_value,
                                key=checkbox_key,
                                help=f"Include {metric} in your custom aspect"
                            )
                            
                            # Update session state and collect selected metrics
                            st.session_state['metric_selections'][checkbox_key] = selected
                            if selected:
                                all_selected_metrics.append(metric)
                else:
                    st.info(f"No {group_name.lower()} metrics found in your dataset.")
        
        # Update selected metrics summary
        st.session_state['selected_metrics'] = all_selected_metrics
        
        # Show currently selected metrics summary
        if all_selected_metrics:
            st.markdown("#### Selected Metrics")
            st.success(f"**{len(all_selected_metrics)} metrics selected:** {', '.join(all_selected_metrics)}")
        else:
            st.info("No metrics selected yet. Use the tabs above to select metrics for your aspect.")
        
        # Buttons for aspect management
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
        
        with btn_col1:
            if st.button("Add Aspect", type="primary"):
                selected_metrics = st.session_state.get('selected_metrics', [])
                if aspect_name and selected_metrics:
                    # Check if aspect name already exists
                    existing_names = [asp['name'] for asp in st.session_state['aspects']]
                    if aspect_name in existing_names:
                        st.error(f"Aspect '{aspect_name}' already exists. Please use a different name.")
                    else:
                        st.session_state['aspects'].append({
                            'name': aspect_name,
                            'metrics': selected_metrics
                        })
                        st.session_state['aspect_weights'][aspect_name] = 0
                        st.success(f"Added aspect: {aspect_name} with {len(selected_metrics)} metrics")
                        # Clear selections for next aspect
                        st.session_state['metric_selections'] = {}
                        st.session_state['selected_metrics'] = []
                        st.rerun()
                elif not aspect_name:
                    st.error("Please enter an aspect name.")
                elif not selected_metrics:
                    st.error("Please select at least one metric from the tabs above.")
        
        with btn_col2:
            if st.button("Clear Selection"):
                # Clear all metric checkbox selections
                st.session_state['metric_selections'] = {}
                st.session_state['selected_metrics'] = []
                st.success("Cleared metric selection")
                st.rerun()
        
        with btn_col3:
            if st.button("Clear All Aspects"):
                st.session_state['aspects'] = []
                st.session_state['aspect_weights'] = {}
                st.session_state['metric_selections'] = {}
                st.session_state['selected_metrics'] = []
                st.success("Cleared all aspects and selections")
                st.rerun()

        # Show current aspects
        if st.session_state['aspects']:
            st.markdown("#### Current Aspects:")
            for i, asp in enumerate(st.session_state['aspects']):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{i+1}. **{asp['name']}** - Metrics: {', '.join(asp['metrics'])}")
                with col2:
                    if st.button(f"Remove", key=f"remove_{i}"):
                        asp_name = st.session_state['aspects'][i]['name']
                        st.session_state['aspects'].pop(i)
                        st.session_state['aspect_weights'].pop(asp_name, None)
                        st.success(f"Removed aspect: {asp_name}")
                        st.rerun()

        # Aspect weights input (after aspect creation)
        if st.session_state['aspects']:
            st.markdown("### Assign Weights to Aspects (must sum to 100%)")
            cols = st.columns(len(st.session_state['aspects']))
            for i, asp in enumerate(st.session_state['aspects']):
                with cols[i]:
                    weight = st.number_input(
                        f"{asp['name']} (%)", 
                        min_value=0, 
                        max_value=100, 
                        value=st.session_state['aspect_weights'].get(asp['name'], 0), 
                        step=1, 
                        key=f"weight_{asp['name']}"
                    )
                    st.session_state['aspect_weights'][asp['name']] = weight
                    
            total_weight = sum(st.session_state['aspect_weights'].values())
            st.write(f"**Total: {total_weight}%**")
            can_run = (total_weight == 100)
            
            if not can_run:
                st.warning("Aspect weights must sum to 100% to run analysis.")
        else:
            can_run = False

        # Custom role analysis output
        if st.session_state['aspects'] and can_run:
            custom_scores_df = calculate_custom_role_scores(df, position, st.session_state['aspects'], st.session_state['aspect_weights'])
            
            if custom_scores_df is not None:
                st.subheader(f"Custom Role Analysis for {position}")
                st.markdown("**Aspect Weights:**")
                st.write({k: f"{v}%" for k, v in st.session_state['aspect_weights'].items()})
                
                # Show results
                show_cols = ['Player', 'Team within selected timeframe', 'Minutes played'] + [asp['name'] for asp in st.session_state['aspects']] + ['Custom_Role_Fit']
                display_df = custom_scores_df[show_cols].sort_values('Custom_Role_Fit', ascending=False)
                
                # Apply color coding to the Custom_Role_Fit column
                styled_custom_df = style_role_scores(display_df, ['Custom_Role_Fit'])
                st.dataframe(styled_custom_df, use_container_width=True)
                
                # League averages for custom role
                st.header("📊 League Average Custom Role Scores")
                st.markdown(f"Average scores across all {len(custom_scores_df)} {position} players for your custom role")
                
                # Calculate league averages for custom aspects and overall fit
                custom_avg_cols = [asp['name'] for asp in st.session_state['aspects']] + ['Custom_Role_Fit']
                numeric_custom_avg_cols = []
                for col in custom_avg_cols:
                    if col in custom_scores_df.columns and pd.api.types.is_numeric_dtype(custom_scores_df[col]):
                        numeric_custom_avg_cols.append(col)
                
                if len(numeric_custom_avg_cols) > 0:
                    custom_league_averages = {}
                    for col in numeric_custom_avg_cols:
                        avg_score = custom_scores_df[col].mean()
                        display_name = "Overall Custom Role" if col == 'Custom_Role_Fit' else col
                        custom_league_averages[display_name] = round(avg_score, 1)
                    
                    # Display custom league averages
                    if len(custom_league_averages) <= 4:
                        cols = st.columns(len(custom_league_averages))
                        for i, (aspect_name, avg_score) in enumerate(custom_league_averages.items()):
                            with cols[i]:
                                st.metric(f"{aspect_name}", f"{avg_score}")
                    else:
                        # For many aspects, use table format
                        custom_avg_df = pd.DataFrame([
                            {"Aspect/Role": aspect_name, "League Average": avg_score} 
                            for aspect_name, avg_score in custom_league_averages.items()
                        ])
                        st.dataframe(custom_avg_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No numeric custom role columns found for league averages.")
                
                # CSV download for custom analysis
                csv = custom_scores_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download custom role data as CSV",
                    data=csv,
                    file_name=f"{position}_custom_role_scores.csv",
                    mime='text/csv'
                )
                
                # Team summary for custom roles
                st.header("🏆 Team-Based Custom Role Summary")
                st.markdown("Average custom role scores by team for this position")
                
                # Calculate team averages for custom role scores
                custom_role_cols = [asp['name'] for asp in st.session_state['aspects']] + ['Custom_Role_Fit']
                numeric_custom_cols = []
                for col in custom_role_cols:
                    if col in custom_scores_df.columns and pd.api.types.is_numeric_dtype(custom_scores_df[col]):
                        numeric_custom_cols.append(col)
                
                if len(numeric_custom_cols) > 0:
                    custom_team_summary = custom_scores_df.groupby('Team within selected timeframe')[numeric_custom_cols].agg(['mean', 'count']).round(1)
                    
                    # Flatten column names
                    custom_team_summary.columns = [f'{col[0]}_{col[1]}' if col[1] == 'count' else f'{col[0]}_avg' for col in custom_team_summary.columns]
                    
                    # Add player count
                    first_count_col = [col for col in custom_team_summary.columns if col.endswith('_count')][0]
                    custom_team_summary['Players'] = custom_team_summary[first_count_col]
                    
                    # Keep only averages and player count
                    custom_avg_cols = [col for col in custom_team_summary.columns if col.endswith('_avg')]
                    custom_team_display = custom_team_summary[custom_avg_cols + ['Players']].copy()
                    
                    # Calculate average level for custom aspects (exclude the final Custom_Role_Fit)
                    aspect_avg_cols = [col for col in custom_avg_cols if not col.startswith('Custom_Role_Fit')]
                    if len(aspect_avg_cols) > 0:
                        custom_team_display['Average_Level'] = custom_team_display[aspect_avg_cols].mean(axis=1).round(1)
                    
                    # Clean column names
                    custom_team_display.columns = [col.replace('_avg', '') for col in custom_team_display.columns]
                    
                    # Sort by Custom_Role_Fit if available
                    if 'Custom_Role_Fit' in custom_team_display.columns:
                        custom_team_display = custom_team_display.sort_values('Custom_Role_Fit', ascending=False)
                    
                    # Apply color coding to custom team display
                    score_cols = [col for col in custom_team_display.columns 
                                 if col not in ['Players'] and pd.api.types.is_numeric_dtype(custom_team_display[col])]
                    styled_custom_team_display = style_role_scores(custom_team_display, score_cols)
                    st.dataframe(styled_custom_team_display, use_container_width=True)
                    
                    # Visualizations for custom roles
                    if len(custom_team_display) > 1:
                        st.subheader("📊 Custom Role Team Comparison")
                        
                        viz_cols = [col for col in custom_team_display.columns if col not in ['Players']]
                        if len(viz_cols) > 0:
                            selected_custom_viz = st.selectbox(
                                "Select metric for team comparison:",
                                options=viz_cols,
                                key="custom_team_viz"
                            )
                            
                            chart_data = custom_team_display.reset_index()
                            fig_custom = px.bar(
                                chart_data,
                                x='Team within selected timeframe',
                                y=selected_custom_viz,
                                title=f'Team Average: {selected_custom_viz}',
                                color=selected_custom_viz,
                                color_continuous_scale='viridis'
                            )
                            fig_custom.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_custom, use_container_width=True)
                    
                    # Download custom team summary
                    custom_team_csv = custom_team_display.to_csv()
                    st.download_button(
                        label="📥 Download custom team summary as CSV",
                        data=custom_team_csv,
                        file_name=f"{position}_custom_team_summary.csv",
                        mime='text/csv'
                    )
            else:
                st.warning(f"No data available for position: {position}")
        elif not st.session_state['aspects']:
            st.info("Add at least one aspect and assign weights to run custom role analysis.")

    else:  # Team Analysis mode
        st.header(f"🏆 Squad Gap Analysis for {position} Position")
        
        # Calculate role scores for gap analysis
        role_scores_df = calculate_role_scores(df, position, min_minutes=min_minutes)
        
        if role_scores_df is not None:
            # Team selection
            teams = sorted(df['Team within selected timeframe'].unique())
            selected_team = st.selectbox(
                "Select Team to Analyze", 
                options=teams,
                help="Choose the team you want to analyze for strengths and weaknesses"
            )
            
            # Get team data
            team_players = role_scores_df[role_scores_df['Team within selected timeframe'] == selected_team]
            
            if len(team_players) > 0:
                st.subheader(f"📊 {selected_team} - Squad Gap Analysis")
                
                # Calculate team averages and gaps
                role_cols = [col for col in role_scores_df.columns if col.endswith('_Fit') and col != 'Best_Role_Fit']
                
                # Filter to only numeric role columns and ensure data quality
                numeric_role_cols = []
                for col in role_cols:
                    try:
                        if pd.api.types.is_numeric_dtype(role_scores_df[col]):
                            # Additional check: ensure the column actually contains numeric data
                            test_mean = role_scores_df[col].mean()
                            numeric_role_cols.append(col)
                    except (TypeError, ValueError):
                        # Skip any problematic columns
                        continue
                
                if len(numeric_role_cols) == 0:
                    st.error("No valid numeric role columns found! Cannot perform gap analysis.")
                    st.warning("This might be a data processing issue. Please check your data or contact support.")
                    st.stop()
                
                gap_data = []
                team_averages = {}
                league_averages = {}
                
                for role_col in numeric_role_cols:
                    role_name = role_col.replace('_Fit', '')
                    team_avg = team_players[role_col].mean()
                    league_avg = role_scores_df[role_col].mean()
                    
                    # Count players above league average for this role
                    players_above_avg = len(team_players[team_players[role_col] > league_avg])
                    total_team_players = len(team_players)
                    
                    # Store for later use
                    team_averages[role_name] = team_avg
                    league_averages[role_name] = league_avg
                    
                    # New 'past the post' classification system
                    if players_above_avg == 0:
                        status = "🔴 LACK OF COVERAGE"
                        priority = 1
                    elif players_above_avg == 1:
                        status = "✅ SOMEONE CAN DO IT"
                        priority = 3
                    elif players_above_avg >= 2:
                        status = "🟢 EXCELLENT COVERAGE"
                        priority = 4
                    else:
                        status = "⚪ AVERAGE COVERAGE"
                        priority = 2
                    
                    gap_data.append({
                        "Role": role_name,
                        "Players Above Average": f"{players_above_avg}/{total_team_players}",
                        "Team Average": f"{team_avg:.1f}",
                        "League Average": f"{league_avg:.1f}",
                        "Status": status,
                        "Priority": priority,
                        "Players_Count": players_above_avg  # For sorting
                    })
                
                # Create DataFrame for display
                gap_df = pd.DataFrame(gap_data)
                
                # Sorting options
                col1, col2 = st.columns([1, 3])
                with col1:
                    sort_by = st.selectbox(
                        "Sort by:",
                        options=["Gap Severity", "Role Name"],
                        help="Choose how to order the analysis"
                    )
                
                # Apply sorting
                if sort_by == "Gap Severity":
                    gap_df = gap_df.sort_values(['Priority', 'Players_Count'])
                else:
                    gap_df = gap_df.sort_values('Role')
                
                # Display gap analysis table
                display_df = gap_df[['Role', 'Players Above Average', 'Team Average', 'League Average', 'Status']].copy()
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Player breakdown section
                st.subheader(f"👥 {selected_team} Player Breakdown")
                st.markdown("See which players are contributing to each role average:")
                
                # Create expandable sections for each role
                for _, row in gap_df.iterrows():
                    role_name = row['Role']
                    role_col = f"{role_name}_Fit"
                    status_emoji = "🔴" if row['Status'] == "🔴 CRITICAL" else "🟡" if row['Status'] == "🟡 MINOR" else "✅" if row['Status'] == "✅ STRENGTH" else "⚪"
                    
                    with st.expander(f"{status_emoji} {role_name} - Players ({row['Team Average']})", expanded=False):
                        if role_col in team_players.columns:
                            # Get players for this role, sorted by score
                            role_players = team_players[['Player', 'Minutes played', role_col]].copy()
                            role_players = role_players.sort_values(role_col, ascending=False)
                            
                            # Rename columns for display
                            role_players.columns = ['Player', 'Minutes', f'{role_name} Score']
                            role_players[f'{role_name} Score'] = role_players[f'{role_name} Score'].round(1)
                            
                            # Apply color coding to the score column
                            styled_role_players = style_role_scores(role_players, [f'{role_name} Score'])
                            st.dataframe(styled_role_players, use_container_width=True, hide_index=True)
                            
                            # Quick stats
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Players", len(role_players))
                            with col2:
                                best_score = role_players[f'{role_name} Score'].max()
                                st.metric("Best Score", f"{best_score:.1f}")
                            with col3:
                                worst_score = role_players[f'{role_name} Score'].min()
                                st.metric("Worst Score", f"{worst_score:.1f}")
                                
                            # Show how each player compares to league average
                            league_avg = float(row['League Average'])
                            st.markdown("**vs League Average:**")
                            for _, player_row in role_players.iterrows():
                                player_score = player_row[f'{role_name} Score']
                                diff = player_score - league_avg
                                if diff >= 2:
                                    status = "✅ Above"
                                elif diff >= -2:
                                    status = "⚪ Average"
                                else:
                                    status = "🔴 Below"
                                st.write(f"• {player_row['Player']}: {player_score:.1f} ({diff:+.1f}) {status}")
                        else:
                            st.warning(f"No data available for {role_name}")
                
                # Summary metrics
                st.subheader("📈 Gap Analysis Summary")
                
                critical_gaps = len(gap_df[gap_df['Status'] == '🔴 CRITICAL'])
                minor_gaps = len(gap_df[gap_df['Status'] == '🟡 MINOR'])
                strengths = len(gap_df[gap_df['Status'] == '✅ STRENGTH'])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🔴 Critical Gaps", critical_gaps)
                
                with col2:
                    st.metric("🟡 Minor Gaps", minor_gaps)
                
                with col3:
                    st.metric("✅ Strengths", strengths)
                
                with col4:
                    overall_team_avg = np.mean(list(team_averages.values()))
                    overall_league_avg = np.mean(list(league_averages.values()))
                    overall_gap = overall_team_avg - overall_league_avg
                    st.metric("📊 Overall Gap", f"{overall_gap:+.1f}")
                
                # Priority recommendations
                critical_roles = gap_df[gap_df['Status'] == '🔴 CRITICAL']['Role'].tolist()
                minor_roles = gap_df[gap_df['Status'] == '🟡 MINOR']['Role'].tolist()
                
                if critical_gaps > 0 or minor_gaps > 0:
                    st.subheader("🎯 Priority Areas for Improvement")
                    
                    if critical_gaps > 0:
                        st.markdown("**🔴 Critical Priority:**")
                        for role in critical_roles:
                            role_gap = gap_df[gap_df['Role'] == role]['Gap_Value'].iloc[0]
                            st.write(f"• **{role}**: {role_gap:.1f} below league average")
                    
                    if minor_gaps > 0:
                        st.markdown("**🟡 Secondary Priority:**")
                        for role in minor_roles:
                            role_gap = gap_df[gap_df['Role'] == role]['Gap_Value'].iloc[0]
                            st.write(f"• **{role}**: {role_gap:.1f} below league average")
                
                else:
                    st.success("🎉 **Excellent!** Your team is at or above league average in all roles!")
                
                # Export functionality
                gap_export_df = gap_df[['Role', 'Team Average', 'League Average', 'Gap', 'Status']].copy()
                gap_csv = gap_export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download gap analysis as CSV",
                    data=gap_csv,
                    file_name=f"{selected_team}_{position}_gap_analysis.csv",
                    mime='text/csv'
                )
                
            else:
                st.warning(f"No {position} players found for {selected_team} with ≥{min_minutes} minutes played")
        
        else:
            st.warning(f"No data available for position: {position}")

else:
    st.info("👆 Please upload a Wyscout data file to begin analysis") 