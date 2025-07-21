# RSA - Advanced Role Score Analysis

A comprehensive football analytics tool for player role analysis and team gap identification.

## 🚀 Features

### 🎯 **Predefined Role Analysis**
- Analyze players using expert-defined role profiles for each position
- CB roles: Ball Player, Aggressor, Protector
- CM roles: Ball-Winner, Conductor, Attacking Box-Crasher, Attacking Creator
- WM, CF, FB roles available
- Role fit scoring and rankings

### 🛠️ **Custom Role Builder**
- Create your own custom roles using any combination of metrics
- Group metrics by category (Ball Carrying, Passing, Defending, etc.)
- Weight different aspects to build unique role profiles
- Save and load custom roles for reuse

### 🏆 **Team Analysis & Gap Identification**
- Automatic identification of team weaknesses vs league average
- Gap severity classification (Critical, Minor, Strengths)
- Individual player breakdown for each role
- Priority recommendations for improvement

### ⚽ **Possession Adjustment**
- Fair comparison across different playing styles
- Normalize defensive stats to 50% possession baseline
- Account for tactical differences between teams
- Improve accuracy of role assessments

## 📊 **Data Support**

- **Default**: Premier League 24/25 season data included
- **Upload**: Support for custom Wyscout exports (Excel/CSV)
- **Possession**: Integrated FBRef possession data for 290+ teams
- **Export**: Download analysis results as CSV

## 🎮 **How to Use**

1. **Select Analysis Mode**: Choose between predefined roles, custom builder, or team analysis
2. **Set Filters**: Adjust minimum minutes and enable possession adjustment if desired
3. **Choose Position**: Select the position you want to analyze
4. **Analyze**: View results, identify gaps, and export data

## 🔧 **Technical Details**

- **Built with**: Streamlit, Pandas, NumPy, Plotly, Scikit-learn
- **Role Scoring**: Z-score based with min-max normalization
- **Custom Metrics**: EFx Duels, P2P ratios, and advanced calculations
- **Performance**: Cached calculations for smooth user experience

## 📈 **Use Cases**

- **Recruitment**: Identify transfer targets based on role fit
- **Squad Planning**: Understand team strengths and weaknesses
- **Player Development**: Track progress in specific role attributes
- **Opposition Analysis**: Compare teams across different playing styles

## 🏅 **Perfect For**

- Football analysts and scouts
- Performance analysts
- Coaches and tactical staff
- Data-driven recruitment teams

---

*Developed for modern football analytics with a focus on role-based player evaluation and team analysis.* 