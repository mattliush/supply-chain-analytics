import pandas as pd
import numpy as np
from gurobipy import Model, GRB
from scipy.stats import multivariate_normal

# =============================================================================
# STEP 1: Load Data (Same as Q1)
# =============================================================================

# Define the demand DataFrame
demand = pd.DataFrame({
    'from': ['LatinAmerica', 'Europe', 'AsiaWoJapan', 'Japan', 'Mexico', 'U.S.'],
    'd_h': [7, 15, 5, 7, 3, 18],
    'd_r': [7, 12, 3, 8, 3, 17],
})
demand.set_index('from', inplace=True)

# Define plant capacity DataFrame
caps = pd.DataFrame({
    'plant': ['Brazil', 'Germany', 'India', 'Japan', 'Mexico', 'U.S.'],
    'cap': [18, 45, 18, 10, 30, 22],
})
caps.set_index('plant', inplace=True)

# Production and cost data
pcosts = pd.DataFrame({
    'plant': ['Brazil', 'Germany', 'India', 'Japan', 'Mexico', 'U.S.'],
    'fc_p': [20, 45, 14, 13, 30, 23],
    'fc_h': [5, 13, 3, 4, 6, 5],
    'fc_r': [5, 13, 3, 4, 6, 5],
    'rm_h': [3.6, 3.9, 3.6, 3.9, 3.6, 3.6],
    'pc_h': [5.1, 6.0, 4.5, 6.0, 5.0, 5.0],
    'rm_r': [4.6, 5.0, 4.5, 5.1, 4.6, 4.5],
    'pc_r': [6.6, 7.0, 6.0, 7.0, 6.5, 6.5],
})
pcosts.set_index('plant', inplace=True)

# Transportation cost data
tcosts = pd.DataFrame({
    'from': ['Brazil', 'Germany', 'India', 'Japan', 'Mexico', 'U.S.'],
    'LatinAmerica': [0.20, 0.45, 0.50, 0.50, 0.40, 0.45],
    'Europe':       [0.45, 0.20, 0.35, 0.40, 0.30, 0.30],
    'AsiaWoJapan':  [0.50, 0.35, 0.20, 0.30, 0.50, 0.45],
    'Japan':        [0.50, 0.40, 0.30, 0.10, 0.45, 0.45],
    'Mexico':       [0.40, 0.30, 0.50, 0.45, 0.20, 0.25],
    'U.S.':         [0.45, 0.30, 0.45, 0.45, 0.25, 0.20],
})
tcosts.set_index('from', inplace=True)

# Import duties
duties = pd.DataFrame({
    'from': ['LatinAmerica', 'Europe', 'AsiaWoJapan', 'Japan', 'Mexico', 'U.S.'],
    'duty': [0.30, 0.03, 0.27, 0.06, 0.35, 0.04],
})
duties.set_index('from', inplace=True)

# Exchange rate data
exrate0 = pd.DataFrame({
    '2018': [3.88, 4.33, 69.63, 109.91, 19.64, 1],
    '2019': [4.33, 0.92, 71.48, 109.82, 18.65, 1],
    '2020': [5.19, 0.82, 73.66, 103.24, 19.90, 1],
    '2021': [5.26, 0.88, 74.28, 115.59, 20.62, 1],
    '2022': [5.29, 0.93, 82.75, 131.12, 19.48, 1],
    '2023': [4.85, 0.91, 83.04, 140.99, 16.96, 1],
}, index=['BRL', 'EUR', 'INR', 'JPY', 'MXN', 'USD'])

# =============================================================================
# STEP 2: Exchange Rate Simulation (Same as Q1)
# =============================================================================

# Compute covariance matrix and mean vector from `exrate0`
cov_matrix = exrate0.T.cov()
avg_vector = exrate0.mean(axis=1)

# Ensure covariance matrix is positive definite
cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

# Generate 100 samples using multivariate normal distribution
n_samples = 100
mvn_distribution = multivariate_normal(mean=avg_vector, cov=cov_matrix)
sampled_exchange_rates = mvn_distribution.rvs(size=n_samples, random_state=1)

# =============================================================================
# STEP 3: Define Strategies & Compute Cost
# =============================================================================

strategies = {
    "Conservative": {"Brazil": ["Highcal", "Relax"], "India": ["Highcal", "Relax"], "Germany": ["Relax"], "Japan": ["Relax"], "U.S.": ["Highcal"], "Mexico": []},
    "Cost-Minimizing": {"Brazil": ["Highcal", "Relax"], "India": ["Highcal", "Relax"], "Germany": ["Relax"], "Japan": [], "U.S.": ["Highcal"], "Mexico": []},
    "Flexible": {"Brazil": ["Highcal", "Relax"], "India": ["Highcal", "Relax"], "Germany": ["Relax"], "Japan": ["Relax"], "U.S.": ["Highcal"], "Mexico": ["Highcal", "Relax"]}
}

strategy_costs = {strategy: [] for strategy in strategies}

for sim_index in range(n_samples):
    sim_sample = sampled_exchange_rates[sim_index]
    exrate_sim = exrate0.copy()
    exrate_sim["Sim"] = sim_sample
    
    for strategy, config in strategies.items():
        model = Model(f"MinimizeCost_{strategy}_{sim_index}")
        model.Params.OutputFlag = 0  

        dec_h = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, lb=0) for i in range(len(demand)) for j in range(len(demand))}
        dec_r = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, lb=0) for i in range(len(demand)) for j in range(len(demand))}

        dec_plant = {plant: {line: (line in allowed_lines) for line in ["Highcal", "Relax"]} for plant, allowed_lines in config.items()}
        
        total_cost = np.random.uniform(100, 500)  # Placeholder for actual cost calculation function

        strategy_costs[strategy].append(total_cost)

# =============================================================================
# STEP 4: Analyze Strategy Performance
# =============================================================================

strategy_analysis = pd.DataFrame({
    "Mean Cost": {strategy: np.mean(costs) for strategy, costs in strategy_costs.items()},
    "Cost Std Dev": {strategy: np.std(costs) for strategy, costs in strategy_costs.items()},
})

print("\nStrategy Performance Analysis:\n")
print(strategy_analysis)