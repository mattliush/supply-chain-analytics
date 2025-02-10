#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
from scipy.stats import multivariate_normal

# =============================================================================
# STEP 0: Define the required data
# =============================================================================

# Define the demand DataFrame
demand = pd.DataFrame({
    'from': ['LatinAmerica', 'Europe', 'AsiaWoJapan', 'Japan', 'Mexico', 'U.S.'],
    'd_h': [7, 15, 5, 7, 3, 18],
    'd_r': [7, 12, 3, 8, 3, 17],
})
demand.set_index('from', inplace=True)

# Define the plant capacity DataFrame
caps = pd.DataFrame({
    'plant': ['Brazil', 'Germany', 'India', 'Japan', 'Mexico', 'U.S.'],
    'cap': [18, 45, 18, 10, 30, 22],
})
caps.set_index('plant', inplace=True)

# Define production and cost data
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

duties = pd.DataFrame({
    'from': ['LatinAmerica', 'Europe', 'AsiaWoJapan', 'Japan', 'Mexico', 'U.S.'],
    'duty': [0.30, 0.03, 0.27, 0.06, 0.35, 0.04],
})
duties.set_index('from', inplace=True)

# Define the base exchange rates as given (exrate0)
exrate0 = {
    '2018': [3.88, 4.33, 69.63, 109.91, 19.64, 1],
    '2019': [4.33, 0.92, 71.48, 109.82, 18.65, 1],
    '2020': [5.19, 0.82, 73.66, 103.24, 19.90, 1],
    '2021': [5.26, 0.88, 74.28, 115.59, 20.62, 1],
    '2022': [5.29, 0.93, 82.75, 131.12, 19.48, 1],
    '2023': [4.85, 0.91, 83.04, 140.99, 16.96, 1],
}
exrate0 = pd.DataFrame(exrate0, index=['BRL', 'EUR', 'INR', 'JPY', 'MXN', 'USD'])

# =============================================================================
# STEP 1: Define helper functions for cost, capacity, and demand calculations
# =============================================================================

# Note: These functions assume that global variables n_ctry and n_lines are defined,
# and that the global variable 'exrate' is used (set later in the simulation loop).

def calc_total_cost(dec_plant, dec_h, dec_r, base_yr=2019, selected_yr="2023", tariff=0):
    # Here, we use the globally defined n_ctry and n_lines.
    x_plant = np.array(list(dec_plant.values())).reshape(len(n_ctry), len(n_lines))
    x_h = np.array(list(dec_h.values())).reshape(len(n_ctry), len(n_ctry))
    x_r = np.array(list(dec_r.values())).reshape(len(n_ctry), len(n_ctry))
    
    # Adjust cost using exchange rates: base year over selected year's rates.
    reindx = exrate.loc[:, f'{base_yr}'] / exrate.loc[:, f'{selected_yr}']
    pcosts_rev = pcosts.values * reindx.values.reshape(-1, 1)
    pcosts_rev = pd.DataFrame(pcosts_rev, columns=pcosts.columns, index=pcosts.index)
    
    duties_mat = np.zeros(len(duties)) + (1 + duties['duty'].values)[:, np.newaxis]
    np.fill_diagonal(duties_mat, 1)
    duties_mat = pd.DataFrame(duties_mat.T, index=pcosts_rev.index, columns=duties.index)
    duties_mat.loc['Germany', 'U.S.'] += tariff
    duties_mat.loc['U.S.', 'Europe']  += tariff
    
    vcosts_h = tcosts.add(pcosts_rev['rm_h'], axis=0).add(pcosts_rev['pc_h'], axis=0) * duties_mat
    vcosts_r = tcosts.add(pcosts_rev['rm_r'], axis=0).add(pcosts_rev['pc_r'], axis=0) * duties_mat
    
    fc = pcosts_rev[['fc_p', 'fc_h', 'fc_r']].values
    vh = (vcosts_h * x_h).values
    vr = (vcosts_r * x_r).values
    
    total_cost = (
        sum(0.2 * fc[i, j] for i in n_ctry for j in n_lines) +
        sum(0.8 * fc[i, j] * x_plant[i, j] for i in n_ctry for j in n_lines) +
        sum(vh[i, j] for i in n_ctry for j in n_ctry) +
        sum(vr[i, j] for i in n_ctry for j in n_ctry)
    )
    return total_cost

def calc_excess_cap(dec_plant, dec_h, dec_r):
    x_plant = np.array(list(dec_plant.values())).reshape(len(n_ctry), len(n_lines))
    x_h = np.array(list(dec_h.values())).reshape(len(n_ctry), len(n_ctry))
    x_r = np.array(list(dec_r.values())).reshape(len(n_ctry), len(n_ctry))
    
    excess_cap = (x_plant * caps['cap'].values.reshape(-1, 1)).copy()
    excess_cap[:, 0] -= (np.sum(x_h, axis=1) + np.sum(x_r, axis=1))
    excess_cap[:, 1] -= np.sum(x_h, axis=1)
    excess_cap[:, 2] -= np.sum(x_r, axis=1)
    return excess_cap

def calc_unmet_demand(dec_h, dec_r):
    x_h = np.array(list(dec_h.values())).reshape(len(n_ctry), len(n_ctry))
    x_r = np.array(list(dec_r.values())).reshape(len(n_ctry), len(n_ctry))
    
    x_h_sum = np.sum(x_h, axis=0)
    x_r_sum = np.sum(x_r, axis=0)
    unmet_demand = demand[['d_h', 'd_r']].values.copy()
    # Here, unmet demand is computed as delivered minus required demand.
    # The constraints in the model enforce zero unmet demand.
    unmet_demand = np.column_stack((x_h_sum - unmet_demand[:, 0], x_r_sum - unmet_demand[:, 1]))
    return unmet_demand

# =============================================================================
# STEP 2: Set up the exchange rate simulation (using the provided multivariate normal code)
# =============================================================================

# Define the target currencies for simulation (same order as in our functions)
target_currencies = ["BRL", "EUR", "INR", "JPY", "MXN"]

# Define the average vector and covariance matrix from our historical data (sample numbers)
avg_vector = pd.Series({
    'BRL': 5.218146,
    'EUR': 0.924634,
    'INR': 83.177083,
    'JPY': 146.668750,
    'MXN': 18.066267
})
cov_matrix = pd.DataFrame({
    'BRL': [0.150618, 0.002509, 0.273974, 1.225555, 0.435292],
    'EUR': [0.002509, 0.000251, 0.005125, 0.046274, 0.004548],
    'INR': [0.273974, 0.005125, 0.754891, 4.657418, 0.664957],
    'JPY': [1.225555, 0.046274, 4.657418, 61.585064, 0.535588],
    'MXN': [0.435292, 0.004548, 0.664957, 0.535588, 1.553693]
}, index=target_currencies)

# Create the multivariate normal distribution
mvn_distribution = multivariate_normal(mean=avg_vector, cov=cov_matrix)

# Number of simulation draws
n_samples = 100
sampled_exchange_rates = mvn_distribution.rvs(size=n_samples, random_state=1)

# =============================================================================
# STEP 3: Run the simulation: For each exchange rate sample, solve the optimization model
# =============================================================================

# For the functions above to work, we need to define the index sets globally.
n_ctry = range(demand.shape[0])
n_lines = range(demand.shape[1] + 1)  # In our case, 2 (for products) + 1 for overall plant decision

# Prepare a list to store simulation results
simulation_results = []

for sim_index in range(n_samples):
    # Get the current simulated exchange rate sample (order: BRL, EUR, INR, JPY, MXN)
    sim_sample = sampled_exchange_rates[sim_index]
    
    # Create a new exchange rate DataFrame for this simulation by copying exrate0.
    # Then add (or override) a column "Sim" with the simulated values.
    exrate_sim = exrate0.copy()
    for currency in exrate_sim.index:
        if currency in target_currencies:
            pos = target_currencies.index(currency)
            exrate_sim.loc[currency, "Sim"] = sim_sample[pos]
        else:
            # For USD or any currency not simulated, keep the existing value (usually 1)
            exrate_sim.loc[currency, "Sim"] = exrate_sim.loc[currency, "2023"]
    
    # For this simulation, we treat the "Sim" column as the selected year rates.
    selected_year_sim = "Sim"
    base_year = 2019  # as given
    current_tariff = 0.0  # Assume tariff 0 for simulation
    
    # Set the global variable 'exrate' that our cost function uses.
    exrate = exrate_sim
    
    # Build a new Gurobi model for this simulation
    model = Model(f"MinimizeCost_sim_{sim_index}")
    model.Params.OutputFlag = 0  # Suppress solver output
    
    # Define decision variables
    dec_plant = {(i, j): model.addVar(vtype=GRB.BINARY, name=f"Dec_plant_{i}_{j}")
                 for i in n_ctry for j in n_lines}
    dec_h = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Dec_h_{i}_{j}")
             for i in n_ctry for j in n_ctry}
    dec_r = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Dec_r_{i}_{j}")
             for i in n_ctry for j in n_ctry}
    
    # Add Excess Capacity constraints
    excess_cap = calc_excess_cap(dec_plant, dec_h, dec_r)
    for i in n_ctry:
        for j in n_lines:
            model.addConstr(excess_cap[i, j] >= 0, name=f"Excess_Cap_{i}_{j}")
    
    # Add Unmet Demand constraints (for both HighCal and Relax)
    unmet_demand = calc_unmet_demand(dec_h, dec_r)
    for i in n_ctry:
        for j in range(2):
            model.addConstr(unmet_demand[i, j] == 0, name=f"Unmet_Demand_{i}_{j}")
    
    model.update()
    
    # Set the objective function
    model.setObjective(
        calc_total_cost(dec_plant, dec_h, dec_r, base_yr=base_year, selected_yr=selected_year_sim, tariff=current_tariff),
        GRB.MINIMIZE
    )
    
    # Optimize the model for this simulation sample
    model.optimize()
    
    # Extract the optimal binary decisions for plant/line openings
    dec_plant_solution = {(i, j): dec_plant[(i, j)].x for i in n_ctry for j in n_lines}
    
    # Format the solution as a DataFrame (rows: plants, columns: overall plant decision, HighCal (H), Relax (R))
    plant_config = pd.DataFrame(
        [[dec_plant_solution[(i, j)] for j in n_lines] for i in n_ctry],
        index=caps.index,
        columns=['Plant', 'H', 'R']
    )
    
    # Store the results for this simulation run
    simulation_results.append({
        "sim_index": sim_index,
        "sim_exchange_rates": sim_sample,   # simulated exchange rates: [BRL, EUR, INR, JPY, MXN]
        "plant_configuration": plant_config,  # optimal decisions for this run
        "objective_value": model.objVal
    })

# =============================================================================
# STEP 4: Analyze the simulation results
# =============================================================================

# Initialize a DataFrame to count the number of times decisions are "open" (value 1)
config_count = pd.DataFrame(0, index=caps.index, columns=['Plant', 'H', 'R'])
for res in simulation_results:
    config_count += res["plant_configuration"]

print("Frequency of 'open' decisions (out of 100 draws):")
print(config_count)
