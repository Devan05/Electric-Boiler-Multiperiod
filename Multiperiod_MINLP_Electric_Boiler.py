from ast import Expression
import pandas as pd
import numpy as np
#import tqdm
#from tqdm import tqdm
from pyomo.environ import (
    log,
    TerminationCondition,
    ConcreteModel,
    Set,
    Var,
    Reals,
    ConstraintList,
    Objective,
    minimize,
    SolverFactory,
    value,
    maximize,
    Constraint,
    sqrt,
    NonNegativeReals,
)
import time
import matplotlib.pyplot as plt
import os
import pickle
import multiprocessing as mp
import logging
import platform
import csv
import random
import gurobipy
from pyomo.environ import *
from pyomo.opt import SolverFactory
from IPython.display import display
import sympy as sp
import math
from datetime import datetime


# path where data of multiple parameters lives output_data_0.819_1
data_path = "data/Gas_boiler_data.csv"
# number of data to take from this file
#instance_number = 490
#skiprows=range(1, 120),
data = pd.read_csv(data_path)#, nrows=instance_number)
boiler_names = data['Technology'].values
boiler_energy_required = data['Gas Consumption (MWh/yr)'].values
boilers = len(boiler_names)
percentage_uncertainty = 0.01

Total_Cost_Bound = 150000000
Target_MS = 0.5
Period = 1
# cluster_factors = {'Cluster_1': 0.7898399542794716*0.9659363289248456*0.9465508226401592, 'Cluster_5': 0.7822094430678808*0.837137116364921*0.8223401594268891*0.8567988323372042*0.8705505632961241*0.9072878561917624*0.9465508226401592,
#                    'Cluster_9': 0.8348143654846168*0.9072878561917624, 'Cluster_15': 0.8796356706393289*0.9330329915368074*0.9330329915368074, 'Cluster_30': 0.9226808345905884*0.9330329915368074, 'Cluster_60': 0.8959584598407622*0.8831631456895738*0.9659363289248456*0.9465508226401592}
cluster_factors = {'Cluster_1':1, 'Cluster_5': 1,
                   'Cluster_9': 1, 'Cluster_15': 1, 'Cluster_30': 1, 'Cluster_60': 1}
Attempt = 1

#1%	5%	14%	30%	50%	70%	86%	95%	99%	100%
#0.1	0.2	0.3	0.4	0.5	0.6	0.7	0.8	0.9	1


x = {
    ": Gas Tax (£/yr)": [0, 0.2],
    ": Electricity Subsidy (£/MWh)": [0, 100],
    ": Carbon Tax (£/tCO2e)": [0, 378],
    ": Annual Grant (%/100)": [0, 0.5],
    ": diff": [0, 150000],
}
x["t"] = [-1e20, 1e20]
x_bin = {}
# times = [1,2,3,4,5]
# for t in times:
#     for boiler in boiler_names:
#         x_bin[boiler + " bv" + str(t)] = [0, 1]
for boiler in boiler_names:
    x_bin[boiler + " bv"] = [0, 1]

#start period 2023? 
Projected_e_values = [130, 121, 124,	131, 131, 128, 128,	128, 130, 130, 125, 125, 125, 114]
Projected_g_values = [29, 31, 32, 33, 34, 35, 36, 36, 36, 36, 36, 36]
Projected_e_emissions = [0.146, 0.127, 0.095, 0.07, 0.061, 0.052, 0.048, 0.04, 0.032, 0.025, 0.02, 0.019]
Carbon_price = [256,260,264,268,272,276,280,285,289,293,298,302]
Bell_Shape = [0.01,	0.04, 0.09,	0.16, 0.2, 0.2,	0.16, 0.09, 0.04, 0.01]
Years = [2024, 2025, 2026, 2027, 2028,	2029, 2030,	2031, 2032,	2033, 2034]

period_to_value_e = {period: value for period, value in enumerate(Projected_e_values, start=1)}
period_to_value_g = {period: value for period, value in enumerate(Projected_g_values, start=1)}
period_to_value_emissions = {period: value for period, value in enumerate(Projected_e_emissions, start=1)}
period_to_bell = {period: value for period, value in enumerate(Bell_Shape, start=1)}
period_to_year= {period: value for period, value in enumerate(Years, start=1)}

cost_e = period_to_value_e[Period]
cost_g = period_to_value_g[Period]
emissions_e = period_to_value_emissions[Period]
bell_curve = period_to_bell[Period]
Year = period_to_year[Period]

all_percentage = False
p = {}
p["Natural Gas Price with CCL (£/MWh)"] = {"val": cost_g,'unc':0}
p["Electricity Price with CCL (£/MWh)"] = {"val":cost_e,'unc':0}
p["Natural Gas Boiler Efficiency"] = {"val": 0.9,'unc':0.0}
p["Electric Boiler Efficiency"] = {"val": 0.99,'unc':0.0}
p['Natural Gas Boiler Lifespan (Year)'] = {"val": 30,'unc':0}
p['Electric Boiler Lifespan (Year)'] = {"val": 20,'unc':0}
p["Operating Hours (h)"] = {"val": 3260,'unc':0}
p["Natural Gas Boiler Availability"] = {"val": 0.9,'unc':0}
p["Electric Boiler Availability"] = {"val": 0.99,'unc':0}
p["Natural Gas Emission Factor (t CO2e/MWh)"] = {"val": 0.184,'unc':0}
p["Electric Emission Factor (t CO2e/MWh)"] = {"val": emissions_e,'unc':0}
p["Natural Gas Boiler Discount Rate"] = {"val": 0.035,'unc':0}
p["Electric Boiler Discount Rate"] = {"val": 0.05,'unc':0}
p["Natural Gas Boiler CAPEX (£/kW)"] = {"val": 166,'unc':0}
p["Electric Boiler CAPEX A"] = {"val": 187.593553855194,'unc':0}
p["Electric Boiler CAPEX B"] = {"val": -0.370877617229971,'unc':0.0}
p["Natural Gas Boiler Fixed O&M Cost (£/kW/Yr)"] = {"val": 0.02,'unc':0.0}
p["Electric Boiler Fixed O&M Cost (£/kW/Yr)"] = {"val": 0.01,'unc':0.0}
#p["Electric Boiler Grant (of CAPEX)"] = {"val": 0.5,'unc':0.0}
p["Learning Rate"] = {"val": 0.05,'unc':0.0}

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

for i in range(boilers):
    p[boiler_names[i] + ": Gas Consumption (MWh/year)"] = {
        "val": boiler_energy_required[i], 'unc': boiler_energy_required[i] * 0.1
    }

# iteratively define the uncertainty to be a percentage, or just keep 
# the value if it already exists
for k, v in p.items(): #k = each key-value pair, in dictionary p. items() to iterate over each key-value pair in p, and for each iteration, the key is assigned to the variable k and the corresponding value is assigned to the variable v. 
    try:
         # Check if the key already has an uncertainty value
        key_test = p[k]['unc']
    except KeyError:  # If the key doesn't have an uncertainty value, calculate it as a percentage of the key's value and add it to the dictionary
        p[k]["unc"] = p[k]["val"] * percentage_uncertainty / 100
        # If all_percentage is True, overwrite any existing uncertainty values with the percentage value
    if all_percentage is True:
        p[k]["unc"] = p[k]["val"] * percentage_uncertainty / 100



con_list = []
ACOHe2 = []
def make_c1(i):
    def c1(x, x_bin, p):
        boiler = boiler_names[i]
        boiler_capacity_gas = (p[boiler + ": Gas Consumption (MWh/year)"] / (p['Operating Hours (h)'] * p['Natural Gas Boiler Availability']))
        ICg = (p['Natural Gas Boiler CAPEX (£/kW)'] * boiler_capacity_gas * 1000)
        LIg = (ICg * p["Natural Gas Boiler Discount Rate"] * ((1 + p["Natural Gas Boiler Discount Rate"]) ** p['Natural Gas Boiler Lifespan (Year)'])) /(((1 + p["Natural Gas Boiler Discount Rate"]) ** p["Natural Gas Boiler Lifespan (Year)"]) - 1)
        VOMg = (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Price with CCL (£/MWh)"])
        FOMg = ((ICg * p["Natural Gas Boiler Fixed O&M Cost (£/kW/Yr)"]))
        boiler_capacity_electric1 = (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"])/(p["Operating Hours (h)"] * p["Electric Boiler Availability"])
        def custom_round(y):
            if y <= 1:
                return 1
            elif 1 < y <= 5:
                return 5
            elif 5 < y <= 9:
                return 9
            elif 9 < y <= 15:
                return 15
            elif 15 < y <= 30:
                return 30
            else:
                return 60
        boiler_capacity_electric =  (custom_round(round(boiler_capacity_electric1, 2)))
        NI_factor = (boiler_capacity_electric * cluster_factors[f'Cluster_{boiler_capacity_electric}'])
        MICe = (p['Electric Boiler CAPEX A'] * (boiler_capacity_electric ** p['Electric Boiler CAPEX B']))
        ICe = (MICe * NI_factor * 1000)
        LIe = (ICe * p["Electric Boiler Discount Rate"] * ((1 + p["Electric Boiler Discount Rate"]) ** p['Electric Boiler Lifespan (Year)'])) /(((1 + p["Electric Boiler Discount Rate"]) ** p['Electric Boiler Lifespan (Year)']) - 1)
        VOMe = (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"]) * (p["Electricity Price with CCL (£/MWh)"])
        FOMe = (ICe * p["Electric Boiler Fixed O&M Cost (£/kW/Yr)"])

        ACHe2 = (VOMe + FOMe + LIe)
        ACHg2 = (VOMg + FOMg)

        CTaxg = (x[": Carbon Tax (£/tCO2e)"] * p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Emission Factor (t CO2e/MWh)"]) 
        CTaxe = (x[": Carbon Tax (£/tCO2e)"] * (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * p["Electric Emission Factor (t CO2e/MWh)"] * p["Electric Boiler Efficiency"]))
        Grant = (x[": Annual Grant (%/100)"] * ICe)
        Taxg = (x[": Gas Tax (£/yr)"] * VOMg)
        FiTe = (x[": Electricity Subsidy (£/MWh)"] * (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * p["Electric Boiler Efficiency"]))
        
        ACOHe = (ACHe2 - FiTe + CTaxe - Grant)
        ACOHg = (ACHg2 + CTaxg + Taxg)
        return (((ACOHe) - (ACOHg)))
    return c1
for i in range(boilers):
    c11 = make_c1(i)
    con_list = con_list + [c11]

def make_c1(i):
    def c1(x, x_bin, p):
        # name of boiler at index i
        boiler = boiler_names[i]
        boiler_capacity_gas = (p[boiler + ": Gas Consumption (MWh/year)"] / (p['Operating Hours (h)'] * p['Natural Gas Boiler Availability']))
        ICg = (p['Natural Gas Boiler CAPEX (£/kW)'] * boiler_capacity_gas * 1000)
        LIg = (ICg * p["Natural Gas Boiler Discount Rate"] * ((1 + p["Natural Gas Boiler Discount Rate"]) ** p['Natural Gas Boiler Lifespan (Year)'])) /(((1 + p["Natural Gas Boiler Discount Rate"]) ** p['Natural Gas Boiler Lifespan (Year)']) - 1)
        VOMg = (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Price with CCL (£/MWh)"])
        FOMg = ((ICg * p["Natural Gas Boiler Fixed O&M Cost (£/kW/Yr)"]))

        boiler_capacity_electric1 = (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"])/(p["Operating Hours (h)"] * p["Electric Boiler Availability"])
        def custom_round(y):
            if y < 1:
                return 1
            elif 1 < y <= 5:
                return 5
            elif 5 < y <= 9:
                return 9
            elif 9 < y <= 15:
                return 15
            elif 15 < y <= 30:
                return 30
            else:
                return 60
        boiler_capacity_electric = (custom_round(round(boiler_capacity_electric1, 2)))
        NI_factor = (boiler_capacity_electric * cluster_factors[f'Cluster_{boiler_capacity_electric}'])
        MICe = (p['Electric Boiler CAPEX A'] * (boiler_capacity_electric ** p['Electric Boiler CAPEX B']))
        ICe = (MICe * NI_factor * 1000)
        LIe = (ICe * p["Electric Boiler Discount Rate"] * ((1 + p["Electric Boiler Discount Rate"]) ** p['Electric Boiler Lifespan (Year)'])) /(((1 + p["Electric Boiler Discount Rate"]) ** p['Electric Boiler Lifespan (Year)']) - 1)
        VOMe = (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"]) * (p["Electricity Price with CCL (£/MWh)"])
        FOMe = (ICe * p["Electric Boiler Fixed O&M Cost (£/kW/Yr)"])

        ACHe2 = (VOMe + FOMe + LIe)
        ACHg2 = (VOMg + FOMg)

        CTaxg = (x[": Carbon Tax (£/tCO2e)"] * p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Emission Factor (t CO2e/MWh)"]) 
        CTaxe = (x[": Carbon Tax (£/tCO2e)"] * (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * p["Electric Emission Factor (t CO2e/MWh)"] * p["Electric Boiler Efficiency"]))
        Grant = (x[": Annual Grant (%/100)"] * ICe)
        Taxg = (x[": Gas Tax (£/yr)"] * VOMg)
        FiTe = (x[": Electricity Subsidy (£/MWh)"] * (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * p["Electric Boiler Efficiency"]))
        
        ACOHe = (ACHe2 - FiTe + CTaxe - Grant)
        ACOHg = (ACHg2 + CTaxg + Taxg)
        #x[": ACOHe_ACOHg"]
        return ((-((ACOHe)-(ACOHg))) - x[": diff"])
    return c1
for i in range(boilers):
    c11 = make_c1(i)
    con_list = con_list + [c11]

def c2(x, x_bin, p):
    VOMg = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Price with CCL (£/MWh)"]  for j in range(boilers)]
    boiler_capacity_electric1 = [(p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"])/(p["Operating Hours (h)"] * p["Electric Boiler Availability"]) for j in range(boilers)]
    def custom_round(y):
        if y < 1:
            return 1
        elif 1 < y <= 5:
            return 5
        elif 5 < y <= 9:
            return 9
        elif 9 < y <= 15:
            return 15
        elif 15 < y <= 30:
            return 30
        else:
            return 60
    boiler_capacity_electric = [custom_round(round(capacity, 2)) for capacity in boiler_capacity_electric1]
    NI_factor = [value * cluster_factors[f'Cluster_{value}'] for value in boiler_capacity_electric]
    MICe = [p['Electric Boiler CAPEX A'] * (boiler_capacity_electric[j] ** p['Electric Boiler CAPEX B']) for j in range(boilers)]
    ICe = [mice * capacity * 1000 for mice, capacity in zip(MICe, NI_factor)]# for j in range(boilers)]

    CTaxg = [x[": Carbon Tax (£/tCO2e)"] * p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Emission Factor (t CO2e/MWh)"] for j in range(boilers)]
    CTaxe = [x[": Carbon Tax (£/tCO2e)"] * val for val in (p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * p["Electric Emission Factor (t CO2e/MWh)"] * p["Electric Boiler Efficiency"] for j in range(boilers))]
    Grant = [x[": Annual Grant (%/100)"] * ice for ice in ICe]
    Taxg = [(vomg * (1 + x[": Gas Tax (£/yr)"])) - vomg for vomg in VOMg]
    FiTe = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * x[": Electricity Subsidy (£/MWh)"] * p["Electric Boiler Efficiency"] for j in range(boilers)]

    BV = [x_bin[boiler_names[j] + " bv"] for j in range(boilers)]

    Cost1g = [((bv * ((bv * fite) - (bv * ctaxe) - ((1-bv) * ctaxg) - ((1-bv) * taxg))) + (bv * grant)) for fite, ctaxe, bv, ctaxg, taxg, grant in zip(FiTe, CTaxe, BV, CTaxg, Taxg, Grant)]
    SumFTCg = sum(Cost1g)
    return ((SumFTCg)-(Total_Cost_Bound))
con_list = con_list + [c2]

boiler_capacity_electric2=[]
Investment_Cost_E2 = []
Emission_savings2 = []
def c3(x, x_bin, p):
    boiler_capacity_gas = [(p[boiler_names[j] + ": Gas Consumption (MWh/year)"] / (p['Operating Hours (h)'] * p['Natural Gas Boiler Availability'])) for j in range(boilers)]
    ICg = [p['Natural Gas Boiler CAPEX (£/kW)'] * boiler_capacity_gas[j] * 1000 for j in range(boilers)]
    VOMg = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Price with CCL (£/MWh)"]  for j in range(boilers)]
    FOMg = [(ICg[j] * p["Natural Gas Boiler Fixed O&M Cost (£/kW/Yr)"]) for j in range(boilers)]

    boiler_capacity_electric1 = [(p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"])/(p["Operating Hours (h)"] * p["Electric Boiler Availability"]) for j in range(boilers)]
    def custom_round(y):
        if y < 1:
            return 1
        elif 1 < y <= 5:
            return 5
        elif 5 < y <= 9:
            return 9
        elif 9 < y <= 15:
            return 15
        elif 15 < y <= 30:
            return 30
        else:
            return 60
    boiler_capacity_electric = [custom_round(round(capacity, 2)) for capacity in boiler_capacity_electric1]
    NI_factor = [value * cluster_factors[f'Cluster_{value}'] for value in boiler_capacity_electric]
    MICe = [p['Electric Boiler CAPEX A'] * (boiler_capacity_electric[j] ** p['Electric Boiler CAPEX B']) for j in range(boilers)]
    ICe = [mice * capacity * 1000 for mice, capacity in zip(MICe, NI_factor)]# for j in range(boilers)]
    LIe = [(ice * p["Electric Boiler Discount Rate"] * ((1 + p["Electric Boiler Discount Rate"]) ** p['Electric Boiler Lifespan (Year)'])) / (((1 + p["Electric Boiler Discount Rate"]) ** p['Electric Boiler Lifespan (Year)']) - 1) for ice in ICe]
    VOMe = [(p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"]) * (p["Electricity Price with CCL (£/MWh)"])  for j in range(boilers)]
    FOMe = [ICe[j] * p["Electric Boiler Fixed O&M Cost (£/kW/Yr)"] for j in range(boilers)]
    boiler_capacity_electric2.extend(boiler_capacity_electric)
    Investment_Cost_E2.extend(ICe)
    CO2_diff = [(p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Emission Factor (t CO2e/MWh)"]) - (p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * p["Electric Emission Factor (t CO2e/MWh)"] * p["Electric Boiler Efficiency"]) for j in range(boilers)]
    Emission_savings2.extend(CO2_diff)
    BV = [x_bin[boiler_names[j]+ " bv"] for j in range(boilers)]

    CTaxg = [x[": Carbon Tax (£/tCO2e)"] * val for val in (p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Emission Factor (t CO2e/MWh)"] for j in range(boilers))]
    CTaxe = [x[": Carbon Tax (£/tCO2e)"] * val for val in (p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * p["Electric Emission Factor (t CO2e/MWh)"] * p["Electric Boiler Efficiency"] for j in range(boilers))]
    Grant = [x[": Annual Grant (%/100)"] * ice for ice in ICe]
    Taxg = [x[": Gas Tax (£/yr)"] * vomg for vomg in VOMg]
    FiTe = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * x[": Electricity Subsidy (£/MWh)"] * p["Electric Boiler Efficiency"] for j in range(boilers)]

    ACHe2 = [(vome + fome + lie) for vome, fome, lie in zip(VOMe, FOMe, LIe)]
    ACHg2 = [(vomg + fomg) for vomg, fomg in zip(VOMg, FOMg)]

    Cost1i = [(bv* ((bv * ctaxe) + bv*(ache2 - fite - grant - achg2) + ((1 - bv) * ctaxg) + ((1 - bv) * taxg))) for ctaxe, ache2, fite, grant, achg2, bv, ctaxg, taxg in zip(CTaxe, ACHe2, FiTe, Grant, ACHg2, BV, CTaxg, Taxg)]
    SumFTCi = sum(Cost1i)
    return ((SumFTCi) - (Total_Cost_Bound))
con_list = con_list + [c3]

MarketShare2 = []
def c5(x, x_bin, p):
    boiler_energy_total = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * (p["Natural Gas Boiler Efficiency"]) for j in range(boilers)]
    NMS2 = [boiler_energy_total[j] / sum(boiler_energy_total) for j in range(boilers)]
    MarketShare2.extend(NMS2)
    BV = [x_bin[boiler_names[j]+ " bv"] for j in range(boilers)]
    B3 = sum([nms2 * bv for nms2, bv in zip(NMS2, BV)])
    return ((B3) - Target_MS)
con_list = con_list + [c5]

def c6(x, x_bin, p): 
    boiler_energy_total = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * (p["Natural Gas Boiler Efficiency"]) for j in range(boilers)]    
    MS = [x_bin[boiler_names[j]+ " bv"] *boiler_energy_total[j] / sum(boiler_energy_total) for j in range(boilers)]
    BV = [x_bin[boiler_names[j]+ " bv"] for j in range(boilers)]
    B4 = sum([nms3 * bv for nms3, bv in zip(MS, BV)])
    return x['t'] - (B4)
con_list = con_list + [c6]

def create_multi_period_model(num_periods, x, x_bin, p):

    def obj(x):
        return -x['t']
    def var_bounds(m_upper, i):
        return (x[i][0], x[i][1])
    #nominal values minus uncertain parameters
    def uncertain_bounds(m, i):
        return (p[i]["val"] - p[i]["unc"], p[i]["val"] + p[i]["unc"])

    snom = time.time()
    m_upper = ConcreteModel()

    
    #m_upper.T = RangeSet(1, num_periods)

    m_upper.x = Set(initialize=x.keys())
    m_upper.x_bin = Set(initialize=x_bin.keys())
    m_upper.x_v = Var(m_upper.x, domain=Reals, bounds=var_bounds)
    m_upper.x_bin_v = Var(m_upper.x_bin, domain=Binary)
    p_nominal = {}
    for pk,pi in p.items():
        p_nominal[pk] = pi['val']
    m_upper.cons = ConstraintList()

    for con in con_list:
        m_upper.cons.add(expr=con(m_upper.x_v, m_upper.x_bin_v, p_nominal) <= 0)

    m_upper.obj = Objective(expr=obj(m_upper.x_v), sense=minimize)
    return m_upper
m_upper = create_multi_period_model(Period, x, x_bin, p)

res = SolverFactory('mindtpy').solve(m_upper,
                                strategy='OA',
                                init_strategy='FP', 
                                iteration_limit=100000, time_limit=600,
                                mip_solver='cplex', mip_solver_args={'timelimit': 600},
                                nlp_solver='ipopt', nlp_solver_args={'timelimit': 600, 'options':{'max_iter':100000}}, tee=True, mip_solver_tee=True, nlp_solver_tee=True,)

nominal_obj = value(m_upper.obj)
term_con = res.solver.termination_condition
x_opt_bin2 = value(m_upper.x_bin_v[:])
x_opt = value(m_upper.x_v[:])
enom = time.time()
m_upper.display()

x_opt_bin1 = {}
my_dict = {"x_opt_bin": x_opt}


TCg = Total_Cost_Bound + value(m_upper.cons[len(m_upper.cons) -2])
TCi = Total_Cost_Bound + value(m_upper.cons[len(m_upper.cons) -3])

x_opt_nominal = value(m_upper.x_v[:])
x_opt = {}
for v in range(len(x)):
    x_opt[list(x.keys())[v]] = x_opt_nominal[v]
x_opt_bin_nominal = value(m_upper.x_bin_v[:])
x_opt_bin = {}
for v in range(len(x_bin)):
    x_opt_bin[list(x_bin.keys())[v]] = x_opt_bin_nominal[v]

print(x_opt)
Boiler_data = zip(boiler_names, boiler_energy_required)
dict1 = {}
for n, v in zip(boiler_names, boiler_energy_required):
    dict1.setdefault(n, []).append(v)

boilers_to_delete = [key.replace(' bv', '') for key, value in x_opt_bin.items() if value == 1.0]
new_dict = {key: value for key, value in dict1.items() if key not in boilers_to_delete}

print(new_dict)
folder_path = r"C:\Users\Devan\Desktop\rmpa\data"  # Replace this with the actual folder path
csv_filename = os.path.join(folder_path, f"output_data_LR_{Target_MS}_{Attempt}.csv")
data_list = [['Technology', 'Gas Consumption (MWh/yr)']]  # Initialize the list with header row

for boiler, value in new_dict.items():
    data_list.append([boiler, value[0]])

# Export the data to the CSV file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data_list)

print(f"Data exported to '{csv_filename}' successfully.")

df = pd.DataFrame(x_opt, index=[0])
def t1(x, my_dict, p):
    #boiler = boiler_names[i]
    boiler_capacity_gas = [(p[boiler_names[j] + ": Gas Consumption (MWh/year)"]['val'] / (p['Operating Hours (h)']['val'] * p['Natural Gas Boiler Availability']['val'])) for j in range(boilers)]

    ICg = [p['Natural Gas Boiler CAPEX (£/kW)']['val'] * boiler_capacity_gas[j] * 1000 for j in range(boilers)]
    VOMg = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"]['val'] * p["Natural Gas Price with CCL (£/MWh)"]['val'] for j in range(boilers)]
    FOMg = [ICg[j] * p["Natural Gas Boiler Fixed O&M Cost (£/kW/Yr)"]['val'] for j in range(boilers)]
    boiler_capacity_electric1 = [(p[boiler_names[j] + ": Gas Consumption (MWh/year)"]['val'] * p["Natural Gas Boiler Efficiency"]['val'])/(p["Operating Hours (h)"]['val'] * p["Electric Boiler Availability"]['val']) for j in range(boilers)]
    def custom_round(y):
        if y < 1:
            return 1
        elif 1 < y <= 5:
            return 5
        elif 5 < y <= 9:
            return 9
        elif 9 < y <= 15:
            return 15
        elif 15 < y <= 30:
            return 30
        else:
            return 60
    boiler_capacity_electric = [custom_round(round(capacity, 2)) for capacity in boiler_capacity_electric1]
    NI_factor = [value * cluster_factors[f'Cluster_{value}'] for value in boiler_capacity_electric]
    MICe = [p['Electric Boiler CAPEX A']['val'] * (boiler_capacity_electric[j] ** p['Electric Boiler CAPEX B']['val']) for j in range(boilers)]
    ICe = [mice * capacity * 1000 for mice, capacity in zip(MICe, NI_factor)]
    LIe = [(ice * p["Electric Boiler Discount Rate"]['val'] * ((1 + p["Electric Boiler Discount Rate"]['val']) ** p['Electric Boiler Lifespan (Year)']['val'])) / (((1 + p["Electric Boiler Discount Rate"]['val']) ** p['Electric Boiler Lifespan (Year)']['val']) - 1) for ice in ICe]
    VOMe = [(p[boiler_names[j] + ": Gas Consumption (MWh/year)"]['val'] * p["Natural Gas Boiler Efficiency"]['val']) * (p["Electricity Price with CCL (£/MWh)"]['val']) for j in range(boilers)]
    FOMe = [ICe[j] * p["Electric Boiler Fixed O&M Cost (£/kW/Yr)"]['val'] for j in range(boilers)]

    ACHe2 = [vome + fome + lie for vome, fome, lie in zip(VOMe, FOMe, LIe)]
    ACHg2 = [vomg + fomg for vomg, fomg in zip(VOMg, FOMg)  ]

    CTaxg = [x[": Carbon Tax (£/tCO2e)"] * val for val in (p[boiler_names[j] + ": Gas Consumption (MWh/year)"]['val'] * p["Natural Gas Emission Factor (t CO2e/MWh)"]['val'] for j in range(boilers))]
    CTaxe = [x[": Carbon Tax (£/tCO2e)"] * val for val in (p[boiler_names[j] + ": Gas Consumption (MWh/year)"]['val'] * p["Natural Gas Boiler Efficiency"]['val'] * p["Electric Emission Factor (t CO2e/MWh)"]['val'] * p["Electric Boiler Efficiency"]['val'] for j in range(boilers))]
    Grant = [x[": Annual Grant (%/100)"] * ice for ice in ICe]
    Taxg = [x[": Gas Tax (£/yr)"] * vomg for vomg in VOMg]
    FiTe = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"]['val'] * p["Natural Gas Boiler Efficiency"]['val'] * x[": Electricity Subsidy (£/MWh)"] * p["Electric Boiler Efficiency"]['val'] for j in range(boilers)]
    
    ACOHe = [(ache2 - fite + ctaxe - grant) for ache2, fite, ctaxe, grant in zip(ACHe2, FiTe, CTaxe, Grant)]
    ACOHg = [(achg2 + ctaxg + taxg) for achg2, ctaxg, taxg in zip(ACHg2, CTaxg, Taxg)]
    results = {
    'ACOHe': ACOHe,
    'ACOHg': ACOHg,}
    return results
result2 = t1(x_opt, my_dict, p)

# Create a DataFrame with the substituted values
df2 = pd.DataFrame({'Periods' : [Period],
                    'Target Market Share' : [Target_MS],
                    'Total Cost Gov' : [TCg],
                    'Total Cost Ind' : [TCi],
                    'Upper Boundary' : [Total_Cost_Bound],
                    #'(ACOHe) - (ACOHg)' : [ACOHe_ACOHg],
                    'LP' : 0.05,
                    })

New_Market_Share = [x * y for x, y in zip(x_opt_bin2, MarketShare2)]
Installed_capacity_per_boiler = [x1 * y1 for x1, y1 in zip(x_opt_bin2, boiler_capacity_electric2)]
Emission_savings = [x2 * y2 for x2, y2 in zip(Emission_savings2, x_opt_bin2)]
total_saved = sum(Emission_savings)
print(total_saved)
def sum_specific_numbers(Installed_capacity_per_boiler):
    specific_numbers = [1, 5, 9, 15, 30, 60]
    number_counts = {number: 0 for number in specific_numbers}
    for number in Installed_capacity_per_boiler:
        if number in specific_numbers:
            number_counts[number] += 1
    results = {}
    for number, count in number_counts.items():
        results[f'Cluster_{number}'] = number * count
    return results
Clusters = sum_specific_numbers(Installed_capacity_per_boiler)

divisors = [1, 5, 9, 15, 30, 60]
result = {key: Clusters[key] / divisors[i] for i, key in enumerate(Clusters)}

pairs = zip(boiler_capacity_electric2, Investment_Cost_E2)
dictionary = {}
for n, v in zip(boiler_capacity_electric2, Investment_Cost_E2):
    dictionary.setdefault(n, []).append(v)
for number in divisors:
    if number not in dictionary:
        dictionary[number] = [0]
#print(dictionary)
def apply_formula_to_clusters(Clusters, p):
    new_dict = {}
    for cluster, value in Clusters.items():
        cluster_name = cluster.split('_')[1]  # Extract the numeric part from the cluster name
        learning_rate = p["Learning Rate"]["val"]  # Default to 1.0 if not found in p dictionary
        if int(cluster_name) == 0 or value == 0:
            # If cluster_name is 0 or value is 0, set new_value to 1
            new_dict[cluster] = 1
        else:
            new_value = (value / int(cluster_name)) ** (-learning_rate)
            new_dict[cluster] = new_value
    return new_dict
cluster_factors2 = apply_formula_to_clusters(Clusters, p)
print(cluster_factors2)
NIC = {}
for key, value_list in dictionary.items():
    cluster_key = f'Cluster_{key}'
    multiplier = cluster_factors2.get(cluster_key, 0)
    multiplied_values = [value * multiplier for value in value_list]
    NIC[key] = multiplied_values
#print(NIC)
dict = {'Boiler number' : boiler_names,
        'Gas Consumption (MWh/yr)' : boiler_energy_required, 
        'Boiler Capacity (MW)' : boiler_capacity_electric2,
        'Binary variable': x_opt_bin2,
        'Boiler New Market Share' : New_Market_Share,
        'ACOHg' : result2['ACOHg'],
        'ACOHe' : result2['ACOHe'],
        'Investment Cost (£)' : Investment_Cost_E2,
        'Installed capacity per boiler (MW)' : Installed_capacity_per_boiler,
        }
df1 = pd.DataFrame(dict)
df3 = pd.DataFrame.from_dict(NIC, orient='index').transpose()
cluster_factors2 = {f'{k}_LF': [v] for k, v in cluster_factors2.items()}
result = {k: [v] for k, v in result.items()}

df4 = pd.DataFrame(cluster_factors2)
df5 = pd.DataFrame(result)
df6 = pd.DataFrame({'CO2 savings (t CO2e/yr)': [total_saved]})
df7 = pd.DataFrame({ 'Electricity Price': [cost_e],
                     'Gas Price': [cost_g],
                     'Emission Factor':  [emissions_e],
                     'Bell Curve Market Share' : [bell_curve],
                     'Year' : [Year]})

combined_df = pd.concat([df, df2, df1, df3,df4,df5,df6,df7], axis=1)
display(combined_df)
print(TCg)
print(TCi)
print(Total_Cost_Bound)
folder_path = r"C:\Users\Devan\Desktop\rmpa\rmpa\Electric Boiler Multiperiod\NMS Data\\S Curve 2.5LR\\"
filename = f'Cluster_Test_{Target_MS}_{Attempt}.xlsx'
file_path = folder_path + filename
combined_df.to_excel(file_path, index=False)

