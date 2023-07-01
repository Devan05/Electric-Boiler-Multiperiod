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
p = {}
# parameters that appear in the problem must be in this format
# New Parameters:
p["Natural Gas Price with CCL (£/MWh)"] = {"val": 26.7,'unc':5}
p["Electricity Price with CCL (£/MWh)"] = {"val":128.2,'unc':5}
p["Natural Gas Boiler Efficiency"] = {"val": 0.9,'unc':0.05}
p["Electric Boiler Efficiency"] = {"val": 0.99,'unc':0.01}
p["Natural Gas Boiler Lifespan (Year)"] = {"val": 30,'unc':5}
p["Electric Boiler Lifespan (Year)"] = {"val": 20,'unc':2}
p["Operating Hours (h)"] = {"val": 3260,'unc':300}
p["Natural Gas Boiler Availability"] = {"val": 0.9,'unc':0.05}
p["Electric Boiler Availability"] = {"val": 0.99,'unc':0.01}
p["Natural Gas Emission Factor (t CO2e/MWh)"] = {"val": 0.184,'unc':0.05}
p["Electric Emission Factor (t CO2e/MWh)"] = {"val": 0.121,'unc':0.05}
p["Natural Gas Boiler Discount Rate"] = {"val": 0.035,'unc':0.005}
p["Electric Boiler Discount Rate"] = {"val": 0.05,'unc':0.005}
p["Natural Gas Boiler CAPEX (£/kW)"] = {"val": 166,'unc':5}
p["Electric Boiler CAPEX A"] = {"val": 187.59,'unc':5}
p["Electric Boiler CAPEX B"] = {"val": -0.37,'unc':0.05}
p["Natural Gas Boiler Fixed O&M Cost (£/kW/Yr)"] = {"val": 0.02,'unc':0.005}
p["Electric Boiler Fixed O&M Cost (£/kW/Yr)"] = {"val": 0.01,'unc':0.005}
p["Electric Boiler Grant (of CAPEX)"] = {"val": 0.5,'unc':0.05}
p["New Market Share"] = {"val": 0.04,'unc':0}

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

# path where data of multiple parameters lives
data_path = "data/Gas_boiler_data.csv"
# number of data to take from this file
instance_number = 50

data = pd.read_csv(data_path,nrows=instance_number)
boiler_names = data['Technology'].values
boiler_energy_required = data['Gas Consumption (MWh/yr)'].values
boilers = len(boiler_names)
# iteratively create parameters from the list of energies imported earlier
for i in range(boilers):
    p[boiler_names[i] + ": Gas Consumption (MWh/year)"] = {
        "val": boiler_energy_required[i], 'unc': boiler_energy_required[i] * 0.05
    }
# iteratively define the uncertainty to be a percentage, or just keep 
# the value if it already exists
percentage_uncertainty = 5
all_percentage = False

for k, v in p.items(): #k = each key-value pair, in dictionary p. items() to iterate over each key-value pair in p, and for each iteration, the key is assigned to the variable k and the corresponding value is assigned to the variable v. 
    try:
         # Check if the key already has an uncertainty value
        key_test = p[k]['unc']
    except KeyError:  # If the key doesn't have an uncertainty value, calculate it as a percentage of the key's value and add it to the dictionary
        p[k]["unc"] = p[k]["val"] * percentage_uncertainty / 100
        # If all_percentage is True, overwrite any existing uncertainty values with the percentage value
    if all_percentage is True:
        p[k]["unc"] = p[k]["val"] * percentage_uncertainty / 100

x = {
    ": Gas Tax (£/yr)": [0, 0.2],
    ": Electricity Subsidy (£/MWh)": [0, 70],
    ": Carbon Tax (£/yr)": [0, 300],
    ": Annual Grant (%/100)": [0, 0.5],
}
x["t"] = [-1e20, 1e20]


x_bin = {"t": [0, 1]}

#x = {} # create empty dictionary to store dicision variable x 
#x_bin = {}
x_bin["t"] = [-1e20, 1e20] # t can be any real number 
#x["t"] = [-1e20, 1e20]

for boiler in boiler_names:
    x_bin[boiler + "bv"] = [0, 1] 
for boiler in boiler_names: #Loops adds decision varaible for each boiler to assign a market share 
    x[boiler + ": Market Share"] = [0, 1]

# this is needed to store all constraints
con_list = []

def make_c1(i):
    def c1(x, x_bin, p):
        # name of boiler at index i
        boiler = boiler_names[i]
        boiler_capacity_gas = (p[boiler + ": Gas Consumption (MWh/year)"] / (p['Operating Hours (h)'] * p['Natural Gas Boiler Availability']))
        ICg = (p['Natural Gas Boiler CAPEX (£/kW)'] * boiler_capacity_gas * 1000)
        LIg = (ICg * p["Natural Gas Boiler Discount Rate"] * ((1 + p["Natural Gas Boiler Discount Rate"]) ** p['Natural Gas Boiler Lifespan (Year)'])) /(((1 + p["Natural Gas Boiler Discount Rate"]) ** p["Natural Gas Boiler Lifespan (Year)"]) - 1)
        VOMg = (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Price with CCL (£/MWh)"])
        FOMg = ((ICg * p["Natural Gas Boiler Fixed O&M Cost (£/kW/Yr)"]))

        boiler_capacity_electric = (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"])/(p["Operating Hours (h)"] * p["Electric Boiler Availability"])
        MICe = (p['Electric Boiler CAPEX A'] * (boiler_capacity_electric ** p['Electric Boiler CAPEX B']))
        ICe = (MICe * boiler_capacity_electric * 1000)
        LIe = (ICe * p["Electric Boiler Discount Rate"] * ((1 + p["Electric Boiler Discount Rate"]) ** p['Electric Boiler Lifespan (Year)'])) /(((1 + p["Electric Boiler Discount Rate"]) ** p['Electric Boiler Lifespan (Year)']) - 1)
        VOMe = (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"]) * (p["Electricity Price with CCL (£/MWh)"])
        FOMe = (ICe * p["Electric Boiler Fixed O&M Cost (£/kW/Yr)"])

        ACHe2 = (VOMe + FOMe + LIe)
        ACHg2 = (VOMg + FOMg + LIg * 0)

        CTaxg = (x[": Carbon Tax (£/yr)"] * p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Emission Factor (t CO2e/MWh)"]) 
        CTaxe = (x[": Carbon Tax (£/yr)"] * (p[boiler + ": Gas Consumption (MWh/year)"] * p["Electric Emission Factor (t CO2e/MWh)"] * p["Electric Boiler Efficiency"]))
        Grant = (x[": Annual Grant (%/100)"] * ICe)
        Taxg = ((VOMg * (1 + x[": Gas Tax (£/yr)"]) - VOMg))
        FiTe = (p[boiler + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * x[": Electricity Subsidy (£/MWh)"] * p["Electric Boiler Efficiency"])
        
        ACOHe = (ACHe2 - FiTe + CTaxe - Grant)
        ACOHg = (ACHg2 + CTaxg + Taxg )
        return ACOHe - ACOHg
    return c1
for i in range(boilers):
    c11 = make_c1(i)
    con_list = con_list + [c11]

def c2(x, x_bin, p):
    #boiler = boiler_names[i]
    VOMg = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Price with CCL (£/MWh)"]  for j in range(boilers)]
    boiler_capacity_electric = [(p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"])/(p["Operating Hours (h)"] * p["Electric Boiler Availability"]) for j in range(boilers)]
    MICe = [p['Electric Boiler CAPEX A'] * (boiler_capacity_electric[j] ** p['Electric Boiler CAPEX B']) for j in range(boilers)]
    ICe = [mice * capacity * 1000 for mice, capacity in zip(MICe, boiler_capacity_electric)]

    CTaxg = [x[": Carbon Tax (£/yr)"] * p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Emission Factor (t CO2e/MWh)"] for j in range(boilers)]
    CTaxe = [x[": Carbon Tax (£/yr)"] * val for val in (p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * p["Electric Emission Factor (t CO2e/MWh)"] * p["Electric Boiler Efficiency"] for j in range(boilers))]
    Grant = [x[": Annual Grant (%/100)"] * ice for ice in ICe]
    Taxg = [(vomg * (1 + x[": Gas Tax (£/yr)"])) - vomg for vomg in VOMg]
    FiTe = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * x[": Electricity Subsidy (£/MWh)"] * p["Electric Boiler Efficiency"] for j in range(boilers)]

    BV = [x_bin[boiler_names[j] + "bv"] for j in range(boilers)]

    Cost1g = [((bv * ((bv * fite) - (bv * ctaxe) - ((1-bv) * ctaxg) - ((1-bv) * taxg))) + (bv * grant)) for fite, ctaxe, bv, ctaxg, taxg, grant in zip(FiTe, CTaxe, BV, CTaxg, Taxg, Grant)]
    SumFTCg = sum(Cost1g)
    return SumFTCg - 1000000

con_list = con_list + [c2]

def c3(x, x_bin, p):
    boiler_capacity_gas = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] / (p['Operating Hours (h)'] * p['Natural Gas Boiler Availability']) for j in range(boilers)]
    ICg = [p['Natural Gas Boiler CAPEX (£/kW)'] * boiler_capacity_gas[j] * 1000 for j in range(boilers)]
    VOMg = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Price with CCL (£/MWh)"]  for j in range(boilers)]
    FOMg = [(ICg[j] * p["Natural Gas Boiler Fixed O&M Cost (£/kW/Yr)"]) for j in range(boilers)]

    boiler_capacity_electric = [(p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"])/
                                (p["Operating Hours (h)"] * p["Electric Boiler Availability"]) for j in range(boilers)]
    MICe = [p['Electric Boiler CAPEX A'] * (boiler_capacity_electric[j] ** p['Electric Boiler CAPEX B']) for j in range(boilers)]
    ICe = [mice * capacity * 1000 for mice, capacity in zip(MICe, boiler_capacity_electric)]# for j in range(boilers)]
    LIe = [(ice * p["Electric Boiler Discount Rate"] * ((1 + p["Electric Boiler Discount Rate"]) ** p['Electric Boiler Lifespan (Year)'])) / (((1 + p["Electric Boiler Discount Rate"]) ** p['Electric Boiler Lifespan (Year)']) - 1) for ice in ICe]
    VOMe = [(p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"]) * (p["Electricity Price with CCL (£/MWh)"])  for j in range(boilers)]
    FOMe = [ICe[j] * p["Electric Boiler Fixed O&M Cost (£/kW/Yr)"] for j in range(boilers)]

    BV = [x_bin[boiler_names[j]+ "bv"] for j in range(boilers)]
    CTaxg = [x[": Carbon Tax (£/yr)"] * val for val in (p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Emission Factor (t CO2e/MWh)"] for j in range(boilers))]
    CTaxe = [x[": Carbon Tax (£/yr)"] * val for val in (p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Electric Emission Factor (t CO2e/MWh)"] * p["Electric Boiler Efficiency"] for j in range(boilers))]
    Grant = [x[": Annual Grant (%/100)"] * ice for ice in ICe]
    Taxg = [vomg * (1 + x[": Gas Tax (£/yr)"]) - vomg for vomg in VOMg]
    FiTe = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Boiler Efficiency"] * x[": Electricity Subsidy (£/MWh)"] * p["Electric Boiler Efficiency"] for j in range(boilers)]

    ACHe2 = [(vome + fome + lie) for vome, fome, lie in zip(VOMe, FOMe, LIe)]
    ACHg2 = [(vomg + fomg) for vomg, fomg in zip(VOMg, FOMg)]

    Cost1i = [(bv* ((bv * ctaxe) + bv*(ache2 - fite - grant - achg2) + ((1 - bv) * ctaxg) + ((1 - bv) * taxg))) for ctaxe, ache2, fite, grant, achg2, bv, ctaxg, taxg in zip(CTaxe, ACHe2, FiTe, Grant, ACHg2, BV, CTaxg, Taxg)]  
    SumFTCi = sum(Cost1i)
    return SumFTCi - 500000

con_list = con_list + [c3]

def make_c4(i):
    def c4(x, x_bin, p):
        # name of boiler at index i
        boiler = boiler_names[i]
        #BV = x_bin[boiler+ "bv"]
        boiler_energy_total = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * (p["Natural Gas Boiler Efficiency"]) for j in range(boilers)]
        D1 = (x[boiler + ": Market Share"] - boiler_energy_total[i] / sum(boiler_energy_total))
        D2 = x_bin[boiler+ "bv"]
        #B = x_bin[boiler+ "bv"]*(x[boiler + ": Market Share"] - boiler_energy_total[i] / sum(boiler_energy_total))
        return D1 * D2 
    return c4

for i in range(boilers):
    c44 = make_c4(i)
    con_list += [c44]

def c5(x, x_bin, p):
    boiler_energy_total = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * (p["Natural Gas Boiler Efficiency"]) for j in range(boilers)]
    NMS2 = [boiler_energy_total[j] / sum(boiler_energy_total) for j in range(boilers)]
    BV = [x_bin[boiler_names[j]+ "bv"] for j in range(boilers)]
    B3 = sum([nms2 * bv for nms2, bv in zip(NMS2, BV)])
    # TotalNMS = [x[boiler_names[j]+ ": Market Share"]  for j in range(boilers)]
    # BV = [x_bin[boiler_names[j]+ "bv"] for j in range(boilers)]
    # B5 = sum([nms3 * bv for nms3, bv in zip(TotalNMS, BV)])
    return (B3) - 0.082
con_list = con_list + [c5]

def c6(x, x_bin, p): 
    '''
    OBJECTIVE CONSTRAINT GOES HERE:
    Maximize the the market share of electric  
    This should have the form: x['t'] - obj(x,p)
    If you want to know why look up 'epigraph form' 
    of an uncertain optimisation problem
    '''
    #boiler_energy_total = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * (p["Natural Gas Boiler Efficiency"]) for j in range(boilers)]    
    #MS = [x_bin[boiler_names[j]+ "bv"] * boiler_energy_total[j] / sum(boiler_energy_total) for j in range(boilers)]
    TotalNMS = [x[boiler_names[j]+ ": Market Share"]  for j in range(boilers)]
    BV = [x_bin[boiler_names[j]+ "bv"] for j in range(boilers)]
    B4 = sum([nms3 * bv for nms3, bv in zip(TotalNMS, BV)])
    return x['t'] - (B4)
con_list = con_list + [c6]

'''
The objective function obj is defined as the negative of the t variable, 
and the var_bounds function provides the variable bounds for each variable in x. 
The uncertain_bounds function provides the bounds for the uncertain parameter p 
for each constraint in the con_list.
'''

def obj(x):
    return -x['t']

def var_bounds(m, i):
     return (x[i][0], x[i][1])

#nominal values minus uncertain parameters
def uncertain_bounds(m, i):
    return (p[i]["val"] - p[i]["unc"], p[i]["val"] + p[i]["unc"])

'''
The uncertain parameters p are used in the ConstraintList to define the constraints. 
The constraints are added to the model by calling the add method of m_upper.cons with each constraint expression. 
The objective function obj is added to the model as the obj attribute of m_upper.
'''

snom = time.time()
#solver = "bonmin"
m_upper = ConcreteModel()
m_upper.x = Set(initialize=x.keys())
m_upper.x_bin = Set(initialize=x_bin.keys(),)
m_upper.x_v = Var(m_upper.x, domain=Reals, bounds=var_bounds)
m_upper.x_bin_v = Var(m_upper.x_bin, domain=Binary)

p_nominal = {}
for pk,pi in p.items():
    p_nominal[pk] = pi['val']
m_upper.cons = ConstraintList()

for con in con_list:
    m_upper.cons.add(expr=con(m_upper.x_v, m_upper.x_bin_v, p_nominal) <= 0)

m_upper.obj = Objective(expr=obj(m_upper.x_v), sense=minimize)
#m_upper.pprint()
res=SolverFactory('mindtpy').solve(m_upper, mip_solver='glpk', nlp_solver='ipopt') 

#res = SolverFactory(solver).solve(m_upper)
print("res start")
print(res)
print("res end")
nominal_obj = value(m_upper.obj)
print(nominal_obj)
term_con = res.solver.termination_condition
print(term_con)
x_opt = value(m_upper.x_v[:])
x_opt_bin = value(m_upper.x_bin_v[:])
print(x_opt)
print(x_opt_bin)
print(res)
enom = time.time()

# global solve_subproblem

# epsilon = 1e-4
# def solve_subproblem(i, x_opt):
#     s = time.time()
#     con = con_list[i]
#     m = ConcreteModel()

#     m.p = Set(initialize=p.keys())
#     m.p_v = Var(m.p, domain=Reals, bounds=uncertain_bounds)

#     m.obj = Objective(expr=con(x_opt, m.p_v), sense=maximize)
#     try:
#         solvern = SolverFactory("bonmin")
#         solvern.options["max_iter"] = 10000
#         solvern.solve(m)
#         p_opt_list = value(m.p_v[:])
#         p_opt = {}
#         p_keys = list(p.keys())
#         for k in range(len(p_opt_list)):
#             if p_opt_list[k] is None:
#                 p_opt_list[k] = p_nominal[p_keys[k]]
#             p_opt[p_keys[k]] = p_opt_list[k]
#         if value(m.obj) > epsilon:
#             #print('Solved subproblem ',i,' in',time.time()-s,' seconds')
#             return [value(m.obj), p_opt]
#         else:
#             #print('Solved subproblem ',i,' in',time.time()-s,' seconds')
#             return [value(m.obj)]
#     except ValueError:
#         #print('Failed to solve subproblem ',i,' ...')
#         return [None]

# #pool = mp.Pool(mp.cpu_count()-1)
# spt = []
# while True:

#     x_opt_nominal = value(m_upper.x_v[:])
#     x_opt = {}
#     for v in range(len(x)):
#         x_opt[list(x.keys())[v]] = x_opt_nominal[v]
#     robust = True
#     s_s = time.time()
#     res = []
#     for i in range(len(con_list)):
#         res.append(solve_subproblem(i, x_opt))
#     #res = pool.starmap(solve_subproblem, [(i, x_opt) for i in range(len(con_list))])
#     e_s = time.time()
#     spt.append(e_s - s_s)

#     robust = True
#     mcv = 0 
#     for i in range(len(res)):
#         if len(res[i]) > 1:
#             robust = False
#             if res[i][0] > mcv:
#                 mcv = res[i][0]
#             con = con_list[i]
#             m_upper.cons.add(expr=con(m_upper.x_v, res[i][1]) <= 0)
#     print('Maximum constraint violation: ',mcv)

#     if robust is True:
#         res = {}
#         nominal_solution = {}
#         for v in range(len(x)):
#             robust_solution = x_opt
#             nominal_solution[list(x.keys())[v]] = x_opt_nominal[v]
#         print("Problem is robustly feasible")
#         res["robust_solution"] = robust_solution
#         res["nominal_solution"] = nominal_solution
#         res["robust_objective"] = value(m_upper.obj)
#         res["nominal_objective"] = nominal_obj

#         break
#     print("Solving upper level problem")
#     res = SolverFactory(solver).solve(m_upper)
#     term_con = res.solver.termination_condition
#     if term_con is TerminationCondition.infeasible:
#         print("Problem is robustly infeasible...")
#         res = {}
#         nominal_solution = {}
#         for v in range(len(x)):
#             nominal_solution[list(x.keys())[v]] = x_opt_nominal[v]
#         res["robust_solution"] = None
#         res["nominal_solution"] = x_opt_nominal
#         res["robust_objective"] = None
#         res["nominal_objective"] = nominal_obj
#         break
# print(res)
