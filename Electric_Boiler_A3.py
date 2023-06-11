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
)
import time
import matplotlib.pyplot as plt
import os
import pickle
import multiprocessing as mp
import logging
import platform

logging.getLogger("pyomo.core").setLevel(logging.ERROR)
os.system("clear")

'''
This code was created by Tom Savage 13/10/2022
A1 - Version Devan Patel 04/04/2023
> 
For help email trs20@ic.ac.uk or ring 07446880063. 
'''

# path where data of multiple parameters lives
data_path = "data/Gas_boiler_data.csv"

# number of data to take from this file
instance_number = 50
data = pd.read_csv(data_path,nrows=instance_number)
boiler_names = data['Technology'].values
boiler_energy_required = data['Gas Consumption (MWh/yr)'].values
boilers = len(boiler_names)

# NOTE 
# To define uncertainty enter the range of each parameter below.
# If no value for 'unc' is entered then a value will be assigned based 
# on the following variable
percentage_uncertainty = 5

# if all_percentage is 'True', given uncertainties are disregarded
# and they are all given a base percentage 
all_percentage = False

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
p["Natural Gas Boiler CAPEX (£/kW)"] = {"val": 168,'unc':5}
p["Electric Boiler CAPEX A"] = {"val": 187.57,'unc':5}
p["Electric Boiler CAPEX B"] = {"val": -0.37,'unc':0.05}
p["Natural Gas Boiler Fixed O&M Cost (£/kW/Yr)"] = {"val": 0.02,'unc':0.005}
p["Electric Boiler Fixed O&M Cost (£/kW/Yr)"] = {"val": 0.01,'unc':0.005}
p["Electric Boiler Grant (of CAPEX)"] = {"val": 0.5,'unc':0.05}
p["New Market Share"] = {"val": 0.04,'unc':0}

# here we iteratively create parameters from the list of energies imported earlier
    # contains key (string of boiler name...) - value (dictionary containing val & unc) pairs representing the thermal energy 
    # required from fuel for each boiler, along with their associated uncertainties. 
    # The number of key-value pairs added to p is equal to the value of boilers
    # ADD EXplaination & HOW to add differemt colour commenst to VSC
#Add eqvivalent electric boiler sizing (Groups 1 -5)...Cluster (Location)? 

for i in range(boilers):
    p[boiler_names[i] + ": Gas Consumption (MWh/year)"] = {
        "val": boiler_energy_required[i],'unc': boiler_energy_required[i] * 0.05
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

# Assign decision variables here. 
# The name of each variable is prepended with the name of the boiler and each
# is associated with upper and lower bounds
x = {} # create empty dictionary to store dicision variable x 
x["t"] = [-1e20, 1e20] # t can be any real number 

'''
 OTHER Policy Below 
Carbon Tax of CO2 emissions
feed-in tariff on electricity
Grant for electric boiler 50% of investment cost
Overall a list containing two elements  
'''
for boiler in boiler_names: 
    x[boiler + ": Electricity Subsidy (£/MWh)"] = [0, 100]
for boiler in boiler_names: 
    x[boiler + ": Carbon Tax (£/yr)"] = [0, 400]
    #x[boiler + ": Renewable hydrogen cost savings (£/yr)"] = [0, 0.9 * gas_consumption[j] * x[boiler + ": Electricity Subsidy (£/MWh)"][0]]
for boiler in boiler_names: 
    x[boiler + ": Gas Tax (£/yr)"] = [0, 0.2]
for boiler in boiler_names: 
    x[boiler + ": Annual Grant (£/yr)"] = [0, 0.5]

# Market share of each boiler.
for boiler in boiler_names: #Loops adds decision varaible for each boiler to assign a market share 
    x[boiler + ": Market Share"] = [0, 1] #make binary? or can lie bewteen 0 and 1

# this is needed to store all constraints
con_list = []

# this iteratively defines constraint functions based on an index i
# defines a function called make_c that creates and returns another function c. 
# The purpose of c is to represent a constraint in an optimization problem.

#def make_c(i):
 #   def c(x, p): #take two argument x and p
#
 #       return 
  #  return c    

#for i in range(boilers):
 #   c = make_c(i)
  #  con_list += [c]

def make_c(i):
    def c(x, p):
        # Market share 
        # name of boiler at index i... boiler efficieny baked in from above.
        boiler = boiler_names[i]
        #Electricity_required = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * (p["Natural Gas Boiler Efficiency"])/(p["Electric Boiler Efficiency"]) for j in range(boilers)]
        boiler_energy_total = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * (p["Natural Gas Boiler Efficiency"]) for j in range(boilers)]
        return x[boiler + ": Market Share"] - boiler_energy_total[i] / sum(boiler_energy_total)
    return c

# this adds constraints for indexes to the list
# ... don't touch!...will need to change to make multi period...?
for i in range(boilers):
    c = make_c(i)
    con_list += [c]


def c(x, p):
    # Market share 
    # name of boiler at index i... boiler efficieny baked in from above.
    # Market share set to a fixed value!!! p["Market Share (%)"]
    #boiler = boiler_names[i] boiler_energy_total = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * (p["Natural Gas Boiler Efficiency"]) for j in range(boilers)]
    A = sum([x[boiler_names[j]+ ": Market Share"]  for j in range(boilers)])
    return p["New Market Share"] - A


# this adds constraints for indexes to the list
# ... don't touch!
for i in range(boilers):
    c = make_c(i)
    con_list += [c]

#def c(x, p):
    '''
    CONSTRAINT GOES HERE 
    New invetsment Cost of electric boiler based on capacity installed perviously and learning rate
    Complex...due to clustering of electric boiler capacity 
    '''
    # name of boiler at index i
    #boiler_energy_total = [p[boiler_names[j] + ": Thermal Energy required from fuel (MJ/year)"]/(1000*p["Hydrogen LHV (MJ/kg)"]) for j in range(boilers)]
    #A = p['A0'] + sum([x[boiler_names[j]+ ": Market Share"]  for j in range(boilers)]) * sum(boiler_energy_total)
    #UC = p['UC0'] * (A / p['A0']) ** (log(1 - p['Learning Rate']) / log(2)) 
    #return (p['UC0'] - UC) / p['UC0'] - 0.04736
# adding constraint to list
#con_list += [c] # '+=' concatentate single element to a list or append contraint c to the con_list

def c(x, p):
    '''
    OBJECTIVE CONSTRAINT GOES HERE:
    Maximize the the market share of electric  
    This should have the form: x['t'] - obj(x,p)
    If you want to know why look up 'epigraph form' 
    of an uncertain optimisation problem
    '''
    # name of boiler at index i 
    #boiler = boiler_names[i]
    boiler_capacity_gas = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"]/
                            (p['Operating Hours (h)'] * p['Natural Gas Boiler Availability']) for j in range(boilers)]
    # Investment Cost gas
    ICg = [p['Natural Gas Boiler CAPEX (£/kW)'] * boiler_capacity_gas[j] * 1000 for j in range(boilers)]
    # Levelised Investment Cost using the parameters in p gas.
    #LIg = [(ICg[j] * p['Natural Gas Boiler Discount Rate'] * (1 + p['Natural Gas Boiler Discount Rate'] ** (p['Natural Gas Boiler Lifespan (Year)']))) / (1 + p['Natural Gas Boiler Discount Rate'] ** (p['Natural Gas Boiler Lifespan (Year)'])) - 1 for j in range(boilers)]
    LIg = [(icg * p["Natural Gas Boiler Discount Rate"] * (1 + (p["Natural Gas Boiler Discount Rate"] ** (p['Natural Gas Boiler Lifespan (Year)'])))) / ((1 + (p["Natural Gas Boiler Discount Rate"])) ** (p["Natural Gas Boiler Lifespan (Year)"] - 1)) for icg in ICg]

    # Variable O&M cost gas.
    VOMg = [boiler_capacity_gas[j] * (p["Natural Gas Price with CCL (£/MWh)"])  for j in range(boilers)]
    # Fixed O&M cost gas.
    FOMg = [(ICg[j] * p["Natural Gas Boiler Fixed O&M Cost (£/kW/Yr)"]) for j in range(boilers)]
    # Annualised cost of heating gas boiler, multiple LIg by 0 as gas boiler already exist. 
    ACHg = (VOMg + FOMg + LIg * 0)  

    # pair energy requirements to electric boiler 
    # Final electric thermal Boiler capacity rounded to 1, 5, 9, 15, 30, 60 MW 
    #boiler_Electricity_required = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"]]
    boiler_capacity_electric = [p[boiler_names[j] + ": Gas Consumption (MWh/year)"]/(p["Operating Hours (h)"] * p["Electric Boiler Availability"]) for j in range(boilers)]
    # Round capacity of each electric boiler to form clusters of similar capacity 
    # Marginal investment cost of each electric boiler
    MICe = [p['Electric Boiler CAPEX A'] * (capacity ** p['Electric Boiler CAPEX B']) for capacity in boiler_capacity_electric]
    # Investment Cost electric
    ICe = [mice * capacity * 1000 for mice, capacity in zip(MICe, boiler_capacity_electric)]# for j in range(boilers)]
    # Levelised Investment Cost using the parameters in p gas.
    LIe = [(ice * p["Electric Boiler Discount Rate"] * (1 + p["Electric Boiler Discount Rate"] ** (p['Electric Boiler Lifespan (Year)'])))/ ((1 + p["Electric Boiler Discount Rate"]) ** ((p['Electric Boiler Lifespan (Year)']-1))) for ice in ICe]
    # Variable O&M cost gas
    VOMe = [boiler_capacity_gas[j] * (p["Electricity Price with CCL (£/MWh)"])  for j in range(boilers)]
    # Fixed O&M cost gas
    FOMe = [ICe[j] * p["Electric Boiler Fixed O&M Cost (£/kW/Yr)"] for j in range(boilers)]
    # Annualised cost of heating gas boiler,  multiple LIg by 1 as electric boiler requires investment  
    ACHe = VOMe + FOMe + LIe
    
    # <= 0 ...AND This line returns the value of the constraint. The constraint is represented as an inequality, with the left-hand side 
    # Cost for industry for a given boiler switching from gas to electric
    #boiler = boiler_names[i]

    MS = x[boiler + ": Market Share"]
    CTaxg = [(1-MS) * x[boiler + ": Carbon Tax (£/yr)"] * val * p["Natural Gas Emission Factor (t CO2e/MWh)"] for val in (p[boiler_names[j] + ": Gas Consumption (MWh/year)"] for j in range(boilers))] 
    #CTaxg2 = [(val/(1-MS)) for val in CTaxg] 
    #CTaxe = x[boiler + ": Carbon Tax (£/yr)"] * [p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Electric Emission Factor (t CO2e/MWh)"] * p["Electric Boiler Efficiency"] for j in range(boilers)]
    CTaxe = [MS * x[boiler + ": Carbon Tax (£/yr)"] * val for val in (p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Electric Emission Factor (t CO2e/MWh)"] * p["Electric Boiler Efficiency"] for j in range(boilers))]
    B = [MS * p[boiler_names[j] + ": Gas Consumption (MWh/year)"] * p["Natural Gas Emission Factor (t CO2e/MWh)"] * x[boiler + ": Carbon Tax (£/yr)"] * p["Electric Boiler Efficiency"] for j in range(boilers)]
    ACHe2 = [ache * MS for ache in ACHe]
    ACHg2 = [achg * MS for achg in ACHg]
    Grant = [MS * p["Electric Boiler Grant (of CAPEX)"] * ice for ice in ICe]
    Taxg = [(1 - MS) * vomg * (1 + x[boiler + ": Gas Tax (£/yr)"]) - vomg for vomg in VOMg]

    Costg = (x - y - w - z for x, y, w, z in zip(B, CTaxe, CTaxg, Taxg))
    Costi = (q + (r - s - t - u) + v + w for q, r, s, t, u, v, w in zip(CTaxe, ACHe2, ACHg2, B, Grant, Taxg, CTaxg))
    Total_Cost = list(Costg) + list(Costi)
    Total_Cost_expr = sum(Total_Cost) 
    return x['t'] - Total_Cost_expr

# adding objective constraint to list
con_list += [c]

# Don't touch anything from here! This should... all work when run
# defines concerete optimisation model'm_upper' 
'''
The objective function obj is defined as the negative of the t variable, 
and the var_bounds function provides the variable bounds for each variable in x. 
The uncertain_bounds function provides the bounds for the uncertain parameter p 
for each constraint in the con_list.
'''

def obj(x):
    return -x['t']
#
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
solver = "ipopt"
m_upper = ConcreteModel()
m_upper.x = Set(initialize=x.keys())
m_upper.x_v = Var(m_upper.x, domain=Reals, bounds=var_bounds)
p_nominal = {}
for pk,pi in p.items():
    p_nominal[pk] = pi['val']
m_upper.cons = ConstraintList()
for con in con_list:
    m_upper.cons.add(expr=con(m_upper.x_v, p_nominal) <= 0)
m_upper.obj = Objective(expr=obj(m_upper.x_v), sense=minimize)
res = SolverFactory(solver).solve(m_upper)
nominal_obj = value(m_upper.obj)
term_con = res.solver.termination_condition
x_opt = value(m_upper.x_v[:])

enom = time.time()


global solve_subproblem

epsilon = 1e-4
def solve_subproblem(i, x_opt):
    s = time.time()
    con = con_list[i]
    m = ConcreteModel()
    m.p = Set(initialize=p.keys())
    m.p_v = Var(m.p, domain=Reals, bounds=uncertain_bounds)
    m.obj = Objective(expr=con(x_opt, m.p_v), sense=maximize)
    try:
        solvern = SolverFactory("ipopt")
        solvern.options["max_iter"] = 10000
        solvern.solve(m)
        p_opt_list = value(m.p_v[:])
        p_opt = {}
        p_keys = list(p.keys())
        for k in range(len(p_opt_list)):
            if p_opt_list[k] is None:
                p_opt_list[k] = p_nominal[p_keys[k]]
            p_opt[p_keys[k]] = p_opt_list[k]
        if value(m.obj) > epsilon:
            #print('Solved subproblem ',i,' in',time.time()-s,' seconds')
            return [value(m.obj), p_opt]
        else:
            #print('Solved subproblem ',i,' in',time.time()-s,' seconds')
            return [value(m.obj)]
    except ValueError:
        #print('Failed to solve subproblem ',i,' ...')
        return [None]

#pool = mp.Pool(mp.cpu_count()-1)
spt = []
while True:

    x_opt_nominal = value(m_upper.x_v[:])
    x_opt = {}
    for v in range(len(x)):
        x_opt[list(x.keys())[v]] = x_opt_nominal[v]
    robust = True
    s_s = time.time()
    res = []
    for i in range(len(con_list)):
        res.append(solve_subproblem(i, x_opt))
    #res = pool.starmap(solve_subproblem, [(i, x_opt) for i in range(len(con_list))])
    e_s = time.time()
    spt.append(e_s - s_s)


    robust = True
    mcv = 0 
    for i in range(len(res)):
        if len(res[i]) > 1:
            robust = False
            if res[i][0] > mcv:
                mcv = res[i][0]
            con = con_list[i]
            m_upper.cons.add(expr=con(m_upper.x_v, res[i][1]) <= 0)
    print('Maximum constraint violation: ',mcv)

    if robust is True:
        res = {}
        nominal_solution = {}
        for v in range(len(x)):
            robust_solution = x_opt
            nominal_solution[list(x.keys())[v]] = x_opt_nominal[v]
        print("Problem is robustly feasible")
        res["robust_solution"] = robust_solution
        res["nominal_solution"] = nominal_solution
        res["robust_objective"] = value(m_upper.obj)
        res["nominal_objective"] = nominal_obj

        break
    print("Solving upper level problem")
    res = SolverFactory(solver).solve(m_upper)
    term_con = res.solver.termination_condition
    if term_con is TerminationCondition.infeasible:
        print("Problem is robustly infeasible...")
        res = {}
        nominal_solution = {}
        for v in range(len(x)):
            nominal_solution[list(x.keys())[v]] = x_opt_nominal[v]
        res["robust_solution"] = None
        res["nominal_solution"] = x_opt_nominal
        res["robust_objective"] = None
        res["nominal_objective"] = nominal_obj
        break
