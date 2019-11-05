from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import psycopg2
import psycopg2.extras
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import urllib
import json
import operator
import ast
import random
from scipy.stats import norm
import sys

# global variables
nReconfig = 4
nTime = 10
nCapLevle = 9

T = range(1, nTime+1)
R = range(1, nCapLevle+1)
#LAND_COST = [0,0,0,0,178174.74,80754.26,0,0]
LAND_COST = [653399.8,
269495.6,
174239.6,
418176.2,
174240.4,
78852.8,
27442.6,
564539.2]

REV = [423, 471, 228] # revenue per visit per service

############# Node set ############
# Here are the nodes in each period from t= 1 to t = 10
# Nodes in t = 0 & t = 11 are (0,0) respectively
PB = {(i,j) for i in R for j in R if i<=j} | {(0,0)}

############# Arc set ############
# 1. from sink node to period 1 (t = 0)
# 2. from last period to sink node (t = 10)
# 3. building capacity does not change, people capacity changes (could stay the same, expand or contract) (t = 1..9)
# this also includes (t, 0, 0, 0, 0)
# 4. building capacity expands, and people capacity does not decrease (could stay the same or expand) (t = 1..9)

# ARCS = {(0,0,0,i2,j2) for (i2,j2) in PB} \
# | {(nTime, i1, i2, 0,0) for (i1, i2) in PB} \
# | {(t, i1, j1, i2, j1) for t in range(1,nTime) for (i1, j1) in PB for (i2, j1) in PB} \
# | {(t, i1, j1, i2, j2) for t in range(1,nTime) for (i1, j1) in PB for (i2, j2) in PB if (j2>j1) and (i2>=i1)}

# 1. open from (0,0) to (i2, j2) (t = 0..9), which also includes "stay closed" (t,0,0,0,0) for t = 0..9
# 2. close from (i1, j1) to (0,0) (t = 1..10), which also include "stay closed" for period nTime (10, 0,0,0,0)
# 3. building capacity does not change, people capacity changes (could stay the same, expand or contract) (t = 1..9)
# 4. building capacity expands, and people capacity does not decrease (could stay the same or expand) (t = 1..9)
ARCS = {(t,0,0,i2,j2) for t in range(0, nTime) for (i2,j2) in PB} \
| {(t, i1, j1, 0,0) for t in T for (i1, j1) in PB} \
| {(t, i1, j1, i2, j2) for t in range(1,nTime) for (i1, j1) in PB for (i2, j2) in PB if j1 == j2} \
| {(t, i1, j1, i2, j2) for t in range(1,nTime) for (i1, j1) in PB for (i2, j2) in PB if (j2>j1) and (i2>=i1)}

# read in data for fixed cost calculation
df = pd.read_csv('cstaff.csv')
# find number of patient visits that each capacity level could handle per year
df['capacity'] = df['Number of Physicians'] * 4000

demand = pd.read_csv('demand.csv')

# FIXED COST of arcs, FC is an array of three dictionaries.
# Each dictionary carries the fixed arc cost mapping of one service.
FC = []
# family practice
cost = {}
for (t, i1, j1, i2, j2) in ARCS:
    cost[t,i1, j1, i2, j2] = 0
    if t <= nTime - 1:
        if i2 > 0 or j2 > 0: # reconfiguration cost
            # administration cost for hiring/lay-off (diff between i2 and i1)
            # building size changing cost (diff between j2 and j1)
            # compensation of people (i2)
            # operation cost of building (j2)
            cost[t,i1, j1, i2, j2] += abs(df.loc[i2, 'Family Practice'] - df.loc[i1, 'Family Practice']) * 0.5 \
                + 78910 * abs(df.loc[j2, 'Number of Physicians'] - df.loc[j1, 'Number of Physicians']) \
                    + df.loc[i2, 'Family Practice'] \
                        + 0.054 * (78910 * df.loc[j2, 'Number of Physicians']  + 1221400 )
            # more adiministration cost to establish a new service
            if i1 == 0 and j1 ==0:
                cost[t,i1, j1, i2, j2] += df.loc[i2, 'Family Practice']
    if i1>0 and i2 == 0 and j2 == 0: # closing cost
        # administration cost for lay-off (-) (i1)
        # building's salvage value upon closing (+) (j1)
        #####################################################
        ######### NOTE: the first item should be (-), temporarily change to plus to match the number in old .scs file
        ####################################################
        cost[t,i1, j1, i2, j2] += + df.loc[i1, 'Family Practice'] * 0.5 \
            + ( 1-0.08*(t-1) ) * (78910 * df.loc[j1, 'Number of Physicians'] + 1221400)

# for (i2,j2) in PB:
#     cost[0,0,0,i2,j2] += LAND_COST[l-1]

FC.append(cost)
# internal medicine
cost = {}
for (t, i1, j1, i2, j2) in ARCS:
    cost[t,i1, j1, i2, j2] = 0
    if t <= nTime - 1:
        if i2 > 0 or j2 > 0: # reconfiguration cost
            # administration cost for hiring/lay-off
            # building size changing cost
            # compensation of people
            # operation cost of building
            cost[t,i1, j1, i2, j2] += abs(df.loc[i2, 'Internal Medincine'] - df.loc[i1, 'Internal Medincine']) * 0.5 \
                + 78910 * abs(df.loc[j2, 'Number of Physicians'] - df.loc[j1, 'Number of Physicians']) \
                    + df.loc[i2, 'Internal Medincine'] \
                        + 0.054 * (78910 * df.loc[j2, 'Number of Physicians']  + 1221400 )
        if i1>0 and i2 == 0 and j2 == 0: # closing cost
            # administration cost for lay-off (-) (i1)
            # building's salvage value upon closing (+) (j1)
            cost[t,i1, j1, i2, j2] += - df.loc[i1, 'Internal Medincine'] * 0.5 \
                + ( 1-0.08*(t-1) ) * (78910 * df.loc[j1, 'Number of Physicians'] + 1221400)

FC.append(cost)
# pediatrics
cost = {}
for (t, i1, j1, i2, j2) in ARCS:
    cost[t,i1, j1, i2, j2] = 0
    if t <= nTime - 1:
        if i2 > 0 or j2 > 0: # reconfiguration cost
            # administration cost for hiring/lay-off
            # building size changing cost
            # compensation of people
            # operation cost of building
            cost[t,i1, j1, i2, j2] += abs(df.loc[i2, 'Pediatrics'] - df.loc[i1, 'Pediatrics']) * 0.5 \
                + 78910 * abs(df.loc[j2, 'Number of Physicians'] - df.loc[j1, 'Number of Physicians']) \
                    + df.loc[i2, 'Pediatrics'] \
                        + 0.054 * (78910 * df.loc[j2, 'Number of Physicians']  + 1221400 )
        if i1>0 and i2 == 0 and j2 == 0: # closing cost
            # administration cost for lay-off (-) (i1)
            # building's salvage value upon closing (+) (j1)
            cost[t,i1, j1, i2, j2] += - df.loc[i1, 'Pediatrics'] * 0.5 \
                + ( 1-0.08*(t-1) ) * (78910 * df.loc[j1, 'Number of Physicians'] + 1221400)
FC.append(cost)

class RSCPP_LS(object):
    def __init__(self, location, service, assignment_vector):
        self.l = location
        self.s = service
        self.pVector = assignment_vector
        self.model = ConcreteModel()
        self.exp_excess_rev = 0.0
        self.results = {} # results from model solution
        
    def find_demand(self):
        pind = [ i+1 for i in range(len(self.pVector)) if self.pVector[i] == 1]
        self.dt = demand.loc[(demand['s'] == self.s) & (demand['p'].isin(pind))].drop({'s','p'}, axis=1).sum(axis = 0)
        
    def model_rscpp(self):
        model = self.model
        ####### sets ######
        # time
        model.T = RangeSet(nTime, ordered=True) # t is like [1, 2, ...,10], not including source (0) and sink (11)

        # arcs
        model.A = Set(dimen=5, initialize = sorted(ARCS)) # arc set includes the arcs from source to period 1 nodes, and arcs from last period nodes to sink

        # nodes in each period from t= 1 to t = 10 (not including source and sink node)
        model.N = Set(dimen = 2, initialize = sorted(PB))

        # capacity levels
        model.K = RangeSet(0, nCapLevle, ordered = True) # capacity set starts from 0

        ####### parameters ######
        def find_arc_fixed_cost(model, t, i1, j1, i2, j2):
            return FC[self.s][t,i1, j1, i2, j2]
        model.arc_fixed_cost = Param(model.A, within = Reals, initialize = find_arc_fixed_cost)

        def capacity_level(model,  k):
            return df.loc[k, 'Number of Physicians']
        model.num_phycisians = Param(model.K, within = NonNegativeReals, initialize = capacity_level)

        def capacity_in_each_level(model, k):
            return df.loc[k, 'capacity']
        model.capacity = Param(model.K, within = NonNegativeReals,initialize = capacity_in_each_level)

        def demand_per_time_period(model, t):
            return self.dt[t]
        model.demand = Param(model.T, within = NonNegativeReals, initialize = demand_per_time_period)

        # expected revenue, expected excess demand, expected excess capacity
        def compute_node_cost(model, t, k):
            if k == 0:
                return 0.0
            else:
                dd = model.demand[t]
                kk = model.capacity[k]
                theta = dd**(1/2.0)
                z = (kk - dd) / theta
                ED = theta * norm.pdf(z) + (dd - kk) * (1-norm.cdf(z))
                return (REV[self.s] * (dd - ED) - REV[self.s] * 1.2 * ED - REV[self.s] *(ED - dd + kk))
        model.expected_excess_revenue = Param(model.T, model.K, initialize = compute_node_cost)

        def find_arc_cost(model, t, i1, j1, i2, j2):
            if t == model.T.last():
                return -model.arc_fixed_cost[t, i1, j1, i2, j2]
            else:
                return -model.arc_fixed_cost[t, i1, j1, i2, j2] + model.expected_excess_revenue[t+1, i2]
        model.arc_cost = Param(model.A,  within = Reals, initialize = find_arc_cost)


        ######## variables ######
        # x[a] = 1 if arc a is selected in the shortest path
        model.x = Var(model.A, within=Binary)

        # people capacity (# of physicians) configured in each time period
        model.p_tilde = Var(model.T, within=NonNegativeReals)

        # building capacity (in terms of # of physicians that this facility can accomodate)
        model.b_tilde = Var(model.T, within=NonNegativeReals)

        ######## objective ######
        def obj_rule(model):
            return sum(model.arc_cost[t,i1,j1,i2,j2] * model.x[t,i1,j1,i2,j2] for (t,i1,j1,i2,j2) in model.A)
        model.obj = Objective(rule = obj_rule, sense = maximize)

        ######## constraints ######
        # (1) source node. one unit flow out of source node
        def source_node_rule(model):
            return sum(model.x[0,0,0,i2,j2] for (i2, j2) in model.N) == 1
        model.source_node_constraint = Constraint(rule = source_node_rule)

        # (2) sink node. one unit flow into the sink node
        def sink_node_rule(model):
            return sum(model.x[model.T.last(), i1, j1, 0, 0] for (i1, j1) in model.N) == 1
        model.sink_node_constraint = Constraint(rule = sink_node_rule)

        #(3) capacity planning node. flow-in = flow-out    
        def capacity_planning_node_rule(model, t, i, j):
            return sum(model.x[t-1, i1, j1, i, j] for (i1, j1) in PB if (t-1, i1, j1, i, j) in model.A) - sum(model.x[t, i, j, i2, j2] for (i2, j2) in PB if (t, i, j, i2, j2) in model.A) == 0
        model.capacity_planning_node_constraint = Constraint(model.T, model.N, rule = capacity_planning_node_rule)

        # (4) number of expansion/contraction limit
        def max_number_of_reconfig_rule(model):
            return sum(model.x[t, i1, j1, i2, j2] for (t, i1, j1, i2, j2) in model.A if ((i1!=i2) and (t<= nTime - 1))) <= nReconfig
        model.max_number_of_reconfig_constraint = Constraint(rule = max_number_of_reconfig_rule)

        #(7) people capacity p_tilde
        def people_capacity_rule(model,t):
            return sum(model.x[t, i, j, i2, j2] * model.num_phycisians[i] for (i,j) in model.N for (i2,j2) in model.N if (t, i, j, i2, j2) in model.A) - model.p_tilde[t] == 0
        model.people_capacity_constraint = Constraint(model.T, rule = people_capacity_rule)

        #  #(8) building capacity b_tilde
        def building_capacity_rule(model,t):
            return sum(model.x[t, i, j, i2, j2] * model.num_phycisians[j] for (i,j) in model.N for (i2,j2) in model.N if (t, i, j, i2, j2) in model.A) - model.b_tilde[t] == 0
        model.building_capacity_constraint = Constraint(model.T, rule = building_capacity_rule)
    
    def solve_rscpp(self):
        solver = SolverFactory('cplex')
        self.results = solver.solve(self.model)
        if (self.results.solver.status != SolverStatus.ok) or (self.results.solver.termination_condition != TerminationCondition.optimal):
        	sys.exit('rcspp is not solved to optimal')
        elif (self.s == 0):
        	self.exp_excess_rev = round(value(self.model.obj) - LAND_COST[self.l-1], 0)
        else:
        	self.exp_excess_rev = round(value(self.model.obj),0)