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

class GEN_COL(object):
    def __init__(self, max_depth):
        self.model = ConcreteModel()
        self.max_depth = max_depth
        self.results = {}
        self.reduced_cost = 0.0
        self.col = []
        
    # construct column generation problem for a location
    # w_p is a list of duals associated with cover_pop_center constraints
    # alpha is the dual variable for convex combiantion constraint for location l
    # slack is the slack variables associated with cover_pop_center constraints
    # bfs_list is the bfs list of population centers
    def gen_col_for_a_location(self, w, alpha_l, slack, bfs):
        ############ set ############
        self.model.P = RangeSet(len(slack), ordered=True)
        ########### decision variable ############
        self.model.y = Var(self.model.P, within = Binary)
        ########### paramters ###########
        # use this MUTABLE parameter to avoid trivia max-depth constraint
        def find_depth_per_pop_center(model, p):
            return bfs[p-1]
        self.model.d = Param(self.model.P, within = NonNegativeIntegers, initialize = find_depth_per_pop_center, mutable = True)
        
        # derive zeta values
        def find_obj_coef(model, p):
            if slack[p-1] > 0:
                return 1.0
            else: return 0.0
        self.model.zeta = Param(self.model.P, within = NonNegativeReals, initialize = find_obj_coef)
        ########## objective function ###########
        def obj_rule(model):
            return sum(model.zeta[p] * model.y[p] for p in model.P)
        self.model.obj = Objective(rule = obj_rule, sense = maximize)
        
        ########## Constraints ##############
        # assigned population center should not exceed max depth
        def max_depth_rule(model, p):
            return model.d[p] * model.y[p] <= self.max_depth
        self.model.max_depth_constr = Constraint(self.model.P, rule = max_depth_rule)
        
        # population centers in depth d can be assigned to this location
        # only if all population centers in depths [0, 1, ...,d-1] are assigned to this location
        def depth_order_rule(model, p1, p2):
            if (p1!= p2) and (bfs[p1-1] < bfs[p2-1]):
                return model.y[p1] - model.y[p2] >=0
            else:
                return Constraint.Skip
        self.model.depth_order_constr = Constraint(self.model.P, self.model.P, rule = depth_order_rule)
        solver = SolverFactory('cplex')
        self.results = solver.solve(self.model)
        for p in self.model.P:
            self.col.append(value(self.model.y[p]))
    
    def check_reduced_cost(self, w, alpha_l, exp_excess_rev):
        self.reduced_cost = exp_excess_rev - sum(value(self.model.y[p]) * w[p-1] for p in self.model.P) - alpha_l

