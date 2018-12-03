# import gurobi libraries
from gurobipy import *
m = Model('NN')

def connectRelu(m, H, RELU_H, a, b):
	if b <= 0:
		m.addConstr(RELU_H == 0)
	elif a >= 0:
		m.addConstr(RELU_H == H)
	else:
		print("COOL")
		m.addConstr(RELU_H >= 0)
		m.addConstr(RELU_H >= H)
		alpha = b / (b - a)
		m.addConstr(RELU_H <= alpha * (H - a))

# Add Variables
H = m.addVar(vtype=GRB.CONTINUOUS, name="H")
RELU_H = m.addVar(vtype=GRB.CONTINUOUS, name="RELU_H")

m.update()

connectRelu(m, H, RELU_H, -2, 2)

c = m.addConstr(H >= -1)
c = m.addConstr(H <= 1)

# Add Objective Function
m.setObjective(RELU_H, GRB.MAXIMIZE)
# m.setObjective(RELU_H, GRB.MINIMIZE)

# Optimize m
m.optimize()

# Output formatted solution
for v in m.getVars():
    print(v.varName, v.x)
    print('Obj :', m.objVal)
