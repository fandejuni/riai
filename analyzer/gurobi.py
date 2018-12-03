# import gurobi libraries
from gurobipy import *
from math import inf

def connectRelu(m, H, RELU_H, a, b):
	if b <= 0:
		m.addConstr(RELU_H == 0)
	elif a >= 0:
		m.addConstr(RELU_H == H)
	else:
		m.addConstr(RELU_H >= 0)
		m.addConstr(RELU_H >= H)
		alpha = b / (b - a)
		m.addConstr(RELU_H <= alpha * (H - a))

def getMin(bound):
    return bound.contents.inf.contents.val.dbl

def getMax(bound):
    return bound.contents.sup.contents.val.dbl

def createNetwork(nn, x_min, x_max, bounds_before, k=0):
    # k = 0 -> from beginning
    # k = 0 -> x_* = image_*
    m = Model('NN')

    x = []
    for i in range(len(x_min)):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, name="x_" + str(i)))
    current_layer = x

    m.update()

    i = 0
    for a, b in zip(x_min, x_max):
        m.addConstr(x[i] >= a)
        m.addConstr(x[i] <= b)
        i += 1

    for i in range(k, nn.numlayer):

        w = nn.weights[i]
        RELU_H = []

        for j in range(len(list(w))):
            name = "h_" + str(i) + "_" + str(j)
            h = m.addVar(-inf, vtype=GRB.CONTINUOUS, name=name)
            relu_h = m.addVar(0, vtype=GRB.CONTINUOUS, name="RELU_" + name)
            value = 0
            for l in range(len(current_layer)):
                value += w[j][l] * current_layer[l]
            value += nn.biases[i][j]
            m.update()
            m.addConstr(value == h)

            bound = bounds_before[i][j]
            connectRelu(m, h, relu_h, getMin(bound), getMax(bound))

            RELU_H.append(relu_h)

        current_layer = RELU_H

    # return m
    # m.setObjective(current_layer[7], GRB.MAXIMIZE)
    # m.optimize()
        # print('Obj: %g' % m.objVal)
    # elif status == GRB.Status.INFEASIBLE:
        # print('Optimization was stopped with status %d' % status)
        # m.computeIIS()
        # for c in m.getConstrs():
            # if c.IISConstr:
                # print('%s' % c.constrName)

    # for v in m.getVars():
        # print(v.varName, v.x)
    # print('Obj :', m.objVal)

    # to_max = current_layer[7] - max_(current_layer[:7] + current_layer[8:])
    final_result = m.addVar(-inf, vtype=GRB.CONTINUOUS, name="final")

    m.update()

    for i in range(10):
        if i != 7:
            m.addConstr(final_result <= current_layer[7] - current_layer[i])

    m.setObjective(final_result, GRB.MAXIMIZE)

    # Optimize m
    m.optimize()
    status = m.status
    #if status == GRB.Status.OPTIMAL:
    for v in m.getVars():
        # print(v.varName)
        if v.varName[:3] == "h_2" or v.varName[:8] == "RELU_h_2":
            print('%s %g' % (v.varName, v.x))
    print("Result: ", m.objVal)

# m = Model('NN')

# Add Variables
# H = m.addVar(vtype=GRB.CONTINUOUS, name="H")
# RELU_H = m.addVar(vtype=GRB.CONTINUOUS, name="RELU_H")

# m.update()

# connectRelu(m, H, RELU_H, -2, 2)
#connectRelu(m, H, RELU_H, 0, 2)

# c = m.addConstr(H >= -1)
# c = m.addConstr(H <= 1)

# Add Objective Function
# m.setObjective(RELU_H, GRB.MAXIMIZE)
# m.setObjective(RELU_H, GRB.MINIMIZE)

