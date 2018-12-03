import analyzer
import gurobipy as g
from math import inf

def connectRelu(m, H, RELU_H, a, b):
	if b <= 0:
		m.addConstr(RELU_H == 0)
	elif a >= 0:
		m.addConstr(RELU_H == H)
	else:
		m.addConstr(RELU_H >= H)
		alpha = b / (b - a)
		m.addConstr(RELU_H <= alpha * (H - a))

def getMin(bound):
    return bound.contents.inf.contents.val.dbl

def getMax(bound):
    return bound.contents.sup.contents.val.dbl

def createNetwork(nn, x_min, x_max, bounds_before, label, k=0):
    # k = 0 -> from beginning
    # k = 0 -> x_* = image_*
    m = g.Model('NN')

    x = []
    for i in range(len(x_min)):
        x.append(m.addVar(vtype=g.GRB.CONTINUOUS, name="x_" + str(i)))
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
            h = m.addVar(-inf, vtype=g.GRB.CONTINUOUS, name=name)
            relu_h = m.addVar(0, vtype=g.GRB.CONTINUOUS, name="RELU_" + name)
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

    maxi = m.addVar(-inf, vtype=g.GRB.CONTINUOUS, name="maxi")
    final_result = m.addVar(-inf, vtype=g.GRB.CONTINUOUS, name="final")

    m.update()

    l = current_layer[:label] + current_layer[label + 1:]
    m.addConstr(maxi == g.max_(l))
    m.addConstr(final_result == current_layer[label] - maxi)

    m.setObjective(final_result, g.GRB.MINIMIZE)

    m.optimize()
    status = m.status

    return m.objVal

def do(netname, specname, epsilon):

    nn, image, bounds_before, bounds_after = analyzer.doAnalysis(netname, specname, epsilon)

    def read(x, bounds):
        a = []
        b = []
        for i in range(10):
            a.append(bounds[x][i].contents.inf.contents.val.dbl)
            b.append(bounds[x][i].contents.sup.contents.val.dbl)
        return a, b

    results = []
    a, b = image
    for k in range(nn.numlayer):
        results.append(createNetwork(nn, a, b, bounds_before, 7, k=k))
        a, b = read(k, bounds_after)

    print(results)


netname = "../mnist_nets/mnist_relu_3_10.txt"
specname = "../mnist_images/img0.txt"
epsilon = 0.0344

do(netname, specname, epsilon)
