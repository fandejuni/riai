import sys
sys.path.insert(0, '../../ELINA/python_interface/')

import numpy as np
import re
import csv
from elina_box import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from elina_scalar import *
from elina_interval import *
from elina_linexpr0 import *
from elina_lincons0 import *
import ctypes
from ctypes.util import find_library
from gurobipy import *
import time

from math import inf

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')

lower_before, lower_after, upper_before, upper_after = None, None, None, None
netstring, specstring, nn, classified_label = None, None, None, None

small_value = 1e-8
n_to_treat = 200
debug = True
chrono = True

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.numlayer = 0
        self.ffn_counter = 0

def parse_bias(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    #return v.reshape((v.size,1))
    return v

def parse_vector(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    return v.reshape((v.size,1))
    #return v

def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []
    while i < len(text):
        if text[i] == '[':
            bal += 1
        elif text[i] == ']':
            bal -= 1
        elif text[i] == ',' and bal == 0:
            result.append(text[start:i])
            start = i+1
        i += 1
    if start < i:
        result.append(text[start:i])
    return result

def parse_matrix(text):
    i = 0
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    return np.array([*map(lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1]))])

def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = layers()
    while i < len(lines):
        if lines[i] in ['ReLU', 'Affine']:
            W = parse_matrix(lines[i+1])
            b = parse_bias(lines[i+2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer+= 1
            i += 3
        else:
            raise Exception('parse error: '+lines[i])
    return res
   
def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    with open('dummy', 'w') as my_file:
        my_file.write(text)
    data = np.genfromtxt('dummy', delimiter=',',dtype=np.double)
    low = np.copy(data[:,0])
    high = np.copy(data[:,1])
    return low,high

def get_perturbed_image(x, epsilon):
    image = x[1:len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon

    for i in range(num_pixels):
        if(LB_N0[i] < 0):
            LB_N0[i] = 0
        if(UB_N0[i] > 1):
            UB_N0[i] = 1
    return LB_N0, UB_N0


def generate_linexpr0(weights, bias, size):
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, bias)
    for i in range(size):
        elina_linexpr0_set_coeff_scalar_double(linexpr0,i,weights[i])
    return linexpr0

def performIntersection(man, element, i, a, b):

    #create an array of two linear constraints
    lincons0_array = elina_lincons0_array_make(2)

    #Create a greater than or equal to inequality for the lower bound
    lincons0_array.p[0].constyp = c_uint(ElinaConstyp.ELINA_CONS_SUPEQ)
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
    cst = pointer(linexpr0.contents.cst)

    #plug the lower bound “a” here
    elina_scalar_set_double(cst.contents.val.scalar, -a)
    linterm = pointer(linexpr0.contents.p.linterm[0])

    #plug the dimension “i” here
    linterm.contents.dim = ElinaDim(i)
    coeff = pointer(linterm.contents.coeff)
    elina_scalar_set_double(coeff.contents.val.scalar, 1)
    lincons0_array.p[0].linexpr0 = linexpr0

    #create a greater than or equal to inequality for the upper bound
    lincons0_array.p[1].constyp = c_uint(ElinaConstyp.ELINA_CONS_SUPEQ)
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
    cst = pointer(linexpr0.contents.cst)

    #plug the upper bound “b” here
    elina_scalar_set_double(cst.contents.val.scalar, b)
    linterm = pointer(linexpr0.contents.p.linterm[0])

    #plug the dimension “i” here
    linterm.contents.dim = ElinaDim(i)
    coeff = pointer(linterm.contents.coeff)
    elina_scalar_set_double(coeff.contents.val.scalar, -1)
    lincons0_array.p[1].linexpr0 = linexpr0

    #perform the intersection
    element = elina_abstract0_meet_lincons_array(man,True,element,lincons0_array)

    return element

def analyze(nn, LB_N0, UB_N0, label, old_lower_before=None, old_upper_before=None):
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer
    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

    all_bounds_before = []
    all_bounds_after = []

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv,num_pixels)
    for layerno in range(numlayer):
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
           weights = nn.weights[nn.ffn_counter]
           biases = nn.biases[nn.ffn_counter]
           dims = elina_abstract0_dimension(man,element)
           num_in_pixels = dims.intdim + dims.realdim
           num_out_pixels = len(weights)

           dimadd = elina_dimchange_alloc(0,num_out_pixels)    
           for i in range(num_out_pixels):
               dimadd.contents.dim[i] = num_in_pixels
           elina_abstract0_add_dimensions(man, True, element, dimadd, False)
           elina_dimchange_free(dimadd)
           np.ascontiguousarray(weights, dtype=np.double)
           np.ascontiguousarray(biases, dtype=np.double)
           var = num_in_pixels
           # handle affine layer

           for i in range(num_out_pixels):

               tdim= ElinaDim(var)
               linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
               element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)

               if old_lower_before is not None and layerno >= 1:
                   a = old_lower_before[layerno][i]
                   b = old_upper_before[layerno][i]
                   element = performIntersection(man, element, var, a, b)

               var+=1
 
           dimrem = elina_dimchange_alloc(0,num_in_pixels)
           for i in range(num_in_pixels):
               dimrem.contents.dim[i] = i
           elina_abstract0_remove_dimensions(man, True, element, dimrem)
           elina_dimchange_free(dimrem)

           bound = elina_abstract0_to_box(man, element)
           # for i in range(2):
               # a = bound[i].contents.inf.contents.val.dbl
               # b = bound[i].contents.sup.contents.val.dbl

           all_bounds_before.append(bound)

           # handle ReLU layer 
           if(nn.layertypes[layerno]=='ReLU'):
              element = relu_box_layerwise(man,True,element,0, num_out_pixels)
           nn.ffn_counter+=1 

           all_bounds_after.append(elina_abstract0_to_box(man, element))

        else:
           print(' net type not supported')

    dims = elina_abstract0_dimension(man,element)
    output_size = dims.intdim + dims.realdim
    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man,element)

    # if epsilon is zero, try to classify else verify robustness 

    verified_flag = True
    predicted_label = 0
    if(LB_N0[0]==UB_N0[0]):
        for i in range(output_size):
            inf = bounds[i].contents.inf.contents.val.dbl
            flag = True
            for j in range(output_size):
                if(j!=i):
                   sup = bounds[j].contents.sup.contents.val.dbl
                   if(inf<=sup):
                      flag = False
                      break
            if(flag):
                predicted_label = i
                break
    else:
        inf = bounds[label].contents.inf.contents.val.dbl
        for j in range(output_size):

            print(bounds[j].contents.inf.contents.val.dbl)
            assert(bounds[j].contents.inf.contents.val.dbl >= -small_value)

            if(j!=label):
                sup = bounds[j].contents.sup.contents.val.dbl
                if(inf<=sup):
                    predicted_label = label
                    verified_flag = False
                    break

    elina_interval_array_free(bounds,output_size)
    elina_abstract0_free(man,element)
    elina_manager_free(man)
    return predicted_label, verified_flag, all_bounds_before, all_bounds_after

def doInterval(netname, specname, epsilon, lower_before=None, upper_before=None):

    global netstring, specstring, nn, classified_label

    #c_label = int(argv[4])
    if netstring is None:
        with open(netname, 'r') as netfile:
            netstring = netfile.read()
    if specstring is None:
        with open(specname, 'r') as specfile:
            specstring = specfile.read()
    if nn is None:
        nn = parse_net(netstring)
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low,0)

    if classified_label is None:
        label, _, _, _ = analyze(nn,LB_N0,UB_N0,0)
        classified_label = label
    else:
        label = classified_label

    start = time.time()

    verified_flag = None
    bounds_before = None
    bounds_after = None

    if(label==int(x0_low[0])):
        correctly_classified = True
        LB_N0, UB_N0 = get_perturbed_image(x0_low,epsilon)
        _, verified_flag, bounds_before, bounds_after = analyze(nn,LB_N0,UB_N0,label, lower_before, upper_before)
    else:
        correctly_classified = False
        print("image not correctly classified by the network. expected label ",int(x0_low[0]), " classified label: ", label)
    end = time.time()

    if chrono:
        print("analysis time: ", (end-start), " seconds")

    return verified_flag, correctly_classified, nn, (LB_N0, UB_N0), bounds_before, bounds_after, label

# ------------------------------------------
# OUR CODE
# ------------------------------------------

def connectRelu(m, H, RELU_H, a, b):
	if b <= 0:
		m.addConstr(RELU_H == 0)
	elif a >= 0:
		m.addConstr(RELU_H == H)
	else:
		m.addConstr(RELU_H >= H)
		alpha = b / (b - a)
		m.addConstr(RELU_H <= alpha * (H - a))

def createNetwork(nn, x_min, x_max, lower_before, upper_before, label, k=0, last_layer=None):

    # k = 0 -> from beginning
    # k = 0 -> x_* = image_*
    m = Model('NN')
    m.setParam( 'OutputFlag', False )

    x = []
    for i in range(len(x_min)):
        x.append(m.addVar(lb=x_min[i], ub=x_max[i], vtype=GRB.CONTINUOUS, name="x_" + str(i)))
    previous_layer = None
    current_layer = x

    m.update()

    if last_layer is None:
        last_layer = nn.numlayer

    for i in range(k, last_layer):

        H = []
        RELU_H = []

        for j in range(len(list(nn.weights[i]))):

            a = lower_before[i][j]
            b = upper_before[i][j]

            name = "h_" + str(i) + "_" + str(j)
            h = m.addVar(lb=a, ub=b, vtype=GRB.CONTINUOUS, name=name)
            m.update()

            value = 0
            for l in range(len(current_layer)):
                value += nn.weights[i][j][l] * current_layer[l]
            value += nn.biases[i][j]
            m.addConstr(value == h)

            relu_h = m.addVar(lb=0, ub=max(b, 0), vtype=GRB.CONTINUOUS, name="RELU_" + name)
            m.update()
            connectRelu(m, h, relu_h, a, b)

            H.append(h)
            RELU_H.append(relu_h)

        previous_layer = H
        current_layer = RELU_H

    return m, previous_layer, current_layer

def improveBounds(nn, x_min, x_max, lower_before, upper_before, label, k, last_layer):

    m, current_layer, _ = createNetwork(nn, x_min, x_max, lower_before, upper_before, label, k, last_layer=last_layer)

    current_lower = lower_before[last_layer - 1]
    current_upper = upper_before[last_layer - 1]

    lower_bounds = [elem for elem in current_lower]
    upper_bounds = [elem for elem in current_upper]

    m.Params.DualReductions = 0

    length = len(current_layer)

    if debug:
        print("improve...", k, "->", last_layer)

    indices = list(range(length))
    indices.sort(key = lambda i: max(current_lower[i], 0) - current_upper[i])
    indices[:n_to_treat]

    for i in indices:

        if current_upper[i] > 0:

            m.setObjective(current_layer[i], GRB.MINIMIZE)
            m.optimize()
            value = printResults(m)
            lower_bounds[i] = value

            assert(current_lower[i] <= value + small_value)

            m.setObjective(current_layer[i], GRB.MAXIMIZE)
            m.optimize()
            value = printResults(m)
            upper_bounds[i] = value

            assert(value <= current_upper[i] + small_value)

    return lower_bounds, upper_bounds

def printResults(m):

    value = -1

    # DEBUG if INFEASIBLE OR UNBOUNDED
    status = m.status
    if status == GRB.Status.OPTIMAL:
        value = m.objVal
    elif status == GRB.Status.INFEASIBLE:
        print("ERROR: INFEASIBLE")
        m.computeIIS()
        for c in m.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
        for v in m.getVars():
            if v.IISLB:
                print("Lower bound", v.varName, v.LB)
            if v.IISUB:
                print("Upper bound", v.varName, v.UB)
        m.write("file.lp")
        sys.exit()
    elif status == GRB.Status.UNBOUNDED:
        print("ERROR: UNBOUNDED")

    return value

def analyzeEnd(nn, x_min, x_max, lower_before, lower_after, label, k=0):

    m, _, current_layer = createNetwork(nn, x_min, x_max, lower_before, lower_after, label, k)

    maxi = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="maxi")
    final_result = m.addVar(lb=-inf, vtype=GRB.CONTINUOUS, name="final")
    m.update()

    l = current_layer[:label] + current_layer[label + 1:]
    m.addConstr(maxi == max_(l))
    m.addConstr(final_result == current_layer[label] - maxi)

    m.setObjective(final_result, GRB.MINIMIZE)

    m.Params.DualReductions = 0
    m.optimize()

    value = printResults(m)

    return value

def convertBounds(bounds, nn):
    lower_bounds = []
    upper_bounds = []
    for i in range(nn.numlayer):
        lower_bounds.append([])
        upper_bounds.append([])
        for j in range(len(nn.weights[i])):
            lower_bounds[i].append(bounds[i][j].contents.inf.contents.val.dbl)
            upper_bounds[i].append(bounds[i][j].contents.sup.contents.val.dbl)
    return lower_bounds, upper_bounds

def doAnalysis(netname, specname, epsilon):

    global lower_before, lower_after, upper_before, upper_after

    def victory(verified):
        if verified:
            if chrono:
                print("Total time: ", time.time() - t0)
            print("verified")
            exit()

    def printTime(string, time):
        if chrono:
            print(string, time)

    t0 = time.time()
    verified_flag, correctly_classified, nn, image, bounds_before, bounds_after, label = doInterval(netname, specname, epsilon)
    t1 = time.time()

    if not correctly_classified:
        if debug:
            print("interval: can not be verified")
        exit()

    if verified_flag:
        if debug:
            print("interval: verified")
    else:
        if debug:
            print("interval: not verified")

    victory(verified_flag)

    lower_before, upper_before = convertBounds(bounds_before, nn)
    lower_after, upper_after = convertBounds(bounds_after, nn)

    def improveFromTo(start, end):
        t0 = time.time()
        if start == 0:
            a, b = image
        else:
            a = lower_after[start-1]
            b = upper_after[start-1]

        lower, upper = improveBounds(nn, a, b, lower_before, upper_before, label, start, end)
        lower_before[end - 1] = lower
        upper_before[end - 1] = upper
        printTime("improveFromTo", time.time() - t0)

    def doIntervalAgain():
        start = time.time()
        global lower_before, lower_after, upper_before, upper_after
 
        verified_flag, _, nn, image, bounds_before, bounds_after, label = doInterval(netname, specname, epsilon, lower_before, upper_before)
        lower_before, upper_before = convertBounds(bounds_before, nn)
        lower_after, upper_after = convertBounds(bounds_after, nn)

        if verified_flag:
            if debug:
                print("Now verified by interval!")
        else:
            if debug:
                print("Still not verified by interval")

        printTime("New Interval", time.time() - start)

        victory(verified_flag)

    def tryToFinishFrom(k):
        start = time.time()
        if k == 0:
            a, b = image
        else:
            a = lower_after[k-1]
            b = upper_after[k-1]
        r = analyzeEnd(nn, a, b, lower_before, upper_before, label, k=k)
        if debug:
            print("try to finish from", k, ": ", r)
        printTime("tryToFinish time", time.time() - start)    
        victory(r > 0)


    def finishSeq(n):
        for k in range(n - 1, -1, -1):
            tryToFinishFrom(k)


    # STRATEGIES / HEURISTICS

    printTime("First interval", t1 - t0)

    def stratOpti(n):
        for i in range(2, n+1):
            improveFromTo(0, i)
            doIntervalAgain()
            tryToFinishFrom(0)

    def stratMedium(n):
        for i in range(n-1):
            improveFromTo(i, i+2)
            doIntervalAgain()
            k = max(n-6, 0)
            tryToFinishFrom(k)
        if n > 6:
            tryToFinishFrom(0)
        for i in range(n//3):
            improveFromTo(i*3, (i+1)*3)
            doIntervalAgain()
            tryToFinishFrom(0)

    def startHard(n):
        improveFromTo(0,2)
        doIntervalAgain()
        tryToFinishSeq(n)

    neurons = len(nn.biases[0])
    layers = nn.numlayer

    if neurons < 400:
        stratOpti(layers)
    else:
        stratHard(layers)
        stratMedium(layers)
        stratOpti(layers)

    printTime("Total", time.time() - t0)

    print("can not be verified")

if __name__ == '__main__':
    from sys import argv
    if len(argv) < 3 or len(argv) > 4:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])

    doAnalysis(netname, specname, epsilon)
