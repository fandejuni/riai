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

def analyze(nn, LB_N0, UB_N0, label):   
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
               var+=1
           dimrem = elina_dimchange_alloc(0,num_in_pixels)
           for i in range(num_in_pixels):
               dimrem.contents.dim[i] = i
           elina_abstract0_remove_dimensions(man, True, element, dimrem)
           elina_dimchange_free(dimrem)

           all_bounds_before.append(elina_abstract0_to_box(man, element))

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

def doInterval(netname, specname, epsilon):
    #c_label = int(argv[4])
    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    nn = parse_net(netstring)
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low,0)

    label, _, _, _ = analyze(nn,LB_N0,UB_N0,0)
    start = time.time()

    verified_flag = None
    bounds_before = None
    bounds_after = None

    if(label==int(x0_low[0])):
        correctly_classified = True
        LB_N0, UB_N0 = get_perturbed_image(x0_low,epsilon)
        _, verified_flag, bounds_before, bounds_after = analyze(nn,LB_N0,UB_N0,label)
        # if(verified_flag):
            # print("verified")
        # else:
            # print("can not be verified")
    else:
        correctly_classified = False
        print("image not correctly classified by the network. expected label ",int(x0_low[0]), " classified label: ", label)
    end = time.time()
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

def getMin(bound):
    return bound.contents.inf.contents.val.dbl

def getMax(bound):
    return bound.contents.sup.contents.val.dbl

def createNetwork(nn, x_min, x_max, bounds_before, label, k=0):
    # k = 0 -> from beginning
    # k = 0 -> x_* = image_*
    m = Model('NN')

    x = []
    for i in range(len(x_min)):
        x.append(m.addVar(lb=x_min[i], ub=x_max[i], vtype=GRB.CONTINUOUS, name="x_" + str(i)))
    current_layer = x

    m.update()

    for i in range(k, nn.numlayer):

        RELU_H = []

        for j in range(len(list(nn.weights[i]))):

            bound = bounds_before[i][j]
            a = getMin(bound)
            b = getMax(bound)

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

            RELU_H.append(relu_h)

        current_layer = RELU_H

    maxi = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="maxi")
    final_result = m.addVar(lb=-inf, vtype=GRB.CONTINUOUS, name="final")
    m.update()

    l = current_layer[:label] + current_layer[label + 1:]
    m.addConstr(maxi == max_(l))
    m.addConstr(final_result == current_layer[label] - maxi)

    m.setObjective(final_result, GRB.MINIMIZE)

    m.Params.DualReductions = 0
    m.optimize()

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

def doAnalysis(netname, specname, epsilon):

    seq_tactic = False

    verified_flag, correctly_classified, nn, image, bounds_before, bounds_after, label = doInterval(netname, specname, epsilon)

    if not correctly_classified:
        print("interval: can not be verified")
        return
    if verified_flag:
        print("interval: verified")
        # return
    else:
        print("interval: not verified")

    def read(nn, x, bounds):
        a = []
        b = []
        # for bo in bounds[x]:
        # for i in range(len(bounds[x])):
        print("BEFORE")
        for i in range(len(nn.weights[x+1][0])):
            a.append(bounds[x][i].contents.inf.contents.val.dbl)
            # a.append(bo.contents.inf.contents.val.dbl)
            b.append(bounds[x][i].contents.sup.contents.val.dbl)
            # b.append(bo.contents.sup.contents.val.dbl)
        print("AFTER")
        return a, b

    if seq_tactic:
        for k in range(nn.numlayer - 1, -1, -1):
            print("Trying k = " + str(k) + "...")
            if k == 0:
                a, b = image
            else:
                a, b = read(nn, k-1, bounds_after)
            r = createNetwork(nn, a, b, bounds_before, label, k=k)

            if r > 0:
                print("verified")
                return
    else:
        comp_k = 1
        go_on = True
        while go_on:
            k = nn.numlayer - comp_k
            print("Trying k = " + str(k) + "...")
            if k == 0:
                a, b = image
            else:
                a, b = read(nn, k-1, bounds_after)
                print(a)
            r = createNetwork(nn, a, b, bounds_before, label, k=k)

            print("verification: k = " + str(k) + ", r = " + str(r))
            if r > 0:
                print("verified")
                return

            if comp_k == nn.numlayer:
                go_on = False
            else:
                comp_k = comp_k * 2
                if comp_k > nn.numlayer:
                    comp_k = nn.numlayer

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
