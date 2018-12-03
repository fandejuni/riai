netname = "../mnist_nets/mnist_relu_3_10.txt"
specname = "../mnist_images/img0.txt"
epsilon = 0.0178
epsilon = 0

import analyzer

nn, image, bounds_before, bounds_after = analyzer.doAnalysis(netname, specname, epsilon)

def read(x, bounds):
    a = []
    b = []
    for i in range(10):
        a.append(bounds[x][i].contents.inf.contents.val.dbl)
        b.append(bounds[x][i].contents.sup.contents.val.dbl)
    return a, b

x, _ = read(1, bounds_after)
real_y, _ = read(2, bounds_before)

y = [0.0 for _ in range(10)]
for i in range(10):
    for j in range(10):
        y[i] += nn.weights[2][i][j] * x[j]
    y[i] += nn.biases[2][i]
