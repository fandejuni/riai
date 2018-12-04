import analyzer

netname = "../mnist_nets/mnist_relu_3_10.txt"

specname = "../mnist_images/img0.txt"
epsilon = 0.0344

specname = "../mnist_images/img1.txt"
epsilon = 0.0397

netname = "../mnist_nets/mnist_relu_6_20.txt"
specname = "../mnist_images/img24.txt"
epsilon = 0.00001

_, _, nn, image, bounds_before, bounds_after, label = analyzer.doInterval(netname, specname, epsilon)
