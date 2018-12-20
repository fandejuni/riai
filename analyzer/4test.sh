for image in ../mnist_images/*; do
    echo "verif: $image"
    python3 analyzer.py ../mnist_nets/mnist_relu_4_1024.txt ../mnist_images/$image 0.006
    python3 analyzer.py ../mnist_nets/mnist_relu_4_1024.txt ../mnist_images/$image 0.005
done
