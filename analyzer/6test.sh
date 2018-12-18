for image in ../mnist_images/*; do
    echo "verif: $image"
    python3 analyzer.py ../mnist_nets/mnist_relu_6_200.txt ../mnist_images/$image 0.01
done
