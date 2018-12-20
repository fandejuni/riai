for net in ../mnist_nets/*; do
    echo "network: $net"
    for image in ../mnist_images/*; do
        echo "image: $image"
        python3 analyzer.py ../mnist_nets/$net ../mnist_images/$image 0.03 | grep -v Academic
    done
done
