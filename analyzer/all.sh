for image in ../mnist_images/*; do
    echo "verif: $image"
    for net in ../mnist_nets/*; do
        echo "verif: $net"
        eval "python3 analyzer.py ../mnist_nets/$net ../mnist_images/$image 0.005"
    done
done
