echo "verif: image0"
for net in ../mnist_littlenets/*; do
    echo "Total: $net"
    python3 analyzer.py ../mnist_nets/$net ../mnist_images/img0.txt 0.1
done
