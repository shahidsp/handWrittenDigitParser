/*
This network is a 2 layer network. The hidden layer has 16 neurons and the
output layer has 10 which correspond to the 10 digits. The activation function
I'm using is the sigmoid function for all the layers. I heard ReLu is better,
but since this is my first network, I wanted to use something that I understand.

The network runs for a number of cycles (defined by the sampleSetSize const).
The derivative of the cost function with respect to each parameter matrix is
calculated for each training example and is added together.

Once a sampleSetSize number of cycles have run, the average derivative for each
parameter (weight and biases for each layer) is calculated and scaled by the
learningRate constant. The backProp() func is then called and the scaled average
derivatives are subtracted from the current network's weights and biases
matrices. The average cost for this sampleSet is also calculated and printed to
the console.

This full cycle runs for a trainingSetSize number of iterations. Finally, the
average cost over the entire training set is printed to the console.

In its current state, the networks cost is not decreasing and is instead
fluctuating by about +- 0.2 every cycle from the initial cost value. I think the
problem is occurring in calculus portion of the derivative calculation, but I
don't know what the problem is specifically. It could also be any number of
other problems as I am self-taught and I built network based on videos by
3blue1brown.

My "sources"
	https://www.3blue1brown.com/topics/neural-networks
	https://www.youtube.com/watch?v=w8yWXqWQYmU
*/

package main

import "handwrittenDigitParser/goMLNetwork"

func main() {
	//nonGoMLNetwork.Run()
	goMLNetwork.Run()
}
