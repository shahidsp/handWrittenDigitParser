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

import (
	"fmt"
	"github.com/moverest/mnist"
	"gonum.org/v1/gonum/mat"
	"log"
	"math"
	"math/rand"
	"time"
)

const (
	sampleSetSize         = 1000
	learningRate  float64 = 0.1
)

func main() {
	rand.Seed(time.Now().UnixNano())

	trainingImages, err := mnist.LoadImageFile("./MNISTDataset/" + mnist.TrainingImageFileName) //loads MNIST training images
	if err != nil {
		log.Panic(err)
	}

	trainingLabels, err := mnist.LoadLabelFile("./MNISTDataset/" + mnist.TrainingLabelFileName) //loads MNIST training labels
	if err != nil {
		log.Panic(err)
	}

	trainingSetSize := len(trainingLabels)

	network := initNetwork() //initializes network with random weights and biases

	var totalCost, setCost float64

	for i := 0; i < trainingSetSize; i++ {
		//initializes the matrices and vectors that are used later
		//when calculating the average derivatives for each bias
		totalDerivativesOfw1 := mat.NewDense(16, 784, nil)
		totalDerivativesOfw2 := mat.NewDense(10, 16, nil)
		totalDerivativesOfb1 := mat.NewVecDense(16, nil)
		totalDerivativesOfb2 := mat.NewVecDense(10, nil)

		//selects random image and corresponding label
		randomNum := rand.Intn(len(trainingImages))
		correctLabel := trainingLabels[randomNum]
		//converts the image into a vector
		a0 := initImageVector(trainingImages[randomNum])

		//runs a forward propagation and stores the calculated cost of the training example
		cost0 := network.forwardProp(a0, correctLabel)
		totalCost += cost0
		setCost += cost0

		//calculates the derivatives for each parameter matrix for this training example
		derivativesOfw1, derivativeOfw2, derivativeOfb1, derivativeOfb2 := network.calcDerivatives(correctLabel, a0)

		//adds up each derivative for each training example
		totalDerivativesOfw1.Add(totalDerivativesOfw1, derivativesOfw1)
		totalDerivativesOfw2.Add(totalDerivativesOfw2, derivativeOfw2)
		totalDerivativesOfb1.AddVec(totalDerivativesOfb1, derivativeOfb1)
		totalDerivativesOfb2.AddVec(totalDerivativesOfb2, derivativeOfb2)

		//runs every sampleSetSize number of cycles
		if (i+1)%sampleSetSize == 0 {
			w1Rows, w1Columns := totalDerivativesOfw1.Caps()
			w2Rows, w2Columns := totalDerivativesOfw2.Caps()

			//initializes the matrices and vectors that store the average derivative
			//values
			averageDerivativeOfw1 := mat.NewDense(w1Rows, w1Columns, nil)
			averageDerivativeOfw2 := mat.NewDense(w2Rows, w2Columns, nil)
			averageDerivativeOfb1 := mat.NewVecDense(totalDerivativesOfb1.Len(), nil)
			averageDerivativeOfb2 := mat.NewVecDense(totalDerivativesOfb2.Len(), nil)

			//averages derivatives over the value of sampleSetSize
			averageDerivativeOfw1.Scale(1/sampleSetSize, totalDerivativesOfw1)
			averageDerivativeOfw2.Scale(1/sampleSetSize, totalDerivativesOfw2)
			averageDerivativeOfb1.ScaleVec(1/sampleSetSize, totalDerivativesOfb1)
			averageDerivativeOfb2.ScaleVec(1/sampleSetSize, totalDerivativesOfb2)

			network.backProp(averageDerivativeOfw1, averageDerivativeOfw2, averageDerivativeOfb1, averageDerivativeOfb2)

			//calculates and prints the average cost for this sample set
			averageCost := setCost / sampleSetSize
			fmt.Printf("AVERAGE COST OF SAMPLE %v: %v\n", i+1, averageCost)
			setCost = 0

			//fmt.Println(network.a2)
		}

		//removes the training image and label used to ensure it isn't used again
		trainingImages = append(trainingImages[:randomNum], trainingImages[randomNum+1:]...)
		trainingLabels = append(trainingLabels[:randomNum], trainingLabels[randomNum+1:]...)

	}

	//calculates and prints the average cost over the whole training set
	averageCost := totalCost / float64(trainingSetSize)
	fmt.Println("FINAL AVERAGE COST: ", averageCost)
}

/*
This function initializes the network in the form of a *Network. The network
is assigned random values for each parameter
*/
func initNetwork() (network *Network) {
	//creates []float64 with enough storage for all weights of layer 1
	wHiddenData := make([]float64, 12544)
	for i := range wHiddenData {
		//assigns a random value between -1 and 1 for each index in the slice
		wHiddenData[i] = -1 + rand.Float64()*2
	}

	//creates []float64 with enough storage for all weights of layer 2
	wOutputData := make([]float64, 160)
	for i := range wOutputData {
		//assigns a random value between -1 and 1 for each index in the slice
		wOutputData[i] = -1 + rand.Float64()*2
	}

	//creates []float64 with enough storage for all biases of layer 1
	bHiddenData := make([]float64, 16)
	for i := range bHiddenData {
		//assigns a random value between -1 and 1 for each index in the slice
		bHiddenData[i] = -1 + rand.Float64()*2
	}

	//creates []float64 with enough storage for all biases of layer 2
	bOutputData := make([]float64, 10)
	for i := range bOutputData {
		//assigns a random value between -1 and 1 for each index in the slice
		bOutputData[i] = -1 + rand.Float64()*2
	}

	//creates and matrix or vector for each parameter and using relevant data slice
	wHidden := mat.NewDense(16, 784, wHiddenData)
	wOutput := mat.NewDense(10, 16, wOutputData)
	bHidden := mat.NewVecDense(16, bHiddenData)
	bOutput := mat.NewVecDense(10, bOutputData)

	//creates *Network with relevant parameters
	network = &Network{
		bHidden: bHidden,
		bOutput: bOutput,
		wHidden: wHidden,
		wOutput: wOutput,
	}
	return network
}

// implementation of the sigmoid activation function
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// returns a vector representation of an MNIST image
func initImageVector(image *mnist.Image) *mat.VecDense {
	vectorData := make([]float64, 784)
	for i := range vectorData {
		//normalizes the value of each pixel to a float64 between 0 and 1
		vectorData[i] = float64(image[i]) / 255
	}

	return mat.NewVecDense(784, vectorData)
}

/*
this method runs a forward propagation cycle and returns the cost of the
training sample. The function also calculate a1 and a2 (the activation of each
neuron in each layer, represented as vectors), and z1 and z2 (the uncorrected
weighted sums for each neuron, represented as vectors) and stores them as fields
in the *Network
*/
func (network *Network) forwardProp(input *mat.VecDense, correctLabel mnist.Label) (cost0 float64) {
	//initializes relevant matrices for layer 1
	z1 := mat.NewVecDense(16, nil)
	a1 := mat.NewVecDense(16, nil)

	//calculates z1 by multiplying the activations of the input layer by the weights
	//of the hidden layer and adding the biases of the hidden layer
	z1.MulVec(network.wHidden, input)
	z1.AddVec(z1, network.bHidden)
	for i := 0; i < 16; i++ {
		//calculates a1 by applying the sigmoid function to each element in z1
		a1.SetVec(i, sigmoid(z1.AtVec(i)))
	}

	//the same steps are repeated to calculate z2 and a2
	z2 := mat.NewVecDense(10, nil)
	a2 := mat.NewVecDense(10, nil)
	z2.MulVec(network.wOutput, a1)
	z2.AddVec(z2, network.bOutput)
	for i := 0; i < 10; i++ {
		a2.SetVec(i, sigmoid(z2.AtVec(i)))
	}

	cost0 = computeCost0(a2, correctLabel)

	//sets the relevant fields in the *Network
	network.z1 = z1
	network.a1 = a1
	network.z2 = z2
	network.a2 = a2

	return cost0
}

// this function computes the cost value given a2 and the correct label
func computeCost0(a2 *mat.VecDense, correctLabel mnist.Label) (cost float64) {
	/*
		creates a slice representation of the label. For example, if the correct label
		is "8", the corresponding slice would be:
			"[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]"
	*/
	correctSlice := make([]float64, 10)
	correctSlice[int(correctLabel)] = 1

	//calculates the cost using the following equation:
	//	cost of training example  = summation of( (a2 at i - correct label at i )^2 )
	//I got this equation from 3blue1brown. It probably has a technical name, but I
	//don't know what it is
	for i := 0; i < len(correctSlice); i++ {
		cost += (a2.AtVec(i) - correctSlice[i]) * (a2.AtVec(i) - correctSlice[i])
	}

	return cost
}

// this method generates new weights and biases for the *Network using the relevant partial derivatives
func (network *Network) backProp(derivativesOfw1, derivativesOfw2 *mat.Dense, derivativesOfb1, derivativesOfb2 *mat.VecDense) {

	//scales the derivatives by learningRate
	derivativesOfw1Scaled := mat.NewDense(16, 784, nil)
	derivativesOfw1Scaled.Scale(learningRate, derivativesOfw1)

	derivativesOfw2Scaled := mat.NewDense(10, 16, nil)
	derivativesOfw2Scaled.Scale(learningRate, derivativesOfw2)

	derivativesOfb1Scaled := mat.NewVecDense(16, nil)
	derivativesOfb1Scaled.ScaleVec(learningRate, derivativesOfb1)

	derivativesOfb2Scaled := mat.NewVecDense(10, nil)
	derivativesOfb2Scaled.ScaleVec(learningRate, derivativesOfb2)

	//subtracts the weights and biases by their relevant derivatives
	network.wHidden.Sub(network.wHidden, derivativesOfw1Scaled)
	network.wOutput.Sub(network.wOutput, derivativesOfw2Scaled)
	network.bHidden.SubVec(network.bHidden, derivativesOfb1Scaled)
	network.bOutput.SubVec(network.bOutput, derivativesOfb2Scaled)
}

/*
Network

Abstract representation of a neural network. wHidden and wOutput correspond to
the weights for the hidden and output layer respectively. bHidden and bOutput
represent the biases for the hidden and output layers respectively. z1 and z2
represent the weighted sums for the hidden and output layers respectively, and
a1 and a2 represent the activation values for the hidden and output layers
respectively
*/
type Network struct {
	bHidden, bOutput, z1, a1, z2, a2 *mat.VecDense
	wHidden, wOutput                 *mat.Dense
}

/*
This method calculates the partial derivatives of the cost function with respect
to the weights and biases of *Network. I believe that the current issue with the
network lies somewhere in this method, however I don't have the knowledge to see
where. It could also totally not be in here and in a completely different part
of the code. Again, I don't know.
*/
func (network *Network) calcDerivatives(
	correctLabel mnist.Label,
	a0 *mat.VecDense,
) (
	derivativesOfw1, derivativesOfw2 *mat.Dense,
	derivativesOfb1, derivativesOfb2 *mat.VecDense,
) {
	/*
		creates a slice representation of the label. For example, if the correct label
		is "8", the corresponding slice would be:
			"[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]"
	*/
	correctSlice := make([]float64, 10)
	correctSlice[int(correctLabel)] = 1

	const numNeuronsOutputLayer = 10
	rowsW2, columnsW2 := network.wOutput.Caps()
	rowsW1, columnsW1 := network.wHidden.Caps()
	//initializes data slices for the derivative matrices and vectors
	var w2DerivativesData, w1DerivativesData, b2DerivativeData, b1DerivativeData []float64

	//calculates the derivative of the cost function with respect to the weights in layer 2
	for j := 0; j < rowsW2; j++ { //integrates through each row in the weight matrix (each neuron in the second layer)
		for k := 0; k < columnsW2; k++ { //iterates through each column of the weight matrix (each connection to a neuron in the previous layer)
			/*
				this formula implements the chain rule to calculate the partial derivatives of
				the cost function with respect to each weight. This formula was interpreted
				from this website by 3blue1brown:
				https://www.3blue1brown.com/lessons/backpropagation-calculus
				the function is as follows
					dC/dw = dz/dw * da/dw * dc/da
					(c) = cost of training example
					(w) = weight value
					(z) = weighted sum
					(a) = activation value
				In the way I coded it, A = (dz/dw), B = (da/dz), and C = (dc/dw)
			*/
			A := network.a1.AtVec(k)
			B := sigmoidDerivative(network.z2.AtVec(j))
			C := 2 * (network.a2.AtVec(j) - correctSlice[j])
			weightDerivative := A * B * C

			//adds the derivative to the data slice
			w2DerivativesData = append(w2DerivativesData, weightDerivative)
		}
	}
	//creates a matrix using the data slice
	derivativesOfw2 = mat.NewDense(10, 16, w2DerivativesData)

	for j := 0; j < network.bOutput.Len(); j++ { //iterates through rows in bias vector (each neuron in current layer)
		/*
			this formula implements the chain rule to calculate the partial derivatives of
			the cost function with respect to each bias. This formula was interpreted
			from this website by 3blue1brown:
			https://www.3blue1brown.com/lessons/backpropagation-calculus
			the function is as follows
				dc/db = dz/db * da/dw * dc/da
				(c) = cost of training example
				(b) = bias value
				(z) = weighted sum
				(a) = activation value
			In the way I coded it, A = (dz/db), B = (da/dz), and C = (dc/dw)
			A is always equal to 1, so it was left out
		*/
		B := sigmoidDerivative(network.z2.AtVec(j))
		C := 2 * (network.a2.AtVec(j) - correctSlice[j])
		biasDerivative := B * C

		b2DerivativeData = append(b2DerivativeData, biasDerivative)
	}
	derivativesOfb2 = mat.NewVecDense(len(b2DerivativeData), b2DerivativeData)

	/*
		The following loops calculate the partial derivatives with respect to the
		weights and biases for the hidden layer (layer 1). These formulas are more
		complex as each parameter affects all the output neurons. These formulas were
		also interpreted form 3blue1brown's website:
		https://www.3blue1brown.com/lessons/backpropagation-calculus

		The formula for the derivatives of the weights is as follows:
			dc/dw = cz/dw * da/dz * (SUMMATION(dc/da))/numOfNeuronsInOutputLayer
			c = cost
			w = weight
			z = weighted sum
			a = activation value

			All dc/da values must be calculated and averaged to determine
			the overall effect of the weight on the cost (I think)

		In the eay I coded it, A = dz/dw, B = da/dz, and
		C = (SUMMATION(dc/da))/numOfNeuronsInOutputLayer
	*/
	for j := 0; j < rowsW1; j++ { //iterates through each row in the weights matrix (each neuron in current layer)
		for k := 0; k < columnsW1; k++ { //iterates through each column in weights matrix (each connection to a neuron in the previous layer)
			//cdeTotal represents SUMMATION(dc/da)
			var cdeTotal float64
			for n := 0; n < numNeuronsOutputLayer; n++ { //this loop calculates dc/da for each neuron in the output layer and adds it to cdeTotal
				/*
					The formula for calculating the partial derivative of the cost function with
					respect to the activation in the output neuron is as follows:
						dc/da(hiddenLayer) = dz(outputLayer)/da(hiddenLayer) * da(outputLayer)/dz(outputLayer) * dc/da(outputLayer)

						cSingle = dz(outputLayer)/da(hiddenLayer)
						dSingle = da(outputLayer)/dz(outputLayer)
						eSingle = dc/da(outputLayer)
						cdeSingle = dc/da(hiddenLayer)
				*/
				cSingle := network.wOutput.At(n, j)
				dSingle := sigmoidDerivative(network.z2.AtVec(n))
				eSingle := 2 * (network.a2.AtVec(n) * correctSlice[n])

				cedSingle := cSingle * dSingle * eSingle
				cdeTotal += cedSingle
			}

			A := a0.AtVec(j)
			B := sigmoidDerivative(network.z1.AtVec(j))
			CDEAverage := cdeTotal / numNeuronsOutputLayer

			derivative := A * B * CDEAverage

			w1DerivativesData = append(w1DerivativesData, derivative)
		}
	}
	derivativesOfw1 = mat.NewDense(rowsW1, columnsW1, w1DerivativesData)

	/*
		same steps only with the biases. cde still represents the same values as before.
		The only difference is that A is always 1 so it is not calculated
	*/
	for j := 0; j < network.bHidden.Len(); j++ {
		var CDETotal float64
		for n := 0; n < numNeuronsOutputLayer; n++ {
			cSingle := network.wOutput.At(n, j)
			dSingle := sigmoidDerivative(network.z2.AtVec(n))
			eSingle := 2 * (network.a2.AtVec(n) * correctSlice[n])

			CDESingle := cSingle * dSingle * eSingle

			CDETotal += CDESingle

		}

		B := sigmoidDerivative(network.z1.AtVec(j))
		CDEAverage := CDETotal / numNeuronsOutputLayer

		derivative := B * CDEAverage

		b1DerivativeData = append(b1DerivativeData, derivative)
	}
	derivativesOfb1 = mat.NewVecDense(16, b1DerivativeData)

	return derivativesOfw1, derivativesOfw2, derivativesOfb1, derivativesOfb2
}

// implementation of the derivative of the sigmoid function
func sigmoidDerivative(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}
