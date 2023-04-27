package nonGoMLNetwork

import (
	"github.com/moverest/mnist"
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

const learningRate float64 = 0.1

/*
Network

Abstract representation of a neural network. wHidden and wOutput correspond to
the weights for the hidden and output layer respectively. bHidden and bOutput
represent the biases for the hidden and output layers respectively. z1 and z2
represent the weighted sums for the hidden and output layers respectively, and
a1 and A2 represent the activation values for the hidden and output layers
respectively
*/
type Network struct {
	bHidden, bOutput, z1, a1, z2, A2 *mat.VecDense
	wHidden, wOutput                 *mat.Dense
}

func InitNetwork() (network *Network) {
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

// ForwardProp
// This method runs a forward propagation cycle and returns the cost
// of the training sample. The function also calculate a1 and a2 (the activation
// of each neuron in each layer, represented as vectors), and z1 and z2 (the
// uncorrected weighted sums for each neuron, represented as vectors) and stores
// them as fields in the *Network
func (network *Network) ForwardProp(input *mat.VecDense, correctLabel mnist.Label) (cost0 float64) {
	//initializes relevant matrices for layer 1
	z1 := mat.NewVecDense(16, nil)
	a1 := mat.NewVecDense(16, nil)

	//calculates z1 by multiplying the activations of the input layer by the weights
	//of the hidden layer and adding the biases of the hidden layer
	z1.MulVec(network.wHidden, input)
	z1.AddVec(z1, network.bHidden)
	for i := 0; i < 16; i++ {
		//calculates a1 by applying the sigmoid function to each element in z1
		a1.SetVec(i, Sigmoid(z1.AtVec(i)))
	}

	//the same steps are repeated to calculate z2 and A2
	z2 := mat.NewVecDense(10, nil)
	a2 := mat.NewVecDense(10, nil)
	z2.MulVec(network.wOutput, a1)
	z2.AddVec(z2, network.bOutput)
	for i := 0; i < 10; i++ {
		a2.SetVec(i, Sigmoid(z2.AtVec(i)))
	}

	cost0 = ComputeCost0(a2, correctLabel)

	//sets the relevant fields in the *Network
	network.z1 = z1
	network.a1 = a1
	network.z2 = z2
	network.A2 = a2

	return cost0
}

// BackProp this method generates new weights and biases for the *Network using the relevant partial derivatives
func (network *Network) BackProp(derivativesOfw1, derivativesOfw2 *mat.Dense, derivativesOfb1, derivativesOfb2 *mat.VecDense) {

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

// CalcDerivatives
// This method calculates the partial derivatives of the cost function with respect
// to the weights and biases of *Network. I believe that the current issue with the
// network lies somewhere in this method, however I don't have the knowledge to see
// where. It could also totally not be in here and in a completely different part
// of the code. Again, I don't know.
func (network *Network) CalcDerivatives(
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
			B := SigmoidDerivative(network.z2.AtVec(j))
			C := 2 * (network.A2.AtVec(j) - correctSlice[j])
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
		B := SigmoidDerivative(network.z2.AtVec(j))
		C := 2 * (network.A2.AtVec(j) - correctSlice[j])
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
				dSingle := SigmoidDerivative(network.z2.AtVec(n))
				eSingle := 2 * (network.A2.AtVec(n) * correctSlice[n])

				cedSingle := cSingle * dSingle * eSingle
				cdeTotal += cedSingle
			}

			A := a0.AtVec(j)
			B := SigmoidDerivative(network.z1.AtVec(j))
			CDEAverage := cdeTotal / numNeuronsOutputLayer

			derivative := A * B * CDEAverage

			w1DerivativesData = append(w1DerivativesData, derivative)
		}
	}
	derivativesOfw1 = mat.NewDense(rowsW1, columnsW1, w1DerivativesData)

	/*
		same steps only with the biases. cde still represents the same values as before.
		The only difference is that A is always 1, so it is not calculated
	*/
	for j := 0; j < network.bHidden.Len(); j++ {
		var CDETotal float64
		for n := 0; n < numNeuronsOutputLayer; n++ {
			cSingle := network.wOutput.At(n, j)
			dSingle := SigmoidDerivative(network.z2.AtVec(n))
			eSingle := 2 * (network.A2.AtVec(n) * correctSlice[n])

			CDESingle := cSingle * dSingle * eSingle

			CDETotal += CDESingle

		}

		B := SigmoidDerivative(network.z1.AtVec(j))
		CDEAverage := CDETotal / numNeuronsOutputLayer

		derivative := B * CDEAverage

		b1DerivativeData = append(b1DerivativeData, derivative)
	}
	derivativesOfb1 = mat.NewVecDense(16, b1DerivativeData)

	return derivativesOfw1, derivativesOfw2, derivativesOfb1, derivativesOfb2
}
