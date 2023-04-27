package nonGoMLNetwork

import (
	"github.com/moverest/mnist"
	"gonum.org/v1/gonum/mat"
	"math"
)

// Sigmoid implementation of the sigmoid activation function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// ComputeCost0 this function computes the cost value given a2 and the correct label
func ComputeCost0(a2 *mat.VecDense, correctLabel mnist.Label) (cost float64) {
	/*
		creates a slice representation of the label. For example, if the correct label
		is "8", the corresponding slice would be:
			"[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]"
	*/
	cost = 0
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

// SigmoidDerivative implementation of the derivative of the sigmoid function
func SigmoidDerivative(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}
