package goMLNetwork

import (
	"fmt"
	"github.com/kujenga/goml/mnist"
	"github.com/kujenga/goml/neural"
	"log"
)

func Run() {

	input := neural.Layer{
		Name:                    "input",
		Width:                   784,
		ActivationFunction:      nil,
		ActivationFunctionDeriv: nil,
	}

	hidden1 := neural.Layer{
		Name:                    "hidden1",
		Width:                   100,
		ActivationFunction:      nil,
		ActivationFunctionDeriv: nil,
	}

	output := neural.Layer{
		Name:                    "output",
		Width:                   10,
		ActivationFunction:      nil,
		ActivationFunctionDeriv: nil,
	}

	layers := make([]*neural.Layer, 3)
	layers[0] = &input
	layers[1] = &hidden1
	layers[2] = &output

	network := neural.MLP{
		Layers:       layers,
		LearningRate: 0.05,
		Introspect: func(step neural.Step) {
			fmt.Println(step.Epoch)
			fmt.Println(step.Loss)
		},
	}

	dataset, err := mnist.Read("MNISTDataset")
	if err != nil {
		log.Panic(err)
	}

	fmt.Println(network.Train(10, dataset.TrainInputs, dataset.TrainLabels))

	predictions := network.Predict(dataset.TestInputs)

	var correct float64 = 0
	for i := 0; i < len(predictions); i++ {
		if predictions[i].MaxVal() == dataset.TestLabels[i].MaxVal() {
			correct++
		} else {
			fmt.Println("***************")
			fmt.Printf("IMAGE NUM: %v\n", i+1)
			fmt.Printf("NETWORK GUESS: %v\n", predictions[i].MaxVal())
			fmt.Printf("CORRECT ANS: %v\n", dataset.TestLabels[i].MaxVal())
			fmt.Println("***************")

		}
	}

	fmt.Println(len(predictions))
	fmt.Println(correct)
	fmt.Println((correct / float64(len(predictions))) * 100)
}
