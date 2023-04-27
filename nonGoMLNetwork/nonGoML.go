package nonGoMLNetwork

import (
	"fmt"
	"github.com/moverest/mnist"
	"gonum.org/v1/gonum/mat"
	"log"
	"math/rand"
	"time"
)

const sampleSetSize = 1000

func Run() {
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

	network := InitNetwork() //initializes network with random weights and biases

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
		cost0 := network.ForwardProp(a0, correctLabel)
		totalCost += cost0
		setCost += cost0

		//calculates the derivatives for each parameter matrix for this training example
		derivativesOfw1, derivativeOfw2, derivativeOfb1, derivativeOfb2 := network.CalcDerivatives(correctLabel, a0)

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

			network.BackProp(averageDerivativeOfw1, averageDerivativeOfw2, averageDerivativeOfb1, averageDerivativeOfb2)

			//calculates and prints the average cost for this sample set
			averageCost := setCost / sampleSetSize
			fmt.Printf("AVERAGE COST OF SAMPLE %v: %v\n", i+1, averageCost)
			fmt.Println(mat.Formatted(network.A2, mat.Squeeze()))
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

// returns a vector representation of an MNIST image
func initImageVector(image *mnist.Image) *mat.VecDense {
	vectorData := make([]float64, 784)
	for i := range vectorData {
		//normalizes the value of each pixel to a float64 between 0 and 1
		vectorData[i] = float64(image[i]) / 255
	}

	return mat.NewVecDense(784, vectorData)
}
