import org.scalatest._
import MathUtils._
import scala.annotation.tailrec
import scala.util.Random

class PerceptronTest extends FunSuite with Matchers with GivenWhenThen with Inspectors {

  test("Should find weights for 4 patterns") {
    //Given
    val patterns = List(
      Pattern(inputs = List(1.0, 0, 0, 0), expectedResult = List(1.0, 0, 0, 0)),
      Pattern(inputs = List(0, 1.0, 0, 0), expectedResult = List(0, 1.0, 0, 0)),
      Pattern(inputs = List(0, 0, 1.0, 0), expectedResult = List(0, 0, 1.0, 0)),
      Pattern(inputs = List(0, 0, 0, 1.0), expectedResult = List(0, 0, 0, 1.0))
    )
    val delta = 0.03
    val perceptron = new Perceptron(patterns, delta)
    //When
    println(s"Training perceptron with patterns:\n\t${patterns.mkString("\n\t")}")
    val trainingResult = perceptron.train()
    println("Trained\n")
    //Then
    forAll(patterns) { pattern =>
      val actualResult = perceptron.propagate(pattern.inputs, trainingResult)
      val expectedResult = pattern.expectedResult
      println(s"Feeding:${pattern.inputs}")
      println(s"Expecting: $expectedResult (+-$delta) ")
      println(s"Actual:$actualResult\n")
        forAll(actualResult.zip(expectedResult)) { case (actual, expected) =>
        actual should be(expected +- delta)
      }
    }
  }

  case class Pattern(inputs: List[Double], expectedResult: List[Double])

  class Perceptron(patterns: List[Pattern], delta: Double) {
    type Weights = List[Double]
    type Inputs = List[Double]
    type Layer = List[Neuron]
    type ErrorSignal = Double
    val Step = 0.1
    val Bias = 1.0
    case class Neuron(weights: Weights)
    case class NeuronResult(weights: Weights, weightedSum: Double, output: Double)
    case class TrainingResult(hiddenLayer: Layer, outputLayer: Layer)

    def train(): TrainingResult = {
      val freshHiddenLayer: List[Neuron] = List.fill(2)(Neuron(List.fill(5)(randomWeight())))
      val freshOutputLayer: List[Neuron] = List.fill(4)(Neuron(List.fill(3)(randomWeight())))
      trainLoop(freshHiddenLayer, freshOutputLayer)
    }

    def propagate(inputs: Inputs, trainingResult: TrainingResult): List[Double] = {
      calcNetwork(inputs, trainingResult.hiddenLayer, trainingResult.outputLayer).map(_.output)
    }

    @tailrec
    private def trainLoop(hiddenLayer: Layer, outputLayer: Layer): TrainingResult = {
      val pattern = patterns(Random.nextInt(patterns.length))
      val inputs: Inputs = pattern.inputs
      val hiddenLayerResult = calcLayer(inputs, hiddenLayer)
      val outputLayerResult = calcLayer(hiddenLayerResult.map(_.output), outputLayer)
      if (isTrained(hiddenLayer, outputLayer)) {
        TrainingResult(hiddenLayer, outputLayer)
      } else {
        val outputLayerErrorSignals = calcOutputLayerSignals(pattern, outputLayerResult)
        val hiddenLayerErrorSignals = calcHiddenLayerSignals(hiddenLayerResult, outputLayer, outputLayerErrorSignals)
        val hiddenLayerNewNeurons = newNeurons(hiddenLayerResult, hiddenLayerErrorSignals, inputs)
        val outputLayerNewNeurons = newNeurons(outputLayerResult, outputLayerErrorSignals, hiddenLayerResult.map(_.output))
        trainLoop(hiddenLayerNewNeurons, outputLayerNewNeurons)
      }
    }

    private def calcNeuron(inputs: Inputs, neuron: Neuron): NeuronResult = {
      val weighSum = weightedSum(inputs, neuron.weights)
      NeuronResult(neuron.weights, weighSum, sigmoid(weighSum))
    }

    private def calcLayer(inputs: Inputs, neurons: List[Neuron]): List[NeuronResult] = neurons.map(n => calcNeuron(inputs :+ Bias, n))

    private def calcNetwork(inputs: Inputs, hiddenLayer: List[Neuron], outputLayer: List[Neuron]): List[NeuronResult] = {
      val hiddenLayerResult = calcLayer(inputs, hiddenLayer)
      val outputLayerResult = calcLayer(hiddenLayerResult.map(_.output), outputLayer)
      outputLayerResult
    }

    private def isTrained(hiddenLayer: Layer, outputLayer: Layer): Boolean = {
      patterns.forall { p =>
        val expectedResults = p.expectedResult
        val results = calcNetwork(p.inputs, hiddenLayer, outputLayer)
        results.zip(expectedResults).forall { case (actual, expected) => Math.abs(actual.output - expected) < delta}
      }
    }

    private def calcOutputLayerSignals(pattern: Pattern, outputLayerResult: List[NeuronResult]): List[ErrorSignal] = {
      outputLayerResult.zip(pattern.expectedResult).map { case (result, desiredOutput) =>
        sigmoidDerivative(result.weightedSum) * (desiredOutput - result.output)
      }
    }

    private def calcHiddenLayerSignals(hiddenLayerResult: List[NeuronResult], outputLayer: Layer, outputLayerErrorSignals: List[ErrorSignal]): List[ErrorSignal] = {
      hiddenLayerResult.zipWithIndex.map { case (neuronResult, neuronIndex) =>
        val correspondingOutputWeights = outputLayer.map(_.weights(neuronIndex))
        sigmoidDerivative(neuronResult.weightedSum) * weightedSum(outputLayerErrorSignals, correspondingOutputWeights)
      }
    }

    private def newNeurons(results: List[NeuronResult], errorSignals: List[ErrorSignal], previousLayerOutputs: List[Double]): List[Neuron] = {
      results.zip(errorSignals).map { case (result, signal) =>
        result.weights.zip(previousLayerOutputs :+ Bias).map { case (weight, output) => weight + Step * signal * output }
      }.map(Neuron)
    }
  }
}

object MathUtils {
  def sigmoid(x: Double): Double = {
    1.0 / (1.0 + Math.exp(-x))
  }

  def sigmoidDerivative(x: Double): Double = {
    Math.exp(x) / Math.pow(Math.exp(x) + 1.0, 2)
  }


  def weightedSum(input: List[Double], weights: List[Double]): Double = {
    input.zip(weights).map { case (value, weight) => value * weight }.sum
  }

  def randomWeight(): Double = {
    (Random.nextDouble() - 0.5) / 2.0
  }
}
