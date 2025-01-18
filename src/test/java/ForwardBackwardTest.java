import Evolution.Constants;
import Evolution.Evolution;
import Evolution.Evolution.EvolutionBuilder;
import Genome.Modifier;
import Genome.NN;
import Genome.enums.Activation;
import Genome.enums.Cost;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ForwardBackwardTest {

    private static final Constants Constants;
    private static final NN defaultNN;

    static {
        Evolution agentFactory = new Evolution.EvolutionBuilder().setInputNum(4).setOutputNum(3)
                .setDefaultHiddenAF(Activation.none).setOutputAF(Activation.arrays.none)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(1).build();
        Constants = agentFactory.Constants;
        defaultNN = NN.getDefaultNeuralNet(Constants);
        defaultNN.nodes.forEach(n -> n.bias = 0);
    }

    @Test
    void calculateTest1() {
        NN Network = (NN) defaultNN.clone();
        for (int i = 0; i < Constants.getInputNum(); i++)
            for (int j = 0; j < Constants.getOutputNum(); j++)
                Modifier.addEdge(Network, 1, i, j + Constants.getInputNum());

        assertArrayEquals(new double[]{4, 4, 4}, Network.calculateWeightedOutput(new double[]{1, 1, 1, 1}));
        Supplier<Double> randNum = () -> new Random().nextDouble(-1e8, 1e8);
        for (int i = 0; i < 1000; i++) {
            double[] input = new double[]{randNum.get(), randNum.get(), randNum.get(), randNum.get()};
            double sum = input[0] + input[1] + input[2] + input[3];
            assertArrayEquals(new double[]{sum, sum, sum}, Network.calculateWeightedOutput(input));
        }
    }

    @Test
    void calculateTest2() {
        NN Network = (NN) defaultNN.clone();
        Supplier<Double> randNum = () -> new Random().nextDouble(-1e8, 1e8);
        double[][] weights = new double[Constants.getInputNum()][Constants.getOutputNum()];
        for (int i = 0; i < weights.length; i++)
            for (int j = 0; j < weights[i].length; j++)
                weights[i][j] = randNum.get();

        for (int i = 0; i < Constants.getInputNum(); i++)
            for (int j = 0; j < Constants.getOutputNum(); j++)
                Modifier.addEdge(Network, weights[i][j], i, j + Constants.getInputNum());

        for (int count = 0; count < 1000; count++) {
            double[] input = new double[]{randNum.get(), randNum.get(), randNum.get(), randNum.get()};
            double[] output = new double[Constants.getOutputNum()];
            for (int i = 0; i < weights.length; i++)
                for (int j = 0; j < weights[i].length; j++) output[j] += weights[i][j] * input[i];
            assertArrayEquals(output, Network.calculateWeightedOutput(input));
        }
    }

    @Test
    void calculateTest3() {
        Constants Constants = new EvolutionBuilder().setInputNum(4).setOutputNum(3)
                .setDefaultHiddenAF(Activation.none).setOutputAF(Activation.arrays.sigmoid)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(1).build().Constants;
        NN Network = NN.getDefaultNeuralNet(Constants);
        Supplier<Double> randNum = () -> new Random().nextDouble(-1e8, 1e8);
        double[][] weights = new double[Constants.getInputNum()][Constants.getOutputNum()];
        for (int i = 0; i < weights.length; i++)
            for (int j = 0; j < weights[i].length; j++)
                weights[i][j] = randNum.get();

        for (int i = 0; i < Constants.getInputNum(); i++)
            for (int j = 0; j < Constants.getOutputNum(); j++)
                Modifier.addEdge(Network, weights[i][j], i, j + Constants.getInputNum());

        for (int count = 0; count < 1000; count++) {
            double[] input = new double[]{randNum.get(), randNum.get(), randNum.get(), randNum.get()};
            double[] output = new double[Constants.getOutputNum()];

            for (int i = 0; i < weights.length; i++)
                for (int j = 0; j < weights[i].length; j++)
                    output[j] += weights[i][j] * input[i];

            double[] outputCopy = Activation.arrays.sigmoid.calculate(output);
            for (int i = 0; i < output.length; i++) output[i] = Activation.sigmoid.calculate(output[i]);
            assertArrayEquals(outputCopy, output);
            assertArrayEquals(output, Network.calculateWeightedOutput(input));
        }
    }

    //the backpropagate test written for ML_Optimizers, adapted to be identical to
    @RepeatedTest(1000)
    void testBackPropagateANDNetwork() {
        Constants Constants = new EvolutionBuilder().setInputNum(2).setOutputNum(2)
                .setDefaultHiddenAF(Activation.sigmoid).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(1).build().Constants;
        NN Network = NN.getDefaultNeuralNet(Constants);

        for (int i = 0; i < 6; i++) {
            Modifier.addEdge(Network, Constants.getInitializedValue(), 0, 2 + i);
            Modifier.splitEdge(Network, Constants.getInitializedValue(), Constants.getDefaultHiddenAF(), 0);
        }

        for (int i : new int[]{1, 3})
            for (int j = 2; j < 8; j++)
                Modifier.addEdge(Network, Constants.getInitializedValue(), i, j);

        final int iterations = 1000;
        for (int i = 0; i < iterations; i++) {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == 1 && testInput[1] == 1 ? 1 : 0] = 1;

            NN.learn(Network, 0.3, 0.9, 0.9, 1e-4, new double[][]{testInput}, new double[][]{testOutput});

            if (evaluate(new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}}, new double[][]{
                    Network.calculateWeightedOutput(new double[]{0, 0}),
                    Network.calculateWeightedOutput(new double[]{0, 1}),
                    Network.calculateWeightedOutput(new double[]{1, 0}),
                    Network.calculateWeightedOutput(new double[]{1, 1})}, 1e-2)) break;
        }

        assertTrue(evaluate(new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}}, new double[][]{
                Network.calculateWeightedOutput(new double[]{0, 0}),
                Network.calculateWeightedOutput(new double[]{0, 1}),
                Network.calculateWeightedOutput(new double[]{1, 0}),
                Network.calculateWeightedOutput(new double[]{1, 1})}, 1e-2));
    }

    private boolean evaluate(double[][] expectedOutputs, double[][] actualOutputs, double delta) {
        assert expectedOutputs.length == actualOutputs.length;

        for (int i = 0; i < expectedOutputs.length; i++) {
            for (int j = 0; j < expectedOutputs[i].length; j++) {
                if (Math.abs(expectedOutputs[i][j] - actualOutputs[i][j]) > delta) {
//                    System.err.println("expected: " + expectedOutputs[i][j] + " but was: " + actualOutputs[i][j]);
                    return false;
                }
            }
        }
        return true;
    }
}
