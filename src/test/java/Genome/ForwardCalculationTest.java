package Genome;

import Evolution.Constants;
import Evolution.Evolution;
import Evolution.Evolution.EvolutionBuilder;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ForwardCalculationTest {

    private static final Constants Constants;
    private static final NN defaultNN;

    static {
        Evolution agentFactory = null;
        try {
            agentFactory = new EvolutionBuilder().setInputNum(4).setOutputNum(3)
                    .setDefaultHiddenAF(Activation.none).setOutputAF(Activation.arrays.none)
                    .setCostFunction(Cost.crossEntropy).setNumSimulated(1).setInitialMutation(0).build();
        } catch (EvolutionBuilder.MissingInformation e) {
            throw new RuntimeException(e);
        }
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
    void calculateTest3() throws Evolution.EvolutionBuilder.MissingInformation {
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
}
