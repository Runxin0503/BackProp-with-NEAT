import Evolution.Constants;
import Evolution.Evolution;
import Genome.*;
import org.junit.jupiter.api.RepeatedTest;

import static org.junit.jupiter.api.Assertions.assertTrue;

//the backpropagate AND test written for ML_Optimizers on all Optimizers, adapted for the TWEANN network structure.
public class FixedNetworkBackpropagationTest {

    @RepeatedTest(1000)
    void testBackPropagateANDNetworkSGD() {
        Constants Constants = new Evolution.EvolutionBuilder().setInputNum(2).setOutputNum(2).setOptimizer(Optimizer.SGD)
                .setDefaultHiddenAF(Activation.sigmoid).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(1).setInitialMutation(0).build().Constants;
        NN Network = NN.getDefaultNeuralNet(Constants);

        for (int i = 0; i < 6; i++) {
            Modifier.addEdge(Network, Constants.getInitializedValue(), 0, 2 + i);
            Modifier.splitEdge(Network, Constants.getInitializedValue(), Constants.getDefaultHiddenAF(), 0);
        }
        for (int j = 2; j < 8; j++)
            Modifier.addEdge(Network, Constants.getInitializedValue(), 1, j);
        for (int i = 2; i < 8; i++)
            Modifier.addEdge(Network, Constants.getInitializedValue(), i, 9);

        final int iterations = 1000;
        for (int i = 0; i < iterations; i++) {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == 1 && testInput[1] == 1 ? 1 : 0] = 1;

            NN.learn(Network, 0.5, 0, 0, 0, new double[][]{testInput}, new double[][]{testOutput});

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

    @RepeatedTest(1000)
    void testBackPropagateANDNetworkSGDMomentum() {
        Constants Constants = new Evolution.EvolutionBuilder().setInputNum(2).setOutputNum(2).setOptimizer(Optimizer.SGD_MOMENTUM)
                .setDefaultHiddenAF(Activation.sigmoid).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(1).setInitialMutation(0).build().Constants;
        NN Network = NN.getDefaultNeuralNet(Constants);

        for (int i = 0; i < 6; i++) {
            Modifier.addEdge(Network, Constants.getInitializedValue(), 0, 2 + i);
            Modifier.splitEdge(Network, Constants.getInitializedValue(), Constants.getDefaultHiddenAF(), 0);
        }
        for (int j = 2; j < 8; j++)
            Modifier.addEdge(Network, Constants.getInitializedValue(), 1, j);
        for (int i = 2; i < 8; i++)
            Modifier.addEdge(Network, Constants.getInitializedValue(), i, 9);

        final int iterations = 1000;
        for (int i = 0; i < iterations; i++) {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == 1 && testInput[1] == 1 ? 1 : 0] = 1;

            NN.learn(Network, 0.75, 0.9, 0, 0, new double[][]{testInput}, new double[][]{testOutput});

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

    @RepeatedTest(1000)
    void testBackPropagateANDNetworkRMSProp() {
        Constants Constants = new Evolution.EvolutionBuilder().setInputNum(2).setOutputNum(2).setOptimizer(Optimizer.RMS_PROP)
                .setDefaultHiddenAF(Activation.sigmoid).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(1).setInitialMutation(0).build().Constants;
        NN Network = NN.getDefaultNeuralNet(Constants);

        for (int i = 0; i < 6; i++) {
            Modifier.addEdge(Network, Constants.getInitializedValue(), 0, 2 + i);
            Modifier.splitEdge(Network, Constants.getInitializedValue(), Constants.getDefaultHiddenAF(), 0);
        }
        for (int j = 2; j < 8; j++)
            Modifier.addEdge(Network, Constants.getInitializedValue(), 1, j);
        for (int i = 2; i < 8; i++)
            Modifier.addEdge(Network, Constants.getInitializedValue(), i, 9);

        final int iterations = 1000;
        for (int i = 0; i < iterations; i++) {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == 1 && testInput[1] == 1 ? 1 : 0] = 1;

            NN.learn(Network, 0.75, 0, 0.9, 1e-4, new double[][]{testInput}, new double[][]{testOutput});

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

    @RepeatedTest(1000)
    void testBackPropagateANDNetworkADAM() {
        Constants Constants = new Evolution.EvolutionBuilder().setInputNum(2).setOutputNum(2).setOptimizer(Optimizer.ADAM)
                .setDefaultHiddenAF(Activation.sigmoid).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(1).setInitialMutation(0).build().Constants;
        NN Network = NN.getDefaultNeuralNet(Constants);

        for (int i = 0; i < 6; i++) {
            Modifier.addEdge(Network, Constants.getInitializedValue(), 0, 2 + i);
            Modifier.splitEdge(Network, Constants.getInitializedValue(), Constants.getDefaultHiddenAF(), 0);
        }
        for (int j = 2; j < 8; j++)
            Modifier.addEdge(Network, Constants.getInitializedValue(), 1, j);
        for (int i = 2; i < 8; i++)
            Modifier.addEdge(Network, Constants.getInitializedValue(), i, 9);

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
