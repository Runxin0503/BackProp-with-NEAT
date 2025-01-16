import Evolution.Agent;
import Evolution.Constants;
import Evolution.Evolution;
import Evolution.Evolution.EvolutionBuilder;
import Genome.NN;
import Genome.edge;
import Genome.enums.Activation;
import Genome.enums.Cost;
import Genome.node;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
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
            for (int j = 0; j < Constants.getOutputNum(); j++) {
                Network.nodes.get(i).addOutgoingEdgeIndex(Network.genome.size());
                Network.nodes.get(j + Constants.getInputNum()).addIncomingEdgeIndex(Network.genome.size());
                Network.genome.add(new edge(
                        0, 1, true, i, j + Constants.getInputNum(),
                        i - Constants.getInputNum() - Constants.getOutputNum(), j - Constants.getOutputNum()));
            }

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
            for (int j = 0; j < weights[i].length; j++) weights[i][j] = randNum.get();

        for (int i = 0; i < Constants.getInputNum(); i++)
            for (int j = 0; j < Constants.getOutputNum(); j++) {
                Network.nodes.get(i).addOutgoingEdgeIndex(Network.genome.size());
                Network.nodes.get(j + Constants.getInputNum()).addIncomingEdgeIndex(Network.genome.size());
                Network.genome.add(new edge(
                        0, weights[i][j], true, i, j + Constants.getInputNum(),
                        i - Constants.getInputNum() - Constants.getOutputNum(), j - Constants.getOutputNum()));
            }

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
            for (int j = 0; j < weights[i].length; j++) weights[i][j] = randNum.get();

        for (int i = 0; i < Constants.getInputNum(); i++)
            for (int j = 0; j < Constants.getOutputNum(); j++) {
                Network.nodes.get(i).addOutgoingEdgeIndex(Network.genome.size());
                Network.nodes.get(j + Constants.getInputNum()).addIncomingEdgeIndex(Network.genome.size());
                Network.genome.add(new edge(
                        0, weights[i][j], true, i, j + Constants.getInputNum(),
                        i - Constants.getInputNum() - Constants.getOutputNum(), j - Constants.getOutputNum()));
            }

        for (int count = 0; count < 1000; count++) {
            double[] input = new double[]{randNum.get(), randNum.get(), randNum.get(), randNum.get()};
            double[] output = new double[Constants.getOutputNum()];
            for (int i = 0; i < weights.length; i++)
                for (int j = 0; j < weights[i].length; j++) output[j] += weights[i][j] * input[i];
            double[] outputCopy = Activation.arrays.sigmoid.calculate(output);
            for (int i = 0; i < output.length; i++) output[i] = Activation.sigmoid.calculate(output[i]);
            assertArrayEquals(outputCopy, output);
            assertArrayEquals(output, Network.calculateWeightedOutput(input));
        }
    }

    @RepeatedTest(1000)
    void testBackPropagateANDNetwork() {
        Constants Constants = new EvolutionBuilder().setInputNum(2).setOutputNum(2)
                .setDefaultHiddenAF(Activation.sigmoid).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(1).build().Constants;
        NN Network = NN.getDefaultNeuralNet(Constants);

        for(int i=2;i<8;i++) {
            node newNode = new node(0,Activation.sigmoid,Constants.getInitializedValue());
            newNode.x = 0.5;
            newNode.y = (i-1.0)/7;
            Network.nodes.add(i,newNode);
        }
        for (int i = 0; i < 2; i++)
            for (int j = 2; j < 8; j++) {
                Network.nodes.get(i).addOutgoingEdgeIndex(Network.genome.size());
                Network.nodes.get(j).addIncomingEdgeIndex(Network.genome.size());
                Network.genome.add(new edge(
                        0, Constants.getInitializedValue(), true, i, j,
                        i - 4, 0));
            }
        for (int i = 2; i < 8; i++)
            for (int j = 8; j < 10; j++) {
                Network.nodes.get(i).addOutgoingEdgeIndex(Network.genome.size());
                Network.nodes.get(j).addIncomingEdgeIndex(Network.genome.size());
                Network.genome.add(new edge(
                        0, Constants.getInitializedValue(), true, i, j,
                        0, j - 10));
            }
        final int iterations = 1000;

        for (int i = 0; i < iterations; i++) {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == 1 && testInput[1] == 1 ? 1 : 0] = 1;

            NN.learn(Network, 0.3, 0.9,0.9, 1e-4,new double[][]{testInput}, new double[][]{testOutput});

            if (isCorrect(new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}},new double[][]{
                    Network.calculateWeightedOutput(new double[]{0, 0}),
                    Network.calculateWeightedOutput(new double[]{0, 1}),
                    Network.calculateWeightedOutput(new double[]{1, 0}),
                    Network.calculateWeightedOutput(new double[]{1, 1})},1e-2)) break;
        }

        assertTrue(isCorrect(new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}},new double[][]{
                Network.calculateWeightedOutput(new double[]{0, 0}),
                Network.calculateWeightedOutput(new double[]{0, 1}),
                Network.calculateWeightedOutput(new double[]{1, 0}),
                Network.calculateWeightedOutput(new double[]{1, 1})},1e-2));
    }

    /** Test Procedure: When input is 0, predict 1. When input is 1, predict 0 */
    @Test
    void trainNOTNeuralNetwork() {
        final Evolution agentFactory = new EvolutionBuilder().setInputNum(1).setOutputNum(2)
                .setDefaultHiddenAF(Activation.ReLU).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(100).build();
        final int trainingIterations = 1000;

        Thread[] workerThreads = new Thread[agentFactory.agents.length];
        while (true) {
            for (int i = 0; i < workerThreads.length; i++) {
                final Agent agent = agentFactory.agents[i];
                final NN agentGenome = agent.getGenomeClone();
                workerThreads[i] = new Thread(null, () -> {
                    for (int j = 0; j < trainingIterations; j++) {
                        int testInput = (int) Math.round(Math.random());
                        double[] testOutput = new double[2];
                        testOutput[testInput == 1 ? 0 : 1] = 1;

                        NN.learn(agentGenome, 0.5, 0.9, 0.9, 1e-8, new double[][]{{testInput}}, new double[][]{testOutput});
                    }
                    agent.setScore(-Math.log(evaluate(agentGenome,
                            new double[][]{{1}, {0}},
                            new double[][]{{1, 0}, {0, 1}})) + 5);
                    if (isCorrect(new double[][]{{1, 0}, {0, 1}}, new double[][]{
                            agentGenome.calculateWeightedOutput(new double[]{0}),
                            agentGenome.calculateWeightedOutput(new double[]{1})}, 1e-2)) {
                        System.out.println("Output of 0: " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0})));
                        System.out.println("Output of 1: " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1})));
                    }
                }, "Worker Thread " + i);
                workerThreads[i].start();
            }

            for (Thread t : workerThreads) {
                try {
                    t.join();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            for (Agent agent : agentFactory.agents)
                if (agent.getScore() > 1e8) {
                    System.out.println("Agent Score: " + agent.getScore());
                    return;
                }

            agentFactory.nextGen();
        }
    }

    /** Test Procedure: AND. When input is both 1, predict 1, otherwise predict 0 */
    @Test
    void trainANDNeuralNetwork() {
        final Evolution agentFactory = new EvolutionBuilder().setInputNum(2).setOutputNum(2)
                .setDefaultHiddenAF(Activation.sigmoid).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(100).build();
        final int trainingIterations = 1000;

        Thread[] workerThreads = new Thread[agentFactory.agents.length];
        while (true) {
            for (int i = 0; i < workerThreads.length; i++) {
                final Agent agent = agentFactory.agents[i];
                final NN agentGenome = agent.getGenomeClone();
                workerThreads[i] = new Thread(null, () -> {
                    for (int j = 0; j < trainingIterations; j++) {
                        double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
                        double[] testOutput = new double[2];
                        testOutput[testInput[0] == 1 && testInput[1] == 1 ? 1 : 0] = 1;

                        NN.learn(agentGenome, 0.3, 0.9, 0.9, 1e-8, new double[][]{testInput}, new double[][]{testOutput});
                    }
                    agent.setScore(evaluate(agentGenome,
                            new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                            new double[][]{{0, 1}, {0, 1}, {0, 1}, {1, 0}}));
                    if (isCorrect(new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}}, new double[][]{
                            agentGenome.calculateWeightedOutput(new double[]{0, 0}),
                            agentGenome.calculateWeightedOutput(new double[]{0, 1}),
                            agentGenome.calculateWeightedOutput(new double[]{1, 0}),
                            agentGenome.calculateWeightedOutput(new double[]{1, 1})}, 0.1)) {
                        System.out.println("Output of 0: " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0})));
                        System.out.println("Output of 1: " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1})));
                    }
                }, "Worker Thread " + i);
                workerThreads[i].start();
            }

            for (Thread t : workerThreads) {
                try {
                    t.join();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }

            agentFactory.nextGen();
        }
    }
//
//    /** Test Procedure: OR. When either input is 1, predict 1, otherwise predict 0 */
//    @Test
//    void trainORNeuralNetwork() {
//        final NN linearNN = new NN(Activation.sigmoid, Activation.softmax, Cost.crossEntropy, 2, 4, 2);
//        final int iterations = 1000;
//
//        for (int i = 0; i < iterations; i++) {
//            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
//            double[] testOutput = new double[2];
//            testOutput[testInput[0] == 1 || testInput[1] == 1 ? 1 : 0] = 1;
//
//            NN.learn(linearNN, 0.4, 0.8,0.9, 1e-4,new double[][]{testInput}, new double[][]{testOutput});
//
//            if (evaluate(1e-2, new double[][]{{1, 0}, {0, 1}, {0, 1}, {0, 1}},
//                    linearNN.calculateOutput(new double[]{0, 0}),
//                    linearNN.calculateOutput(new double[]{0, 1}),
//                    linearNN.calculateOutput(new double[]{1, 0}),
//                    linearNN.calculateOutput(new double[]{1, 1}))) break;
//        }
//
//        assertTrue(evaluate(1e-2, new double[][]{{1, 0}, {0, 1}, {0, 1}, {0, 1}},
//                linearNN.calculateOutput(new double[]{0, 0}),
//                linearNN.calculateOutput(new double[]{0, 1}),
//                linearNN.calculateOutput(new double[]{1, 0}),
//                linearNN.calculateOutput(new double[]{1, 1})));
//    }
//
//    /** Test Procedure: XOR. When both inputs are 1,1 or 0,0 predict 0, otherwise predict 1 */
//    @Test
//    void trainXORNeuralNetwork() {
//        final NN semiComplexNN = new NN(Activation.tanh, Activation.softmax, Cost.crossEntropy, 2, 8, 2);
//        final int iterations = 1000;
//
//        for (int i = 0; i < iterations; i++) {
//            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
//            double[] testOutput = new double[2];
//            testOutput[testInput[0] == testInput[1] ? 0 : 1] = 1;
////            System.out.println(Arrays.toString(testInput));
//
//            NN.learn(semiComplexNN, 0.015, 0.96,0.9, 1e-4,new double[][]{testInput}, new double[][]{testOutput});
//
//            if (evaluate(1e-2, new double[][]{{1, 0}, {1, 0}, {0, 1}, {0, 1}},
//                    semiComplexNN.calculateOutput(new double[]{1, 1}),
//                    semiComplexNN.calculateOutput(new double[]{0, 0}),
//                    semiComplexNN.calculateOutput(new double[]{0, 1}),
//                    semiComplexNN.calculateOutput(new double[]{1, 0}))) break;
//        }
////        System.out.println();
//
//        assertTrue(evaluate(1e-2, new double[][]{{1, 0}, {1, 0}, {0, 1}, {0, 1}},
//                semiComplexNN.calculateOutput(new double[]{1, 1}),
//                semiComplexNN.calculateOutput(new double[]{0, 0}),
//                semiComplexNN.calculateOutput(new double[]{0, 1}),
//                semiComplexNN.calculateOutput(new double[]{1, 0})));
//    }

    private double evaluate(NN genome, double[][] inputs, double[][] expectedOutputs) {
        assert inputs.length == expectedOutputs.length;

        double cost = 0;
        for (int i = 0; i < expectedOutputs.length; i++) {
            double[] input = inputs[i], expectedOutput = expectedOutputs[i];
            cost += genome.calculateCost(input, expectedOutput);
        }
        return cost;
    }

    private boolean isCorrect(double[][] expectedOutputs, double[][] actualOutputs, double delta) {
        assert expectedOutputs.length == actualOutputs.length;

        for (int i = 0; i < expectedOutputs.length; i++) {
            for (int j = 0; j < expectedOutputs[i].length; j++) {
                if (Math.abs(expectedOutputs[i][j] - actualOutputs[i][j]) > delta) {
                    System.err.println("expected: " + expectedOutputs[i][j] + " but was: " + actualOutputs[i][j]);
                    return false;
                }
            }
        }
        return true;
    }
}
