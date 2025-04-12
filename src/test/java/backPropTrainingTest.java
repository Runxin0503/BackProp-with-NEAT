import Evolution.Agent;
import Evolution.Evolution;
import Genome.NN;
import Genome.Activation;
import Genome.Cost;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.function.Consumer;
import java.util.function.Function;

public class backPropTrainingTest {

    /** Test Procedure: When input is 0, predict 1. When input is 1, predict 0 */
    @Test
    void trainNOTNeuralNetwork() {
        final Evolution agentFactory = new Evolution.EvolutionBuilder().setInputNum(1).setOutputNum(2)
                .setDefaultHiddenAF(Activation.reLU).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(100).build();
        final int trainingIterations = 1000;

        Consumer<NN> trainAgents = agentGenome -> {
            int testInput = (int) Math.round(Math.random());
            double[] testOutput = new double[2];
            testOutput[testInput == 1 ? 0 : 1] = 1;

            NN.learn(agentGenome, 0.5, 0.9, 0.9, 1e-8, new double[][]{{testInput}}, new double[][]{testOutput});
        };

        Function<NN, Double> evaluateScore =
                agentGenome -> -Math.log(evaluate(agentGenome,
                        new double[][]{{1}, {0}},
                        new double[][]{{1, 0}, {0, 1}})) + 5;

        Function<NN, Boolean> evaluateCorrectness = agentGenome -> isCorrect(
                new double[][]{{0, 1}, {1, 0}}, new double[][]{
                        agentGenome.calculateWeightedOutput(new double[]{0}),
                        agentGenome.calculateWeightedOutput(new double[]{1})}, 1e-2);

        Consumer<NN> print = agentGenome -> {
            System.out.println("Output of 0: " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0})));
            System.out.println("Output of 1: " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1})));
        };

        trainNeuralNetwork(agentFactory, trainingIterations, trainAgents, evaluateScore, evaluateCorrectness, print);
    }

    /** Test Procedure: AND. When input is both 1, predict 1, otherwise predict 0 */
    @Test
    void trainANDNeuralNetwork() {
        final Evolution agentFactory = new Evolution.EvolutionBuilder().setInputNum(2).setOutputNum(2)
                .setDefaultHiddenAF(Activation.sigmoid).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(100).build();
        agentFactory.Constants.mutationSynapseProbability = 0.1;
        final int trainingIterations = 1000;

        Consumer<NN> trainAgents = agentGenome -> {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == 1 && testInput[1] == 1 ? 1 : 0] = 1;

            NN.learn(agentGenome, 0.6, 0.9, 0.9, 1e-8, new double[][]{testInput}, new double[][]{testOutput});
        };

        Function<NN, Double> evaluateScore =
                agentGenome -> -Math.log(evaluate(agentGenome,
                        new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                        new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}})) + 5;

        Function<NN, Boolean> evaluateCorrectness = agentGenome -> isCorrect(
                new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}}, new double[][]{
                        agentGenome.calculateWeightedOutput(new double[]{0, 0}),
                        agentGenome.calculateWeightedOutput(new double[]{0, 1}),
                        agentGenome.calculateWeightedOutput(new double[]{1, 0}),
                        agentGenome.calculateWeightedOutput(new double[]{1, 1})}, 1e-2);

        Consumer<NN> print = agentGenome -> {
            System.out.println("Output of (0,0): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0, 0})));
            System.out.println("Output of (0,1): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0, 1})));
            System.out.println("Output of (1,0): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1, 0})));
            System.out.println("Output of (1,1): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1, 1})));
        };

        trainNeuralNetwork(agentFactory, trainingIterations, trainAgents, evaluateScore, evaluateCorrectness, print);

    }

    /** Test Procedure: OR. When either input is 1, predict 1, otherwise predict 0 */
    @Test
    void trainORNeuralNetwork() {
        final Evolution agentFactory = new Evolution.EvolutionBuilder().setInputNum(2).setOutputNum(2)
                .setDefaultHiddenAF(Activation.sigmoid).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(100).build();
        final int trainingIterations = 1000;

        Consumer<NN> trainAgents = agentGenome -> {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == 1 || testInput[1] == 1 ? 1 : 0] = 1;

            NN.learn(agentGenome, 0.4, 0.8, 0.9, 1e-4, new double[][]{testInput}, new double[][]{testOutput});
        };

        Function<NN, Double> evaluateScore =
                agentGenome -> -Math.log(evaluate(agentGenome,
                        new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                        new double[][]{{1, 0}, {0, 1}, {0, 1}, {0, 1}})) + 5;

        Function<NN, Boolean> evaluateCorrectness = agentGenome -> isCorrect(
                new double[][]{{1, 0}, {0, 1}, {0, 1}, {0, 1}}, new double[][]{
                        agentGenome.calculateWeightedOutput(new double[]{0, 0}),
                        agentGenome.calculateWeightedOutput(new double[]{0, 1}),
                        agentGenome.calculateWeightedOutput(new double[]{1, 0}),
                        agentGenome.calculateWeightedOutput(new double[]{1, 1})}, 1e-2);

        Consumer<NN> print = agentGenome -> {
            System.out.println("Output of (0,0): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0, 0})));
            System.out.println("Output of (0,1): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0, 1})));
            System.out.println("Output of (1,0): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1, 0})));
            System.out.println("Output of (1,1): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1, 1})));
        };

        trainNeuralNetwork(agentFactory, trainingIterations, trainAgents, evaluateScore, evaluateCorrectness, print);
    }

    /** Test Procedure: XOR. When both inputs are 1,1 or 0,0 predict 0, otherwise predict 1 */
    @Test
    void trainXORNeuralNetwork() {
        final Evolution agentFactory = new Evolution.EvolutionBuilder().setInputNum(2).setOutputNum(2)
                .setDefaultHiddenAF(Activation.tanh).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(100).build();
        agentFactory.Constants.mutationSynapseProbability = 0.15;
        final int trainingIterations = 100;

        Consumer<NN> trainAgents = agentGenome -> {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == testInput[1] ? 0 : 1] = 1;

            NN.learn(agentGenome, 0.4, 0.96, 0.9, 1e-4, new double[][]{testInput}, new double[][]{testOutput});
        };

        Function<NN, Double> evaluateScore =
                agentGenome -> -Math.log(evaluate(agentGenome,
                        new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                        new double[][]{{1, 0}, {0, 1}, {0, 1}, {1, 0}})) + 5;

        Function<NN, Boolean> evaluateCorrectness = agentGenome -> isCorrect(
                new double[][]{{1, 0}, {0, 1}, {0, 1}, {1, 0}}, new double[][]{
                        agentGenome.calculateWeightedOutput(new double[]{0, 0}),
                        agentGenome.calculateWeightedOutput(new double[]{0, 1}),
                        agentGenome.calculateWeightedOutput(new double[]{1, 0}),
                        agentGenome.calculateWeightedOutput(new double[]{1, 1})}, 1e-2);

        Consumer<NN> print = agentGenome -> {
            System.out.println("Output of (0,0): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0, 0})));
            System.out.println("Output of (0,1): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0, 1})));
            System.out.println("Output of (1,0): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1, 0})));
            System.out.println("Output of (1,1): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1, 1})));
        };

        trainNeuralNetwork(agentFactory, trainingIterations, trainAgents, evaluateScore, evaluateCorrectness, print);
    }

    /** Test Procedure: When input is 0, predict 1. When input is 1, predict 0 */
    @Test
    void trainNOTNeuralNetworkMiniBatch() {
        final Evolution agentFactory = new Evolution.EvolutionBuilder().setInputNum(1).setOutputNum(2)
                .setDefaultHiddenAF(Activation.reLU).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(100).build();
        final int trainingIterations = 1000;

        Consumer<NN> trainAgents = agentGenome -> {
            double[][] testInputs = new double[][]{{0}, {1}};
            double[][] testOutputs = new double[][]{{0, 1}, {1, 0}};

            NN.learn(agentGenome, 0.5, 0.9, 0.9, 1e-8, testInputs, testOutputs);
        };

        Function<NN, Double> evaluateScore =
                agentGenome -> -Math.log(evaluate(agentGenome,
                        new double[][]{{1}, {0}},
                        new double[][]{{1, 0}, {0, 1}})) + 5;

        Function<NN, Boolean> evaluateCorrectness = agentGenome -> isCorrect(
                new double[][]{{0, 1}, {1, 0}}, new double[][]{
                        agentGenome.calculateWeightedOutput(new double[]{0}),
                        agentGenome.calculateWeightedOutput(new double[]{1})}, 1e-2);

        Consumer<NN> print = agentGenome -> {
            System.out.println("Output of 0: " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0})));
            System.out.println("Output of 1: " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1})));
        };

        trainNeuralNetwork(agentFactory, trainingIterations, trainAgents, evaluateScore, evaluateCorrectness, print);
    }

    /** Test Procedure: AND. When input is both 1, predict 1, otherwise predict 0 */
    @Test
    void trainANDNeuralNetworkMiniBatch() {
        final Evolution agentFactory = new Evolution.EvolutionBuilder().setInputNum(2).setOutputNum(2)
                .setDefaultHiddenAF(Activation.sigmoid).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(100).build();
        agentFactory.Constants.mutationSynapseProbability = 0.1;
        final int trainingIterations = 100;

        Consumer<NN> trainAgents = agentGenome -> {
            double[][] testInputs = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            double[][] testOutputs = new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}};

            NN.learn(agentGenome, 0.6, 0.9, 0.9, 1e-8, testInputs, testOutputs);
        };

        Function<NN, Double> evaluateScore =
                agentGenome -> -Math.log(evaluate(agentGenome,
                        new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                        new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}})) + 5;

        Function<NN, Boolean> evaluateCorrectness = agentGenome -> isCorrect(
                new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}}, new double[][]{
                        agentGenome.calculateWeightedOutput(new double[]{0, 0}),
                        agentGenome.calculateWeightedOutput(new double[]{0, 1}),
                        agentGenome.calculateWeightedOutput(new double[]{1, 0}),
                        agentGenome.calculateWeightedOutput(new double[]{1, 1})}, 1e-2);

        Consumer<NN> print = agentGenome -> {
            System.out.println("Output of (0,0): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0, 0})));
            System.out.println("Output of (0,1): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0, 1})));
            System.out.println("Output of (1,0): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1, 0})));
            System.out.println("Output of (1,1): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1, 1})));
        };

        trainNeuralNetwork(agentFactory, trainingIterations, trainAgents, evaluateScore, evaluateCorrectness, print);

    }

    /** Test Procedure: OR. When either input is 1, predict 1, otherwise predict 0 */
    @Test
    void trainORNeuralNetworkMiniBatch() {
        final Evolution agentFactory = new Evolution.EvolutionBuilder().setInputNum(2).setOutputNum(2)
                .setDefaultHiddenAF(Activation.sigmoid).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(75).build();
        agentFactory.Constants.mutationSynapseProbability = 0.15;
        final int trainingIterations = 100;

        Consumer<NN> trainAgents = agentGenome -> {
            double[][] testInputs = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            double[][] testOutputs = new double[][]{{1, 0}, {0, 1}, {0, 1}, {0, 1}};

            NN.learn(agentGenome, 0.4, 0.8, 0.9, 1e-4, testInputs, testOutputs);
        };

        Function<NN, Double> evaluateScore =
                agentGenome -> -Math.log(evaluate(agentGenome,
                        new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                        new double[][]{{1, 0}, {0, 1}, {0, 1}, {0, 1}})) + 5;

        Function<NN, Boolean> evaluateCorrectness = agentGenome -> isCorrect(
                new double[][]{{1, 0}, {0, 1}, {0, 1}, {0, 1}}, new double[][]{
                        agentGenome.calculateWeightedOutput(new double[]{0, 0}),
                        agentGenome.calculateWeightedOutput(new double[]{0, 1}),
                        agentGenome.calculateWeightedOutput(new double[]{1, 0}),
                        agentGenome.calculateWeightedOutput(new double[]{1, 1})}, 1e-2);

        Consumer<NN> print = agentGenome -> {
            System.out.println("Output of (0,0): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0, 0})));
            System.out.println("Output of (0,1): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0, 1})));
            System.out.println("Output of (1,0): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1, 0})));
            System.out.println("Output of (1,1): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1, 1})));
        };

        trainNeuralNetwork(agentFactory, trainingIterations, trainAgents, evaluateScore, evaluateCorrectness, print);
    }

    /** Test Procedure: XOR. When both inputs are 1,1 or 0,0 predict 0, otherwise predict 1 */
    @Test
    void trainXORNeuralNetworkMiniBatch() {
        final Evolution agentFactory = new Evolution.EvolutionBuilder().setInputNum(2).setOutputNum(2)
                .setDefaultHiddenAF(Activation.tanh).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(50).build();
        agentFactory.Constants.mutationSynapseProbability = 0.15;
        final int trainingIterations = 100;

        Consumer<NN> trainAgents = agentGenome -> {
            double[][] testInputs = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            double[][] testOutputs = new double[][]{{1, 0}, {0, 1}, {0, 1}, {1, 0}};

            NN.learn(agentGenome, 0.4, 0.96, 0.9, 1e-4, testInputs, testOutputs);
        };

        Function<NN, Double> evaluateScore =
                agentGenome -> -Math.log(evaluate(agentGenome,
                        new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                        new double[][]{{1, 0}, {0, 1}, {0, 1}, {1, 0}})) + 5;

        Function<NN, Boolean> evaluateCorrectness = agentGenome -> isCorrect(
                new double[][]{{1, 0}, {0, 1}, {0, 1}, {1, 0}}, new double[][]{
                        agentGenome.calculateWeightedOutput(new double[]{0, 0}),
                        agentGenome.calculateWeightedOutput(new double[]{0, 1}),
                        agentGenome.calculateWeightedOutput(new double[]{1, 0}),
                        agentGenome.calculateWeightedOutput(new double[]{1, 1})}, 1e-2);

        Consumer<NN> print = agentGenome -> {
            System.out.println("Output of (0,0): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0, 0})));
            System.out.println("Output of (0,1): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{0, 1})));
            System.out.println("Output of (1,0): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1, 0})));
            System.out.println("Output of (1,1): " + Arrays.toString(agentGenome.calculateWeightedOutput(new double[]{1, 1})));
        };

        trainNeuralNetwork(agentFactory, trainingIterations, trainAgents, evaluateScore, evaluateCorrectness, print);
    }

    /**
     * A Helper Function that greatly reduces repetitive code by
     * forming the outlines of the training regiment every agentFactory goes through.
     */
    private void trainNeuralNetwork(
            Evolution agentFactory, int trainingIterations, Consumer<NN> trainAgent, Function<NN, Double> evaluateScore,
            Function<NN, Boolean> evaluateCorrectness, Consumer<NN> printWhenCorrect) {
        Thread[] workerThreads = new Thread[agentFactory.agents.length];
        while (true) {
            for (int i = 0; i < workerThreads.length; i++) {
                final Agent agent = agentFactory.agents[i];
                final NN agentGenome = agent.getGenomeClone();
                workerThreads[i] = new Thread(null, () -> {
                    for (int j = 0; j < trainingIterations; j++) {
                        trainAgent.accept(agentGenome);
                    }
                    agent.setScore(evaluateScore.apply(agentGenome));
                    if (evaluateCorrectness.apply(agentGenome)) {
                        printWhenCorrect.accept(agentGenome);
                        System.out.println("Agent has passed the test, score of " + agent.getScore());
                        agent.setScore(Double.POSITIVE_INFINITY);
                        System.out.println("\n" + agentGenome + "\n\n\n\n\n\n");
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

    /**
     * Returns the cost of {@code genome} on the inputs and expectedOutputs,
     * <br>Lower the cost, better the performance
     */
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
//                    System.err.println("expected: " + expectedOutputs[i][j] + " but was: " + actualOutputs[i][j]);
                    return false;
                }
            }
        }
        return true;
    }
}
