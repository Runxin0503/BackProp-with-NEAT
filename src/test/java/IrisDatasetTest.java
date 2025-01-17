import Evolution.Agent;
import Evolution.Evolution;
import Evolution.Evolution.EvolutionBuilder;
import Genome.NN;
import Genome.enums.Activation;
import Genome.enums.Cost;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class IrisDatasetTest {

    private static final int Iris_Size = 150;
    private static final HashMap<double[], Integer> featuresToCategories = new HashMap<>(Iris_Size);
    private static final List<double[]> features;
    private static final ArrayList<String> names = new ArrayList<>();

    static {
        try (BufferedReader br = new BufferedReader(new FileReader("src/test/resources/iris.data"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double[] features = new double[4];
                for (int i = 0; i < 4; i++) {
                    features[i] = Double.parseDouble(parts[i]);
                }
                String label = parts[4];
                if (!names.contains(label)) names.add(label);
                featuresToCategories.put(features, names.indexOf(label));
            }
            assert featuresToCategories.size() == Iris_Size;
            features = featuresToCategories.keySet().stream().toList();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void testDataset() {
        Evolution agentFactory = new EvolutionBuilder()
                .setInputNum(4).setOutputNum(3)
                .setOutputAF(Activation.arrays.softmax).setNumSimulated(50)
                .setCostFunction(Cost.crossEntropy).setDefaultHiddenAF(Activation.sigmoid).build();
        agentFactory.Constants.mutationSynapseProbability = 0.1;

        Thread[] workerThreads = new Thread[agentFactory.agents.length];
        while (true) {
            for (int i = 0; i < agentFactory.agents.length; i++) {
                Agent agent = agentFactory.agents[i];
                workerThreads[i] = new Thread(() -> agent.setScore(trainAgent(agent)));
                workerThreads[i].start();
            }

            for (Thread thread : workerThreads)
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

            for (Agent agent : agentFactory.agents)
                if (agent.getScore() == Double.POSITIVE_INFINITY)
                    return;


            agentFactory.nextGen();
        }
    }

    /** Trains the agent on backpropagation and returns its performance rating */
    private double trainAgent(Agent agent) {
        final int trainingIterations = 10;
        final int batchSize = 15;

        NN genome = agent.getGenomeClone();

        for (int loopedIterations = 0; loopedIterations < trainingIterations; loopedIterations++) {
            for (int trainingIndex = 0; trainingIndex < Iris_Size - batchSize; trainingIndex += batchSize) {
                double[][] trainBatchInputs = new double[batchSize][4];
                double[][] trainBatchOutputs = new double[batchSize][names.size()];
                for (int i = 0; i < batchSize; i++) {
                    trainBatchInputs[i] = features.get(trainingIndex + i);
                    trainBatchOutputs[i][featuresToCategories.get(trainBatchInputs[i])] = 1;
                }

                NN.learn(genome, 0.3, 0.9, 0.9, 1e-4, trainBatchInputs, trainBatchOutputs);
            }
        }

        return reportPerformance(genome);
    }

    /**
     * Reports and returns the performance of {@code agent} on the IRIS dataset
     */
    private static double reportPerformance(NN NeuralNetwork) {
        double cost = 0;
        int accuracy = 0;
        for (int i = 0; i < Iris_Size; i++) {
            double[] feature = features.get(i);
            int category = featuresToCategories.get(feature);
            double[] expectedOutput = new double[names.size()];
            expectedOutput[category] = 1;
            cost += NeuralNetwork.calculateCost(feature, expectedOutput);
            if (evaluateOutput(NeuralNetwork.calculateWeightedOutput(feature), category)) accuracy++;
        }
        double trainAccuracy = accuracy * 10000 / (Iris_Size * 100.0);
        double trainCost = (int) (cost * 100) / (Iris_Size * 100.0);
        System.out.println("Train Accuracy: " + trainAccuracy + "%\t\tAvg Cost: " + trainCost);

        double score = trainAccuracy - Math.log(trainCost * trainCost);
        if (trainAccuracy > 95) {
            System.out.println("Agent has passed the test, score of " + score);
            System.out.println("\n" + NeuralNetwork + "\n\n\n\n\n");
            score = Double.POSITIVE_INFINITY;
        }
        System.out.println(score);
        return score;
    }

    private static boolean evaluateOutput(double[] output, int answer) {
        return getOutput(output) == answer;
    }

    private static int getOutput(double[] output) {
        int guess = 0;
        for (int j = 0; j < output.length; j++) {
            if (output[j] > output[guess]) guess = j;
        }
        return guess;
    }
}
