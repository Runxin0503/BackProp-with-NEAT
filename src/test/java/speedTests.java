import Evolution.Constants;
import Evolution.Evolution;
import Genome.Mutation;
import Genome.NN;
import Genome.Activation;
import Genome.Cost;

import java.io.FileWriter;
import java.io.IOException;

public class speedTests {

    private static final Constants Constants;

    static {
        Evolution agentFactory = new Evolution.EvolutionBuilder().setInputNum(10).setOutputNum(10)
                .setDefaultHiddenAF(Activation.none).setOutputAF(Activation.arrays.none)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(1).build();
        Constants = agentFactory.Constants;
    }

//    @Test
    @Deprecated
    void testCalculateSpeed() {
        NN Network = NN.getDefaultNeuralNet(Constants);
        final int calculationIterations = 100_000;

        try (FileWriter csvWriter = new FileWriter("src/test/resources/calculateSpeedDirectReference.csv")) {
            csvWriter.append("Iteration,Nodes,Edges,Time\n");

            for (int i = Constants.getInputNum() + Constants.getOutputNum(); i < 1_000; i++) {
                System.out.println("Iteration "+i);
                int parameters = Network.nodes.size() + Network.genome.size();
                while (parameters < i) {
                    int initialEdges = Network.genome.size();
                    Mutation.mutateSynapse(Network);
                    parameters = Network.nodes.size() + Network.genome.size();
                    if (Network.genome.size() == initialEdges && parameters < i)
                        Mutation.mutateNode(Network);
                }

                long avgTime = 0;
                for (int j = 0; j < calculationIterations; j++) {
                    double[] randomInput = new double[Constants.getInputNum()];
                    for (int k = 0; k < randomInput.length; k++)
                        randomInput[k] = (Math.random() * 2 - 1) * 1e20;
                    long startTime = System.nanoTime();
                    Network.calculateWeightedOutput(randomInput);
                    long endTime = System.nanoTime();
                    avgTime += endTime - startTime;
                }
                avgTime /= calculationIterations;

                csvWriter.append(String.valueOf(i))
                        .append(',').append(String.valueOf(Network.nodes.size()))
                        .append(',').append(String.valueOf(Network.genome.size()))
                        .append(',').append(String.valueOf(avgTime))
                        .append('\n');
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

//    @Test
    @Deprecated
    void testBackPropagateSpeed() {
        NN Network = NN.getDefaultNeuralNet(Constants);
        final int calculationIterations = 20_000;

        try (FileWriter csvWriter = new FileWriter("src/test/resources/backpropagationSpeedDirectReference.csv")) {
            csvWriter.append("Iteration,Nodes,Edges,Time\n");

            for (int i = Constants.getInputNum() + Constants.getOutputNum(); i < 1_000; i++) {
                System.out.println("Iteration "+i);
                int parameters = Network.nodes.size() + Network.genome.size();
                while (parameters < i) {
                    int initialEdges = Network.genome.size();
                    Mutation.mutateSynapse(Network);
                    parameters = Network.nodes.size() + Network.genome.size();
                    if (Network.genome.size() == initialEdges && parameters < i)
                        Mutation.mutateNode(Network);
                }

                long avgTime = 0;
                for (int j = 0; j < calculationIterations; j++) {
                    double[] randomInput = new double[Constants.getInputNum()];
                    for (int k = 0; k < randomInput.length; k++)
                        randomInput[k] = (Math.random() * 2 - 1) * 1e20;
                    long startTime = System.nanoTime();
                    NN.learn(Network,1,0.9,0.9,1e-8,new double[][]{randomInput},new double[][]{randomInput});
                    long endTime = System.nanoTime();
                    avgTime += endTime - startTime;
                }
                avgTime /= calculationIterations;

                csvWriter.append(String.valueOf(i))
                        .append(',').append(String.valueOf(Network.nodes.size()))
                        .append(',').append(String.valueOf(Network.genome.size()))
                        .append(',').append(String.valueOf(avgTime))
                        .append('\n');
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

//    @Test
    @Deprecated
    void testMutateSynapseSpeed() {

    }

//    @Test
    @Deprecated
    void testMutateNodeSpeed() {

    }

//    @Test
    @Deprecated
    void testCloneSpeed() {

    }

//    @Test
    @Deprecated
    void testCrossoverSpeed() {

    }
}
