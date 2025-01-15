import Evolution.Agent;
import Evolution.Constants;
import Evolution.Evolution;
import Genome.Innovation;
import Genome.Mutation;
import Genome.NN;
import Genome.edge;
import Genome.enums.Activation;
import Genome.enums.Cost;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class MutationTest {

    private static final Constants Constants;
    private static final Agent agent;

    static {
        Evolution agentFactory = new Evolution.EvolutionBuilder().setInputNum(4).setOutputNum(3)
                .setDefaultHiddenAF(Activation.ReLU).setOutputAF(Activation.arrays.softmax)
                .setCostFunction(Cost.crossEntropy).setNumSimulated(1).build();
        agent = agentFactory.agents[0];
        Constants = agentFactory.Constants;
    }

    @BeforeEach
    void resetInnovation(){
        Innovation.reset();
    }

    @RepeatedTest(100)
    void testEquals() {
        NN Network = NN.getDefaultNeuralNet(Constants);
        for (int i = 0; i < 1_000; i++) {
            Network.mutate();
            assertTrue(Network.classInv());
            assertEquals(Network, Network.clone());
        }
    }

    @Test
    void testShiftWeightsNoEdge() {
        NN Network = NN.getDefaultNeuralNet(Constants), Compare = (NN) Network.clone();
        Mutation.shiftWeights(Network);
        assertEquals(Network, Compare);
    }

    @Test
    void testShiftWeights1Edge() {
        NN Network = NN.getDefaultNeuralNet(Constants);
        Network.genome.add(new edge(0, 1, true, 3, 4, -4, -3));
        Network.nodes.get(3).addOutgoingEdgeIndex(0);
        Network.nodes.get(4).addIncomingEdgeIndex(0);
        NN Compare = (NN) Network.clone();
        Mutation.shiftWeights(Network);

        assertTrue(Network.classInv());
        assertNotEquals(Compare, Network);
        assertEquals(Compare.toString(), Network.toString());
        assertEquals(1.25, Network.genome.getFirst().getWeight(), 0.75);
    }

    @RepeatedTest(1000)
    void testShiftWeightsGeneralizedEdge() {
        NN Network = NN.getDefaultNeuralNet(Constants);
        List<int[]> edgePairs = new ArrayList<>();
        for (int i = 0; i < Constants.getInputNum(); i++)
            for (int j = Constants.getInputNum(); j < Constants.getInputNum() + Constants.getOutputNum(); j++)
                edgePairs.add(new int[]{i, j});

        int edgeCounts = new Random().nextInt(1, edgePairs.size());
        for (int i = 0; i < edgeCounts; i++) {
            int[] edgePair = edgePairs.remove((int) (Math.random() * edgePairs.size()));
            Network.nodes.get(edgePair[0]).addOutgoingEdgeIndex(Network.genome.size());
            Network.nodes.get(edgePair[1]).addIncomingEdgeIndex(Network.genome.size());
            Network.genome.add(new edge(i, 1, true,
                    edgePair[0], edgePair[1], edgePair[0] - 7, edgePair[1] - 7));
        }
        assertTrue(Network.classInv());

        NN Compare = (NN) Network.clone();
        Mutation.shiftWeights(Network);

        assertNotEquals(Compare, Network);
        int count = 0;
        for (int i = 0; i < Network.genome.size(); i++) {
            if (Network.genome.get(i).getWeight() != Compare.genome.get(i).getWeight()) {
                assertEquals(1.25, Network.genome.get(i).getWeight(), 0.75);
                count++;
            }
        }
        assertEquals(1, count);
    }

    @Test
    void testRandomWeightsNoEdge() {
        NN Network = NN.getDefaultNeuralNet(Constants), Compare = (NN) Network.clone();
        Mutation.randomWeights(Network);
        assertEquals(Network, Compare);
    }

    @Test
    void testRandomWeights1Edge() {
        NN Network = NN.getDefaultNeuralNet(Constants);
        Network.genome.add(new edge(0, new Random().nextDouble(Double.MIN_VALUE, Double.MAX_VALUE), true, 3, 4, -4, -3));
        Network.nodes.get(3).addOutgoingEdgeIndex(0);
        Network.nodes.get(4).addIncomingEdgeIndex(0);
        NN Compare = (NN) Network.clone();
        Mutation.randomWeights(Network);

        assertTrue(Network.classInv());
        assertNotEquals(Compare, Network);
        assertEquals(Compare.toString(), Network.toString());
        assertEquals(0, Network.genome.getFirst().getWeight(), Constants.mutationWeightRandomStrength);
    }

    @RepeatedTest(1000)
    void testRandomWeightsGeneralizedEdge() {
        NN Network = NN.getDefaultNeuralNet(Constants);
        List<int[]> edgePairs = new ArrayList<>();
        for (int i = 0; i < Constants.getInputNum(); i++)
            for (int j = Constants.getInputNum(); j < Constants.getInputNum() + Constants.getOutputNum(); j++)
                edgePairs.add(new int[]{i, j});

        int edgeCounts = new Random().nextInt(1, edgePairs.size());
        for (int i = 0; i < edgeCounts; i++) {
            int[] edgePair = edgePairs.remove((int) (Math.random() * edgePairs.size()));
            Network.nodes.get(edgePair[0]).addOutgoingEdgeIndex(Network.genome.size());
            Network.nodes.get(edgePair[1]).addIncomingEdgeIndex(Network.genome.size());
            Network.genome.add(new edge(i, new Random().nextDouble(Double.MIN_VALUE, Double.MAX_VALUE), true,
                    edgePair[0], edgePair[1], edgePair[0] - 7, edgePair[1] - 7));
        }
        assertTrue(Network.classInv());

        NN Compare = (NN) Network.clone();
        Mutation.randomWeights(Network);

        assertNotEquals(Compare, Network);
        int count = 0;
        for (int i = 0; i < Network.genome.size(); i++) {
            if (Network.genome.get(i).getWeight() != Compare.genome.get(i).getWeight()) {
                assertEquals(0, Network.genome.get(i).getWeight(), Constants.mutationWeightRandomStrength);
                count++;
            }
        }
        assertEquals(1, count);
    }

    @Test
    void testShiftBiasNoEdge() {
        NN Network = NN.getDefaultNeuralNet(Constants), Compare = (NN) Network.clone();
        Mutation.shiftBias(Network);

        assertNotEquals(Network, Compare);
        int difference = 0;
        for(int i = 0; i < Network.nodes.size(); i++) {
            if(Network.nodes.get(i).getBias() != Compare.nodes.get(i).getBias()) {
                assertEquals(Compare.nodes.get(i).getBias(), Network.nodes.get(i).getBias(), Constants.mutationBiasShiftStrength);
                difference++;
            }
        }
        assertEquals(1,difference);
    }

    @Test
    void testShiftBias() {
        NN Network = NN.getDefaultNeuralNet(Constants);
        NN Compare = (NN) Network.clone();
        Mutation.shiftBias(Network);

        assertTrue(Network.classInv());
        assertNotEquals(Compare, Network);
        int count = 0;
        for (int i = 0; i < Network.nodes.size(); i++) {
            if (Network.nodes.get(i).getBias() != Compare.nodes.get(i).getBias()) {
                assertEquals(Compare.nodes.get(i).getBias(), Network.nodes.get(i).getBias(), Constants.mutationBiasShiftStrength);
                count++;
            }
        }
        assertEquals(1, count);
    }

    @RepeatedTest(10)
    void testShiftBiasHiddenNodes() {
        NN Network = NN.getDefaultNeuralNet(Constants);

        int edgeCounts = new Random().nextInt(1, 10000);
        for (int i = 0; i < edgeCounts; i++) {
            Mutation.mutateSynapse(Network);
            Mutation.mutateNode(Network);
        }
        assertTrue(Network.classInv());

        NN Compare = (NN) Network.clone();
        Mutation.shiftBias(Network);

        assertNotEquals(Compare, Network);
        int count = 0;
        for (int i = 0; i < Network.nodes.size(); i++) {
            if (Network.nodes.get(i).getBias() != Compare.nodes.get(i).getBias()) {
                assertEquals(Compare.nodes.get(i).getBias(), Network.nodes.get(i).getBias(), Constants.mutationBiasShiftStrength);
                count++;
            }
        }
        assertEquals(1, count);
    }

    @Test
    void testMutateSynapseNoHiddenNodes() {
        NN Network = NN.getDefaultNeuralNet(Constants);
        for(int i=0;i<1000;i++){
            Mutation.mutateSynapse(Network);
            assertEquals(Math.min(i+1,Constants.getInputNum()*Constants.getOutputNum()),Network.genome.size());
            Network.genome.forEach(e -> {
                assertTrue(e.getPreviousIID() < -Constants.getOutputNum());
                assertTrue(e.getNextIID() < 0 && e.getNextIID() >= -Constants.getOutputNum());
            });
        }
    }

    @Test
    void testMutateSynapse2HiddenNodes() {
        NN Network = NN.getDefaultNeuralNet(Constants);
        Mutation.mutateSynapse(Network);
        Mutation.mutateNode(Network);
        Mutation.mutateNode(Network);


    }

    @Test
    void testMutateNodeNoSynapse() {
        NN Network = NN.getDefaultNeuralNet(Constants), Compare = (NN) Network.clone();
        Mutation.mutateNode(Network);
        assertEquals(Network, Compare);
    }

    @Test
    void testMutateNode1Edge() {
        NN Network = NN.getDefaultNeuralNet(Constants), Compare = (NN) Network.clone();
        Network.genome.add(new edge(Innovation.getEdgeInnovationID(-4, -3), new Random().nextDouble(Double.MIN_VALUE, Double.MAX_VALUE), true, 3, 4, -4, -3));
        Network.nodes.get(3).addOutgoingEdgeIndex(0);
        Network.nodes.get(4).addIncomingEdgeIndex(0);
        Mutation.mutateNode(Network);

        assertNotEquals(Network, Compare);
        assertEquals(3, Network.genome.size());
        assertTrue(Network.genome.getFirst().isDisabled());
        assertFalse(Network.genome.get(1).isDisabled());
        assertFalse(Network.genome.get(2).isDisabled());
        assertEquals(Network.genome.getFirst().getWeight(), Network.genome.get(1).getWeight());
        assertEquals(1, Network.genome.get(2).getWeight());
    }

    @Test
    void testMutateNodeGeneralizedEdge() {
        NN Network = NN.getDefaultNeuralNet(Constants);
        List<int[]> edgePairs = new ArrayList<>();
        for (int i = 0; i < Constants.getInputNum(); i++)
            for (int j = Constants.getInputNum(); j < Constants.getInputNum() + Constants.getOutputNum(); j++)
                edgePairs.add(new int[]{i, j});

        int edgeCounts = new Random().nextInt(1, edgePairs.size());
        for (int i = 0; i < edgeCounts; i++) {
            int[] edgePair = edgePairs.remove((int) (Math.random() * edgePairs.size()));
            Network.nodes.get(edgePair[0]).addOutgoingEdgeIndex(Network.genome.size());
            Network.nodes.get(edgePair[1]).addIncomingEdgeIndex(Network.genome.size());
            Network.genome.add(new edge(Innovation.getEdgeInnovationID(edgePair[0] - 7, edgePair[1] - 7), new Random().nextDouble(Double.MIN_VALUE, Double.MAX_VALUE), true,
                    edgePair[0], edgePair[1], edgePair[0] - 7, edgePair[1] - 7));
        }
        NN Compare = (NN) Network.clone();
        Mutation.mutateNode(Network);

        assertNotEquals(Network, Compare);
        assertEquals(Compare.genome.size()+2,Network.genome.size());

        int newNodeIID = Network.nodes.get(4).innovationID,prevCount = 0,nextCount = 0;
        for(edge e : Network.genome) {
            if(e.getPreviousIID()==newNodeIID) prevCount++;
            else if(e.getNextIID()==newNodeIID) nextCount++;
        }
        assertEquals(1,prevCount);
        assertEquals(1,nextCount);
    }

    @Test
    void testCrossoverEmptyGenome() {
        NN Network = NN.getDefaultNeuralNet(Constants);
        assertTrue(NN.crossover(Network, (NN) Network.clone(), 0, 1).classInv());
    }

    @RepeatedTest(1000)
    void testRandomCrossoverClassInv() {
        NN parent1 = NN.getDefaultNeuralNet(Constants);
        NN parent2 = NN.getDefaultNeuralNet(Constants);

        int randomMutation = new Random().nextInt(1, 1000);
        while (randomMutation-- > 0) parent1.mutate();

        randomMutation = new Random().nextInt(1, 1000);
            while (randomMutation-- > 0) parent2.mutate();

        assertTrue(parent1.classInv());
        assertTrue(parent2.classInv());

        assertTrue(NN.crossover(parent1, parent2, 0, 1).classInv());
        assertTrue(NN.crossover(parent1, parent2, 1, 0).classInv());
        assertTrue(NN.crossover(parent1, parent2, 0, 0).classInv());
        assertTrue(NN.crossover(parent1, parent2, 1, 1).classInv());
    }
}
