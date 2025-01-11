package Genome;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;

/** Package-private Static class that mutates the genome of any given {@link NN} */
public class Mutation {

    /** Chooses a random synapse (if there is one) to shift its weight by a random amount */
    public static void shiftWeights(NN nn) {
        if (nn.genome.isEmpty()) return;
        for (int count = 0; count < 100; count++) {
            edge e = nn.genome.get((int) (Math.random() * nn.genome.size()));

            if (e.shiftWeights(nn.Constants.mutationWeightShiftStrength)) return;
        }
    }

    /** Chooses a random synapse (if there is one) to randomly set its weight */
    public static void randomWeights(NN nn) {
        if (nn.genome.isEmpty()) return;
        for (int count = 0; count < 100; count++) {
            edge e = nn.genome.get((int) (Math.random() * nn.genome.size()));

            if (e.randomWeights(nn.Constants.mutationWeightRandomStrength)) return;
        }
    }

    /** Chooses a random node to shift its bias by a random amount */
    public static void shiftBias(NN nn) {
        for (int count = 0; count < 100; count++) {
            //randomly picks an index for all nodes except an input node
            int nodeIndex = (int) (Math.random() * (nn.nodes.size() - nn.Constants.getInputNum()) + nn.Constants.getInputNum());

            node n = nn.nodes.get(nodeIndex);
            if (n.shiftBias(nn.Constants)) return;
        }
    }

    /** Chooses two random nodes that aren't directly connected and create a synapse between them */
    public static void mutateSynapse(NN nn) {
        for (int count = 0; count < 100; count++) {
            int i1 = (int) (Math.random() * (nn.nodes.size() - nn.Constants.getOutputNum())), i2 = (int) (Math.random() * (nn.nodes.size() - nn.Constants.getInputNum())) + nn.Constants.getInputNum();

            if (i1 == i2 || (i1 > i2 && isLooping(i1, i2, nn)))
                continue;

            node n1 = nn.nodes.get(i1), n2 = nn.nodes.get(i2);
            int edgeIID = Innovation.getEdgeInnovationID(n1.getInnovationID(), n2.getInnovationID());

            int edgeIndex = nn.genome.size();
            edge newEdge = new edge(edgeIID, nn.Constants.getInitializedValue(), true, n1.getInnovationID(), n2.getInnovationID());
            newEdge.prevIndex = i1;
            newEdge.nextIndex = i2;

            if(nn.genome.contains(newEdge)) continue;
            nn.genome.add(newEdge);

            n1.addOutgoingEdgeIndex(edgeIndex);
            n2.addIncomingEdgeIndex(edgeIndex);

            // if i1 > i2, move and insert i1 at i2's position and push i2 back
            // if i1 < i2 no sorting required
            if (i1 > i2) {
                //moves i1 node to i2's index and shifts every node in [i2,i1) to the right
                nn.nodes.add(i2, nn.nodes.remove(i1));

                for (edge e : nn.genome) {
                    //shifts node index to the right if its within [i2,i1)
                    //sets node index to i2 if it equals i1
                    int prevIndex = e.prevIndex, nextIndex = e.nextIndex;
                    if (prevIndex == i1) e.prevIndex = i2;
                    else if (prevIndex < i1 && prevIndex >= i2) e.prevIndex = prevIndex + 1;
                    if (nextIndex == i1) e.nextIndex = i2;
                    else if (nextIndex < i1 && nextIndex >= i2) e.nextIndex = prevIndex + 1;
                }
            }

            return;
        }
        System.err.println("failed");
    }

    /**
     * Chooses a random synapse (if there is one) and insert a new Node directly in the middle<br>
     * The two previously connected nodes will now connect through this new node.<br>
     * The previous synapse will be removed. Two new synapses will be created connecting the 3 nodes
     */
    public static void mutateNode(NN nn) {
        if (nn.genome.isEmpty()) return;
        for (int count = 0; count < 100; count++) {
            //any edge in the genome is valid for node splitting
            edge edge = nn.genome.get((int) (Math.random() * nn.genome.size()));
            if (edge.isDisabled()) continue;

            edge.disable();

            node newNode = new node(edge.getInnovationID(), nn.Constants.getDefaultHiddenAF(), nn.Constants.getInitializedValue());

            node prevNode = nn.nodes.get(edge.prevIndex), nextNode = nn.nodes.get(edge.nextIndex);
            int prevIID = prevNode.getInnovationID(), midIID = newNode.getInnovationID(), nextIID = nextNode.getInnovationID();
            edge edge1 = new edge(Innovation.getEdgeInnovationID(prevIID, midIID), nn.Constants.getInitializedValue(), true, prevIID, midIID);
            edge edge2 = new edge(Innovation.getEdgeInnovationID(midIID, nextIID), nn.Constants.getInitializedValue(), true, midIID, nextIID);

            //add indices to nodes
            prevNode.addOutgoingEdgeIndex(nn.genome.size());
            newNode.addIncomingEdgeIndex(nn.genome.size());
            newNode.addOutgoingEdgeIndex(nn.genome.size() + 1);
            nextNode.addIncomingEdgeIndex(nn.genome.size() + 1);
            nn.genome.add(edge1);
            nn.genome.add(edge2);

            // insert the brand new node right after prevNode index, push every other node back and re-index
            edge1.prevIndex = edge.prevIndex;
            edge2.nextIndex = edge.nextIndex;

            //new node index can't be in between input node indexes
            int newNodeIndex = Math.max(edge.prevIndex + 1,nn.Constants.getInputNum());

            //increase every edge's nodeIndex by one if their index is after the new node index
            for (edge e : nn.genome) {
                if (e.prevIndex >= newNodeIndex) e.prevIndex++;
                if (e.nextIndex >= newNodeIndex) e.nextIndex++;
            }
            nn.nodes.add(newNodeIndex, newNode);

            edge1.nextIndex = newNodeIndex;
            edge2.prevIndex = newNodeIndex;

            Innovation.resetNodeCoords(nn);

            return;
        }
    }

    /** Returns whether introducing an edge from {@code rootNodeIndex} to {@code newEdgeNodeIndex} will introduce a cycle */
    private static boolean isLooping(int rootNodeIndex, int newEdgeNodeIndex, NN nn) {
        HashSet<Integer> visitedNodes = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        queue.add(newEdgeNodeIndex);
        visitedNodes.add(newEdgeNodeIndex);
        while (!queue.isEmpty()) {
            int nodeIndex = queue.remove();
            for (int edgeIndex : nn.nodes.get(nodeIndex).getOutgoingEdgeIndices()) {
                int nextNode = nn.genome.get(edgeIndex).nextIndex;
                if (nextNode == rootNodeIndex) return true;
                if (!visitedNodes.contains(nextNode)) queue.add(nextNode);
            }
            visitedNodes.add(nodeIndex);
        }

        return false;
    }
}
