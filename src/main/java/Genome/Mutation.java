package Genome;

import java.util.*;

/** Package-private Static class that mutates the genome of any given {@link NN} */
public class Mutation {

    /** Chooses a random synapse (if there is one) to shift its weight by a random amount */
    public static void shiftWeights(NN nn) {
        if (nn.genome.isEmpty()) return;
        for (int count = 0; count < 100; count++) {
            edge e = nn.genome.get((int) (Math.random() * nn.genome.size()));

            if (e.shiftWeights(nn.Constants.mutationWeightShiftStrength)) {
                assert nn.classInv() : nn.toString();
                return;
            }
        }
    }

    /** Chooses a random synapse (if there is one) to randomly set its weight */
    public static void randomWeights(NN nn) {
        if (nn.genome.isEmpty()) return;
        for (int count = 0; count < 100; count++) {
            edge e = nn.genome.get((int) (Math.random() * nn.genome.size()));

            if (e.randomWeights(nn.Constants.mutationWeightRandomStrength)) {
                assert nn.classInv() : nn.toString();
                return;
            }
        }
    }

    /** Chooses a random node to shift its bias by a random amount */
    public static void shiftBias(NN nn) {
        //randomly picks an index for all nodes except an input node
        int nodeIndex = (int) (Math.random() * (nn.nodes.size() - nn.Constants.getInputNum()) + nn.Constants.getInputNum());

        node n = nn.nodes.get(nodeIndex);
        n.shiftBias(nn.Constants);
        assert nn.classInv() : nn.toString();
    }

    /** Chooses two random nodes that aren't directly connected and create a synapse between them */
    public static void mutateSynapse(NN nn) {
        for (int count = 0; count < 100; count++) {
            int i1 = (int) (Math.random() * (nn.nodes.size() - nn.Constants.getOutputNum())), i2 = (int) (Math.random() * (nn.nodes.size() - nn.Constants.getInputNum())) + nn.Constants.getInputNum();

            if (i1 == i2 || (i1 > i2 && isLooping(i1, i2, nn)))
                continue;

            node n1 = nn.nodes.get(i1), n2 = nn.nodes.get(i2);
            int edgeIID = Innovation.getEdgeInnovationID(n1.getInnovationID(), n2.getInnovationID());

            edge newEdge = new edge(edgeIID, nn.Constants.getInitializedValue(), true, n1.getInnovationID(), n2.getInnovationID());
            newEdge.prevIndex = i1;
            newEdge.nextIndex = i2;

            if (!addEdge(nn, newEdge, n1, n2)) continue;

            // if i1 > i2, topologically sort the entire NN
            // if i1 < i2 no sorting required
            if (i1 > i2) {
                ArrayList<node> newNodes = Innovation.topologicalSort(List.copyOf(nn.nodes),nn.genome,nn.Constants);
                nn.nodes.clear();
                nn.nodes.addAll(newNodes);
            }
            Innovation.resetNodeCoords(nn);

            return;
        }
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

            node newNode = new node(Innovation.getSplitNodeInnovationID(edge.innovationID,nn.nodes), nn.Constants.getDefaultHiddenAF(), nn.Constants.getInitializedValue());

            node prevNode = nn.nodes.get(edge.prevIndex), nextNode = nn.nodes.get(edge.nextIndex);
            int prevIID = prevNode.getInnovationID(), midIID = newNode.getInnovationID(), nextIID = nextNode.getInnovationID();
            edge edge1 = new edge(Innovation.getEdgeInnovationID(prevIID, midIID), edge.getWeight(), true, prevIID, midIID);
            edge edge2 = new edge(Innovation.getEdgeInnovationID(midIID, nextIID), 1, true, midIID, nextIID);

            //add edges to genome
            if (!addEdge(nn, edge1, prevNode, newNode) || !addEdge(nn, edge2, newNode, nextNode))
                continue;

            edge.disable();

            //new node index can't be in between input node indexes
            int newNodeIndex = Math.max(edge.prevIndex + 1, nn.Constants.getInputNum());

            //increase every edge's nodeIndex by one if their index is after the new node index
            for (edge e : nn.genome) {
                if (e.prevIndex >= newNodeIndex) e.prevIndex++;
                if (e.nextIndex >= newNodeIndex) e.nextIndex++;
            }
            newNode.getIncomingEdgeIndices().clear();
            newNode.addIncomingEdgeIndex(nn.genome.indexOf(edge1));
            newNode.getOutgoingEdgeIndices().clear();
            newNode.addOutgoingEdgeIndex(nn.genome.indexOf(edge2));
            nn.nodes.add(newNodeIndex, newNode);

            edge1.prevIndex = edge.prevIndex;
            edge2.nextIndex = edge.nextIndex;
            edge1.nextIndex = newNodeIndex;
            edge2.prevIndex = newNodeIndex;

            Innovation.resetNodeCoords(nn);

            assert nn.classInv() : nn.toString();
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

    /**
     * Attempts to add an edge to {@code nn}
     * <br>Effect: Inserts the edge in a sorted manner into nn's genome,
     * re-indexes every node's local reference to their edges after that operation.
     * @param n1 the source node of newEdge
     * @param n2 the destination node of newEdge
     * @return false if nn's genome already contains an identical edge to {@code newEdge}, true otherwise
     */
    private static boolean addEdge(NN nn, edge newEdge, node n1, node n2) {
        int edgeIndex = 0;
        //attempts to insert newEdge at appropriate index and shift node's local references
        // ONLY IF genome is not empty
        if (!nn.genome.isEmpty()) {
            //sets edgeIndex to the first index where the edge has
            // either the same or a larger InnovationID than newEdge, or nn.genome.size() if
            // it has the largest InnovationID
            for (; edgeIndex < nn.genome.size(); edgeIndex++) {
                if (nn.genome.get(edgeIndex).innovationID == newEdge.innovationID) {
                    if (nn.genome.get(edgeIndex).isDisabled()) {
                        nn.genome.set(edgeIndex,newEdge);
                        return true;
                    }
                    else return false;
                }
                else if (nn.genome.get(edgeIndex).innovationID > newEdge.innovationID) break;
            }

            for (node n : nn.nodes) {
                for (int i = 0; i < n.getIncomingEdgeIndices().size(); i++)
                    if (n.getIncomingEdgeIndices().get(i) >= edgeIndex)
                        n.getIncomingEdgeIndices().set(i, n.getIncomingEdgeIndices().get(i) + 1);
                for (int i = 0; i < n.getOutgoingEdgeIndices().size(); i++)
                    if (n.getOutgoingEdgeIndices().get(i) >= edgeIndex)
                        n.getOutgoingEdgeIndices().set(i, n.getOutgoingEdgeIndices().get(i) + 1);
            }
        }
        nn.genome.add(edgeIndex, newEdge);

        n1.addOutgoingEdgeIndex(edgeIndex);
        n2.addIncomingEdgeIndex(edgeIndex);
        return true;
    }
}
