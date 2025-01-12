package Genome;

import Evolution.Constants;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A Static class containing mappings between node and synapses using their InnovationIDs
 */
class Innovation {

    /*
     * Must contain:
     * - A HashMap mapping synapses to pairs of nodes (creating a new synapse from two existing nodes)
     */

    /** Used to obtain the innovationID of the two nodes of any given synapse innovationID */
    private static final HashMap<intPairs, Integer> nodePairsToEdge = new HashMap<>();

    /**
     * Returns the Innovation ID of the edge connecting {@code node1IID} and {@code node2IID}
     */
    public static int getEdgeInnovationID(int node1IID, int node2IID) {
        intPairs pair = new intPairs(node1IID, node2IID);
        Integer edgeIID = nodePairsToEdge.get(pair);
        if (edgeIID != null) return edgeIID;

        nodePairsToEdge.put(pair, nodePairsToEdge.size());
        return nodePairsToEdge.size() - 1;
    }

    /** Re-initializes all hidden nodes to its appropriate values */
    public static void resetNodeCoords(NN nn) {
        int inputNum = nn.Constants.getInputNum(), outputNum = nn.Constants.getOutputNum(), nodesNum = nn.nodes.size();
        for (int i = 0; i < inputNum; i++) {
            nn.nodes.get(i).x = 0;
            nn.nodes.get(i).y = (i + 1) / (inputNum + 1.0);
        }
        for (int i = nodesNum - outputNum; i < nodesNum; i++) {
            nn.nodes.get(i).x = 1;
            nn.nodes.get(i).y = (i - (nodesNum - outputNum) + 1) / (outputNum + 1.0);
        }

        for (int i = inputNum, j = nn.nodes.size() - outputNum - 1; i <= j; i++, j--) {
            node front = nn.nodes.get(i),back = nn.nodes.get(j);
            front.x = (i - inputNum + 1) / (nodesNum - inputNum - outputNum + 1.0);
            back.x = (j - inputNum + 1) / (nodesNum - inputNum - outputNum + 1.0);
            if (i == j) {
                //middle node, y = avg of all prev and next connected nodes
                double y = 0;
                for(int edge : front.getIncomingEdgeIndices())
                    y += nn.nodes.get(nn.genome.get(edge).prevIndex).y;
                for(int edge : front.getOutgoingEdgeIndices())
                    y += nn.nodes.get(nn.genome.get(edge).nextIndex).y;
                y /= front.getIncomingEdgeIndices().length + front.getOutgoingEdgeIndices().length;
                front.y = y;
            } else {
                //1 sided node, y = avg of either prev or next connected nodes
                double frontY = 0, backY = 0;
                for(int edge : front.getIncomingEdgeIndices())
                    frontY += nn.nodes.get(nn.genome.get(edge).prevIndex).y;
                for(int edge : back.getOutgoingEdgeIndices())
                    backY += nn.nodes.get(nn.genome.get(edge).nextIndex).y;
                frontY /= front.getIncomingEdgeIndices().length;
                backY /= back.getOutgoingEdgeIndices().length;
                front.y = frontY;
                back.y = backY;
            }
        }
        assert nn.classInv();
    }

    /**
     * Creates a topologically sorted array of nodes from a list of edges (genomes) used for calculating
     * <br>The nodes should contain identical input/output nodes from {@code dominantNodes}
     * <br>It should also bind all node IDs in edges to the local node Index.
     */
    public static ArrayList<node> constructNetworkFromGenome(ArrayList<edge> genome, ArrayList<node> dominantNodes, ArrayList<node> submissiveNodes, Constants Constants) {
        // Maps Nodes to the number of incoming edges (for topological sorted orders)
        Map<Integer, AtomicInteger> indegree = new HashMap<>();
        //maps innovationID to node clones
        HashMap<Integer, node> innovationIDtoNodes = new HashMap<>();

        // Step 1:
        for (int i = 0; i < genome.size(); i++) {
            node u = getNodeByInnovationID(genome.get(i).getPreviousIID(), dominantNodes, submissiveNodes), v = getNodeByInnovationID(genome.get(i).getNextIID(), dominantNodes, submissiveNodes);

            //puts u and v inside map. sets u and v to appropriate object
            if (!innovationIDtoNodes.containsKey(u.innovationID)) innovationIDtoNodes.put(u.innovationID, u.clone());
            else u = innovationIDtoNodes.get(u.innovationID);
            if (!innovationIDtoNodes.containsKey(v.innovationID)) innovationIDtoNodes.put(v.innovationID, v.clone());
            else v = innovationIDtoNodes.get(v.innovationID);

            u.addOutgoingEdgeIndex(i);
            v.addIncomingEdgeIndex(i);

            indegree.putIfAbsent(u.innovationID, new AtomicInteger(0));
            indegree.putIfAbsent(v.innovationID, new AtomicInteger(0));
        }

        // Step 2: Initialize the queue with all input nodes in order of innovation ID
        Queue<node> queue = new LinkedList<>();
        for (int i = -Constants.getInputNum() - Constants.getOutputNum(); i < -Constants.getOutputNum(); i++)
            queue.add(innovationIDtoNodes.get(i));

        // Step 3: Process the nodes
        ArrayList<node> topologicalOrder = new ArrayList<>();
        while (!queue.isEmpty()) {
            node n = queue.poll();
            topologicalOrder.add(n);

            // For each neighbor, reduce the indegree and add to queue if it becomes 0
            for (int edgeIndex : n.getOutgoingEdgeIndices()) {
                node nextNode = innovationIDtoNodes.get(genome.get(edgeIndex).getNextIID());
                int val = indegree.get(nextNode.innovationID).incrementAndGet();
                if (val == nextNode.getIncomingEdgeIndices().length) {
                    queue.add(nextNode);
                }
            }
        }

        HashMap<Integer, Integer> innovationIDtoLocalIndex = new HashMap<>();
        for (int i = 0; i < topologicalOrder.size(); i++)
            innovationIDtoLocalIndex.put(topologicalOrder.get(i).innovationID, i);

        for (edge e : genome) {
            e.prevIndex = innovationIDtoLocalIndex.get(e.getPreviousIID());
            e.nextIndex = innovationIDtoLocalIndex.get(e.getNextIID());
        }

        //replace the last set of nodes (all output nodes) with a sorted fixed ascending innovation ID order
        topologicalOrder.subList(topologicalOrder.size() - Constants.getOutputNum(), topologicalOrder.size()).clear();
        for (int i = -Constants.getOutputNum(); i < 0; i++) topologicalOrder.add(innovationIDtoNodes.get(i));

        return topologicalOrder;
    }

    /**
     * Attempts to scan through all nodes in {@code nodeLists} and return a node of the right Innovation ID
     * <br>If none are found, throws an unchecked Exception
     * <br>(NOTE: If two nodes have identical InnovationID, whichever is first in an iterator is returned)
     */
    @SafeVarargs
    private static node getNodeByInnovationID(int innovationID, List<node>... nodeLists) {
        for (List<node> nodes : nodeLists) {
            for (node n : nodes) {
                if (n.innovationID == innovationID) {
                    return n;
                }
            }
        }
        throw new RuntimeException("no nodes found");
    }

    /** A Memory efficient object wrapper for two ints */
    private static class intPairs {
        private final long combined;

        private intPairs(int num1, int num2) {
            this.combined = ((long) num1 << 32) | (num2 & 0xFFFFFFFFL);
        }

        private int first() {
            return (int) (combined >> 32);
        }

        private int second() {
            return (int) combined;
        }

        @Override
        public int hashCode() {
            return Long.hashCode(combined);
        }

        @Override
        public boolean equals(Object obj) {
            return obj instanceof intPairs && combined == ((intPairs) obj).combined;
        }
    }
}
