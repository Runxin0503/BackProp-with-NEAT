package Evolution.Genome;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A Static class containing mappings between node and synapses using their InnovationIDs
 */
public class Innovation {

    /*
     * Must contain:
     * - A HashMap mapping synapses to pairs of nodes (creating a new synapse from two existing nodes)
     */

    /** Used to obtain the innovationID of the two nodes of any given synapse innovationID */
    private static final HashMap<intPairs,Integer> nodePairsToEdge = new HashMap<>();

    /**
     * Returns the Innovation ID of the edge connecting {@code node1IID} and {@code node2IID}
     */
    public static int getEdgeInnovationID(int node1IID, int node2IID) {
        intPairs pair = new intPairs(node1IID, node2IID);
        Integer edgeIID = nodePairsToEdge.get(pair);
        if(edgeIID != null) return edgeIID;
        else nodePairsToEdge.put(pair,nodePairsToEdge.size());
        return nodePairsToEdge.size()-1;
    }

    /**
     * Creates a topologically sorted array of nodes from a list of edges (genomes) used for calculating
     * <br>The nodes should contain identical input/output nodes from {@code dominantNodes}
     * <br>It should also bind all node IDs in edges to the local node Index.
     */
    public static ArrayList<node> constructNetworkFromGenome(ArrayList<edge> genome, ArrayList<node> dominantNodes, ArrayList<node> submissiveNodes){
        // Maps Nodes to the number of incoming edges (for topological sorted orders)
        Map<Integer, AtomicInteger> indegree = new HashMap<>();
        //maps innovationID to nodes
        HashMap<Integer,node> innovationIDtoNodes = new HashMap<>();

        // Step 1:
        for(int i=0;i<genome.size();i++){
            node u = getNodeByInnovationID(genome.get(i).getPreviousIID(),dominantNodes,submissiveNodes), v = getNodeByInnovationID(genome.get(i).getNextIID(),dominantNodes,submissiveNodes);

            //puts u and v inside map. sets u and v to appropriate object
            if(!innovationIDtoNodes.containsKey(u.innovationID)) innovationIDtoNodes.put(u.innovationID,u.clone());
            else u = innovationIDtoNodes.get(u.innovationID);
            if(!innovationIDtoNodes.containsKey(v.innovationID)) innovationIDtoNodes.put(v.innovationID,v.clone());
            else v = innovationIDtoNodes.get(v.innovationID);

            u.addOutgoingEdgeIndex(i);
            v.addIncomingEdgeIndex(i);

            indegree.putIfAbsent(u.innovationID,new AtomicInteger(0));
            indegree.putIfAbsent(v.innovationID,new AtomicInteger(0));
        }

        // Step 2: Initialize the queue with all nodes of indegree 0
        Queue<node> queue = new LinkedList<>();
        for(node n : innovationIDtoNodes.values()) if(n.getIncomingEdgeIndices().length == 0) queue.add(n);

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

        HashMap<Integer,Integer> innovationIDtoLocalIndex = new HashMap<>();
        for(int i=0;i<topologicalOrder.size();i++) innovationIDtoLocalIndex.put(topologicalOrder.get(i).innovationID,i);

        for(edge e : genome){
            e.setPreviousIndex(innovationIDtoLocalIndex.get(e.getPreviousIID()));
            e.setNextIndex(innovationIDtoLocalIndex.get(e.getNextIID()));
        }

        return topologicalOrder;
    }

    /**
     * Attempts to scan through all nodes in {@code nodeLists} and return a node of the right Innovation ID
     * <br>If none are found, throws an unchecked Exception
     * <br>(NOTE: If two nodes have identical InnovationID, whichever is first in an iterator is returned)
     */
    @SafeVarargs
    private static node getNodeByInnovationID(int innovationID, List<node>... nodeLists){
        for(List<node> nodes : nodeLists){
            for(node n : nodes){
                if(n.innovationID == innovationID){
                    return n;
                }
            }
        }
        throw new RuntimeException("no nodes found");
    }

    /** A Memory efficient object wrapper for two ints */
    private static class intPairs{
        private final long combined;
        private intPairs(int num1, int num2){
            this.combined = ((long)num1 << 32) | (num2 & 0xFFFFFFFFL);
        }

        private int first(){return (int) (combined >> 32);}
        private int second(){return (int) combined;}

        @Override
        public int hashCode(){return Long.hashCode(combined);}
    }
}
