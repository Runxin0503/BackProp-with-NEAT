package Genome;

import Evolution.Constants;
import Genome.enums.nodeType;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/** Package-private Static class that mutates the genome of any given {@link NN} */
class Mutation {

    /** Chooses a random synapse (if there is one) to shift its weight by a random amount */
    static void shiftWeights(NN nn){
        if(nn.genome.isEmpty())return;
        for(int count = 0; count <100; count++){
            edge e = nn.genome.get((int)(Math.random()*nn.genome.size()));

            if(e.shiftWeights()) return;
        }
    }

    /** Chooses a random synapse (if there is one) to randomly set its weight */
    static void randomWeights(NN nn){
        if(nn.genome.isEmpty())return;
        for(int count = 0; count <100; count++){
            edge e = nn.genome.get((int)(Math.random()*nn.genome.size()));

            if(e.randomWeights()) return;
        }
    }

    /** Chooses a random node to shift its bias by a random amount */
    static void shiftBias(NN nn){
        for(int count = 0; count <100; count++){
            node n = nn.nodes.get((int)(Math.random()*nn.nodes.size()));
            if(n.getType().equals(nodeType.output)) continue;

            if(n.shiftBias()) return;
        }
    }

    /** Chooses two random nodes that aren't directly connected and create a synapse between them */
    static void mutateSynapse(NN nn){
        for(int count = 0; count <100; count++){
            int i1 = (int)(Math.random()*nn.nodes.size()), i2 = (int)(Math.random()*nn.nodes.size());

            if(i1==i2 || (i1 > i2 && isLooping(i1,i2,nn)) || i2 < Constants.inputNum || i1 >= nn.nodes.size()-Constants.outputNum) continue;

            node n1 = nn.nodes.get(i1), n2 = nn.nodes.get(i2);
            int edgeIID = Innovation.getEdgeInnovationID(n1.getInnovationID(),n2.getInnovationID());

            int edgeIndex = nn.genome.size();
            nn.genome.add(new edge(edgeIID,n1.getInnovationID(),n2.getInnovationID()));
            n1.addOutgoingEdgeIndex(edgeIndex);
            n2.addIncomingEdgeIndex(edgeIndex);

            // if i1 > i2, move and insert i1 at i2's position and push i2 back
            // if i1 < i2 no sorting required
            if(i1>i2){
                //moves i1 node to i2's index and shifts every node in [i2,i1) to the right
                nn.nodes.add(i2,nn.nodes.remove(i1));

                for(edge e : nn.genome){
                    //shifts node index to the right if its within [i2,i1)
                    //sets node index to i2 if it equals i1
                    int prevIndex = e.getPreviousIndex(), nextIndex = e.getNextIndex();
                    if(prevIndex==i1) e.setPreviousIndex(i2);
                    else if(prevIndex < i1 && prevIndex >= i2) e.setPreviousIndex(prevIndex+1);
                    if(nextIndex==i1) e.setNextIndex(i2);
                    else if(nextIndex < i1 && nextIndex >= i2) e.setNextIndex(prevIndex+1);
                }
            }

            return;
        }
    }

    /**
     * Chooses a random synapse (if there is one) and insert a new Node directly in the middle<br>
     * The two previously connected nodes will now connect through this new node.<br>
     * The previous synapse will be removed. Two new synapses will be created connecting the 3 nodes
     */
    static void mutateNode(NN nn){
        if(nn.genome.isEmpty())return;
        for(int count = 0; count <100; count++){
            //any edge in the genome is valid for node splitting
            edge edge = nn.genome.get((int)(Math.random()*nn.genome.size()));
            if(edge.isDisabled()) continue;

            edge.disable();

            node newNode = new node(nodeType.hidden,edge.getInnovationID(),Constants.hiddenAF);

            node prevNode = nn.nodes.get(edge.getPreviousIndex()),nextNode = nn.nodes.get(edge.getNextIndex());
            int prevIID = prevNode.getInnovationID(), midIID = newNode.getInnovationID(), nextIID = nextNode.getInnovationID();
            edge edge1 = new edge(Innovation.getEdgeInnovationID(prevIID,midIID),prevIID,midIID);
            edge edge2 = new edge(Innovation.getEdgeInnovationID(midIID,nextIID),midIID,nextIID);

            //add indices to nodes
            prevNode.addOutgoingEdgeIndex(nn.genome.size());
            nextNode.addIncomingEdgeIndex(nn.genome.size()+1);
            nn.genome.add(edge1);
            nn.genome.add(edge2);

            // insert the brand new node right after prevNode index, push every other node back and re-index
            edge1.setPreviousIndex(edge.getPreviousIndex());
            edge2.setNextIndex(edge.getNextIndex());

            int newNodeIndex = edge.getPreviousIndex()+1;

            //increase every edge's nodeIndex by one if their index is after the new node index
            for(edge e : nn.genome){
                if(e.getPreviousIndex() >= newNodeIndex) e.setPreviousIndex(e.getPreviousIndex()+1);
                if(e.getNextIndex() >= newNodeIndex) e.setNextIndex(e.getNextIndex()+1);
            }
            nn.nodes.add(newNodeIndex,newNode);

            edge1.setNextIndex(newNodeIndex);
            edge2.setPreviousIndex(newNodeIndex);

            return;
        }
    }

    /** Returns whether introducing an edge from {@code rootNodeIndex} to {@code newEdgeNodeIndex} will introduce a cycle */
    static boolean isLooping(int rootNodeIndex,int newEdgeNodeIndex,NN nn){
        HashSet<Integer> visitedNodes = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        queue.add(newEdgeNodeIndex);
        visitedNodes.add(newEdgeNodeIndex);
        while(!queue.isEmpty()){
            int nodeIndex = queue.remove();
            for(int edgeIndex : nn.nodes.get(nodeIndex).getOutgoingEdgeIndices()){
                int nextNode = nn.genome.get(edgeIndex).getNextIndex();
                if(nextNode==rootNodeIndex) return true;
                if(!visitedNodes.contains(nextNode)) queue.add(nextNode);
            }
            visitedNodes.add(nodeIndex);
        }

        return false;
    }
}
