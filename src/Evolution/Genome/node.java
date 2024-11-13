package Evolution.Genome;

import Evolution.Constants;
import Genome.hidden;
import Genome.nodeType;

import java.util.ArrayList;
import java.util.List;

/** The Genome representation of a node */
public class node extends Gene {
    /*
     * must contain:
     * - nodeType enum: (input,hidden,output)
     * - innovation number:
     *      - should be the same as synapse split if hidden node ( >= 0)
     *      - if input/output node, should range from -(inputNum+outputNum) to -1
     * - ActivationFunction enum:
     *      - should be null if it's input or output
     *      - can be any hiddenAF enums if its hidden, outputAF are reserved to used in Constants file
     * - bias
     * - boolean activated: true if this node's output is > 0, false otherwise
     * - int array of outgoing connections
     * - int array of incoming connections
     */

    private final nodeType type;
    private hidden AF;
    private double bias;
    private boolean activated;
    private final List<Integer> incomingConnections = new ArrayList<>(),
            outgoingConnections = new ArrayList<>();


    public node(nodeType type, int innovationID, hidden AF, double bias){
        this.type = type;
        this.innovationID = innovationID;
        this.AF = AF;
        this.bias = bias;
        this.activated = false;
    }

    /**
     * Applies the bias and activation function to the given {@code input}.<br>
     * Changes the {@code activated} var to the appropriate value
     */
    public double calculateOutput(double input){
        if(type.equals(nodeType.output)) return input;
        return AF.evaluate(input+bias);
    }

    /**
     * Returns the local indices of all edges pointing into this node.
     * <br>Used for calculating derivative terms in back-propagation
     */
    public Integer[] getIncomingEdgeIndices() {
        return incomingConnections.toArray(new Integer[0]);
    }

    /**
     * Returns the local indices of all edges pointing out from this node.
     * <br>Used for feed-forward calculation
     */
    public Integer[] getOutgoingEdgeIndices(){
        return outgoingConnections.toArray(new Integer[0]);
    }

    /**
     * Attempts to add {@code index} to the array of outgoing edge indices.
     * <br>Returns false if the array already contains the index, true otherwise
     */
    public boolean addOutgoingEdgeIndex(int index){
        if(outgoingConnections.contains(index)) return false;
        outgoingConnections.add(index);
        return true;
    }
    /**
     * Attempts to add {@code index} to the array of incoming edge indices.
     * <br>Returns false if the array already contains the index, true otherwise
     */
    public boolean addIncomingEdgeIndex(int index){
        if(incomingConnections.contains(index)) return false;
        incomingConnections.add(index);
        return true;
    }

    /** Returns the type of this node */
    public nodeType getType() {return type;}

    /**
     * Shifts the Bias of this node by a random amount
     * @return false if this edge can't apply this mutation, true otherwise
     */
    public boolean shiftBias() {
        if(type.equals(nodeType.output)) return false;
        this.bias *= (Math.random()*2-1) * Constants.mutationBiasShiftStrength;
        return true;
    }

    @Override
    public node clone() {return new node(type,innovationID,AF,bias);}

    @Override
    public boolean equals(Object obj){
        return obj instanceof node && ((node)obj).innovationID == innovationID;
    }

}
