package Evolution.Genome;

import Evolution.Constants;

import java.util.List;

/** Genome Blueprint for constructing the {@code sortedNetwork} of NN class */
public class edge extends Gene {

    /*
     * must contain:
     * - innovationID
     * - weight
     * - enabled: If this edge is active or not
     * - index of previous and next node in topological order
     */

    /** The weight of this edge, used in {@link #calculateOutput} */
    private double weight;

    /**
     * If false, this synapse will always return 0 when {@link #calculateOutput} is called.
     * <br>This synapse can also not be split to make new nodes if false.
     */
    private boolean enabled;

    /**
     * The previous and next nodes' 'ID'. Two definitions
     * <br> Refers to the index of the node in topological order
     * <br> Refers to the InnovationID of the nodes explicitly during crossover
     */
    private int previousID,nextID;


    public edge(int innovationID, double weight, boolean enabled, int previousInnovationID, int nextInnovationID) {
        this.innovationID = innovationID;
        this.weight = weight;
        this.previousID = previousInnovationID;
        this.nextID = nextInnovationID;
        this.enabled = enabled;
    }

    public double getWeight(){
        return weight;
    }

    public boolean isDisabled(){
        return !enabled;
    }

    /** Returns the ID (local node index or Innovation ID) of the previous node */
    public int getPreviousID(){return previousID;}
    /** Returns the ID (local node index or Innovation ID) of the next node */
    public int getNextID(){return nextID;}

    /** Used during Neural Network crossover to re-bind node ID from global InnovationID to local Index */
    public void setPreviousID(int previousID){this.previousID = previousID;}
    /** Used during Neural Network crossover to re-bind node ID from global InnovationID to local Index */
    public void setNextID(int nextID){this.nextID = nextID;}

    /** Applies the weight of this edge to the given {@code input} */
    public double calculateOutput(double input){
        if(enabled) return input * weight;
        return 0;
    }

    /**
     * Shifts the weight of this edge by a random amount
     * @return false if this edge can't apply this mutation, true otherwise
     */
    public boolean shiftWeights(){
        if(!enabled) return false;
        this.weight *= Constants.mutationWeightShiftStrength * (Math.random()*2-1);
        return true;
    }

    /**
     * Sets the weight of this edge to a random number
     * @return false if this edge can't apply this mutation, true otherwise
     */
    public boolean randomWeights(){
        if(!enabled) return false;
        this.weight = Constants.mutationWeightRandomStrength * (Math.random());
        return true;
    }

    public edge clone(List<node> nodes) {
        return new edge(innovationID, weight,enabled,nodes.get(previousID).innovationID,nodes.get(nextID).innovationID);
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof edge && ((edge)obj).innovationID == innovationID;
    }
}
