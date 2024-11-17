package Evolution.Genome;

import Evolution.Constants;
import Evolution.NN;

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

    /** The local indices of the previous and next node */
    private int previousIndex, nextIndex;

    /** The absolute Innovation ID of the previous and next node */
    private int previousIID,nextIID;

    /**
     * Used during {@link NN#mutateNode} and {@link NN#mutateSynapse} function to create new edges
     */
    public edge(int innovationID,int previousInnovationID,int nextInnovationID) {
        this.innovationID = innovationID;
        this.weight = 1;
        this.previousIID = previousInnovationID;
        this.nextIID = nextInnovationID;
        this.enabled = true;
    }

    private edge(int innovationID, double weight, boolean enabled, int previousIID, int nextIID) {
        this.innovationID = innovationID;
        this.weight = weight;
        this.previousIID = previousIID;
        this.nextIID = nextIID;
        this.enabled = enabled;
    }

    public double getWeight(){
        return weight;
    }

    public boolean isDisabled(){
        return !enabled;
    }

    /** Returns the local node index of the previous node */
    public int getPreviousIndex(){return previousIndex;}
    /** Returns the ID (local node index or Innovation ID) of the next node */
    public int getNextIndex(){return nextIndex;}

    /** Returns the Innovation ID of the previous node */
    public int getPreviousIID(){return previousIID;}
    /** Returns the Innovation ID of the next node */
    public int getNextIID(){return nextIndex;}

    /** Sets the local node index of the previous node */
    public void setPreviousIndex(int previousIndex){this.previousIndex = previousIndex;}
    /** Sets the local node index of the next node */
    public void setNextIndex(int nextIndex){this.nextIndex = nextIndex;}

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

    /**
     * Disables this synapse from the calculation process
     * <br> Returns false if it was already disabled, true otherwise
     */
    public void disable(){
        enabled = false;
    }

    public edge clone(List<node> nodes) {
        return new edge(innovationID, weight,enabled,nodes.get(previousIndex).innovationID,nodes.get(nextIndex).innovationID);
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof edge && ((edge)obj).innovationID == innovationID;
    }
}
