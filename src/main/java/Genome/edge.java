package Genome;

import java.util.List;

/** TODO */
class edge extends Gene {

    /** The weight of this edge */
    double weight;

    /**
     * If false, this synapse should not be considered in forward-feeding / back-propagation calculations.
     * <br>This synapse can also not be split to make new nodes if false.
     */
    boolean enabled;

    /** The local indices of the previous and next node */
    public int prevIndex, nextIndex;

    /** The absolute Innovation ID of the previous and next node */
    private final int previousIID, nextIID;

    edge(int innovationID, double weight, boolean enabled, int previousIID, int nextIID, Optimizer optimizer) {
        super(optimizer);
        this.innovationID = innovationID;
        this.weight = weight;
        this.previousIID = previousIID;
        this.nextIID = nextIID;
        this.enabled = enabled;
    }

    /** Returns the weight of this edge */
    double getWeight() {
        return weight;
    }

    /** Returns true if this edge is disabled, false otherwise */
    boolean isDisabled() {
        return !enabled;
    }

    /** Returns the Innovation ID of the previous node */
    int getPreviousIID() {
        return previousIID;
    }

    /** Returns the Innovation ID of the next node */
    int getNextIID() {
        return nextIID;
    }

    @Override
    protected void addValue(double deltaValue) {
        this.weight += deltaValue;
    }

    /**
     * Shifts the weight of this edge by a random amount
     * @return false if this edge can't apply this mutation, true otherwise
     */
    boolean shiftWeights(double mutationWeightShiftStrength) {
        if (!enabled) return false;
        this.weight *= Math.pow(mutationWeightShiftStrength, Math.random() * 2 - 1);
        return true;
    }

    /**
     * Sets the weight of this edge to a random number
     * @return false if this edge can't apply this mutation, true otherwise
     */
    boolean randomWeights(double mutationWeightRandomStrength) {
        if (!enabled) return false;
        this.weight = mutationWeightRandomStrength * (Math.random() * 2 - 1);
        return true;
    }

    /** TODO */
    public edge clone(List<node> nodes,Optimizer optimizer) {
        return new edge(innovationID, weight, enabled, nodes.get(prevIndex).innovationID, nodes.get(nextIndex).innovationID,optimizer);
    }

    /** TODO */
    @Override
    public boolean equals(Object obj) {
        return obj instanceof edge && ((edge) obj).innovationID == innovationID;
    }

    /** TODO */
    boolean identical(edge other) {
        return other.innovationID == innovationID && other.weight == weight && other.enabled == enabled &&
                other.prevIndex == prevIndex && other.nextIndex == nextIndex &&
                other.previousIID == previousIID && other.nextIID == nextIID;
    }
}
