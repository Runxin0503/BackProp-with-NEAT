package Genome;

abstract class Gene {

    /** Identifies the unique ID this gene has.
     * <br>Used for matching nodes against synapses and vice versa.
     * <br>Also used in Neural Network comparisons and crossovers.
     */
    protected int innovationID;

    /** returns the innovation ID of this gene */
    public int getInnovationID(){
        return innovationID;
    }

    /** Implementers should calculate the value of the output of this gene */
    public abstract double calculateOutput(double input);

    @Override
    public int hashCode(){
        return getInnovationID();
    }

    public abstract boolean equals(Object obj);
}
