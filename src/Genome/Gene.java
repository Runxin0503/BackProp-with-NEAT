package Genome;

abstract class Gene {

    /** Identifies the unique ID this gene has.
     * <br>Used for matching nodes against synapses and vice versa.
     * <br>Also used in Neural Network comparisons and crossovers.
     */
    protected int innovationID;

    /** The velocity and squared-velocity used for ADAM optimizer */
    protected double velocity = 0, velocitySquared = 0;

    /** Applies the given {@code Gradient} to the value of this Gene using the ADAM optimizer */
    abstract void applyGradient(double gradient,double adjustedLearningRate, double momentum, double beta, double epsilon);

    /** returns the innovation ID of this gene */
    int getInnovationID() {
        return innovationID;
    }

    /** Implementers should calculate the value of the output of this gene */
    abstract double calculateOutput(double input);

    @Override
    public int hashCode() {
        return getInnovationID();
    }

    @Override
    public abstract boolean equals(Object obj);
}
