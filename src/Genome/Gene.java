package Genome;

abstract class Gene {

    /** Identifies the unique ID this gene has.
     * <br>Used for matching nodes against synapses and vice versa.
     * <br>Also used in Neural Network comparisons and crossovers.
     */
    protected int innovationID;

    /** The velocity and squared-velocity used for ADAM optimizer */
    protected double velocity = 0, velocitySquared = 0;

    /** The Gradient of this particular Gene. */
    protected double gradient = 0;

    /** Clears the Gradient of this particular node */
    void clearGradient(){this.gradient = 0;}

    /** Adds {@code newGradient} to the existing gradient of this Gene */
    synchronized void addGradient(double newGradient){this.gradient += newGradient;}

    /**
     * Applies the given {@code Gradient} to the value of this Gene using the ADAM optimizer
     * @param gradient the derivative of this Gene's value with respect to the loss function
     * @param adjustedLearningRate the learning rate for {@link NN#learn} divided by the batch size
     * @param momentum the momentum hyper-parameter in {@link NN#learn}, used in SGD with momentum and ADAM
     * @param correctionMomentum 1 / (1 - momentum^t), where t is the number of times the Neural Network was trained
     * @param beta the beta hyper-parameter in {@link NN#learn}, used in RMS-Prop and ADAM
     * @param correctionBeta 1 / (1 - beta^t), where t is the number of times the Neural Network was trained
     * @param epsilon a hyper-parameter in {@link NN#learn}, typically a very small value like {@code 1e-8}
     */
    void applyGradient(double gradient,double adjustedLearningRate, double momentum, double beta,double correctionMomentum,double correctionBeta, double epsilon){
        velocity = momentum * velocity + (1 - momentum) * gradient;
        velocitySquared = beta * velocitySquared + (1 - beta) * gradient * gradient;
        double correctedVelocity = velocity * correctionMomentum;
        double correctedVelocitySquared = velocitySquared * correctionBeta;
        addValue(-adjustedLearningRate * correctedVelocity / Math.sqrt(correctedVelocitySquared + epsilon));
    }

    /** Adds {@code deltaValue} to the current value of this Gene, used for back-propagation to tune Gene values */
    abstract void addValue(double deltaValue);

    /** returns the innovation ID of this gene */
    int getInnovationID() {return innovationID;}

    /** Implementers should calculate the value of the output of this gene */
    abstract double calculateOutput(double input);

    @Override
    public int hashCode() {
        return getInnovationID();
    }

    @Override
    public abstract boolean equals(Object obj);
}
