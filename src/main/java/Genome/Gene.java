package Genome;

abstract class Gene {

    /** Identifies the unique ID this gene has.
     * <br>Used for matching nodes against synapses and vice versa.
     * <br>Also used in Neural Network comparisons and crossovers.
     */
    int innovationID;

    /** The velocity and squared-velocity used for ADAM optimizer */
    private double velocity, velocitySquared;

    /** The Gradient of this particular Gene. */
    private double gradient = 0;

    /** Clears the Gradient of this particular node */
    void clearGradient() {
        this.gradient = 0;
    }

    /** Adds {@code deltaGradient} to the existing gradient of this Gene */
    synchronized void addGradient(double deltaGradient) {
        this.gradient += deltaGradient;
    }

    Gene(Optimizer optimizer) {
        if (optimizer == Optimizer.SGD_MOMENTUM || optimizer == Optimizer.ADAM)
            this.velocity = 0;
        if (optimizer == Optimizer.RMS_PROP || optimizer == Optimizer.ADAM)
            this.velocitySquared = 0;
    }

    /**
     * Applies the given {@code Gradient} to the value of this Gene using the ADAM optimizer
     * @param adjustedLearningRate the learning rate for {@link NN#learn} divided by the batch size
     * @param momentum the momentum hyper-parameter in {@link NN#learn}, used in SGD with momentum and ADAM
     * @param correctionMomentum 1 / (1 - momentum^t), where t is the number of times the Neural Network was trained
     * @param beta the beta hyper-parameter in {@link NN#learn}, used in RMS-Prop and ADAM
     * @param correctionBeta 1 / (1 - beta^t), where t is the number of times the Neural Network was trained
     * @param epsilon a hyper-parameter in {@link NN#learn}, typically a very small value like {@code 1e-8}
     */
    void applyGradient(double adjustedLearningRate, double momentum, double correctionMomentum, double beta, double correctionBeta, double epsilon, Optimizer optimizer) {
        switch (optimizer) {
            case SGD -> addValue(-adjustedLearningRate * gradient);
            case SGD_MOMENTUM -> {
                velocity = momentum * velocity + (1 - momentum) * gradient;
                addValue(-adjustedLearningRate * velocity);
            }
            case RMS_PROP -> {
                velocitySquared = beta * velocitySquared + (1 - beta) * gradient * gradient;
                addValue(-adjustedLearningRate * gradient / Math.sqrt(velocitySquared + epsilon));
            }
            case ADAM -> {
                velocity = momentum * velocity + (1 - momentum) * gradient;
                velocitySquared = beta * velocitySquared + (1 - beta) * gradient * gradient;
                double correctedVelocity = velocity / correctionMomentum;
                double correctedVelocitySquared = velocitySquared / correctionBeta;
                addValue(-adjustedLearningRate * correctedVelocity / Math.sqrt(correctedVelocitySquared + epsilon));
            }
            case null, default -> throw new RuntimeException("Unknown Optimizer: " + optimizer);
        }
    }

    /** Adds {@code deltaValue} to the current value of this Gene, used for back-propagation to tune Gene values */
    abstract void addValue(double deltaValue);

    /** returns the innovation ID of this gene */
    int getInnovationID() {
        return innovationID;
    }

    /** TODO */
    @Override
    public int hashCode() {
        return getInnovationID();
    }

    /** TODO */
    @Override
    public abstract boolean equals(Object obj);
}
