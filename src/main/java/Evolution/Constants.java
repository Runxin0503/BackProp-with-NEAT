package Evolution;

import Genome.Activation;
import Genome.Cost;
import Genome.Innovation;
import Genome.Optimizer;

import java.util.function.Supplier;

/**
 * A Constants class containing the default value of the {@link Evolution} class, which
 * is modified to fit the client's needs during the construction process.
 */
public class Constants {
    private final int inputNum, outputNum;
    private final int numSimulated;
    private final Activation.arrays outputAF;

    private final Activation defaultHiddenAF;
    private final Supplier<Double> defaultValueInitializer;
    private final Cost CostFunction;
    private final Optimizer optimizer;

    private final Innovation Innovation = new Innovation();

    public Constants(int inputNum, int outputNum, int numSimulated, Activation defaultHiddenAF, Activation.arrays outputAF, Cost CostFunction, Optimizer Optimizer) {
        this.inputNum = inputNum;
        this.outputNum = outputNum;
        this.numSimulated = numSimulated;
        this.defaultHiddenAF = defaultHiddenAF;
        this.outputAF = outputAF;
        this.CostFunction = CostFunction;
        this.optimizer = Optimizer;
        defaultValueInitializer = Activation.getInitializer(defaultHiddenAF, inputNum, outputNum);
    }

    /**
     * The coefficient for how much excess genes contribute to the compatibility distance between genomes.
     * Larger values make speciation more sensitive to excess genes.
     */
    public double weightedExcess = 1;

    /**
     * The coefficient for how much disjoint genes contribute to the compatibility distance between genomes.
     * Larger values make speciation more sensitive to disjoint genes.
     */
    public double weightedDisjoints = 1;

    /**
     * The coefficient for how much differences in connection weights contribute to the compatibility distance.
     * Larger values make small weight differences matter more in speciation.
     */
    public double weightedWeights = 1;

    /**
     * The threshold for compatibility distance used to determine if two genomes belong to the same species.
     * If distance exceeds this threshold, genomes are separated into different species.
     */
    public double compatibilityThreshold = 4;

    /**
     * The maximum number of generations a species is allowed to stagnate (not improve) before it is removed.
     * Helps keep the population fresh by eliminating non-improving species.
     */
    public double maxStagDropoff = 20;

    /**
     * The probability that a mutation event adds a new synapse (connection) between two nodes.
     */
    public double mutationSynapseProbability = 0.03;


    /**
     * The probability that a mutation event adds a new node by splitting an existing connection.
     */
    public double mutationNodeProbability = 0.2;

    /**
     * The probability that a mutation event shifts (perturbs) the weight of an existing connection slightly.
     */
    public double mutationWeightShiftProbability = 0.06;

    /**
     * The probability that a mutation event assigns a completely new random weight to an existing connection.
     */
    public double mutationWeightRandomProbability = 0.06;

    /**
     * The probability that a mutation event shifts the bias of a node slightly.
     */
    public double mutationBiasShiftProbability = 0.06;

    /**
     * The probability that a mutation event changes a node's activation function to another random function.
     */
    public double mutationChangeAFProbability = 0.06;

    /**
     * The strength (magnitude) of changes when a weight shift mutation occurs.
     */
    public double mutationWeightShiftStrength = 2;

    /**
     * The range of possible values when assigning a random weight during a weight randomization mutation.
     */
    public double mutationWeightRandomStrength = 2;

    /**
     * The strength (magnitude) of changes when a node bias shift mutation occurs.
     */
    public double mutationBiasShiftStrength = 0.3;

    /**
     * The percentage of a species' population that is kept after culling the worst-performing individuals.
     * Helps maintain diversity while emphasizing strong performers.
     */
    public double perctCull = 0.2;

    public int getInputNum() {
        return inputNum;
    }

    public int getOutputNum() {
        return outputNum;
    }

    public int getNumSimulated() {
        return numSimulated;
    }

    public Activation getDefaultHiddenAF() {
        return defaultHiddenAF;
    }

    public double getInitializedValue() {
        return defaultValueInitializer.get();
    }

    public Cost getCostFunction() {
        return CostFunction;
    }

    public Activation.arrays getOutputAF() {
        return outputAF;
    }

    public Innovation getInnovation(){
        return Innovation;
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }
}
