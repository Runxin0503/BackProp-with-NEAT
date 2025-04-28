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
    final int numSimulated;
    private final Activation.arrays outputAF;

    private final Activation defaultHiddenAF;
    private final Supplier<Double> defaultValueInitializer;
    private final Cost CostFunction;
    private final Optimizer optimizer;

    private final Innovation Innovation = new Innovation();

    Constants(int inputNum, int outputNum, int numSimulated, Activation defaultHiddenAF, Activation.arrays outputAF, Cost CostFunction, Optimizer Optimizer) {
        this.inputNum = inputNum;
        this.outputNum = outputNum;
        this.numSimulated = numSimulated;
        this.defaultHiddenAF = defaultHiddenAF;
        this.outputAF = outputAF;
        this.CostFunction = CostFunction;
        this.optimizer = Optimizer;
        defaultValueInitializer = Activation.getInitializer(defaultHiddenAF, inputNum, outputNum);
    }

    /** TODO */
    public double weightedExcess = 1;

    /** TODO */
    public double weightedDisjoints = 1;

    /** TODO */
    public double weightedWeights = 1;

    /** TODO */
    public double compatibilityThreshold = 4;

    /** TODO */
    public double maxStagDropoff = 20;

    /** TODO */
    public double mutationSynapseProbability = 0.03;

    /** TODO */
    public double mutationNodeProbability = 0.2;

    /** TODO */
    public double mutationWeightShiftProbability = 0.06;

    /** TODO */
    public double mutationWeightRandomProbability = 0.06;

    /** TODO */
    public double mutationBiasShiftProbability = 0.06;

    /** TODO */
    public double mutationChangeAFProbability = 0.06;

    /** TODO */
    public double mutationWeightShiftStrength = 2;

    /** TODO */
    public double mutationWeightRandomStrength = 2;

    /** TODO */
    public double mutationBiasShiftStrength = 0.3;

    /** TODO */
    public double perctCull = 0.2;

    public int getInputNum() {
        return inputNum;
    }

    public int getOutputNum() {
        return outputNum;
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
