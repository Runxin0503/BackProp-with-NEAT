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
    //TODO add a constructor for this class with package-private access and change vars like inputNum to be final
    int inputNum = -1, outputNum = -1;
    int numSimulated = -1;
    Activation.arrays outputAF = null;

    Activation defaultHiddenAF = Activation.none;
    Supplier<Double> defaultValueInitializer;
    Cost CostFunction = null;
    Optimizer optimizer = Optimizer.ADAM;

    private final Innovation Innovation = new Innovation();

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
