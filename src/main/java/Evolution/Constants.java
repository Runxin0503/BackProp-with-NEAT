package Evolution;

import Genome.Activation;
import Genome.Cost;
import Genome.Innovation;
import Genome.Optimizer;

import java.util.function.Supplier;

/**
 * A Constants class containing the default value of the {@link Evolution} class, which
 * is modified to fit the client's needs during the construction process
 */
public class Constants {
    int inputNum = -1, outputNum = -1;
    int numSimulated = -1;
    Activation.arrays outputAF = null;

    Activation defaultHiddenAF = Activation.none;
    Supplier<Double> defaultValueInitializer;
    Cost CostFunction = null;
    Optimizer optimizer = Optimizer.ADAM;

    private final Innovation Innovation = new Innovation();

    public double
            weightedExcess = 1,
            weightedDisjoints = 1,
            weightedWeights = 1,
            compatibilityThreshold = 4,
            maxStagDropoff = 20,
            mutationSynapseProbability = 0.03,
            mutationNodeProbability = 0.2,
            mutationWeightShiftProbability = 0.06,
            mutationWeightRandomProbability = 0.06,
            mutationBiasShiftProbability = 0.06,
            mutationChangeAFProbability = 0.06,
            mutationWeightShiftStrength = 2,
            mutationWeightRandomStrength = 2,
            mutationBiasShiftStrength = 0.3,
            perctCull = 0.2;

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
