package Evolution;

import Genome.enums.Activation;

/**
 * A Constants class containing the default value of the {@link Evolution} class, which
 * is modified to fit the client's needs during the construction process
 */
public class Constants {
    int inputNum = -1,outputNum = -1;
    int numSimulated = -1;
    Activation.arrays outputAF = null;

    Activation defaultHiddenAF = Activation.none;
    boolean batchNormalizeLayer = false;

    public double
            weightedExcess = 1,
            weightedDisjoints = 1,
            weightedWeights = 1,
            compatibilityThreshold = 4,
            maxStagDropoff = 20,
            mutationSynapseProbability=0.03,
            mutationNodeProbability=0.2,
            mutationWeightShiftProbability=0.06,
            mutationWeightRandomProbability=0.06,
            mutationBiasShiftProbability=0.06,
            mutationWeightShiftStrength=2,
            mutationWeightRandomStrength=2,
            mutationBiasShiftStrength=0.3,
            perctCull = 0.2;

    public int getNumSimulated() {
        return numSimulated;
    }

    public int getInputNum() {
        return inputNum;
    }

    public int getOutputNum() {
        return outputNum;
    }

    public Activation getDefaultHiddenAF() {
        return defaultHiddenAF;
    }

    public Activation.arrays getOutputAF() {
        return outputAF;
    }
}
