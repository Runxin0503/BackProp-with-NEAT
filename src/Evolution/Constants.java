package Evolution;

import Genome.enums.hidden;
import Genome.enums.output;
import globalGenomes.globalInnovations;
import globalGenomes.globalNodes;

public class Constants {
    public static final int inputNum=4;
    public static final int outputNum=1;
    public static final globalNodes globalNodes= new globalNodes(inputNum, outputNum);
    public static final globalInnovations globalInnovations = new globalInnovations(globalNodes);

    public static final hidden hiddenAF = hidden.sigmoid;
    public static final output outputAF = output.none;

    public static final boolean batchNormalizeLayer = false;

    public static final double weightedExcess = 1;
    public static final double weightedDisjoints = 1;
    public static final double weightedWeights = 1;
    public static final double compatibilityThreshold = 4;
    public static final double maxStagDropoff = 20;

    public static final double mutationSynapseProbability=0.03;
    public static final double mutationNodeProbability=0.2;

    public static final double mutationWeightShiftProbability=0.06;
    public static final double mutationWeightRandomProbability=0.06;
    public static final double mutationBiasShiftProbability=0.06;

    public static final double mutationWeightShiftStrength=2;
    public static final double mutationWeightRandomStrength=2;
    public static final double mutationBiasShiftStrength=0.3;
    public static final double perctCull = 0.2;
}
