package Evolution;

import Genome.*;

public class Agent {
    public Genome.NN NN;
    public double score;

    public Agent(){
        this.score=-1;
        this.NN = Innovation.getDefaultNode();

        //initializes the nodes array with a set of input and a set of output
    }

    /** Resets the score of this Agent */
    public void reset(){
        score=-1;
    }

    public String toString(){
        return NN.toString();
    }

//------------------------------------------------------------output--------------------------------------------------------------------------

    /** Calculates the weighted output of the values using the Neural Network currently in this Agent */
    public double[] calculateWeightedOutput(double[] input) {
        return NN.calculateWeightedOutput(input);
    }
}