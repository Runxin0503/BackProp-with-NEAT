package Evolution;

import Genome.*;

import java.util.ArrayList;
import java.util.HashMap;

public class Agent {
    public NN NN;
    public double score;

    public Agent(){
        this(new NN());
    }

    public Agent(NN NN){
        this.score=-1;
        this.NN = NN;

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
    public double[] calculateWeightedOutput(double[] input){
        if (input.length!= Constants.inputNum){
            return null;
        }
        HashMap<Integer,Integer> countedInputs = new HashMap<Integer,Integer>();
        for(node n : NN.nodes)countedInputs.put(n.innovationID,0);

        for (synapse s : NN.synapses) {
            if (s.enabled) {
                countedInputs.replace(s.to.innovationID, countedInputs.get(s.to.innovationID) + 1);
            }

        }

//        batchNormalization(input);

        ArrayList<synapse> temp = new ArrayList<synapse>(NN.synapses);
        for(int i=0;i<temp.size();i++) if(!temp.get(i).enabled)temp.remove(i--);
        ArrayList<node> scanLayer = new ArrayList<node>();
        for(node n : NN.nodes)if(n.isInput()){
            scanLayer.add(n);
            n.latestOutput = input[n.innovationID];
        }
        ArrayList<synapse> bank = new ArrayList<synapse>();
        double[] outputList = new double[Constants.outputNum];
        while (!scanLayer.isEmpty()){
            ArrayList<synapse> layerConnectedSynapse = new ArrayList<synapse>();
            ArrayList<node> nextScan = new ArrayList<node>();
            for(int i=0;i<temp.size();i++){
                synapse scan = temp.get(i);
                if(scanLayer.contains(scan.from)){
                    scan.latestInput=scan.from.latestOutput;
                    countedInputs.replace(scan.to.innovationID,countedInputs.get(scan.to.innovationID)-1);
                    if(countedInputs.get(scan.to.innovationID)==0 && !nextScan.contains(scan.to)){
                        nextScan.add(scan.to);
                        layerConnectedSynapse.add(temp.remove(i));
                        for(int j=0;j<bank.size();j++){
                            if(bank.get(j).to.equals(scan.to)){
                                layerConnectedSynapse.add(bank.remove(j--));
                            }
                        }
                    }else{
                        bank.add(temp.remove(i));
                    }
                    i--;
                }
            }

            if(Constants.batchNormalizeLayer) batchNormalization(layerConnectedSynapse);//normalize output of previous layer

            for (node n : nextScan) {
                n.latestInputSum=0;
                for (int j = 0; j < layerConnectedSynapse.size(); j++) {
                    if (layerConnectedSynapse.get(j).to.equals(n)) {
                        n.latestInputSum += layerConnectedSynapse.get(j).latestInput * layerConnectedSynapse.remove(j--).weight;
                    }
                }
                n.latestInputSum += n.bias;
            }
            for(node n : nextScan){
                if(n.isOutput()){
                    outputList[n.innovationID-Constants.inputNum]=n.latestInputSum;
                }else{
                    n.latestOutput = Constants.hiddenAF.evaluate(n.latestInputSum);
                }
            }
            for(int i=nextScan.size()-1;i>=0;i--)if(nextScan.get(i).isOutput())nextScan.remove(i);
            scanLayer = nextScan;
        }

        return Constants.outputAF.evaluate(outputList);
    }

    /** Centers the distribution of values to 0 */
    private void batchNormalization(double[] input){
        int len = input.length;
        double sum=0;
        for (double val : input) {
            sum += val;
        }
        double mean = sum/len;
        sum=0;
        for (double val : input) {
            double temp = val - mean;
            sum += temp * temp;
        }
        double ISD = (sum==0.0 ? 1 : 1/Math.sqrt(sum/len));
        for(int i=0;i<len;i++){
            input[i]=(input[i]-mean)*ISD;
        }
    }

    /** Centers the distribution of values to 0 */
    private void batchNormalization(ArrayList<synapse> layerConnectedSynapse){
        double[] input = new double[layerConnectedSynapse.size()];
        for(int i=0;i<layerConnectedSynapse.size();i++)input[i]=layerConnectedSynapse.get(i).latestInput;
        batchNormalization(input);
        for(int i=0;i<layerConnectedSynapse.size();i++)layerConnectedSynapse.get(i).latestInput=input[i];
    }
}