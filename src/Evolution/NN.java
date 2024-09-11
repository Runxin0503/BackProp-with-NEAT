package Evolution;

import Evolution.Evolution;
import Genome.*;

import java.util.ArrayList;
import java.util.HashMap;

public class NN {
    public Genome genome;
    public double score;

    public NN(Genome genome){
        this.score=-1;
        this.genome = genome;

        //initializes the nodes array with a set of input and a set of output
    }

    public void reset(){
        score=-1;
    }

    public String toString(){
        return genome.toString();
    }

//------------------------------------------------------------output--------------------------------------------------------------------------

    public double[] calculateWeightedOutput(double... input){
        if (input.length!= Evolution.inputNum){
            return null;
        }
        HashMap<Integer,Integer> countedInputs = new HashMap<Integer,Integer>();
        for(node n : genome.nodes)countedInputs.put(n.innovationID,0);

        for (synapse s : genome.synapses) {
            if (s.enabled) {
                countedInputs.replace(s.to.innovationID, countedInputs.get(s.to.innovationID) + 1);
            }

        }

//        batchNormalization(input);

        ArrayList<synapse> temp = new ArrayList<synapse>(genome.synapses);
        for(int i=0;i<temp.size();i++) if(!temp.get(i).enabled)temp.remove(i--);
        ArrayList<node> scanLayer = new ArrayList<node>();
        for(node n : genome.nodes)if(n.isInput()){
            scanLayer.add(n);
            n.latestOutput = input[n.innovationID];
        }
        ArrayList<synapse> bank = new ArrayList<synapse>();
        double[] outputList = new double[Evolution.outputNum];
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

            batchNormalization(layerConnectedSynapse);//normalize output of previous layer

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
                    outputList[n.innovationID-Evolution.inputNum]=n.latestInputSum;
                }else{
                    n.latestOutput = activationFunction(n.latestInputSum,false);
                }
            }
            for(int i=nextScan.size()-1;i>=0;i--)if(nextScan.get(i).isOutput())nextScan.remove(i);
            scanLayer = nextScan;
        }

        if(Evolution.outputAF.equalsIgnoreCase("softmax")){
            outputList = softmaxActivationFunction(outputList);
        }else{
            for(int i = 0; i< Evolution.outputNum; i++){
                outputList[i]=activationFunction(outputList[i],true);
            }
        }
        return outputList;
    }

    private double activationFunction(double num, boolean isOutput){
        String function = isOutput ? Evolution.outputAF : Evolution.hiddenAF;
        if(function.equalsIgnoreCase("none")){
            return num;
        }else if(function.equalsIgnoreCase("relu")){
            return (num > 0) ? num : 0;
        }else if(function.equalsIgnoreCase("sigmoid")){
            return 1/(1+Math.pow(Math.E,-num));
        }else if(function.equalsIgnoreCase("tanh")){
            return (Math.pow(Math.E,num)-Math.pow(Math.E,-num))/(Math.pow(Math.E,num)+Math.pow(Math.E,-num));
        }else if(function.equalsIgnoreCase("leaky relu")){
            return Math.max(num,0.1*num);
        }
        System.out.println("ERROR");
        return num;
    }

    private double[] softmaxActivationFunction(double[] nums){
        double sum=0;
        double[] result = new double[nums.length];
        for (double num : nums) {
            sum += num;
        }
        for(int i=0;i<nums.length;i++){
            result[i]= sum==0 ? 0 : nums[i]/sum;
        }
        return result;
    }

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

    private void batchNormalization(ArrayList<synapse> layerConnectedSynapse){
        double[] input = new double[layerConnectedSynapse.size()];
        for(int i=0;i<layerConnectedSynapse.size();i++)input[i]=layerConnectedSynapse.get(i).latestInput;
        batchNormalization(input);
        for(int i=0;i<layerConnectedSynapse.size();i++)layerConnectedSynapse.get(i).latestInput=input[i];
    }
}