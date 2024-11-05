package Genome;

import java.util.ArrayList;

import Evolution.Constants;

public class NN {

    public ArrayList<synapse> synapses;
    public ArrayList<node> nodes;

    public NN(){
        this.synapses = new ArrayList<synapse>();
        this.nodes = new ArrayList<node>();
        for(int i = 0; i < Constants.inputNum+ Constants.outputNum; i++)nodes.add(Constants.globalNodes.get(i));
    }

    public double compare(NN other){
        if(this.synapses.isEmpty()&&other.synapses.isEmpty())return 0;
        NN maxInnoNet = this;
        NN minInnoNet = other;
        int maxInnoNum = maxInnoNet.synapses.isEmpty() ? 0 : maxInnoNet.synapses.get(maxInnoNet.synapses.size()-1).innovationID;
        int minInnoNum = minInnoNet.synapses.isEmpty() ? 0 : minInnoNet.synapses.get(minInnoNet.synapses.size()-1).innovationID;
        if(maxInnoNum<minInnoNum){
            maxInnoNet = other;
            minInnoNet = this;
        }

        int index1 = 0,index2 = 0;

        int disjoint = 0,excess,similar = 0;
        double weight_diff = 0;


        while(index1 < maxInnoNet.synapses.size() && index2 < minInnoNet.synapses.size()){

            synapse gene1 = maxInnoNet.synapses.get(index1);
            synapse gene2 = minInnoNet.synapses.get(index2);

            int firstInnovationID = gene1.innovationID;
            int secondInnovationID = gene2.innovationID;

            if(firstInnovationID == secondInnovationID){
                //similargene
                similar ++;
                weight_diff += Math.abs(gene1.weight - gene2.weight);
                index1++;
                index2++;
            }else if(firstInnovationID > secondInnovationID){
                //disjoint gene of b
                disjoint ++;
                index2++;
            }else{
                //disjoint gene of a
                disjoint ++;
                index1 ++;
            }
        }

        weight_diff /= Math.max(1,similar);
        excess = maxInnoNet.synapses.size() - index1;

        double N = Math.max(maxInnoNet.synapses.size(),minInnoNet.synapses.size());
        if(N < 20) N = 1;

        return Constants.weightedDisjoints * disjoint / N + Constants.weightedExcess * excess / N + Constants.weightedWeights * weight_diff;
    }

    public static NN crossover(NN first, NN second, double firstScore, double secondScore){
        NN NN = new NN();

        int index1 = 0,index2 = 0;
        boolean equalScore = firstScore==secondScore;
        if(firstScore<secondScore){
            NN temp = first;
            first = second;
            second = temp;
        }

        while(index1 < first.synapses.size() && index2 < second.synapses.size()){

            synapse gene1 = first.synapses.get(index1);
            synapse gene2 = second.synapses.get(index2);

            int firstInnovationID = gene1.innovationID;
            int secondInnovationID = gene2.innovationID;

            if(firstInnovationID == secondInnovationID){
                if(Math.random() > 0.5){
                    NN.synapses.add(gene1.clone());
                }else{
                    NN.synapses.add(gene2.clone());
                }
                index1++;
                index2++;
            }else if(firstInnovationID > secondInnovationID){
                if(equalScore) NN.synapses.add(gene2.clone());
                //disjoint gene of b
                index2++;
            }else{
                //disjoint gene of a
                NN.synapses.add(gene1.clone());
                index1++;
            }
        }

        while(index1 < first.synapses.size()){
            synapse gene1 = first.synapses.get(index1);
            NN.addSynapse(gene1.clone());
            index1++;
        }
        if(equalScore){
            while(index2 < second.synapses.size()){
                synapse gene2 = second.synapses.get(index2);
                NN.addSynapse(gene2.clone());
                index2++;
            }
        }

        for(synapse s : NN.synapses){
            s.from = NN.addNode(s.from.clone());
            s.to = NN.addNode(s.to.clone());
        }

//        if(genome.nodes.size() < first.inputNum+first.outputNum){
//            System.out.println(first.nodes+"\n"+first.synapses+"\n\n"+second.nodes+"\n"+second.synapses);
//        }
        return NN;
    }

    public void addSynapse(synapse s){
        if(!synapses.contains(s)){
            if(synapses.isEmpty()||synapses.get(synapses.size()-1).innovationID<s.innovationID){
                synapses.add(s);
            }else{
                for(int i=0;i<synapses.size();i++){
                    if(synapses.get(i).innovationID>s.innovationID){
                        synapses.add(i,s);
                        return;
                    }
                }
            }
        }
    }

    public node addNode(node n){
        if(!nodes.contains(n)){
            if(nodes.isEmpty()||nodes.get(nodes.size()-1).innovationID<n.innovationID){
                nodes.add(n);
                return n;
            }else{
                for(int i=0;i<nodes.size();i++){
                    if(nodes.get(i).innovationID>n.innovationID){
                        nodes.add(i,n);
                        return n;
                    }
                }
                nodes.add(n);
                return n;
            }
        }else{
            return nodes.get(nodes.indexOf(n));
        }
    }

    @Override
    public String toString(){
        return "================================================================\nSynapses("+synapses.size()+"): "+synapses+"\nNodes("+nodes.size()+"): "+nodes+"\n================================================================";
    }

    public void mutate(){
        if(Math.random() < Constants.mutationSynapseProbability) mutateSynapse();
        if(Math.random() < Constants.mutationNodeProbability) mutateNode();
        if(Math.random() < Constants.mutationWeightShiftProbability) shiftWeights(Constants.mutationWeightShiftStrength);
        if(Math.random() < Constants.mutationWeightRandomProbability) randomWeights(Constants.mutationWeightRandomStrength);
        if(Math.random() < Constants.mutationBiasShiftProbability) shiftBias(Constants.mutationBiasShiftStrength);
    }

    public void shiftWeights(double mutationStrength){
        if(synapses.isEmpty())return;
        synapse s = synapses.get((int)(Math.random()*synapses.size()));
        s.weight = s.weight*(Math.random()*mutationStrength);
    }

    public void randomWeights(double mutationStrength){
        if(synapses.isEmpty())return;
        synapse s = synapses.get((int)(Math.random()*synapses.size()));
        s.weight = (Math.random()*2-1)*mutationStrength;
    }

    public void shiftBias(double mutationStrength){
        if(nodes.isEmpty())return;
        for(int i=0;i<100;i++){
            node n = nodes.get((int)(Math.random()*nodes.size()));
            if(n.isOutput())continue;
            n.bias += (Math.random()*2-1)*mutationStrength;
            return;
        }
    }

    public void mutateSynapse(){
        for(int i=0;i<100;i++){
            node from=nodes.get((int)(Math.random() * nodes.size()));
            node to=nodes.get((int)(Math.random() * nodes.size()));
            if (from.equals(to) || from.isOutput() || to.isInput() || isLooping(from, to) || synapses.contains(new synapse(from,to))){
                //failed to add
                continue;
            }
            int innovationID = Constants.globalInnovations.get(from,to);
            synapse newSynapse = new synapse(from, to,(Math.random()*2-1)* Constants.mutationWeightRandomStrength,true,innovationID);
            addSynapse(newSynapse);
            return;
        }
    }

    public void mutateNode(){
        if(synapses.isEmpty())return;
        for(int i=0;i<100;i++){
            synapse s = synapses.get((int)(Math.random()*synapses.size()));
            if(!s.enabled)continue;
            s.enabled=false;
            node newNode = Constants.globalInnovations.getSplitNode(s);
            addNode(newNode);
            addSynapse(new synapse(newNode,s.to,s.weight,true,Constants.globalInnovations.get(newNode,s.to)));
            addSynapse(new synapse(s.from, newNode,Constants.globalInnovations.get(s.from,newNode)));
            return;
        }
    }

    private boolean isLooping(node startingPoint, node to){
        ArrayList<node> stack = new ArrayList<node>();
        stack.add(to);
        while (!stack.isEmpty()){
            node toScan = stack.remove(0);
//            System.out.print(toScan+" ");
            if(toScan.equals(startingPoint)){
                return true;
            }
            if(!toScan.isOutput()){
                for (synapse synapse : synapses){
                    if (synapse.from.equals(toScan)) {
//                        System.out.print("("+synapse.from+"-"+synapse.to+"),");
                        stack.add(0,synapse.to);
                    }
                }
            }
        }
        return false;
    }
}
