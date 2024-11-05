package Evolution;

import Genome.NN;

import java.util.ArrayList;

public class Species {
    private Agent representative;
    public int age,stag;
    public double speciesScore;
    public ArrayList<Agent> NeuralNets = new ArrayList<Agent>();

    public Species(Agent representative){
        this.representative=representative;
        this.NeuralNets.add(representative);
        this.age=0;
        this.stag=0;
        this.speciesScore = representative.score;
    }

    public boolean add(Agent newNeuralNet){
        if(representative.NN.compare(newNeuralNet.NN)< Constants.compatibilityThreshold){
            if(NeuralNets.isEmpty()||NeuralNets.get(NeuralNets.size()-1).score>newNeuralNet.score){
                NeuralNets.add(newNeuralNet);
            }else{
                for(int i=0;i<NeuralNets.size();i++){
                    if(NeuralNets.get(i).score<newNeuralNet.score){
                        NeuralNets.add(i,newNeuralNet);
                        break;
                    }
                }
            }
            return true;
        }
        return false;
    }

    public void calculateScore(){
        speciesScore=0;
        for(Agent agent : NeuralNets){
            speciesScore+= agent.score;
        }
        speciesScore /= NeuralNets.size();
        if(stag>=Constants.maxStagDropoff) speciesScore *= 0.7;
    }

    public void updateStag(){
        double count=0;
        for(Agent agent : NeuralNets){
            count+= agent.score;
        }
        count /= NeuralNets.size();
        if(speciesScore>count)stag++;
        else stag = Math.max(0,(int)Math.round(stag * (1-(count-speciesScore)/speciesScore)));
    }

    public void reset(){
        representative = NeuralNets.get((int)(Math.random() * NeuralNets.size()));
        NeuralNets.clear();
    }

    public void cull(double percentage){
        int numSurvived = (int)(Math.round(NeuralNets.size()*(1-percentage)));
        for(int i=NeuralNets.size()-1;i>numSurvived;i--){
            NeuralNets.remove(i).NN =null;
        }
    }

    public void extinct(){
        for(Agent agent : NeuralNets){
            agent.NN = null;
        }
    }

    public void populateGenome(Agent emptyAgent){
        Agent first = pickReproducer(NeuralNets);
        Agent second = pickReproducer(NeuralNets);
        emptyAgent.NN = NN.crossover(first.NN,second.NN,first.score,second.score);
        emptyAgent.score=0;
        NeuralNets.add(emptyAgent);
    }

    private Agent pickReproducer(ArrayList<Agent> currentPopulation){
        if(speciesScore==0)return currentPopulation.get((int)(Math.random()*currentPopulation.size()));
        double random = Math.random() * speciesScore;
        for(Agent agent : currentPopulation){
            random -= agent.score;
            if(random<=0){
                return agent;
            }
        }
        return currentPopulation.get((int)(Math.random()*currentPopulation.size()));
    }

}
