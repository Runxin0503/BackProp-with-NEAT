package Evolution;

import java.util.ArrayList;

public class Species {
    private NN representative;
    public int age,stag;
    public double speciesScore;
    public ArrayList<NN> NeuralNets = new ArrayList<NN>();

    public Species(NN representative){
        this.representative=representative;
        this.NeuralNets.add(representative);
        this.age=0;
        this.stag=0;
        this.speciesScore = representative.score;
    }

    public boolean add(NN newNeuralNet){
        if(representative.genome.compare(newNeuralNet.genome)< Evolution.compatibilityThreshold){
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
        for(NN nn : NeuralNets){
            speciesScore+=nn.score;
        }
        speciesScore /= NeuralNets.size();
        if(stag>=Evolution.maxStagDropoff) speciesScore *= 0.7;
    }

    public void updateStag(){
        double count=0;
        for(NN nn : NeuralNets){
            count+=nn.score;
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
            NeuralNets.remove(i).genome=null;
        }
    }

    public void extinct(){
        for(NN nn : NeuralNets){
            nn.genome = null;
        }
    }

    public void populateGenome(NN emptyNN){
        NN first = pickReproducer(NeuralNets);
        NN second = pickReproducer(NeuralNets);
        emptyNN.genome = Genome.Genome.crossover(first.genome,second.genome,first.score,second.score);
        emptyNN.score=0;
        NeuralNets.add(emptyNN);
    }

    private NN pickReproducer(ArrayList<NN> currentPopulation){
        if(speciesScore==0)return currentPopulation.get((int)(Math.random()*currentPopulation.size()));
        double random = Math.random() * speciesScore;
        for(NN nn : currentPopulation){
            random -= nn.score;
            if(random<=0){
                return nn;
            }
        }
        return currentPopulation.get((int)(Math.random()*currentPopulation.size()));
    }

}
