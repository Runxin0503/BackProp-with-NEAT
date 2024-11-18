package Evolution;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class Evolution implements Iterator<Agent> {
    private final ArrayList<Species> species = new ArrayList<>();
    private final Agent[] agents;
    private int index = 0;

    public Evolution(int numSimulated){

        Agent first = new Agent();
        agents = new Agent[numSimulated];
        agents[0] = first;
        species.add(new Species(first));
        for(int i=1;i<numSimulated;i++){
            Agent temp = new Agent();
            species.getFirst().add(temp);
            agents[i] = temp;
        }
    }

    public void nextGen(){
        //Species Separation
        for(Species s : species){
            s.reset();
        }

        //assign all agents to a species
        for(Agent agent : agents){
            boolean found = false;
            for(Species s : species){
                if(s.add(agent)){
                    found=true;
                    break;
                }
            }
            if(!found) species.add(new Species(agent));
        }

        //update stagnant count, then cull
        for(Species s : species){
            s.updateStag();
            s.cull();
        }

        //remove empty species
        for(int i = species.size()-1; i>=0; i--){
            Species s = species.get(i);
            if(s.isEmpty()) species.remove(i);

        }

        //calculates population score
        for(Species s : species) s.calculateScore();

        //repopulate Genomes & reproduce
        for(Agent agent : agents){
            if(!agent.hasGenome()){
                WeightedRandom.getRandom(species).populateGenome(agent);
            }
        }

        //mutate
        for(Agent agent : agents){
            agent.mutate();
        }

        //reset
        for(Agent agent : agents){
            agent.reset();
        }
    }

    @Override
    public boolean hasNext() {
        return index < agents.length;
    }

    @Override
    public Agent next() {
        if(hasNext()) return agents[index++];
        throw new NoSuchElementException();
    }
}
