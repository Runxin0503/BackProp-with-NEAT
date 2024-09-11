package globalGenomes;

import Genome.*;
import java.util.ArrayList;
import java.util.HashMap;

public class globalInnovations {
    private final ArrayList<synapse> primitiveSynapses;
    private final HashMap<synapse, node> splitNode;
    private final globalNodes globalNodes;

    public globalInnovations(globalNodes globalNodes){
        this.globalNodes = globalNodes;
        this.splitNode = new HashMap<synapse, node>();
        this.primitiveSynapses=new ArrayList<synapse>();
    }

    public int get(node from, node to){
        synapse s = new synapse(from, to);
        if(primitiveSynapses.contains(s)) return primitiveSynapses.indexOf(s);
        s.innovationID=primitiveSynapses.size();
        primitiveSynapses.add(s);
        return primitiveSynapses.size()-1;
    }

    public node getSplitNode(synapse s){
//        System.out.println(splitNode);
        if(splitNode.containsKey(s))return splitNode.get(s);
        node n = globalNodes.add(s.from,s.to);
        splitNode.put(new synapse(s.from,s.to), n);
        return n;
    }
}
