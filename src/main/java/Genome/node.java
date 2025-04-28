package Genome;

import Evolution.Constants;

import java.util.ArrayList;
import java.util.List;

/** The Genome representation of a node */
final class node extends Gene {

    /** The Activation Function of this neuron */
    private Activation activationFunction;

    /** The bias value of this neuron */
    double bias;

    /** A list of local indices of incoming/outgoing edges to this node */
    private final List<edge> incomingConnections = new ArrayList<>(),
            outgoingConnections = new ArrayList<>();

    /** X, Y coordinates for visualization */
    double x, y;

    node(int innovationID, Activation activationFunction, double bias, Optimizer optimizer) {
        super(optimizer);
        this.innovationID = innovationID;
        this.activationFunction = activationFunction;
        this.bias = bias;
        this.x = 0;
        this.y = 0;
    }

    /**
     * Returns the direct reference to all edges pointing into this node.
     * <br>Used for calculating derivative terms in back-propagation
     */
    List<edge> getIncomingEdges() {
        return incomingConnections;
    }

    /**
     * Returns the direct reference to all edges pointing out from this node.
     * <br>Used for feed-forward calculation
     */
    List<edge> getOutgoingEdges() {
        return outgoingConnections;
    }

    /**
     * Attempts to add {@code e} to the array of outgoing edges.
     * Doesn't do anything if the array already contains the edge
     */
    void addOutgoingEdge(edge e) {
        if (outgoingConnections.contains(e)) return;
        outgoingConnections.add(e);
    }

    /**
     * Attempts to add {@code e} to the array of incoming edges.
     * Doesn't do anything if the array already contains the edge
     */
    void addIncomingEdge(edge e) {
        if (incomingConnections.contains(e)) return;
        incomingConnections.add(e);
    }

    /** Returns the Activation Function of this node. */
    Activation getActivationFunction() {
        return activationFunction;
    }

    /** Shifts the Bias of this node by a random amount. */
    void shiftBias(Constants Constants) {
        this.bias += Constants.mutationBiasShiftStrength * (Math.random() * 2 - 1);
    }

    /** Randomly changes this node's Activation Function. */
    void changeAF() {
        this.activationFunction = Activation.values()[(int) (Math.random() * Activation.values().length)];
    }

    /** Returns an identical node object except {@linkplain #incomingConnections} and
     * {@linkplain #outgoingConnections} are empty. */
    node clone(Optimizer optimizer) {
        return new node(innovationID, activationFunction, bias, optimizer);
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof node && ((node) obj).innovationID == innovationID;
    }

    /** Returns true if {@code other} is identical in every instance field.
     * <br>Use this method instead of {@link #equals(Object)} to compare instance variables beyond InnovationID. */
    boolean identical(node other) {
        return other.innovationID == innovationID && other.activationFunction == activationFunction &&
                other.activated == activated && other.x == x && other.y == y &&
                other.bias == bias && other.incomingConnections.equals(incomingConnections) &&
                other.outgoingConnections.equals(outgoingConnections);
    }

    @Override
    protected void addValue(double deltaValue) {
        this.bias += deltaValue;
    }
}
