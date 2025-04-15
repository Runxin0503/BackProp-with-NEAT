package Genome;

import Evolution.Constants;

import java.util.ArrayList;
import java.util.List;

/** The Genome representation of a node */
//todo made public for testing
public class node extends Gene {
    /*
     * must contain:
     * - innovation number:
     *      - should be the same as synapse split if hidden node ( >= 0)
     *      - if input/output node, should range from -(inputNum+outputNum) to -1
     * - ActivationFunction enum:
     *      - should be null if it's input or output
     *      - can be any hiddenAF enums if its hidden, outputAF are reserved to used in Constants file
     *      - if input/output node, CANNOT have an ActivationFunction enum other than Activation.none
     * - bias
     * - boolean activated: true if this node's output is > 0, false otherwise
     * - int array of outgoing connections (index of edges)
     * - int array of incoming connections (index of edges)
     */

    /** The Activation Function of this neuron */
    private Activation activationFunction;

    /** The bias value of this neuron */
    //todo made public for testing purposes
    public double bias;

    /**
     * True if neuron's output is > 0, false otherwise.
     * <br>Used in visualizing individual neuron firing
     */
    private boolean activated;

    /** A list of local indices of incoming/outgoing edges to this node */
    private final List<edge> incomingConnections = new ArrayList<>(),
            outgoingConnections = new ArrayList<>();

    /** TODO */
    double x, y;

    public node(int innovationID, Activation activationFunction, double bias, Optimizer optimizer) {
        super(optimizer);
        this.innovationID = innovationID;
        this.activationFunction = activationFunction;
        this.bias = bias;
        this.activated = false;
        this.x = 0;
        this.y = 0;
    }

    /** Returns true if the last output of this neuron is > 0, false otherwise */
    public boolean isActivated() {
        return activated;
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

    /** Returns the bias of this node */
    double getBias() {
        return bias;
    }

    /** TODO */
    Activation getActivationFunction() {
        return activationFunction;
    }

    /** Shifts the Bias of this node by a random amount. */
    void shiftBias(Constants Constants) {
        this.bias += Constants.mutationBiasShiftStrength * (Math.random() * 2 - 1);
    }

    /** TODO */
    void changeAF() {
        this.activationFunction = Activation.values()[(int) (Math.random() * Activation.values().length)];
    }

    public node clone(Optimizer optimizer) {
        return new node(innovationID, activationFunction, bias, optimizer);
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof node && ((node) obj).innovationID == innovationID;
    }

    /** TODO */
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
