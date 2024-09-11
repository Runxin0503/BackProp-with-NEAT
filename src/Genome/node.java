package Genome;

public class node {
    public double bias;
    public String type;
    public double latestInputSum;
    public double latestOutput;
    public int innovationID;
    public double x,y;

    public node(String type, int innovationID,double x,double y){
        this.bias=0;
        this.type = type;
        this.innovationID = innovationID;
        this.x=x;
        this.y=y;
    }

    private node(double bias,String type,int innovationID,double x,double y){
        this.bias = bias;
        this.type = type;
        this.innovationID = innovationID;
        this.x=x;
        this.y=y;
    }

    @Override
    public boolean equals(Object other){
        if(!(other instanceof node))return false;
        return this.innovationID == ((node)other).innovationID;
    }

    public boolean isOutput(){
        return this.type.equals("output");
    }

    public boolean isInput(){
        return this.type.equals("input");
    }

    public boolean isHidden(){
        return this.type.equals("hidden");
    }


    @Override
    public node clone(){
        return new node(bias, type,innovationID,x,y);
    }

    @Override
    public String toString(){
        return "(" + (isOutput() ? "^" : (isInput() ? "v" : bias)) + ")";
    }
}
