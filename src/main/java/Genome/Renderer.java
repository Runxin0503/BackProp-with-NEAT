package Genome;

import Evolution.*;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.ResourceBundle;

import Genome.enums.Activation;
import Genome.enums.Cost;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.geometry.Point2D;
import javafx.geometry.Rectangle2D;
import javafx.scene.*;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.layout.AnchorPane;
import javafx.scene.paint.Color;
import javafx.scene.transform.Affine;
import javafx.scene.transform.NonInvertibleTransformException;
import javafx.stage.Stage;
import Evolution.Evolution.EvolutionBuilder;
import javafx.util.Duration;

public class Renderer extends Application implements Initializable {

    @FXML
    private Button rWeights;

    @FXML
    private Button sWeights;

    @FXML
    private Button sBias;

    @FXML
    private Button rEdge;

    @FXML
    private Button rNode;

    @FXML
    private Button mutate;

    @FXML
    private Button calculate;

    @FXML
    private Canvas canvas;

    @FXML
    private AnchorPane canvasScroller;

    private Affine canvasTransform;

    private Evolution agentFactory;
    private Agent agent;
    private NN agentGenome;

    private final double radius = 10;
    private final double MAX_FPS = 120;
    private final double MIN_ZOOM = 0;
    private final double MAX_ZOOM = 10;
    private boolean redrawCanvas = true;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(final Stage stage) {
        try {
            final URL r = getClass().getResource("/Visualizer.fxml");
            if (r == null) {
                System.err.println("No FXML resource found.");
                try {
                    stop();
                } catch (final Exception e) {
                }
            }
            final Parent node = FXMLLoader.load(r);
            final Scene scene = new Scene(node);
            stage.setScene(scene);
            stage.sizeToScene();
            stage.show();
        } catch (final IOException ioe) {
            System.err.println("Can't load FXML file.");
            ioe.printStackTrace();
            System.out.println(ioe);
            try {
                stop();
            } catch (final Exception e) {
            }
        }
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        agentFactory = new EvolutionBuilder()
                .setDefaultHiddenAF(Activation.ReLU).setCostFunction(Cost.crossEntropy).setNumSimulated(1)
                .setInputNum(4).setOutputNum(3).setOutputAF(Activation.arrays.softmax).build();
        agent = agentFactory.agents[0];
        agentGenome = agent.getGenomeClone();

        rWeights.setOnAction(e -> Mutation.randomWeights(agentGenome));

        sWeights.setOnAction(e -> Mutation.shiftWeights(agentGenome));

        sBias.setOnAction(e -> Mutation.shiftBias(agentGenome));

        rEdge.setOnAction(e -> Mutation.mutateSynapse(agentGenome));

        rNode.setOnAction(e -> Mutation.mutateNode(agentGenome));

        mutate.setOnAction(e -> agentGenome.mutate());

        calculate.setOnAction(e -> {
            double[] input = new double[agentFactory.Constants.getInputNum()];
            for(int i=0;i<input.length;i++)input[i]=Math.random() * 20 - 10;
            System.out.print("Input: ");
            for(Double d : input)System.out.print(d+" ");
            System.out.println();
            System.out.println("Output: "+Arrays.toString(agentGenome.calculateWeightedOutput(input)));
        });

        canvasScroller.setOnScroll(ae -> {
            if (ae.getDeltaY() != 0) {  // Only react to vertical scroll
                double zoomFactor = ae.getDeltaY() > 0 ? 1.1 : 0.9;  // Zoom in or out
                if ((canvasTransform.getMxx() >= MAX_ZOOM / 1.1 || canvasTransform.getMyy() >= MAX_ZOOM / 1.1) && zoomFactor > 1)
                    return;
                if ((canvasTransform.getMxx() <= MIN_ZOOM / 0.9 || canvasTransform.getMyy() <= MIN_ZOOM / 0.9) && zoomFactor < 1)
                    return;

                Point2D mouseCoords;
                try{
                    mouseCoords = canvasTransform.inverseTransform(canvas.sceneToLocal(ae.getSceneX(), ae.getSceneY()));
                } catch (NonInvertibleTransformException e) {
                    throw new RuntimeException(e);
                }

                double oldScaleX = canvasTransform.getMxx(), oldScaleY = canvasTransform.getMyy();

                canvasTransform.setMxx(Math.clamp(oldScaleX * zoomFactor, MIN_ZOOM, MAX_ZOOM));
                canvasTransform.setMyy(Math.clamp(oldScaleY * zoomFactor, MIN_ZOOM, MAX_ZOOM));

                // Translate to maintain zoom focus on the mouse position
                canvasTransform.setTx(canvasTransform.getTx() - mouseCoords.getX() * (canvasTransform.getMxx() - oldScaleX));
                canvasTransform.setTy(canvasTransform.getTy() - mouseCoords.getY() * (canvasTransform.getMyy() - oldScaleY));

                redrawCanvas = true;
            }
        });

        final double[] dragAnchor = new double[2]; // To store initial mouse click position
        canvasScroller.setOnMousePressed(ae -> {
            // Store initial mouse position for panning
            dragAnchor[0] = ae.getSceneX() - canvasTransform.getTx();
            dragAnchor[1] = ae.getSceneY() - canvasTransform.getTy();
        });

        canvasScroller.setOnMouseDragged(ae -> {
            // Calculate new position for panning
            double offsetX = ae.getSceneX() - dragAnchor[0];
            double offsetY = ae.getSceneY() - dragAnchor[1];
            canvasTransform.setTx(offsetX);
            canvasTransform.setTy(offsetY);

            redrawCanvas = true;
        });

        canvasScroller.setOnMouseReleased(ae -> {
            Point2D mouseCoords;
            try {
                mouseCoords = canvasTransform.inverseTransform(canvas.sceneToLocal(ae.getSceneX(), ae.getSceneY()));
            } catch (NonInvertibleTransformException e) {
                throw new RuntimeException(e);
            }

            redrawCanvas = true;
        });

        Timeline updateCanvasPeriodically = new Timeline(new KeyFrame(
                Duration.seconds(1.0 / MAX_FPS),
                event -> {
                    if (redrawCanvas) {
                        drawCanvas();
                        redrawCanvas = false;
                    }
                }
        ));

        updateCanvasPeriodically.setCycleCount(Timeline.INDEFINITE);
        updateCanvasPeriodically.play();
    }

    private void drawCanvas() {
        double minX = Math.clamp(
                (radius - canvasTransform.getTx()) / canvasTransform.getMxx(),
                radius,
                canvas.getWidth()-radius
        );
        double minY = Math.clamp(
                (radius - canvasTransform.getTy()) / canvasTransform.getMyy(),
                radius,
                canvas.getHeight()-radius
        );
        double maxX = Math.clamp(
                (canvas.getWidth() - radius - canvasTransform.getTx()) / canvasTransform.getMxx(),
                radius,
                canvas.getWidth()-radius
        );
        double maxY = Math.clamp(
                (canvas.getHeight() - radius - canvasTransform.getTy()) / canvasTransform.getMyy(),
                radius,
                canvas.getHeight()-radius
        );

        Rectangle2D canvasCameraBoundingBox = new Rectangle2D(minX, minY, maxX - minX, maxY - minY);

        GraphicsContext gc = canvas.getGraphicsContext2D();

        gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());

        gc.save();
        gc.translate(canvasTransform.getTx(), canvasTransform.getTy());  // Translate first
        gc.scale(canvasTransform.getMxx(), canvasTransform.getMyy());  // Then apply scale

        gc.setStroke(Color.BLACK);
        for (int i=0; i<agentGenome.nodes.size();i++) {
            //todo loop through all nodes and edges NN and draw them

            if(i<agentFactory.Constants.getInputNum()){
                //input nodes
            }else if(i>agentGenome.nodes.size()-agentFactory.Constants.getOutputNum()-1){
                //output nodes
            }else{
                //hidden nodes
            }
        }

        gc.restore();
    }
}
