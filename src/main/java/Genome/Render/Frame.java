package Genome.Render;

import Evolution.*;
import Genome.NN;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.ResourceBundle;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.*;
import javafx.stage.Stage;

public class Frame extends Application {

    public static void main(String[] args) {
        launch(args);
    }


    @Override
    public void start(final Stage stage) {
        try {
            final URL r = getClass().getResource("/GUI.fxml");
            if (r == null) {
                System.err.println("No FXML resource found.");
                try {
                    stop();
                } catch (final Exception e) {
                }
            }
            final Parent node = FXMLLoader.load(r);
            final Scene scene = new Scene(node);
            stage.setTitle("WALLY'S WORLD");
            stage.setScene(scene);
            stage.sizeToScene();
            stage.show();
//            scene.addEventFilter(MouseEvent.MOUSE_CLICKED, event -> {
//                System.out.println(event.getX() + "," + event.getY());
//            });
//            scene.addEventFilter(MouseEvent.MOUSE_CLICKED, event -> {
//                System.out.println(scene.getWidth() + "," + scene.getHeight());
//            });
            stage.setMinWidth(Constants.MIN_STAGE_WIDTH);
            stage.setMinHeight(Constants.MIN_STAGE_HEIGHT);
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

    }


    public Frame() {
        JButton buttonB = new JButton("random weight");
        buttonB.addActionListener(e -> {
            NN.randomWeights(Constants.mutationWeightRandomStrength);
            repaint();
        });
        menu.add(buttonB);

        JButton buttonZ = new JButton("weight shift");
        buttonZ.addActionListener(e -> {
            NN.shiftWeights(Constants.mutationWeightShiftStrength);
            repaint();
        });
        menu.add(buttonZ);

        JButton buttonC = new JButton("Edge mutate");
        buttonC.addActionListener(e -> {
            NN.mutateSynapse();
            repaint();
        });
        menu.add(buttonC);

        JButton buttonD = new JButton("Node mutate");
        buttonD.addActionListener(e -> {
            NN.mutateNode();
            repaint();
        });
        menu.add(buttonD);

        JButton buttonE = new JButton("Mutate");
        buttonE.addActionListener(e -> {
            NN.mutate();
            repaint();
        });
        menu.add(buttonE);

        JButton buttonF = new JButton("Calculate");
        buttonF.addActionListener(e -> {
            double[] input = new double[Constants.inputNum];
            for(int i=0;i<input.length;i++)input[i]=Math.random() * 20 - 10;
            System.out.print("Input: ");
            for(Double d : input)System.out.print(d+" ");
            System.out.println();
            System.out.println("Output: "+Arrays.toString(agent.calculateWeightedOutput(input)));
            repaint();
        });
        menu.add(buttonF);
    }

}
