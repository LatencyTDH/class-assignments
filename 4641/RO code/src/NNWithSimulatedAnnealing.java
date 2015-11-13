import opt.OptimizationAlgorithm;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.DataSetDescription;
import shared.ErrorMeasure;
import shared.FixedIterationTrainer;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.filt.LabelSplitFilter;
import shared.tester.AccuracyTestMetric;
import shared.tester.ConfusionMatrixTestMetric;
import shared.tester.NeuralNetworkTester;
import shared.tester.TestMetric;
import shared.tester.Tester;
import shared.reader.ArffDataSetReader;
import shared.reader.CSVDataSetReader;
import shared.reader.DataSetLabelBinarySeperator;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;



import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;


public class NNWithSimulatedAnnealing {
    private Instance[] instances = initializeInstances();
    private final String DATASET = "letter_entire_new.txt";
    private double coolingRate;

    private int inputLayer , hiddenLayer , outputLayer , trainingIterations ;
    private BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private ErrorMeasure measure = new SumOfSquaresError();

    private DataSet set = new DataSet(instances);

    private BackPropagationNetwork networks[] = new BackPropagationNetwork[1];
    private NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];

    private OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
    private String[] oaNames = {"SA"};
    private String results = "";

    private DecimalFormat df = new DecimalFormat("0.000");

    public NNWithSimulatedAnnealing(int inputLayer, int hiddenLayer, int outputLayer, int trainingIterations,
                                    double cooling) {
        this.inputLayer = inputLayer;
        this.hiddenLayer = hiddenLayer;
        this.outputLayer = outputLayer;
        this.trainingIterations = trainingIterations;
        this.coolingRate = cooling;
    }
    public void run(String[] args) throws FileNotFoundException{
        /* Logs the data from the run */
        String filename = String.format("sa-cooling-%f-%d.txt", coolingRate, System.currentTimeMillis());
        File file = new File("result_data/" + filename);
        FileOutputStream fo = new FileOutputStream(file);
        PrintStream ps = new PrintStream(fo);
        System.setOut(ps);


        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new SimulatedAnnealing(1E11, coolingRate, nnop[0]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
        System.out.println(String.format("inputLayer=%d, hiddenLayer=%d, outputLayer=%d, trainingIterations=%d" +
                ", coolingRate=%f",
                inputLayer, hiddenLayer, outputLayer, trainingIterations, coolingRate));
        results = "";
    }

    private void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "(Cooling Rate: "
                            + coolingRate +")" + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
    }

    private Instance[] initializeInstances() {
        final int INSTANCES = 20000;
        final int ATTRIBUTES = 16;
        double[][][] attributes = new double[INSTANCES][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("data/"+DATASET)));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[ATTRIBUTES]; // 16 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < ATTRIBUTES; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications are 1 if the letter is 'O' and 0 otherwise
            instances[i].setLabel(new Instance(attributes[i][1][0] == 0 ? 0 : 1));
        }

        return instances;
    }
}
