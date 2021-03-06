import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
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
import shared.reader.CSVDataSetReader;
import shared.reader.ArffDataSetReader;
import shared.reader.DataSetLabelBinarySeperator;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;

public class NNWithRandomHillClimbing {
    public void run(int iterations) throws Exception {
        // 1) Construct data instances for training.  These will also be run
        //    through the network at the bottom to verify the output
		CSVDataSetReader reader = new CSVDataSetReader("data/letter_training_new.data");
        DataSet set = reader.read();
        LabelSplitFilter flt = new LabelSplitFilter();
        flt.filter(set);
        DataSetLabelBinarySeperator.seperateLabels(set);
        DataSetDescription desc = set.getDescription();
        DataSetDescription labelDesc = desc.getLabelDescription();
        
        // 2) Instantiate a network using the FeedForwardNeuralNetworkFactory.  This network
        //    will be our classifier.
        FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
        // 2a) These numbers correspond to the number of nodes in each layer.
        //     This network has 4 input nodes, 3 hidden nodes in 1 layer, and 1 output node in the output layer.
        FeedForwardNetwork network = factory.createClassificationNetwork(new int[] { desc.getAttributeCount(),
                factory.getOptimalHiddenLayerNodes(desc, labelDesc),
                labelDesc.getDiscreteRange() });

        // 3) Instantiate a measure, which is used to evaluate each possible set of weights.
        ErrorMeasure measure = new SumOfSquaresError();
        
        // 4) Instantiate a DataSet, which adapts a set of instances to the optimization problem.
        //DataSet set = new DataSet(patterns);
        
        // 5) Instantiate an optimization problem, which is used to specify the dataset, evaluation
        //    function, mutator and crossover function (for Genetic Algorithms), and any other
        //    parameters used in optimization.
        NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(
            set, network, measure);
        
        // 6) Instantiate a specific OptimizationAlgorithm, which defines how we pick our next potential
        //    hypothesis.
        OptimizationAlgorithm o = new RandomizedHillClimbing(nno);
        
        // 7) Instantiate a trainer.  The FixtIterationTrainer takes another trainer (in this case,
        //    an OptimizationAlgorithm) and executes it a specified number of times.
        FixedIterationTrainer fit = new FixedIterationTrainer(o, iterations);
        
        // 8) Run the trainer.  This may take a little while to run, depending on the OptimizationAlgorithm,
        //    size of the data, and number of iterations.
        fit.train();
        
        // 9) Once training is done, get the optimal solution from the OptimizationAlgorithm.  These are the
        //    optimal weights found for this network.
        Instance opt = o.getOptimal();
        network.setWeights(opt.getData());
        
        //10) Run the training data through the network with the weights discovered through optimization, and
        //    print out the expected label and result of the classifier for each instance.
        int[] labels = {0,1};
        TestMetric acc = new AccuracyTestMetric();
        TestMetric cm  = new ConfusionMatrixTestMetric(labels);
        Tester t = new NeuralNetworkTester(network, acc, cm);
        t.test(set.getInstances());
        
        acc.printResults();
    }
}
