public class RandomizedOptimization {
    public static void main(String args[]) throws Exception {
        // NNWithRandomHillClimbing rhc = new NNWithRandomHillClimbing();

        final int INPUT = 16;
        final int OUTPUT = 1; // # of nodes in the output layer
        final int HIDDEN = 16; // # of nodes in each hidden layer
        final double COOLING = .95; // rate of cooling for SA

        final int POPULATION = 200;
        final int NUM_TO_MATE = 100;
        final int NUM_TO_MUTATE = 200; // number of individuals out of a
                                       // population of 200 to mutate
//        int[] iterations = { 10, 25, 50, 100, 250, 500, 1000, 1500 };
        int[] iterations = { 1500 };
        int[] detailedTestingIterations = { 500 }; // for specific variable
                                                   // testing of RO alg. Eg.
                                                   // cooling rate on SA
                                                   // accuracy

        // //Run all RO algorithms for the neural networks
         for (int it: iterations) {
         LetterTest lt = new LetterTest(INPUT, HIDDEN, OUTPUT, it);
         lt.run(args);
         }

        for (int it2 : detailedTestingIterations) {
            // //lower the cooling rate for simulated annealing by .20, for 500
            // fixed iterations each
            // for (double currentCoolingRate = COOLING; currentCoolingRate > 0;
            // currentCoolingRate -= .20) {
            // NNWithSimulatedAnnealing sa = new NNWithSimulatedAnnealing(INPUT,
            // HIDDEN, OUTPUT, it2, currentCoolingRate);
            // sa.run(args); }
            
            /*
             * First all offspring experience mutations, then fewer individuals
             * experience them, print error rates according to this scheme.
             * Fixed variables population = 200, toMate = 100.
             */
           for (int toMutate = NUM_TO_MUTATE; toMutate > 0; toMutate -= 25) {
               NNWithGeneticAlgorithm ga = new NNWithGeneticAlgorithm(INPUT,HIDDEN,OUTPUT,it2,
                       POPULATION,NUM_TO_MATE,toMutate);
               ga.run(args);
           }
            
            /*
             * Start with everyone mating (all 200 of the pop.), decreasing by
             * 25 until only a small number of individuals can mate. Print the
             * error results of neural net classification using this GA scheme.
             * Fixed variables: population = 200, toMutate = 10
             */
           for (int toMate = 200; toMate > 0; toMate -= 25) {
               NNWithGeneticAlgorithm ga = new NNWithGeneticAlgorithm(INPUT,HIDDEN,OUTPUT,it2,
                       POPULATION,toMate,10);
               ga.setTitle("toMate");
               ga.run(args);
           }
        }
    }
}