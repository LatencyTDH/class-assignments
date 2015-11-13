ASSIGNMENT 2: Randomized Optimization
Sean Dai

=========================================================================
SEC 0. DATASET - LETTER
=========================================================================
The letter dataset, letter_entire_new.txt, from Assignment 1 is used.
ABAGAIL isn't very friendly when it comes to multiclass classification;
it is hard to classify the 26 English capital letters (A-Z). So I 
modified the original letter dataset to return a binary label for 
each instance:

1 - if the letter is the capital letter "O"
0 - otherwise

All other attribute names and their values remain the same as from 
Assignment 1. The new dataset was retested in WEKA (with CV of course),
and appropriate training/test results were logged.

=========================================================================
SEC 1. DEFAULT NEURAL NET SETTINGS
=========================================================================
The default neural net has the following parameters:

16 input nodes in the input layer
1 hidden layer
	16 nodes in each hidden layer
1 output layer with 1 node
	Class: {1 - if the letter is an "O"
		   {0 - otherwise
			
ITERATIONS = 500

=========================================================================
SEC 2. CREATING 3 NNET FROM RHC, SA, GA ALGORITHMS
=========================================================================
We will test neural net classification accuracy as a function of training
iterations. The neural nets are trained for a discrete number of training 
iterations for the 3 randomized optimization algorithms:

int[] iterations = { 10, 25, 50, 100, 250, 500, 1000, 1500 };

Statistics (accuracy, training time, SSE, etc.) are output after each 
iteration.

Cross validation is also performed on the dataset, but what we really care
about are the squared errors that are output after each iteration of the
randomized optimization algorithms.

The default parameters passed into the RO algorithm constructors can be 
found in LetterTest.java.

1) Randomized Hill Climbing:
	RandomizedHillClimbing(HillClimbingProblem hcp), where hcp contains
	the entire set of instances (20000 for LETTER), the network for which
	we wish to find optimal weights, and the error measure (SumOfSquaresError).

2) Simulated Annealing:
	SimulatedAnnealing(double t, double cooling, HillClimbingProblem hcp)
	where t is the starting temperature (for us, t=1E11), cooling rate
	(.95), and hcp is the problem described in 1).

3) Genetic Algorithm:
	StandardGeneticAlgorithm(int populationSize, int toMate, int toMutate,
							GeneticAlgorithmProblem gap)
	where populationSize = 200, toMate is the number of individuals that 
	mate each iteration (set to =100), the number of offspring toMutate
	each iteration (10) --few individuals will mutate, and the genetic
	algorithm problem.

============================================================================
SEC 3. DETAILED TESTING OF THE RO ALGORITHMS
============================================================================
Now that we have created neural nets with weights based on fixed parameters,
let's see what happens when we alter specific parameter values for Simulated
Annealing and Genetic Algorithm. I elected not to make any slight changes
to RHC parameters because it doesn't really have any extra parameters
besides HillClimbingProblem in ABAGAIL.

To allow for fair comparison between each parameter variation, we fix the
number of iterations = 500, which is a reasonable number of iterations.

============================================================================
SEC 3a. Simulated Annealing (Parameter Tweaking):
============================================================================
ABAGAIL's SimulatedAnnealing(double t, double cooling, HillClimbingProblem hcp)
class takes 3 parameters: 
1) double t : initial starting temperature 
2) double cooling : cooling rate over time. 
3) HillClimbingProblem hcp

In this assignment, I vary the one parameter : cooling.
to create optimal weights for the neural net. I will analyze clock time, 
classification accuracy, mean squared error as a result of these parameter 
modifications.

The number of input nodes, hidden layers/hidden nodes, output nodes, and 
# of iterations remain the same as the default neural net described in 
Sec. 1.

vary_cooling: Fixed parameters are t = 1E11.

============================================================================
SEC 3b. Genetic Algorithm (Parameter Tweaking):
============================================================================

ABAGAIL's StandardGeneticAlgorithm class takes 4 parameters: 
1) int populationSize
2) int toMate
3) int toMutate
4) GeneticAlgorithmGap gap (this is the problem to solve)

In this assignment, I vary the parameters toMate & toMutate (one at a time)
to create optimal weights for the neural net. I will analyze clock time, 
classification accuracy, mean squared error as a result of these 
parameter modifications.

The number of input nodes, hidden layers/hidden nodes, output nodes, and 
# of iterations remains the same as the default neural net described in 
Sec. 1.

vary_toMate: Fixed parameters are populationSize = 200 and toMutate = 10
vary_toMutate: Fixed parameters are populationSize = 200 and toMate = 100

===================================================================================
SEC 4. RUNNING - Some Code Taken from https://github.com/tomelm/supervised-learning
===================================================================================
This makes use of the ABAGAIL library. In Eclipse, setup a new project for ABAGAIL 
first. Then setup a new project for the code contained in “RO code/“. Link this 
code  to the ABAGAIL project in eclipse and then run

``RandomizedOptimization.java``
``ContinuousPeaksProblem.java``
``FourPeaksProblem.java``
``KnapsackProblem.java``

=================================================================================
Part 1: The Problems Given to You
=================================================================================

Run ``RandomizedOptimization.java`` inside of Eclipse. This will run random hill 
climbing, simulated annealing, and genetic algorithm on the modified letter 
dataset described in SEC 0.

===============================================================================
Part 2: The Problems You Give Us
===============================================================================
Running ``ContinuousPeaksProblem.java``, ``FourPeaksProblem.java``, and 
``KnapsackProblem.java`` inside of Eclipse will run the three optimization 
problems used for this assignment. 

The code was taken from ABAGAIL'S examples and slightly modified to run 100 times 
and print out relevant statistics (fitness values, average time over 100 iterations).
