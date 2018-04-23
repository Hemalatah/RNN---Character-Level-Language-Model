# RNN---Character-Level-Language-Model

Character level language model - Dinosaurus land

Welcome to Dinosaurus Island! 65 million years ago, dinosaurs existed, and in this assignment they are back. 
You are in charge of a special task. Leading biology researchers are creating new breeds of dinosaurs and bringing them to life on earth, and your job is to give names to these dinosaurs. If a dinosaur does not like its name, it might go beserk, so choose wisely!

Luckily you have learned some deep learning and you will use it to save the day. Your assistant has collected a list of all the dinosaur names they could find, and compiled them into this dataset. (In coursera dataset is given!) To create new dinosaur names, you will build a character level language model to generate new names. Your algorithm will learn the different name patterns, and randomly generate new names. Hopefully this algorithm will keep you and your team safe from the dinosaurs' wrath!

By completing this assignment you will learn:

How to store text data for processing using an RNN
How to synthesize data, by sampling predictions at each time step and passing it to the next RNN-cell unit
How to build a character-level text generation recurrent neural network
Why clipping the gradients is important
We will begin by loading in some functions that we have provided for you in rnn_utils. Specifically, you have access to functions such as rnn_forward and  rnn_backward which are equivalent to those you've implemented in the previous assignment.
(see rnn.py)

1 - Problem Statement
1.1 - Dataset and Preprocessing
Run the following cell to read the dataset of dinosaur names, create a list of unique characters (such as a-z), and compute the dataset and vocabulary size. (see rnn.py)

There are 19909 total characters and 27 unique characters in your data.

The characters are a-z (26 characters) plus the "\n" (or newline character), which in this assignment plays a role similar to the <EOS> (or "End of sentence") token we had discussed in lecture, only here it indicates the end of the dinosaur name rather than the end of a sentence. In the cell below, we create a python dictionary (i.e., a hash table) to map each character to an index from 0-26. We also create a second python dictionary that maps each index back to the corresponding character character. This will help you figure out what index corresponds to what character in the probability distribution output of the softmax layer. Below, char_to_ix and ix_to_char are the python dictionaries. (see rnn.py)

Expected Output:
{0: '\n', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}

1.2 - Overview of the model
Your model will have the following structure:

Initialize parameters
Run the optimization loop
Forward propagation to compute the loss function
Backward propagation to compute the gradients with respect to the loss function
Clip the gradients to avoid exploding gradients
Using the gradients, update your parameter with the gradient descent update rule.
Return the learned parameters
(refer images)

Figure 1: Recurrent Neural Network, similar to what you had built in the previous notebook "Building a RNN - Step by Step".

At each time-step, the RNN tries to predict what is the next character given the previous characters. The dataset X=(x⟨1⟩,x⟨2⟩,...,x⟨Tx⟩)  is a list of characters in the training set, while  Y=(y⟨1⟩,y⟨2⟩,...,y⟨Tx⟩)  is such that at every time-step  t, we have  y⟨t⟩=x⟨t+1⟩ .

2 - Building blocks of the model
In this part, you will build two important blocks of the overall model:

Gradient clipping: to avoid exploding gradients
Sampling: a technique used to generate characters
You will then apply these two functions to build the model.

2.1 - Clipping the gradients in the optimization loop
In this section you will implement the clip function that you will call inside of your optimization loop. Recall that your overall loop structure usually consists of a forward pass, a cost computation, a backward pass, and a parameter update. Before updating the parameters, you will perform gradient clipping when needed to make sure that your gradients are not "exploding," meaning taking on overly large values.

In the exercise below, you will implement a function clip that takes in a dictionary of gradients and returns a clipped version of gradients if needed. There are different ways to clip gradients; we will use a simple element-wise clipping procedure, in which every element of the gradient vector is clipped to lie between some range [-N, N]. More generally, you will provide a maxValue (say 10). In this example, if any component of the gradient vector is greater than 10, it would be set to 10; and if any component of the gradient vector is less than -10, it would be set to -10. If it is between -10 and 10, it is left alone.
(refer images)

Figure 2: Visualization of gradient descent with and without gradient clipping, in a case where the network is running into slight "exploding gradient" problems.

Exercise: Implement the function below to return the clipped gradients of your dictionary gradients. Your function takes in a maximum threshold and returns the clipped versions of your gradients. You can check out this hint for examples of how to clip in numpy. You will need to use the argument out = ....(see rnn.py)

Expected output:

gradients["dWaa"][1][2]	10.0
gradients["dWax"][3][1]	-10.0
gradients["dWya"][1][2]	0.29713815361
gradients["db"][4]	[ 10.]
gradients["dby"][1]	[ 8.45833407]

2.2 - Sampling
Now assume that your model is trained. You would like to generate new text (characters). The process of generation is explained in the picture below:
(refer images)

Figure 3: In this picture, we assume the model is already trained. We pass in  x⟨1⟩=0→  at the first time step, and have the network then sample one character at a time.

Exercise: Implement the sample function below to sample characters. You need to carry out 4 steps:

Step 1: Pass the network the first "dummy" input x⟨1⟩=0→  (the vector of zeros). This is the default input before we've generated any characters. We also set a⟨0⟩=0→ 
Step 2: Run one step of forward propagation to get  a⟨1⟩  and  ŷ ⟨1⟩. Here are the equations:

a⟨t+1⟩=tanh(Waxx⟨t⟩+Waaa⟨t⟩+b)(1)
 
z⟨t+1⟩=Wyaa⟨t+1⟩+by(2)
 
ŷ ⟨t+1⟩=softmax(z⟨t+1⟩)(3)
 
Note that  ŷ ⟨t+1⟩ is a (softmax) probability vector (its entries are between 0 and 1 and sum to 1). y^i⟨t+1⟩  represents the probability that the character indexed by "i" is the next character. We have provided a softmax() function that you can use.

Step 3: Carry out sampling: Pick the next character's index according to the probability distribution specified by  ŷ ⟨t+1⟩. This means that if  y^i⟨t+1⟩=0.16 , you will pick the index "i" with 16% probability. To implement it, you can use np.random.choice.
Here is an example of how to use np.random.choice():

np.random.seed(0)
p = np.array([0.1, 0.0, 0.7, 0.2])
index = np.random.choice([0, 1, 2, 3], p = p.ravel())
This means that you will pick the index according to the distribution:  P(index=0)=0.1,P(index=1)=0.0,P(index=2)=0.7,P(index=3)=0.2.
Step 4: The last step to implement in sample() is to overwrite the variable x, which currently stores  x⟨t⟩, with the value of  x⟨t+1⟩. You will represent  x⟨t+1⟩ by creating a one-hot vector corresponding to the character you've chosen as your prediction. You will then forward propagate  x⟨t+1⟩  in Step 1 and keep repeating the process until you get a "\n" character, indicating you've reached the end of the dinosaur name.
(see rnn.py)

Expected output:

list of sampled indices:	[12, 17, 24, 14, 13, 9, 10, 22, 24, 6, 13, 11, 12, 6, 21, 15, 21, 14, 3, 2, 1, 21, 18, 24, 
7, 25, 6, 25, 18, 10, 16, 2, 3, 8, 15, 12, 11, 7, 1, 12, 10, 2, 7, 7, 11, 5, 6, 12, 25, 0, 0]
list of sampled characters:	['l', 'q', 'x', 'n', 'm', 'i', 'j', 'v', 'x', 'f', 'm', 'k', 'l', 'f', 'u', 'o', 
'u', 'n', 'c', 'b', 'a', 'u', 'r', 'x', 'g', 'y', 'f', 'y', 'r', 'j', 'p', 'b', 'c', 'h', 'o', 
'l', 'k', 'g', 'a', 'l', 'j', 'b', 'g', 'g', 'k', 'e', 'f', 'l', 'y', '\n', '\n']

3 - Building the language model
It is time to build the character-level language model for text generation.

3.1 - Gradient descent
In this section you will implement a function performing one step of stochastic gradient descent (with clipped gradients). You will go through the training examples one at a time, so the optimization algorithm will be stochastic gradient descent. As a reminder, here are the steps of a common optimization loop for an RNN:

Forward propagate through the RNN to compute the loss
Backward propagate through time to compute the gradients of the loss with respect to the parameters
Clip the gradients if necessary
Update your parameters using gradient descent
Exercise: Implement this optimization process (one step of stochastic gradient descent).

We provide you with the following functions:

def rnn_forward(X, Y, a_prev, parameters):
    """ Performs the forward propagation through the RNN and computes the cross-entropy loss.
    It returns the loss' value as well as a "cache" storing values to be used in the backpropagation."""
    ....
    return loss, cache

def rnn_backward(X, Y, parameters, cache):
    """ Performs the backward propagation through time to compute the gradients of the loss with respect
    to the parameters. It returns also all the hidden states."""
    ...
    return gradients, a

def update_parameters(parameters, gradients, learning_rate):
    """ Updates parameters using the Gradient Descent Update Rule."""
    ...
    return parameters
    
 (see rnn.py)
    
    Expected output:

    Loss	126.503975722
    gradients["dWaa"][1][2]	0.194709315347
    np.argmax(gradients["dWax"])	93
    gradients["dWya"][1][2]	-0.007773876032
    gradients["db"][4]	[-0.06809825]
    gradients["dby"][1]	[ 0.01538192]
    a_last[4]	[-1.]

3.2 - Training the model
Given the dataset of dinosaur names, we use each line of the dataset (one name) as one training example. Every 100 steps of stochastic gradient descent, you will sample 10 randomly chosen names to see how the algorithm is doing. Remember to shuffle the dataset, so that stochastic gradient descent visits the examples in random order.

Exercise: Follow the instructions and implement model(). When examples[index] contains one dinosaur name (string), to create an example (X, Y), you can use this:

        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]] 
        Y = X[1:] + [char_to_ix["\n"]]
Note that we use: index= j % len(examples), where j = 1....num_iterations, to make sure that examples[index] is always a valid statement (index is smaller than len(examples)). The first entry of X being None will be interpreted by rnn_forward() as setting x⟨0⟩=0→. Further, this ensures that Y is equal to X but shifted one step to the left, and with an additional "\n" appended to signify the end of the dinosaur name.

(see rnn.py)

Run the following cell, you should observe your model outputting random-looking characters at the first iteration. After a few thousand iterations, your model should learn to generate reasonable-looking names.
(see rnn.py)

You can see that your algorithm has started to generate plausible dinosaur names towards the end of the training. At first, it was generating random characters, but towards the end you could see dinosaur names with cool endings. Feel free to run the algorithm even longer and play with hyperparameters to see if you can get even better results. Our implemetation generated some really cool names like maconucon, marloralus and macingsersaurus. Your model hopefully also learned that dinosaur names tend to end in saurus, don, aura, tor, etc.

If your model generates some non-cool names, don't blame the model entirely--not all actual dinosaur names sound cool. (For example, dromaeosauroides is an actual dinosaur name and is in the training set.) But this model should give you a set of candidates from which you can pick the coolest!

This assignment had used a relatively small dataset, so that you could train an RNN quickly on a CPU. Training a model of the english language requires a much bigger dataset, and usually needs much more computation, and could run for many hours on GPUs. We ran our dinosaur name for quite some time, and so far our favoriate name is the great, undefeatable, and fierce: Mangosaurus!
