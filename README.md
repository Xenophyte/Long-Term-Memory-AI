# Long-Term-Memory-AI
An AI that solves narrow problems but keeps a memory of what previous solutions it has found and uses that knowledge to find new solutions to problems it had never seen before.

# The basic principle
The goal of this project is to prove in a toy model the advantages of using long term memory in machine learning. In this toy model we will create a machine that learns to fit a curve onto one-dimensional data. We will give it nothing more than the four basic operators to start with sum, difference, product and quotient. The machine should eventually be able to develop more and more sophisticated functions to fit the less obvious data.

This principle is the principle behind all evolution of complexity in the universe, be it intelligence, life, technology or knowledge. To educate a child one doesn't sit it in front of a million examples of the Navier-Stokes equation and let it train for 15 years the neural network in its brain on that data; instead, it we give it a few problems it can solve within the reach of its abilities and then use the acquired knowledge to build on top of that new tools and concepts to address new problems. The same is true for technology: each new invention is reused later to make other new, different technologies. And the interesting thing about a system based on this principle is that like with a real person, one can't tell what exactly it will be good at.

# Arquitecture

The machine is composed of three subsystems:

- the knowledge database or long-term memory;
- a neural network that takes the 1D plot of the function and selects the best specimens to use to fit that function (optional);
- a genetic algorithm which mixes specimens together to form new ones.

# The data

The data will be generated using a genetic algorithm from common functions to write differential equations which will then be simulated using random input parameters. 

# The machine's knowledge: the specimen functions and their fitness

A function must contain a skeleton, mentions to x and optimizable parameters. The number of parameters will affect the fitness of the function since the time to optimize will be taken into account. For example, sin(a x) has only one parameter to optimize and so wil converge faster than b sin(a x) / b which is the same function but needlessly more complicated. The quality of an attempt to fit data by a specimen function is given by

attempt_quality = exp(-(mean square error)/hyperparameter1) * sigmoid( -hyperparameter2*(time spent optimizing - hyperparameter3) )

- hyperparameter1: mean square error tolerance;
- hyperparameter2: abruptness of fitness decay after hyperparameter3;
- hyperparameter3: "max" optimization time

The fitness of a specimen will depend its own total score over all the attempts it made to fit itself onto data and on the score of the specimens that make use of it. That way if a specimen isn't very useful on its own but is a great building block for other highly sucessful specimens, it will be kept in the knowledge base of the machine.

fitness = (sum of own attempt_qualities) + hyperparameter4 * (sum of first degree attempt_qualities)

- hyperparameter4: contribution of first order neighbors to the specimen's fitness
