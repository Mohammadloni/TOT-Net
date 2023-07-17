"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function

from evolver import Evolver
from tqdm import tqdm
import logging
from keras import backend as K
import sys
from tensorflow.python.client import device_lib
import tensorflow as tf

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO#,
    #filename='log.txt'
)



#Function to carry out NSGA-II's fast non dominated sort

def train_genomes(genomes, dataset):
    """Train each genome.

    Args:
        networks (list): Current population of genomes
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***train_networks(networks, dataset)***")

    pbar = tqdm(total=len(genomes))

    for genome in genomes:
        genome.train(dataset)
        
        pbar.update(1)

    pbar.close()
 
def get_average_accuracy(genomes):
    total_accuracy = 0
    for genome in genomes:
        total_accuracy += genome.accuracy
    return total_accuracy / len(genomes)

def generate(generations, population, all_possible_genes, dataset):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generation
        all_possible_genes (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***generate(generations, population, all_possible_genes, dataset)***")
    
    evolver = Evolver(all_possible_genes)
    
    genomes = evolver.create_population(population)

    # Evolve the generation.
    for i in range( generations ):

        logging.info("***Now in generation %d of %d***" % (i + 1, generations))
        # Train and get accuracy for networks/genomes.
        train_genomes(genomes, dataset)
        sort = genomes
        

        # for genome in after_sort:
        for genome in sort:
            genome.print_genome()
            
        average_accuracy = get_average_accuracy(sort)
        #Print out the average accuracy each generation.
        print("Gen: %d, Generation average: %.2f" % (i, average_accuracy * 100))


        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Evolve!
            genomes = evolver.evolve(sort)

    sort = genomes
    for genome in sort:
            genome.print_genome()
    

def print_genomes(genomes):

    logging.info('-'*80)
    for genome in genomes:
        genome.print_genome()



def main():

    population = 20 # Number of networks/genomes in each generation.
    #we only need to train the new ones....
    
    ds = 4

    if   (ds == 1):
        dataset = 'mnist_mlp'
    elif (ds == 2):
        dataset = 'mnist_cnn'
    elif (ds == 3):
        dataset = 'cifar10_mlp'
    elif (ds == 4):
        dataset = 'cifar10_cnn'
    else:
        dataset = 'mnist_mlp'

    print("***Dataset:", dataset)

    if dataset == 'mnist_cnn':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [16 ,28,40, 52],
            'nb_layers':  [2,3 ],
            'activationL': ['relu', 'elu', 'tanh', 'sigmoid', 'selu', 'swish'],
            'activationR': ['relu', 'elu', 'tanh', 'sigmoid', 'selu', 'swish'],
            'optimizer': [ 'adam', 'sgd', 'adagrad', 'adamax', 'nadam']
        }
    elif dataset == 'mnist_mlp':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
           
            'nb_neurons': [16, 32, 48, 64, 96, 128, 192 ,256, 512, 768, 1024],#, 128], #, 256, 512, 768, 1024],
            'nb_layers':  [1, 2, 3, 4, 5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
    elif dataset == 'cifar10_mlp':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [64, 128, 192,256, 512, 768, 1024],
            'nb_layers':  [1, 2, 3, 4, 5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
    elif dataset == 'cifar10_cnn':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'activationL': ['relu', 'elu', 'tanh', 'sigmoid', 'selu','swish','prelu','leaky_relu','elish','hardtanh'],
            'activationR': ['relu', 'elu', 'tanh', 'sigmoid', 'selu','swish','prelu','leaky_relu','elish','hardtanh'],
          
        }        

    else:
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [64, 128, 256, 512, 768, 1024],
            'nb_layers':  [1, 2, 3, 4, 5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
            
    print("***Evolving for %d generations with population size = %d***" % (generations, population))

    generate(generations, population, all_possible_genes, dataset)

if __name__ == '__main__':
    #print(device_lib.list_local_devices())
    main()
