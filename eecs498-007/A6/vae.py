from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


def hello_vae():
    print("Hello from vae.py!")


class VAE(nn.Module):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.hidden_dim = None # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ############################################################################################
        # TODO: Implement the fully-connected encoder architecture described in the notebook.      #
        # Specifically, self.encoder should be a network that inputs a batch of input images of    #
        # shape (N, 1, H, W) into a batch of hidden features of shape (N, H_d). Set up             #
        # self.mu_layer and self.logvar_layer to be a pair of linear layers that map the hidden    #
        # features into estimates of the mean and log-variance of the posterior over the latent    #
        # vectors; the mean and log-variance estimates will both be tensors of shape (N, Z).       #
        ############################################################################################
        # Replace "pass" statement with your code

        # Define the hidden dimension as shown in course's lecture example.
        # Source: https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture20.pdf
        # Slide: 12/120.
        self.hidden_dim = 400

        # Define the encoder: A three Linear+ReLU layers neural network.
        self.encoder = nn.Sequential(
          nn.Flatten(),
          nn.Linear(self.input_size, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          nn.ReLU()
        )

        # Define the mu_layer, with input (N, H_d) and output (N, Z)
        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_size)
        # Define the logvar_layer, with input (N, H_d) and output (N, Z)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_size)

        ############################################################################################
        # TODO: Implement the fully-connected decoder architecture described in the notebook.      #
        # Specifically, self.decoder should be a network that inputs a batch of latent vectors of  #
        # shape (N, Z) and outputs a tensor of estimated images of shape (N, 1, H, W).             #
        ############################################################################################
        # Replace "pass" statement with your code

        self.decoder = nn.Sequential(
          nn.Linear(self.latent_size, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, self.input_size),
          nn.Sigmoid(),
          nn.Unflatten(dim=1, unflattened_size=(1, 28, 28))
        )

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################


    def forward(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the input batch through the encoder model to get posterior mu and logvariance   #
        # (2) Reparametrize to compute  the latent vector z                                        #
        # (3) Pass z through the decoder to resconstruct x                                         #
        ############################################################################################
        # Replace "pass" statement with your code
        
        # Pass input images "x" to the encoder. Output shape is: (N, H_d)
        encoder_out = self.encoder(x)

        # Get the posterior mu from the encoder's output. Its shape is: (N, Z)
        mu = self.mu_layer(encoder_out)
        # Get the posterior logvariance from the encoder's output. Its shape is: (N, Z)
        logvar = self.logvar_layer(encoder_out)

        # Reparametrize to compute the latent vector "z", of shape (N, Z)
        z = reparametrize(mu, logvar)

        # Pass "z" through the decoder to resconstruct "x", the "x_hat".
        x_hat = self.decoder(z)

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.num_classes = num_classes # C
        self.hidden_dim = None # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ############################################################################################
        # TODO: Define a FC encoder as described in the notebook that transforms the image--after  #
        # flattening and now adding our one-hot class vector (N, H*W + C)--into a hidden_dimension #               #
        # (N, H_d) feature space, and a final two layers that project that feature space           #
        # to posterior mu and posterior log-variance estimates of the latent space (N, Z)          #
        ############################################################################################
        # Replace "pass" statement with your code
        
        # Define the hidden dimension as shown in course's lecture example.
        # Source: https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture20.pdf
        # Slide: 12/120.
        self.hidden_dim = 400

        # Define the encoder: A three Linear+ReLU layers neural network.
        # The encoder's input has shape of (N, H*W + C), i.e. The flattened images and their classes.
        # Note that the concatenation (between images and classes) is done in the "forward" function.
        self.encoder = nn.Sequential(
          nn.Linear(self.input_size + self.num_classes, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          nn.ReLU()
        )

        # Define the mu_layer, with input (N, H_d) and output (N, Z)
        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_size)
        # Define the logvar_layer, with input (N, H_d) and output (N, Z)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_size)

        ############################################################################################
        # TODO: Define a fully-connected decoder as described in the notebook that transforms the  #
        # latent space (N, Z + C) to the estimated images of shape (N, 1, H, W).                   #
        ############################################################################################
        # Replace "pass" statement with your code

        self.decoder = nn.Sequential(
          nn.Linear(self.latent_size + self.num_classes, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, self.input_size),
          nn.Sigmoid(),
          nn.Unflatten(dim=1, unflattened_size=(1, 28, 28))
        )

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    def forward(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Input data for this timestep of shape (N, 1, H, W)
        - c: One hot vector representing the input class (0-9) (N, C)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the concatenation of input batch and one hot vectors through the encoder model  #
        # to get posterior mu and logvariance                                                      #
        # (2) Reparametrize to compute the latent vector z                                         #
        # (3) Pass concatenation of z and one hot vectors through the decoder to resconstruct x    #
        ############################################################################################
        # Replace "pass" statement with your code
        
        # Flatten the "height" and "width" of the input batch 'x'.
        # Input shape (N, 1, H, W). Output shape: (N, H*W)
        x_flat = torch.flatten(x, start_dim=1, end_dim=-1)

        # Create the encoder's input 'enc_in' by concatenating flattened images and their classes.
        # Output shape is: (N, H*W + C)
        enc_in = torch.cat((x_flat, c), dim=1)

        # Pass 'enc_in' to the encoder. Output shape is: (N, H_d)
        enc_out = self.encoder(enc_in)

        # Get the posterior mu from the encoder's output. Its shape is: (N, Z)
        mu = self.mu_layer(enc_out)
        # Get the posterior logvariance from the encoder's output. Its shape is: (N, Z)
        logvar = self.logvar_layer(enc_out)

        # Reparametrize to compute the latent vector "z", of shape (N, Z)
        z = reparametrize(mu, logvar)

        # Create the decoder's input 'dec_in' by concatenating the latent vector and its classes.
        # Output shape is: (N, Z + C)
        dec_in = torch.cat((z, c), dim=1)

        # Pass "dec_in" through the decoder to resconstruct "x", the "x_hat".
        # x.shape == x_hat.shape == (N, 1, H, W)
        x_hat = self.decoder(dec_in)

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar



def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns: 
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None
    ################################################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and scaling by          #
    # posterior mu and sigma to estimate z                                                         #
    ################################################################################################
    # Replace "pass" statement with your code

    # Convert the "log of the variance" to "sigma" (standard deviation).
    sigma = torch.sqrt(torch.exp(logvar))

    # Compute 'z'.
    # Epsilon is a Tensor that contains random samples from a standard normal
    # distribution (mu=0, std=1)
    z = sigma * torch.randn_like(mu) + mu

    ################################################################################################
    #                              END OF YOUR CODE                                                #
    ################################################################################################
    return z


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).

    Inputs:
    - x_hat: Reconstruced input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    loss = None
    ################################################################################################
    # TODO: Compute negative variational lowerbound loss as described in the notebook              #
    ################################################################################################
    # Replace "pass" statement with your code
    
    # Get the minibatch size
    N = mu.shape[0]

    # Compute the reconstruction loss term, using Binary Cross Entropy (BCE) loss.
    # The "BCE loss" have to be adapted to the "reconstruction loss" (Expectation) by:
    # - Changing the reduction mode from 'mean' (default) to 'sum' (used in the Expectation).
    # - The input to the BCE is 'x_hat' and the target is 'x'. This can be done because we are
    # operating on MNIST dataset, where each pixel is either 0 or 1.
    # Note that the minus sign is handled by the BCE loss itself.
    rec_term = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

    # Compute the KL divergence term (kldiv_term).
    kldiv_term = 1 + logvar - mu**2 - torch.exp(logvar)
    kldiv_term = -0.5 * kldiv_term.sum()

    # Final loss is the sum of "reconstruction loss term" and "KL divergence term".
    loss = rec_term + kldiv_term

    # Average the loss across samples in the minibatch.
    loss /= N

    ################################################################################################
    #                            END OF YOUR CODE                                                  #
    ################################################################################################
    return loss

