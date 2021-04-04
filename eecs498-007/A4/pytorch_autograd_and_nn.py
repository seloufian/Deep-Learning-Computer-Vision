"""
Implements pytorch autograd and nn in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from a4_helper import *
import torch.nn.functional as F
import torch.optim as optim

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from pytorch_autograd_and_nn.py!')



################################################################################
# Part II. Barebones PyTorch                         
################################################################################
# Before we start, we define the flatten function for your convenience.
def flatten(x, start_dim=1, end_dim=-1):
  return x.flatten(start_dim=start_dim, end_dim=end_dim)


def three_layer_convnet(x, params):
  """
  Performs the forward pass of a three-layer convolutional network with the
  architecture defined above.

  Inputs:
  - x: A PyTorch Tensor of shape (N, C, H, W) giving a minibatch of images
  - params: A list of PyTorch Tensors giving the weights and biases for the
    network; should contain the following:
    - conv_w1: PyTorch Tensor of shape (channel_1, C, KH1, KW1) giving weights
      for the first convolutional layer
    - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
      convolutional layer
    - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
      weights for the second convolutional layer
    - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
      convolutional layer
    - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
      figure out what the shape should be?
    - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
      figure out what the shape should be?
  
  Returns:
  - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
  """
  conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
  scores = None
  ##############################################################################
  # TODO: Implement the forward pass for the three-layer ConvNet.              
  # The network have the following architecture:                               
  # 1. Conv layer (with bias) with 32 5x5 filters, with zero-padding of 2     
  #   2. ReLU                                                                  
  # 3. Conv layer (with bias) with 16 3x3 filters, with zero-padding of 1     
  # 4. ReLU                                                                   
  # 5. Fully-connected layer (with bias) to compute scores for 10 classes    
  # Hint: F.linear, F.conv2d, F.relu, flatten (implemented above)                                   
  ##############################################################################
  # Replace "pass" statement with your code

  # (1) and (2): Conv layer + ReLU
  x = F.conv2d(x, conv_w1, bias=conv_b1, padding=2)
  x = F.relu(x)
  # (3) and (4): Conv layer + ReLU
  x = F.conv2d(x, conv_w2, bias=conv_b2, padding=1)
  x = F.relu(x)
  # Flatten 'channels', 'height' and 'width' of "x" (i.e. keep only batches dim [N]).
  x = flatten(x)
  # (5): FC layer.
  scores = F.linear(x, fc_w, bias=fc_b)

  ##############################################################################
  #                                 END OF YOUR CODE                             
  ##############################################################################
  return scores


def initialize_three_layer_conv_part2(dtype=torch.float, device='cpu'):
  '''
  Initializes weights for the three_layer_convnet for part II
  Inputs:
    - dtype: A torch data type object; all computations will be performed using
        this datatype. float is faster but less accurate, so you should use
        double for numeric gradient checking.
      - device: device to use for computation. 'cpu' or 'cuda'
  '''
  # Input/Output dimenssions
  C, H, W = 3, 32, 32
  num_classes = 10

  # Hidden layer channel and kernel sizes
  channel_1 = 32
  channel_2 = 16
  kernel_size_1 = 5
  kernel_size_2 = 3

  # Initialize the weights
  conv_w1 = None
  conv_b1 = None
  conv_w2 = None
  conv_b2 = None
  fc_w = None
  fc_b = None

  ##############################################################################
  # TODO: Define and initialize the parameters of a three-layer ConvNet           
  # using nn.init.kaiming_normal_. You should initialize your bias vectors    
  # using the zero_weight function.                         
  # You are given all the necessary variables above for initializing weights. 
  ##############################################################################
  # Replace "pass" statement with your code

  # "conv1_shape" is a 4-D tensor of shape (out_channels, in_channels, kH, kW).
  conv1_shape = (channel_1, C, kernel_size_1, kernel_size_1)
  conv_w1 = nn.init.kaiming_normal_(torch.empty(conv1_shape, dtype=dtype, device=device))
  # "conv_b1" is a 1-D tensor of shape (out_channels,)
  conv_b1 = nn.init.zeros_(torch.empty(conv1_shape[0], dtype=dtype, device=device))
  # Compute Conv1 layer's output height/width. This is need for next operations.
  # Conv1 output height/width = 1 + (H - k1 + 2*(padding_conv1)) / stride_conv1
  #                           = 1 + (H - k1 + 2*2) / 1
  # That is, Conv1 output is a 3-D tensor of shape (channel_1, HW_1, HW_1)
  # Note that, the batch size is not mentioned and "output's height" = "output's width".
  HW_1 = 1 + (H - kernel_size_1 + 2*2)

  # Compute "conv_w2" and "conv_b2" with the same way as for conv1 parameters.
  conv2_shape = (channel_2, channel_1, kernel_size_2, kernel_size_2)
  conv_w2 = nn.init.kaiming_normal_(torch.empty(conv2_shape, dtype=dtype, device=device))
  conv_b2 = nn.init.zeros_(torch.empty(conv2_shape[0], dtype=dtype, device=device))
  # Conv2 output height/width = 1 + (HW_1 - k2 + 2*(padding_conv2)) / stride_conv2
  #                           = 1 + (HW_1 - k2 + 2*1) / 1
  # That is, Conv2 output is a 3-D tensor of shape (channel_2, HW_2, HW_2)
  HW_2 = 1 + (HW_1 - kernel_size_2 + 2)

  fc_shape = (num_classes, channel_2 * HW_2 * HW_2)
  fc_w = nn.init.kaiming_normal_(torch.empty(fc_shape, dtype=dtype, device=device))
  fc_b = nn.init.zeros_(torch.empty(fc_shape[0], dtype=dtype, device=device))

  # Mark all weight and bias tensors as trainable (i.e. Requires gradients).
  for tensor in [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]:
    tensor.requires_grad = True

  ##############################################################################
  #                                 END OF YOUR CODE                            
  ##############################################################################
  return [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]




################################################################################
# Part III. PyTorch Module API                         
################################################################################

class ThreeLayerConvNet(nn.Module):
  def __init__(self, in_channel, channel_1, channel_2, num_classes):
    super().__init__()
    ############################################################################
    # TODO: Set up the layers you need for a three-layer ConvNet with the       
    # architecture defined below. You should initialize the weight  of the
    # model using Kaiming normal initialization, and zero out the bias vectors.     
    #                                       
    # The network architecture should be the same as in Part II:          
  #   1. Convolutional layer with channel_1 5x5 filters with zero-padding of 2  
    #   2. ReLU                                   
    #   3. Convolutional layer with channel_2 3x3 filters with zero-padding of 1
    #   4. ReLU                                   
    #   5. Fully-connected layer to num_classes classes               
    #                                       
    # We assume that the size of the input of this network is `H = W = 32`, and   
    # there is no pooing; this information is required when computing the number  
    # of input channels in the last fully-connected layer.              
    #                                         
    # HINT: nn.Conv2d, nn.init.kaiming_normal_, nn.init.zeros_            
    ############################################################################
    # Replace "pass" statement with your code
    
    # Define the 1st Conv layer.
    # Input: Tensor of shape (in_channel, 32, 32)
    # Output: Tensor of shape (channel_1, 32, 32)
    self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, padding=2)
    # Initialize Conv1 layer's weights and biases.
    nn.init.kaiming_normal_(self.conv1.weight)
    nn.init.zeros_(self.conv1.bias)

    # Define the 2nd Conv layer.
    # Input: Tensor of shape (channel_1, 32, 32)
    # Output: Tensor of shape (channel_2, 32, 32)
    self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1)
    # Initialize Conv2 layer's weights and biases.
    nn.init.kaiming_normal_(self.conv2.weight)
    nn.init.zeros_(self.conv2.bias)

    # Define the Fully-connected layer.
    self.fc = nn.Linear(channel_2*32*32, num_classes)
    # Initialize FC layer's weights and biases.
    nn.init.kaiming_normal_(self.fc.weight)
    nn.init.zeros_(self.fc.bias)

    ############################################################################
    #                           END OF YOUR CODE                            
    ############################################################################

  def forward(self, x):
    scores = None
    ############################################################################
    # TODO: Implement the forward function for a 3-layer ConvNet. you      
    # should use the layers you defined in __init__ and specify the       
    # connectivity of those layers in forward()   
    # Hint: flatten (implemented at the start of part II)                          
    ############################################################################
    # Replace "pass" statement with your code

    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = flatten(x)
    scores = self.fc(x)

    ############################################################################
    #                            END OF YOUR CODE                          
    ############################################################################
    return scores


def initialize_three_layer_conv_part3():
  '''
  Instantiates a ThreeLayerConvNet model and a corresponding optimizer for part III
  '''

  # Parameters for ThreeLayerConvNet
  C = 3
  num_classes = 10

  channel_1 = 32
  channel_2 = 16

  # Parameters for optimizer
  learning_rate = 3e-3
  weight_decay = 1e-4

  model = None
  optimizer = None
  ##############################################################################
  # TODO: Instantiate ThreeLayerConvNet model and a corresponding optimizer.     
  # Use the above mentioned variables for setting the parameters.                
  # You should train the model using stochastic gradient descent without       
  # momentum, with L2 weight decay of 1e-4.                    
  ##############################################################################
  # Replace "pass" statement with your code

  model = ThreeLayerConvNet(C, channel_1, channel_2, num_classes)

  optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

  ##############################################################################
  #                                 END OF YOUR CODE                            
  ##############################################################################
  return model, optimizer


################################################################################
# Part IV. PyTorch Sequential API                        
################################################################################

# Before we start, We need to wrap `flatten` function in a module in order to stack it in `nn.Sequential`.
# As of 1.3.0, PyTorch supports `nn.Flatten`, so this is not required in the latest version.
# However, let's use the following `Flatten` class for backward compatibility for now.
class Flatten(nn.Module):
  def forward(self, x):
    return flatten(x)


def initialize_three_layer_conv_part4():
  '''
  Instantiates a ThreeLayerConvNet model and a corresponding optimizer for part IV
  '''
  # Input/Output dimenssions
  C, H, W = 3, 32, 32
  num_classes = 10

  # Hidden layer channel and kernel sizes
  channel_1 = 32
  channel_2 = 16
  kernel_size_1 = 5
  pad_size_1 = 2
  kernel_size_2 = 3
  pad_size_2 = 1

  # Parameters for optimizer
  learning_rate = 1e-2
  weight_decay = 1e-4
  momentum = 0.5

  model = None
  optimizer = None
  ##################################################################################
  # TODO: Rewrite the 3-layer ConvNet with bias from Part III with Sequential API and 
  # a corresponding optimizer.
  # You don't have to re-initialize your weight matrices and bias vectors.  
  # Here you should use `nn.Sequential` to define a three-layer ConvNet with:
  #   1. Convolutional layer (with bias) with 32 5x5 filters, with zero-padding of 2 
  #   2. ReLU                                      
  #   3. Convolutional layer (with bias) with 16 3x3 filters, with zero-padding of 1 
  #   4. ReLU                                      
  #   5. Fully-connected layer (with bias) to compute scores for 10 classes        
  #                                            
  # You should optimize your model using stochastic gradient descent with Nesterov   
  # momentum 0.5, with L2 weight decay of 1e-4 as given in the variables above.   
  # Hint: nn.Sequential, Flatten (implemented at the start of Part IV)   
  ####################################################################################
  # Replace "pass" statement with your code

  model = nn.Sequential(
    nn.Conv2d(C, channel_1, kernel_size_1, padding=pad_size_1),
    nn.ReLU(),
    nn.Conv2d(channel_1, channel_2, kernel_size_2, padding=pad_size_2),
    nn.ReLU(),
    Flatten(),
    nn.Linear(channel_2 * H * W, num_classes)
  )

  optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                        weight_decay=weight_decay, momentum=momentum, nesterov=True)

  ################################################################################
  #                                 END OF YOUR CODE                             
  ################################################################################
  return model, optimizer


################################################################################
# Part V. ResNet for CIFAR-10                        
################################################################################

class PlainBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()

    self.net = None
    ############################################################################
    # TODO: Implement PlainBlock.                                             
    # Hint: Wrap your layers by nn.Sequential() to output a single module.     
    #       You don't have use OrderedDict.                                    
    # Inputs:                                                                  
    # - Cin: number of input channels                                          
    # - Cout: number of output channels                                        
    # - downsample: add downsampling (a conv with stride=2) if True            
    # Store the result in self.net.                                            
    ############################################################################
    # Replace "pass" statement with your code

    self.net = nn.Sequential(
      nn.BatchNorm2d(Cin),
      nn.ReLU(),
      nn.Conv2d(Cin, Cout, 3, stride=(2 if downsample else 1), padding=1),
      nn.BatchNorm2d(Cout),
      nn.ReLU(),
      nn.Conv2d(Cout, Cout, 3, stride=1, padding=1)
    )

    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################

  def forward(self, x):
    return self.net(x)


class ResidualBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()

    self.block = None # F
    self.shortcut = None # G
    ############################################################################
    # TODO: Implement residual block using plain block. Hint: nn.Identity()    #
    # Inputs:                                                                  #
    # - Cin: number of input channels                                          #
    # - Cout: number of output channels                                        #
    # - downsample: add downsampling (a conv with stride=2) if True            #
    # Store the main block in self.block and the shortcut in self.shortcut.    #
    ############################################################################
    # Replace "pass" statement with your code

    self.block = PlainBlock(Cin, Cout, downsample)

    if not downsample:
      if Cin == Cout:
        self.shortcut = nn.Identity()
      else:
        self.shortcut = nn.Conv2d(Cin, Cout, 1, stride=1, padding=0)
    else:
      self.shortcut = nn.Conv2d(Cin, Cout, 1, stride=2, padding=0)

    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################
  
  def forward(self, x):
    return self.block(x) + self.shortcut(x)


class ResNet(nn.Module):
  def __init__(self, stage_args, Cin=3, block=ResidualBlock, num_classes=10):
    super().__init__()

    self.cnn = None
    ############################################################################
    # TODO: Implement the convolutional part of ResNet using ResNetStem,       #
    #       ResNetStage, and wrap the modules by nn.Sequential.                #
    # Store the model in self.cnn.                                             #
    ############################################################################
    # Replace "pass" statement with your code

    self.cnn = nn.Sequential(
      ResNetStem(),
      *[ResNetStage(*stage, block) for stage in stage_args]
    )

    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################
    self.fc = nn.Linear(stage_args[-1][1], num_classes)
  
  def forward(self, x):
    scores = None
    ############################################################################
    # TODO: Implement the forward function of ResNet.                          #
    # Store the output in `scores`.                                            #
    ############################################################################
    # Replace "pass" statement with your code

    # Pass "x" [tensor of shape (B, 3, H, W)] into the CNN.
    # The output is a tensor of shape (B, Cout, H', W')
    x = self.cnn(x)

    # Get the height/width of the previous output (i.e. H' and W').
    xh, xw = x.shape[2], x.shape[3]
    # Perform average pooling across the width and height (without padding/stride).
    # That is, we 'average' the width and height into a single value.
    # The output is a tensor of shape (B, Cout, 1, 1)
    x = F.avg_pool2d(x, kernel_size=(xh, xw))

    # Transform "x" from shape (B, Cout, 1, 1) to (B, Cout)
    x = flatten(x)

    # Pass the flattened "x" to the FC layer. Output's shape is (B, <num_classes>)
    scores = self.fc(x)

    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################
    return scores


class ResidualBottleneckBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()

    self.block = None
    self.shortcut = None
    ############################################################################
    # TODO: Implement residual bottleneck block.                               #
    # Inputs:                                                                  #
    # - Cin: number of input channels                                          #
    # - Cout: number of output channels                                        #
    # - downsample: add downsampling (a conv with stride=2) if True            #
    # Store the main block in self.block and the shortcut in self.shortcut.    #
    ############################################################################
    # Replace "pass" statement with your code
    
    # Define the "intermediate Cout".
    coutint = Cout // 4

    self.block = nn.Sequential(
      nn.BatchNorm2d(Cin),
      nn.ReLU(),
      nn.Conv2d(Cin, coutint, 1, stride=(2 if downsample else 1), padding=0),
      nn.BatchNorm2d(coutint),
      nn.ReLU(),
      nn.Conv2d(coutint, coutint, 3, stride=1, padding=1),
      nn.BatchNorm2d(coutint),
      nn.ReLU(),
      nn.Conv2d(coutint, Cout, 1, stride=1, padding=0)
    )

    if not downsample:
      if Cin == Cout:
        self.shortcut = nn.Identity()
      else:
        self.shortcut = nn.Conv2d(Cin, Cout, 1, stride=1, padding=0)
    else:
      self.shortcut = nn.Conv2d(Cin, Cout, 1, stride=2, padding=0)

    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################

  def forward(self, x):
    return self.block(x) + self.shortcut(x)

##############################################################################
# No need to implement anything here                     
##############################################################################
class ResNetStem(nn.Module):
  def __init__(self, Cin=3, Cout=8):
    super().__init__()
    layers = [
        nn.Conv2d(Cin, Cout, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
    ]
    self.net = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.net(x)

class ResNetStage(nn.Module):
  def __init__(self, Cin, Cout, num_blocks, downsample=True,
               block=ResidualBlock):
    super().__init__()
    blocks = [block(Cin, Cout, downsample)]
    for _ in range(num_blocks - 1):
      blocks.append(block(Cout, Cout))
    self.net = nn.Sequential(*blocks)
  
  def forward(self, x):
    return self.net(x)