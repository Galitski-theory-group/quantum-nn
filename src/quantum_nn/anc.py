import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torchvision import datasets, transforms
import json
from tqdm import tqdm
import os

class WMeasure_Rot_Fun(autograd.Function):
    """
    A function implementing a forward step consisting of rotating the qubits and then (weakly) measuring them.
    Implements htanh gradient for backpropagation.

    Methods:
    forward(ctx:torch.Tensor,input:torch.Tensor,angles:torch.Tensor,last_res:torch.Tensor,a:float,g:float) -> torch.Tensor
    backward(ctx:torch.Tensor,grad_output:torch.Tensor) -> torch.Tensor

    """
    @staticmethod
    def forward(ctx:torch.Tensor,
                input:torch.Tensor,
                angles:torch.Tensor,
                last_res:torch.Tensor,
                a:float,
                g:float
                ) -> torch.Tensor:
        """
        Implements a rotation about the y-axis and then a (generally weak) measurement on the qubits.

        Parameters:
            ctx (torch.Tensor): Context of the calculation. Can store data needed for the backward step.
            input (torch.Tensor): Input data
            angles (torch.Tensor): Angles of the qubits
            last_res (torch.Tensor): Previous set of measurement results
            a (float): Controls how much to "stretch" the activation function
            g (float): Controls how much entanglement is created between the neuron and ancilla qubits

        Returns:
            torch.Tensor: Measurement outcomes

        Note:
            Updates angles and last_res parameters to new values
        
        """

        ctx.save_for_backward(input) # save input for backpropagation
        
        # calculate angles of rotation
        if a==0.0:
            thetas=torch.pi/2*(last_res-torch.sign(input))
        else:
            thetas=torch.pi/2*(last_res-F.tanh(input/a))

        temp_angles=angles+thetas # find angles after rotation
        
        betas=torch.sin(temp_angles/2) # beta coeffiecients of angles
        exp_z=torch.cos(temp_angles/2)**2-betas**2 # expectation value of the neuron qubit
        exp_z_anc=exp_z*np.sin(g) # expectation value of the ancilla qubit

        probs=torch.stack([(1+exp_z_anc)/2,(1-exp_z_anc)/2],2) # probabilities of 0 and 1 states respectively
        probs = torch.clamp(probs, min=0) # clamp negative probabilities that may arise from numerical artifacts to 0
        outcomes=torch.tensor([1.0,-1.0]) # the possible measurement outcomes
        meas_res=outcomes[torch.distributions.Categorical(probs,validate_args=False).sample()] # sample the outcomes
        last_res.copy_(meas_res) # update activations
        
        side=betas*torch.sqrt((1-meas_res*np.sin(g))/(1+meas_res*exp_z_anc)) # distances of qubit vectors to real axis
        side=torch.clamp(side,min=-1,max=1) # clamp unphysical values that may arise from numerical artifacts
        new_angles=2*torch.arcsin(side) # calculate new angles
        angles.copy_(new_angles) # update angles

        return meas_res
    
    @staticmethod
    def backward(ctx:torch.Tensor,grad_output:torch.Tensor) -> torch.Tensor:
        """
        Implements the htanh gradient for backpropagation

        Parameters:
        ctx (torch.Tensor): Context in which the forward calculation was made. Should hold the forward input data.
        grad_output (torch.Tensor): gradients from the next step in the network

        Returns:
        torch.Tensor, None, None, None, None: gradients to pass to the previous layer of the network
        """
        input,=ctx.saved_tensors # get the preactivations from the context

        # create indicator for preactivations with magnitude less than 1
        ones=torch.ones_like(input)
        ind=torch.logical_and(input>-ones,input<ones).float()

        return ind*grad_output, None, None, None, None # return None for all the parameters of the forward method we don't want to calculate gradients for
    
class WMeasure_Rot(nn.Module):
    """
    A module that rotates the qubits about the y-axis then (weakly) measures them

    Attributes:
        angles (torch.Tensor): Angles of the qubits
        last_res (torch.Tensor): Previous set of measurement results
        a (float): Controls how much to "stretch" the activation function
        g (float): Controls how much entanglement is created between the neuron and ancilla qubits

    Methods:
        forward(self,input:torch.Tensor) -> torch.Tensor: does a forward step, returns activations

    Parameters:
        angles (torch.Tensor): Angles of the qubits
        last_res (torch.Tensor): Previous set of measurement results
        a (float): Controls how much to "stretch" the activation function
        g (float): Controls how much entanglement is created between the neuron and ancilla qubits
    """
    def __init__(self,angles:torch.Tensor,last_res:torch.Tensor,a:float,g:float) -> "WMeasure_Rot":
        super(WMeasure_Rot,self).__init__()
        self.a=a
        self.g=g
        self.angles=angles
        self.last_res=last_res

    def forward(self,input:torch.Tensor) -> torch.Tensor:
        return WMeasure_Rot_Fun.apply(input,self.angles,self.last_res,self.a,self.g)

class QMLP(nn.Module):
    """
    A module implementing a fully-connected quantum multi-layer perceptron that uses superposition and weak measurements.

    Attributes:
        width (int): Width of the hidden layers
        angles (torch.Tensor): Angles describing the state of the qubits
        last_res (torch.Tensor): Results of the previous measurement of the qubits
        net (torch.nn.Module): Module implementing the quantum MLP

    Methods:
        prepare(self,input:torch.Tensor): resizes angles and last_res to accomodate input. Initializes angles to all 0s and last_res to all 1s.
        forward(self,input:torch.Tensor) -> torch.Tensor

    Parameters:
        width (int): width of the hidden layers
        depth (int): depth of the network (not including input layer)
        in_size (int): size of input vectors
        out_size (int): size of output layer/ number of classes
        bias (bool): Optional. Indicates whether to include biases in the network. Default is False.
        a (float): Optional. Controls how much to "stretch" the activation function. Default is 0.
        g (float): Optional. Controls how much entanglement is created between the neuron and ancilla qubits. Default is pi/2.

    Note:
        If a and g parameters are not specified the network will behave classically.
    """
    def __init__(self,
                 width:int,
                 depth:int,
                 in_size:int,
                 out_size:int,
                 bias:bool=False,
                 a:float=0,
                 g:float=np.pi,
                 ) -> "QMLP":
        super(QMLP,self).__init__()
        self.width=width

        # initialize the angles and measurement results to be empty tensors
        self.angles=torch.tensor([])
        self.last_res=torch.tensor([])
        
        # build the layers of the network
        layers=[nn.Linear(in_size,width,bias=bias)]
        for _ in range(depth-2):
            layers.append(WMeasure_Rot(self.angles,self.last_res,a,g))
            layers.append(nn.Linear(width,width,bias=bias))
        layers.append(WMeasure_Rot(self.angles,self.last_res,a,np.pi/2))
        layers.append(nn.Linear(width,out_size,bias=bias))

        self.net=nn.Sequential(*layers)

    def prepare(self,input:torch.Tensor):
        # reshape to match input and set all angles to 0
        temp_angles=torch.zeros((input.shape[0],self.width))
        self.angles.resize_(temp_angles.shape).copy_(temp_angles)

        # reshape to match input and set all measurement results to 1
        temp_res=torch.ones((input.shape[0],self.width))
        self.last_res.resize_(temp_res.shape).copy_(torch.ones_like(temp_res))
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        self.prepare(input)
        return self.net(input)

def init_weights(model:torch.nn.Module):
    # Kaiming initializes all weights of the model
    if isinstance(model, nn.Linear):
        nn.init.kaiming_normal_(model.weight,nonlinearity='linear')

class StepHTFunction(autograd.Function):
    """
    A function implementing a forward step with sign activations. Implements htanh for backpropagation

    Methods:
        forward(ctx, input:torch.Tensor) -> torch.Tensor
        backward(ctx, grad_output:torch.Tensor) -> torch.Tensor
    """
    @staticmethod
    def forward(ctx, input:torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input) # save input for backpropagation
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output:torch.Tensor) -> torch.Tensor:
        input,=ctx.saved_tensors # get input from context

        # create indicator for inputs with magnitude less than 1
        ones=torch.ones(input.size())
        ind=torch.logical_and(input>-ones,input<ones).float()

        return ind*grad_output
  
class StepHT(nn.Module):
    """
    A module that uses sign activations going forward and has htanh gradients for backpropagation

    Methods:
        forward(self, input:torch.Tensor) -> torch.Tensor: does a forward step, returns activations
    """
    def __init__(self):
        super(StepHT, self).__init__()

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        return StepHTFunction.apply(input)

class Det_MLP(nn.Module):
    """
    A module that implements a fully connected classical binarized multi-layer perceptron.

    Attributes:
        net (torch.nn.Module): Module implementing the MLP

    Methods:
        forward(self,input) -> torch.Tensor

    Parameters:
        width (int): width of the hidden layers
        depth (int): depth of the network (not including input layer)
        in_size (int): size of input vectors
        out_size (int): size of output layer/ number of classes
        bias (bool): Optional. Indicates whether to include biases in the network. Default is False.
    """
    def __init__(self,
                 width:int,
                 depth:int,
                 in_size:int,
                 out_size:int,
                 bias:bool=False
                 ) -> "Det_MLP":
        super(Det_MLP,self).__init__()

        # build the layers of the network
        layers=[nn.Linear(in_size,width,bias=bias),StepHT()]
        for _ in range(depth-2):
            layers.append(nn.Linear(width,width,bias=bias))
            layers.append(StepHT())
        layers.append(nn.Linear(width,out_size,bias=bias))

        self.net=nn.Sequential(*layers)

    def forward(self,input) -> torch.Tensor:
        return self.net(input)

def prep_data(train_size:int,
              batch_size_train:int,
              test_size:int,
              batch_size_test:int
              ) -> tuple[torch.utils.data.dataloader.DataLoader]:
    """
    Prepares training and testing data

    Parameters:
        train_size (int): the number of training data to prepare
        batch_size_train (int): the batch size for the training data
        test_size (int): the number of testing data to prepare
        batch_size_test (int): the batch size for the testing data

    Returns:
        tuple[torch.utils.data.dataloader.DataLoader]: contains 2 dataloaders, the first for training, the second for testing
    """
    # prepare training data
    train_set = datasets.MNIST('../../data/mnist_data', train=True, download=True, 
                            transform=transforms.Compose([
                               transforms.ToTensor(),
                                torch.flatten
                             ]))
    train_set=torch.utils.data.Subset(train_set,range(0,train_size))
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size_train, shuffle=True)

    # prepare testing data
    test_set = datasets.MNIST('../../data/mnist_data', train=False, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                                 torch.flatten
                             ]))
    test_set=torch.utils.data.Subset(test_set,range(0,test_size))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)

    return train_loader,test_loader

def get_err_from_out(output:torch.Tensor,labels:torch.Tensor,num_shots:int=1) -> float:
    """
    Finds the error rate between model output and the corresponding labels

    Parameters:
        output (torch.Tensor): model outputs
        labels (torch.Tensor): the labels corresponding to the outputs
        num_shots (int): Optional. The number of predictions to make for each datum. Default is 1.

    Returns:
        float: the error rate
    """
    preds_coll=[torch.argmax(output,axis=1) for _ in range(num_shots)] # make a list containing num_shots predictions for each image
    preds=torch.tensor(scipy.stats.mode(preds_coll, axis=0, keepdims=False)[0]) # select the most common prediction for each image
    vec=(preds!=labels) # make indicator for all the wrong predictions
    return float(sum(vec)/len(output))

def get_err(loader:torch.utils.data.dataloader.DataLoader,model:torch.nn.Module,num_shots:int=1) -> float:
    """
    Finds the error rate for a model using the test data in a dataloader

    Parameters:
       loader (torch.utils.data.dataloader.Dataloader): a dataloader containing testing data
       model (torch.nn.Module): a module implementing a neural network
       num_shots (int): Optional. The number of predictions to make for each datum. Default is 1.

    Returns:
        float: the error rate
    """
    batch_err=[]

    # find error rates for each batch in loader
    for imgs, labels in loader:
        output=model(imgs)
        batch_err.append(get_err_from_out(output,labels,num_shots)) # append the batch error rate to the batch_err list

    return np.mean(batch_err)

def get_loss(loader:torch.utils.data.dataloader.DataLoader,net:torch.nn.Module,loss:torch.nn.modules.loss._Loss) -> float:
    """
    Calculates the loss for data in a dataloader for a particular model and loss function

    Parameters:
        loader (torch.utils.data.dataloader.DataLoader): dataloader containing a dataset with labels
        net (torch.nn.Module): a model
        loss (torch.nn.modules.Loss): a loss function

    Returns:
        float: the loss
    """
    losses=[]
    for (imgs,labels) in loader:
        losses.append(loss(imgs,labels).detach().numpy())
    
    return np.mean(losses)

def train(net:torch.nn.Module,
          train_loader:torch.utils.data.dataloader.DataLoader,
          test_loader:torch.utils.data.dataloader.DataLoader,
          learning_rate:float,
          momentum:float,
          num_shots:int,
          num_epochs:int,
          step:int
          ) -> dict:
    """
    Trains a network

    Parameters:
       train_loader (torch.utils.data.dataloader.DataLoader): dataloader containing the training data
       test_loader (torch.utils.data.dataloader.DataLoader): dataloader containing the test data
       learning_rate (float)
       momentum (float)
       num_shots (int): number of predictions to make for each datum
       num_epochs (int): number of epochs to train the model
       step (int): number of epochs between recording loss, training error rate, and test error rate

    Returns:
        dict: contains the keys "test_accuracy", "losses", "err_rates", "train_err_rates", "first_params", and "last_params"
    """
    optimizer=optim.SGD(net.parameters(),lr=learning_rate,momentum=momentum) # set up stochastic gradient descent optimizer
    loss=nn.CrossEntropyLoss()

    first_params=[param.detach().tolist() for param in net.parameters()] # record initial weights
    losses=[]
    train_err_rates=[]
    err_rates=[get_err(test_loader,net,num_shots)] # calculate initial test error rate

    first=True # indicates whether we are on the first step in the training
    for epoch in tqdm(range(num_epochs)):
        record=(epoch%step==step-1 or epoch==num_epochs-1) # indicates if data should be recorded this epoch

        for (imgs,labels) in train_loader:
            optimizer.zero_grad()
            preds=net(imgs)
            loss_step=loss(preds,labels)

            # do this only on first step
            if first:
                train_err_rates.append(get_err_from_out(preds,labels,1))
                losses.append(float(loss_step.detach().numpy()))
            first=False

            loss_step.backward() # backpropagate gradients
            optimizer.step()
            
        # calculate and record data
        if record:
            err_rates.append(float(get_err(test_loader,net,num_shots)))
            train_err_rates.append(get_err_from_out(preds,labels,1))
            losses.append(float(loss_step.detach().numpy()))

    last_params=[param.detach().tolist() for param in net.parameters()] # record final weights
    acc=1-get_err(test_loader,net,num_shots) # calculate accuracy

    # create dictionary of data to return
    out_dict={
        "test_accuracy": acc,
        "losses": losses,
        "err_rates": err_rates,
        "train_err_rates": train_err_rates,
        "first_params":first_params,
        "last_params":last_params
    }

    return out_dict     

def record(out_dict:dict,directory:str,filename:str):
    """
    Writes contents of a dictionary to a json file

    Parameters:
        out_dict (dict): dictionary to write to file
        directory (str)
        filename (str)
    """
    os.makedirs(directory,exist_ok=True)
    with open(directory+'/'+filename,'w') as outfile:
        json.dump(out_dict,outfile)

