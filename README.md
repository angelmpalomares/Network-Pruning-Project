# Network-Pruning-Project
Basic project that test the effects of different pruning techniques on different Neural Network architectures
## How to get the code
In order to run this code, first clone this repository
```
git clone https://github.com/angelmpalomares/Network-Pruning-Project.git
```
Then go to its directory, create a virtual environment, activate it and install the required python modules, which can be found in the requirements.txt
##How to run it
Code runs with several arguments, namely:

'mode': where the choices are 'train', 'eval' or 'prune'

'--backbone': determines the architecture that will be used, to choose between 'MLP', 'CNN' or 'ResNet'

'--epochs': epochs that will be used for training. Must be an integer

'--lr': learning rate that will be used for training

'--device': device that will be used for computations. Choices can be cpu, cuda:0, cuda:1, ... Note that
the default is cuda:0, so if the user doesn't have a working GPU, this parameter MUST be changed

'--pruning': pruning technique that will be used in 'prune' mode, to choose 
between 'random', 'unstructured', 'structured'


So for example if someone would like to train a CNN with the GPU, 10 epochs and a learning rate of 0.01, 
he would have to write in the terminal: 
```
python main.py train --backbone 'CNN' --device 'cuda:0'	--epochs 10 --lr 0.01
```
##Structure
Structure of this code is fairly simple, since it consists in only one module, main.py.
Inside main.py the class Classifier can be found, which implements the different architectures and training methods, evaluation, and also the pruning methods and its performace evaluator functions.
After the definition of the Classifier, the main function is called to make the appropiate computations.
