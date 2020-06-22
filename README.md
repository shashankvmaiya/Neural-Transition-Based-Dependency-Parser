# Neural-Transition-Based-Dependency-Parser

In this project, we implement a neural-network based dependency parser, with the goal of maximizing performance on the UAS (Unlabeled Attachemnt Score) metric. There are two parts to this implementation
- Transition based parser: This incrementally builds up a dependency parse one step at a time. At every step it maintains a partial parse, which is represented as follows:
	- A stack of words that are currently being processed. 
	- A buffer of words yet to be processed.
	- A list of dependencies predicted by the parser.

- Neural Network Classifier: This NN predicts which of the 3 transitions (SHIFT, LEFT-ARC or RIGHT-ARC) the parser needs to apply in its next step


- To train the neural network model and compute the predictions on the test data from Penn Treebank (annotated with Universal Dependencies)
```
python run.py
```
