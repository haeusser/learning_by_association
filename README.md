This repository contains code for the paper "Learning by Association - A versatile semi-supervised training method for neural networks."

It is implemented with TensorFlow. Please refer to the TensorFlow documentation for further information.

The core functions are implemented in semisup.py.
The files train.py and eval.py demonstrate how to use them. A quick example is contained in mnist_train_eval.py.

In order to reproduce the results from the paper, please use the architectures and pipelines from the {stl10,svhn,synth}_tools.py. They are loaded automatically bu setting the flag 'package' in {train,eval}.py accordingly.

If you use the code, please cite the paper "Learning by Association - A versatile semi-supervised training method for neural networks." (arXiv/bibTeX t.b.a.).

For questions please contact Philip Haeusser (haeusser@cs.tum.edu) or Alexander Mordvintsev (moralex@google.com).
