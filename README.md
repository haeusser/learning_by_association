This repository contains code for the paper [Learning by Association - A versatile semi-supervised training method for neural networks (CVPR 2017)](https://vision.in.tum.de/_media/spezial/bib/haeusser_cvpr_17.pdf) and the follow-up work [Associative Domain Adaptation (ICCV 2017)](https://vision.in.tum.de/_media/spezial/bib/haeusser_iccv_17.pdf)

It is implemented with TensorFlow. Please refer to the [TensorFlow documentation](https://www.tensorflow.org/install/) for further information.

The core functions are implemented in `semisup/backend.py`.
The files `train.py` and `eval.py` demonstrate how to use them. A quick example is contained in `mnist_train_eval.py`.

In order to reproduce the results from the paper, please use the architectures and pipelines from the `{stl10,svhn,synth}_tools.py`. They are loaded automatically by setting the flag `package` in `{train,eval}.py` accordingly.

Before you get started, please make sure to add the following to your `~/.bashrc`:
```
export PYTHONPATH=/path/to/learning_by_association:$PYTHONPATH
```

Copy the file `semisup/tools/data_dirs.py.template` to `semisup/tools/data_dirs.py`, adapt the paths and .gitignore this file.

If you use the code, please cite the paper "Learning by Association - A versatile semi-supervised training method for neural networks" or "Associative Domain Adaptation":
```
@string{cvpr="IEEE Conference on Computer Vision and Pattern Recognition (CVPR)"}
@InProceedings{haeusser-cvpr-17,
  author = 	 "P. Haeusser and A. Mordvintsev and D. Cremers",
  title = 	 "Learning by Association - A versatile semi-supervised training method for neural networks",
  booktitle = cvpr,
  year = 	 "2017",
}

@string{iccv="IEEE International Conference on Computer Vision (ICCV)"}
@InProceedings{haeusser-iccv-17,
  author = 	 "P. Haeusser and T. Frerix and A. Mordvintsev and D. Cremers",
  title = 	 "Associative Domain Adaptation",
  booktitle = iccv,
  year = 	 "2017",
}
```

For questions please contact Philip Haeusser (haeusser@cs.tum.edu).

