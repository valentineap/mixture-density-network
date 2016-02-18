# README #
***
Andrew Valentine, Universiteit Utrecht, 2016.
a.p.valentine@uu.nl
***

This package implements the Mixture Density Network (MDN) of Bishop (1995). Implementation choices are inspired by (but not identical to) those described in Kaufl (2015) and Kaufl et al. (2016). Specifically, the package currently includes:

* 'Committees' of independent MDNs, with
* Randomised network sizes (numbers of Gaussian kernels and hidden units);
* Network training via the L-BFGS algorithm;
* Optimal weight selection via 'early stopping', based on an independent set of monitoring data;
* Support for parallel training of committee members via OpenMP.

## Package contents ##

* `mdn_train` - Construct and train a committee of MDNs
* `mdn_apply` - Apply a trained MDN committee to make predictions for a given dataset

### mdn_train ###

Basic usage:

    mdn_train [options] inputfile monitorfile

This constructs an MDN based on training data from `inputfile` (see below for a description of the file format). Data from `monitorfile` is used to prevent overtraining. All other parameters are set to default values. A full list of options, and default values, can be obtained by running

    mdn_train --help

The number of independent MDNs contained within the committee may be specified in two equivalent ways, both resulting (here) in a 25-member committee:

    mdn_train -n 25 inputfile monitorfile
    mdn_train --n-ct-members=25 inputfile monitorfile

Each MDN represents the posterior PDF via a Gaussian mixture model. The number of Gaussian kernels is randomly chosen, from a uniform distribution, with the range controlled by two option flags specifying the lower and upper limits. Thus, to use between 5 and 15 Gaussian kernels (inclusive), we would specify

    mdn_train -j 5 -k 15 inputfile monitorfile
    mdn_train --n-kern-min=5 --nkern-max=15 inputfile monitorfile

Of course, the minimum must be less than or equal to the maximum.

Each MDN is based upon a neural network with one hidden layer. The number of nodes within this layer is also selected randomly. To set this range so that there are between 20 and 50 neurons within the hidden layer, we choose

    mdn_train -l 20 -u 50 inputfile monitorfile
    mdn_train --n-hid-min=20 --n-hid-max 50 inputfile monitorfile

At present, a fixed number of training iterations is performed. To request 500 L-BFGS iterations per MDN, specify

    mdn_train -t 500 inputfile monitorfile
    mdn_train --max-iter=500 inputfile monitorfile

At present there are no criteria to enable training to terminate before this number of iterations has been performed. This is obviously a potential performance limitation, and should be rectified in the future.

Once training is complete, the committee is written to disk. The filename can be specified

    mdn_train -o network.out inputfile monitorfile
    mdn_train --output-file=network.out inputfile monitorfile

This can then be used by mdn_apply.

### mdn_apply ###

Basic usage:

    mdn_apply [options] networkfile datafile

This reads the MDN committee from `networkfile`, and applies it to the data in `datafile`. Again, see below for a description of the file format. If no further options are given, all parameters take default values. For a quick summary, see

    mdn_apply --help

By default, it is assumed that `datafile` contains inputs, but no targets. If `datafile` does, in fact, also contain target values, this can be specified using

    mdn_apply --has-targets networkfile datafile

If this option is not provided, mdn_apply will exit with an error, since it will interpret the targets as an additional input parameter for each example.

Two forms of output are available from mdn_apply; both are written to a file as specified by

    mdn_apply -o out.dat networkfile datafile
    mdn_apply --output-file=out.dat

By default, the program writes out the weights, means, and standard deviations required to fully specify the Gaussian Mixture Model corresponding to each set of inputs in `datafile`. The n-th line of the output file corresponds to the example in the n-th line of `datafile`, and contains a sequence

    wt1 mean1 std1 wt2 mean2 std2 ... wtN meanN stdN

 In many cases, it will be convenient to copy the inputs (and/or targets) from `datafile` into the output file. This can be achieved using the options

     mdn_apply --copy-inputs networkfile datafile
     mdn_apply --has-targets --copy-targets networkfile datafile
     mdn_apply --has-targets --copy-inputs --copy-targets networkfile datafile

When these options are used, the n-th line of the output files will be, respectively

    inp1 inp2 ...inpN wt1 mean1 std1 ... wtN meanN stdN
    target wt1 mean1 std1 ... wtN meanN stdN
    inp1 inp2 ...inpN target wt1 mean1 std1 ... wtN meanN stdN

The second form of output provides the posterior PDF (p(m|d), i.e. the GMM) evaluated at a sequence of points. This is selected using the `--output-pdf` flag. The evaluation points are chosen to be evenly distributed in the range [LO,UP], with N points in total; to specify these values, run

    mdn_apply --output-pdf --m-lo=LO --m-up=UP --m-num=N networkfile datafile

In this case, the n-th line in the output file will contain N values, with the i-th value corresponding to p( LO + (i-1) x (UP-LO) / (N-1) |d). Again, the options `--copy-inputs` and `--copy-targets` may be used to prepend input and/or target values to each line of output.

### Input data file format ###

The various data files read by `mdn_train` and `mdn_apply` share a common file format. Input files should be in ascii format, and have one example per line. Here, an 'example' consists of an N-dimensional input vector, and (possibly) a one-dimensional target value. Each line of the input file should therefore contain a sequence of values corresponding to the elements of the input vector, separated by whitespace; then, if appropriate, a single target value. A typical file will therefore look something like

    inp11 inp12 ... inp1N target1
    inp21 inp22 ... inp2N target2
      :     :    :    :      :   
    inpM1 inpM2 ... inpMN targetM

There is no need to 'standardise' inputs and targets. `mdn_train` converts all input and target vectors to be zero-mean, with unit standard deviation, and the necessary transformation is treated as an inherent property of the MDN committee. Thus, the same transformation is applied by `mdn_apply`, which then rescales the committee outputs to ensure that they remain valid and correctly normalised in the original (input, unstandardised) data space.

### Parallel operation ###

The code has OpenMP support to exploit any availability of multiple processors. If OpenMP functionality is enabled on a given system, training of individual committee members will be distributed across all available cores. The number of parallel threads is (typically) governed by the environment variable `OMP_NUM_THREADS`. In principle, this should be set to be equal to the number of available cores. However, a lower number may result in better performance in cases where the ratio of cores to available memory is low. At present the code does not detect and handle such a situation.


## To-do ##

* Include more sophisticated heuristics to determine when training should be terminated. One obvious possibility is to track the number of iterations without improved performance on the monitoring dataset, and terminate once this passes some threshold.
* Support for mini-batch training.
* Committee weights should be determined using a (third) independent dataset, and not the monitoring set as at present.
* Support for binary data file formats.
* More intelligent setting of OpenMP options.

## References ##

* Bishop, C.M., 1995. Neural networks for pattern recognition, Oxford University Press.
* Kaufl, P., 2015. Rapid probabilistic source inversion using pattern recognition, PhD Thesis, Universiteit Utrecht.
* Kaufl, P., Valentine, A.P., de Wit, R.W.L. and Trampert, J., 2016. Solving probabilistic inverse problems rapidly with prior samples, Geophys. J. Int.