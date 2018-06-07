A test suite for full model FDR and power comparing
selective (conditional) inference and knockoffs.

To install
----------

```
pip install traitlets
git submodule update --init;
cd R-selective;
git submodule update --init;
make install;
cd ../regreg;
git submodule update --init;
pip install -r requirements.txt;
pip install -e .;
cd ../selective-inference;
git submodule update --init;
pip install -r requirements.txt;
pip install -e .;
cd ..;
```

Testing
-------

If installed correctly, the following should give you some results for a small problem:

```
python compare_fdr.py --instance AR_instance --nsample 100 --nfeature 50 --nsignal 10 --methods lee_theory liu_theory --htmlfile AR.html --csvfile AR.csv --verbose --nsim 20 --rho 0. --signal 4.
```

For a grid of signal and / or AR(1) parameters try

```
python compare_fdr.py --instance AR_instance --nsample 100 --nfeature 50 --nsignal 10 --methods lee_theory liu_theory --htmlfile AR.html --signal 3 4 5 --rho 0 0.5 --csvfile AR_grid.csv --verbose --nsim 20
```

Also, try 

```
python compare_fdr.py --help
python compare_fdr.py --list_instances
python compare_fdr.py --list_methods
 ```

