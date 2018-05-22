#! /bin/bash

# python liu_compare.py   --instance liu_high_CV --csvfile liu_high90_CV.csv --htmlfile liu_high90_CV.html --verbose --nfeature 90 --nsample 100 --nsim 1000

# python liu_compare.py   --instance liu_high_CV --csvfile liu_high50_CV.csv --htmlfile liu_high50_CV.html --verbose --nfeature 50 --nsample 100 --nsim 1000

# python liu_compare.py   --instance liu_low_CV --csvfile liu_low90_CV.csv --htmlfile liu_low90_CV.html --verbose --nfeature 90 --nsample 100 --nsim 1000

# python liu_compare.py   --instance liu_low_CV --csvfile liu_low50_CV.csv --htmlfile liu_low50_CV.html --verbose --nfeature 50 --nsample 100 --nsim 1000

python liu_compare.py   --instance liu_high --csvfile liu_high90.csv --htmlfile liu_high90.html --verbose --nfeature 90 --nsample 100 --nsim 1000

python liu_compare.py   --instance liu_high --csvfile liu_high50.csv --htmlfile liu_high50.html --verbose --nfeature 50 --nsample 100 --nsim 1000

python liu_compare.py   --instance liu_low --csvfile liu_low90.csv --htmlfile liu_low90.html --verbose --nfeature 90 --nsample 100 --nsim 1000

python liu_compare.py   --instance liu_low --csvfile liu_low50.csv --htmlfile liu_low50.html --verbose --nfeature 50 --nsample 100 --nsim 1000

