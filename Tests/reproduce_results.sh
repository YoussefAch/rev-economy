#!/usr/bin/env bash


# exctract features, learn classifiers, and save predictions
python extract_learn_store.py --pathToInputParams experiments/experiment1/inputParams.json


# eco-gamma baseline
python test.py --pathToInputParams experiments/experiment1/inputParams.json --pathToOutputs ./experiments/experiment1/


# eco-rev-cu and eco-rev-ca
python test_rev.py --pathToInputParams experiments/experiment2/inputParams.json --pathToOutputs ./experiments/experiment2/


# figures
python Figures_paper_ijcai.py --pathToInputParams experiments/experiment2/inputParams.json


# supplementary material experiments datalink : https://drive.google.com/file/d/1T80jjviBd8nCGYnRnIBshBr85qRseOWp/view?usp=sharing
python test_rev.py --pathToInputParams experiments/experiment4/inputParams.json --pathToOutputs ./experiments/experiment4/

