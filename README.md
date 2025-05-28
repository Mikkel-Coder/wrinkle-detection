
# 1. Generate dataset:
## Install simulation:
```sh
python3.11 -m venv .venv11
source .venv11/bin/activate
pip install -r src/sim/requirements.txt
```
## Generate dataset with:
This takes about 30 min.
```sh
python3.11 -m src.sim.main
```

# 2. Train model
## Installation
Please update the requirements depending on your CUDA version!
```sh
python3.13 -m venv .venv13
source .venv13/bin/activate
pip install -r requirements.txt
```
## Train the model
This takes a **VERY LONG TIME** approximately 4 days!
```sh
python3 -m src.ml.train
```

# 3. Generate data 
using the same `.venv13` run:
```sh
python3 -m src.ml.test
```