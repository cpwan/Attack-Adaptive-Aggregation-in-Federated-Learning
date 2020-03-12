# Robust-Federated-Learning

Program entry: `main.py`
Usage:
- FedAvg no attack:
`python main.py --GAR fedavg`

- DeepGAR no attack:
`python main.py --GAR deepGAR`

To invoke an attack, set `n_attacker_XXXX` to a positive value. e.g.
`python main.py --GAR fedavg --n_attacker_backdoor 1`
This will build an experiment with 1 backdoor attacker.

To save the model weight for training the aggregation network, turn on the `save_model_weights` flag
`python main.py --save_model_weights`

For more configuration, do `python main.py --help`

>
The Aggregation network is not yet finalized. To train the aggregation network, 
`python train_aggNet.py`

The trained network will be saved to `./aggNet/`.

To change the aggregation network used for defensing the federated attacks, change the `path_to_aggNet` in server.py
