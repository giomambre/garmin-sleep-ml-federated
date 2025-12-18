# server.py
import copy
import torch


def fed_avg(global_model, client_states, client_sizes):
    # Media pesata dei pesi del modello: chi ha più esempi pesa di più
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]

    new_state = copy.deepcopy(global_model.state_dict())
    for k in new_state.keys():
        stacked = torch.stack([cs[k] * w for cs, w in zip(client_states, weights)])
        new_state[k] = stacked.sum(dim=0)

    global_model.load_state_dict(new_state)

