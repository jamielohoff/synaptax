from functools import partial
import argparse
# import wandb
import torch
from tqdm import tqdm

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.profiler as profiler

import optax

from synaptax.neuron_models import SNN_LIF, SNN_rec_LIF, SNN_Sigma_Delta, SNN_ALIF
from synaptax.experiments.shd.bptt import make_bptt_step, make_bptt_rec_step, make_bptt_step_ALIF
from synaptax.experiments.shd.eprop import make_eprop_step, make_eprop_rec_step, make_eprop_step_ALIF
from synaptax.custom_dataloaders import load_shd_or_ssc

import yaml

#jax.config.update("jax_disable_jit", True)

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--path", default="./data/shd", type=str, help="Path to the dataset.")
parser.add_argument("-c", "--config", default="./config/params.yaml", type=str, help="Path to the configuration yaml file.")
parser.add_argument("-s", "--seed", default=0, type=int, help="Random seed.")
parser.add_argument("-e", "--epochs", default=100, type=int, help="Number of epochs.")

args = parser.parse_args()

SEED = args.seed
key = jrand.PRNGKey(SEED)
torch.manual_seed(SEED)

# Read experiment config file for parameters
with open(args.config, "r") as file:
    config_dict = yaml.safe_load(file)

neuron_model_dict = {
    "SNN_LIF": SNN_LIF,
    "SNN_rec_LIF": SNN_rec_LIF,
    "SNN_Sigma_Delta": SNN_Sigma_Delta,
    "SNN_ALIF": SNN_ALIF
}

NEURON_MODEL = config_dict["neuron_model"]
LEARNING_RATE = config_dict["hyperparameters"]["learning_rate"]
BATCH_SIZE = config_dict["hyperparameters"]["batch_size"]
NUM_TIMESTEPS = config_dict["hyperparameters"]["timesteps"]
EPOCHS = args.epochs
NUM_HIDDEN = config_dict["hyperparameters"]["hidden"]
PATH = config_dict["dataset"]["folder_path"]
NUM_WORKERS = config_dict["dataset"]["num_workers"]
NUM_LABELS = 20
NUM_CHANNELS = 700
BURNIN_STEPS = config_dict["hyperparameters"]["burnin_steps"]
LOOP_UNROLL = config_dict["hyperparameters"]["loop_unroll"]
TRAIN_ALGORITHM = config_dict["train_algorithm"]


# Initialize wandb:
# wandb.login()

# run = wandb.init(
#     # Set the project where this run will be logged
#     project=config_dict["task"],
#     # Track hyperparameters and run metadata
#     config={
#         "learning_rate": LEARNING_RATE,
#         "epochs": EPOCHS,
#     },
# )


train_loader = load_shd_or_ssc("shd", PATH, "train", BATCH_SIZE, 
                                nb_steps=NUM_TIMESTEPS, shuffle=True,
                                workers=NUM_WORKERS)
test_loader = load_shd_or_ssc("shd", PATH, "test", BATCH_SIZE, 
                                nb_steps=NUM_TIMESTEPS, shuffle=True,
                                workers=NUM_WORKERS)

# Cross-entropy loss
def ce_loss(z, tgt, W_out):
    out = W_out @ z
    probs = jnn.softmax(out) 
    return -jnp.dot(tgt, jnp.log(probs + 1e-8))


def predict(in_seq, model, weights, z0, u0, a0):
    W_out = weights[0]
    Ws = weights[1:]
    def loop_fn(carry, x):
        z, u, a, out_total = carry
        z_next, u_next, a_next = model(x, z, u, a, *Ws)
        out_total += jnp.dot(W_out, z_next)
        carry = (z_next, u_next, a_next, out_total)
        return carry, None

    final_carry, _ = lax.scan(loop_fn, (z0, u0, a0, jnp.zeros(NUM_LABELS)), in_seq)
    out = final_carry[-1]
    return jnp.argmax(out, axis=0)


# Test for one batch:
@partial(jax.jit, static_argnums=2)
def eval_step(in_batch, labels, model, weights, z0, u0, a0):
    preds_batch = jax.vmap(predict, in_axes=(0, None, None, None, None, None))(in_batch, model, weights, z0, u0, a0)
    return (preds_batch == labels).mean()


# Test loop
def eval_model(data_loader, model, weights, z0, u0, a0):
    accuracy_batch, num_iters = 0, 0
    for data, labels, lengths in data_loader:
        in_batch = jnp.array(data.numpy()).squeeze()
        target_batch = jnp.array(labels.numpy())
        accuracy_batch += eval_step(in_batch, target_batch, model, weights, z0, u0, a0)
        num_iters += 1
    return accuracy_batch / num_iters


### Making train loop
z0 = jnp.zeros(NUM_HIDDEN)
u0 = jnp.zeros_like(z0)
a0 = jnp.zeros_like(u0)

wkey, woutkey = jrand.split(key, 2)

init_fn = jnn.initializers.orthogonal(jnp.sqrt(2))
W_out_init_fn = jnn.initializers.xavier_normal()
W = init_fn(wkey, (NUM_HIDDEN, NUM_CHANNELS))
V = jnp.zeros((NUM_HIDDEN, NUM_HIDDEN))
W_out = W_out_init_fn(woutkey, (NUM_LABELS, NUM_HIDDEN))

G_W0 = jnp.zeros((NUM_HIDDEN, NUM_CHANNELS))
G_W_a0 = jnp.zeros_like(G_W0)
G_V0 = jnp.zeros((NUM_HIDDEN, NUM_HIDDEN))
W_out0 = jnp.zeros((NUM_LABELS, NUM_HIDDEN))

optim = optax.chain(optax.adamw(LEARNING_RATE, eps=1e-7, weight_decay=1e-4), 
                    optax.clip_by_global_norm(.5))
# weights = (W, V, W_out) # For recurrence
weights = (W_out, W) # For no recurrence
opt_state = optim.init(weights)
model = neuron_model_dict[NEURON_MODEL]
# step_fn = make_eprop_step(model, optim, ce_loss, unroll=10, burnin_steps=BURNIN_STEPS)
# step_fn = make_eprop_rec_step(model, optim, ce_loss, unroll=10)
# step_fn = make_bptt_step(model, optim, ce_loss, unroll=10, burnin_steps=BURNIN_STEPS)
# step_fn = make_bptt_rec_step(model, optim, ce_loss, unroll=10)
# step_fn = make_bptt_ALIF_step(model, optim, ce_loss, unroll=10)
step_fn = make_eprop_step_ALIF(model, optim, ce_loss, unroll=10, burnin_steps=BURNIN_STEPS)

# Training loop
def run_experiment(partial_step_fn, weights, opt_state):
    for ep in range(EPOCHS):
        pbar = tqdm(train_loader)
        for data, target_batch, lengths in pbar:
            in_batch = jnp.array(data.numpy()).squeeze()
            target_batch = jnp.array(target_batch.numpy())
            target_batch = jnn.one_hot(target_batch, NUM_LABELS)

            loss, weights, opt_state = partial_step_fn(data=in_batch, 
                                                       weights=weights, 
                                                       labels=target_batch, 
                                                       opt_state=opt_state)
            
            pbar.set_description(f"Epoch: {ep + 1}, loss: {loss.mean() / NUM_TIMESTEPS}")
        
        train_acc = eval_model(train_loader, model, weights, z0, u0, a0)
        test_acc = eval_model(test_loader, model, weights, z0, u0, a0)
        print(f"Epoch: {ep + 1}, Train Acc: {train_acc}, Test Acc: {test_acc}")
    
    return weights
        

def run_eprop():
    weights = (W_out, W) # For non-recurrent case.
    opt_state = optim.init(weights)
    step_fn = make_eprop_step(model, optim, ce_loss, 
                              unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
    partial_step_fn = partial(step_fn, z0=z0, u0=u0, G_W0=G_W0, W_out0=W_out0)
    trained_weights = run_experiment(partial_step_fn, weights, opt_state)
    return trained_weights

def run_eprop_rec():
    weights = (W, V, W_out) # For recurrent case.
    opt_state = optim.init(weights)
    step_fn = make_eprop_rec_step(model, optim, ce_loss, 
                                  unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
    partial_step_fn = partial(step_fn, z0=z0, u0=u0, G_W0=G_W0, G_V0=G_V0, W_out0=W_out0)
    trained_weights = run_experiment(partial_step_fn, weights, opt_state)
    return trained_weights

def run_bptt():
    weights = (W_out, W) # For non-recurrent case.
    opt_state = optim.init(weights)
    step_fn = make_bptt_step(model, optim, ce_loss, 
                              unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
    partial_step_fn = partial(step_fn, z0=z0, u0=u0)
    trained_weights = run_experiment(partial_step_fn, weights, opt_state)
    return trained_weights

def run_bptt_rec():
    weights = (W_out, W, V) # For recurrent case.
    opt_state = optim.init(weights)
    step_fn = make_bptt_rec_step(model, optim, ce_loss, 
                              unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
    partial_step_fn = partial(step_fn, z0=z0, u0=u0)
    trained_weights = run_experiment(partial_step_fn, weights, opt_state)
    return trained_weights

def run_eprop_alif():
    weights = (W_out, W) # For recurrent case.
    opt_state = optim.init(weights)
    step_fn = make_eprop_step_ALIF(model, optim, ce_loss, 
                                    unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
    partial_step_fn = partial(step_fn, z0=z0, u0=u0, a0=a0, G_W_u0=G_W0, G_W_a0=G_W0, W_out0=W_out0)
    trained_weights = run_experiment(partial_step_fn, weights, opt_state)
    return trained_weights

def run_bptt_alif():
    weights = (W_out, W) # For non-recurrent case.
    opt_state = optim.init(weights)
    step_fn = make_bptt_step_ALIF(model, optim, ce_loss, 
                              unroll=LOOP_UNROLL, burnin_steps=BURNIN_STEPS)
    partial_step_fn = partial(step_fn, z0=z0, u0=u0, a0=a0)
    trained_weights = run_experiment(partial_step_fn, weights, opt_state)
    return trained_weights

train_algo_dict = {
    "eprop": run_eprop,
    "eprop_rec": run_eprop_rec,
    "bptt": run_bptt,
    "bptt_rec": run_bptt_rec,
    "bptt_alif": run_bptt_alif,
    "eprop_alif": run_eprop_alif
}

# profiler.start_trace("/tmp/tensorboard")
new_weights = train_algo_dict[TRAIN_ALGORITHM]()
# new_weights.block_until_ready()
# profiler.stop_trace()

