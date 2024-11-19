import time
import yaml
import argparse
from functools import partial
from tqdm import tqdm

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import jax.profiler as profiler

import optax

from synaptax.neuron_models import SNN_LIF, SNN_rec_LIF, SNN_Sigma_Delta, SNN_ALIF
from synaptax.experiments.shd.bptt import make_bptt_step, make_bptt_rec_step, make_bptt_step_ALIF
from synaptax.experiments.shd.eprop import make_eprop_step, make_eprop_rec_step, make_eprop_step_ALIF


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="../config/params.yaml", type=str, help="Path to the configuration .yaml file.")
args = parser.parse_args()

NUM_EVALUATIONS = 10
SEED = 42
key = jrand.PRNGKey(SEED)

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
NUM_HIDDEN = config_dict["hyperparameters"]["hidden"]
PATH = config_dict["dataset"]["folder_path"]
NUM_WORKERS = config_dict["dataset"]["num_workers"]
NUM_LABELS = 20
NUM_CHANNELS = 140
BURNIN_STEPS = config_dict["hyperparameters"]["burnin_steps"]
LOOP_UNROLL = config_dict["hyperparameters"]["loop_unroll"]
TRAIN_ALGORITHM = config_dict["train_algorithm"]


# Cross-entropy loss
def ce_loss(z, tgt, W_out):
    out = W_out @ z
    probs = jnn.softmax(out) 
    return -jnp.dot(tgt, jnp.log(probs + 1e-8))


### Making train loop
z0 = jnp.zeros(NUM_HIDDEN)
u0 = jnp.zeros_like(z0)
a0 = jnp.zeros_like(u0)

wkey, woutkey, key = jrand.split(key, 3)

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
model = neuron_model_dict[NEURON_MODEL]

def run_experiment(partial_step_fn, weights, opt_state):
    in_batch = jrand.uniform(key, (BATCH_SIZE, NUM_TIMESTEPS, NUM_CHANNELS))
    target_batch = jrand.uniform(key, (BATCH_SIZE, NUM_LABELS))

    loss, weights, opt_state = jax.jit(partial_step_fn)(data=in_batch, 
                                                        weights=weights, 
                                                        labels=target_batch, 
                                                        opt_state=opt_state)
    
    profiler.start_trace("/tmp/tensorboard")
    start_time = time.time()
    for _ in tqdm(range(NUM_EVALUATIONS)):
        in_batch = jrand.uniform(key, (BATCH_SIZE, NUM_TIMESTEPS, NUM_CHANNELS))
        target_batch = jrand.uniform(key, (BATCH_SIZE, NUM_LABELS))

        loss, weights, opt_state = jax.jit(partial_step_fn)(data=in_batch, 
                                                            weights=weights, 
                                                            labels=target_batch, 
                                                            opt_state=opt_state)
    jax.block_until_ready(weights)
    profiler.stop_trace()
    print("Average time:", (time.time() - start_time)/NUM_EVALUATIONS)

    
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

new_weights = train_algo_dict[TRAIN_ALGORITHM]()

    