import time
import yaml
import argparse
from functools import partial
from tqdm import tqdm

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand


import equinox as eqx

import optax

from synaptax.neuron_models import SNN_LIF, SNN_rec_LIF, SNN_Sigma_Delta, SNN_ALIF
from synaptax.experiments.shd.bptt import make_bptt_step, make_bptt_rec_step, make_bptt_step_ALIF
from synaptax.experiments.shd.eprop import make_eprop_step, make_eprop_rec_step, make_eprop_step_ALIF, make_stupid_eprop_step_ALIF
from synaptax.custom_dataloaders import load_shd_or_ssc

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="../config/params.yaml", type=str, help="Path to the configuration .yaml file.")
args = parser.parse_args()

EPOCHS = 10
SEED = 42
key = jrand.PRNGKey(SEED)
wkey, woutkey, key = jrand.split(key, 3)

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
BATCH_SIZE = 1 # config_dict["hyperparameters"]["batch_size"]
NUM_TIMESTEPS = config_dict["hyperparameters"]["timesteps"]
NUM_HIDDEN = 4 # config_dict["hyperparameters"]["hidden"]
PATH = config_dict["dataset"]["folder_path"]
NUM_WORKERS = config_dict["dataset"]["num_workers"]
NUM_LABELS = 20
INPUT_POOLING = config_dict["dataset"]["input_pooling"]
NUM_CHANNELS = 700//INPUT_POOLING
BURNIN_STEPS = config_dict["hyperparameters"]["burnin_steps"]
LOOP_UNROLL = config_dict["hyperparameters"]["loop_unroll"]
TRAIN_ALGORITHM = config_dict["train_algorithm"]


# Function for downsampling
pooling = eqx.nn.Pool(kernel_size=INPUT_POOLING, stride=INPUT_POOLING, 
                      num_spatial_dims=1, init=0, operation=lax.add)


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


z0 = jnp.zeros(NUM_HIDDEN)
u0 = jnp.zeros_like(z0)
a0 = jnp.zeros_like(u0)

init_fn = jnn.initializers.orthogonal(jnp.sqrt(2))
W_out_init_fn = jnn.initializers.xavier_normal()
W = init_fn(wkey, (NUM_HIDDEN, NUM_CHANNELS))
V = jnp.zeros((NUM_HIDDEN, NUM_HIDDEN))
W_out = W_out_init_fn(woutkey, (NUM_LABELS, NUM_HIDDEN))

G_W0 = jnp.zeros((NUM_HIDDEN, NUM_CHANNELS))
G_W_a0 = jnp.zeros((NUM_HIDDEN, NUM_CHANNELS))
G_W_u0_stupid = jnp.zeros((NUM_HIDDEN, NUM_HIDDEN, NUM_CHANNELS))
G_W_a0_stupid = jnp.zeros((NUM_HIDDEN, NUM_HIDDEN, NUM_CHANNELS))
G_V0 = jnp.zeros((NUM_HIDDEN, NUM_HIDDEN))
W_out0 = jnp.zeros((NUM_LABELS, NUM_HIDDEN))
W0 = jnp.zeros((NUM_HIDDEN, NUM_CHANNELS))

optim = optax.chain(optax.adamw(LEARNING_RATE, eps=1e-7, weight_decay=1e-4), 
                    optax.clip_by_global_norm(.5))
model = neuron_model_dict[NEURON_MODEL]

weights = (W, W_out) # For non-recurrent case.
opt_state = optim.init(weights)

def run_test(weights, opt_state):
    new_opt_state = opt_state
    new_opt_state_stupid = opt_state
    new_weights = weights
    new_weights_stupid = weights

    eprop_step_fn = make_eprop_step_ALIF(SNN_ALIF, optim, ce_loss, unroll=1)
    stupid_eprop_step_fn = make_stupid_eprop_step_ALIF(SNN_ALIF, optim, ce_loss, unroll=1)

    for ep in range(EPOCHS):
        pbar = tqdm(train_loader)
        for data, target_batch, lengths in pbar:
            in_batch = jnp.array(data.numpy()).squeeze()
            in_batch = pooling(in_batch)
            target_batch = jnp.array(target_batch.numpy())
            target_batch = jnn.one_hot(target_batch, NUM_LABELS)

            loss, new_weights, new_opt_state, equivalence_acc = jax.jit(eprop_step_fn)(in_batch, target_batch, new_opt_state, z0, u0, a0, G_W0, G_W_a0, W0, W_out0, new_weights)
            loss_stupid, new_weights_stupid, new_opt_state_stupid, equivalence_acc_stupid = jax.jit(stupid_eprop_step_fn)(in_batch, target_batch, new_opt_state_stupid, z0, u0, a0, G_W_u0_stupid, G_W_a0_stupid, W0, W_out0, new_weights_stupid)

            diff = jnp.abs(equivalence_acc[0] - equivalence_acc_stupid[0])
            print("diff: \n", diff)
            print(jnp.mean(loss))

run_test(weights, opt_state)