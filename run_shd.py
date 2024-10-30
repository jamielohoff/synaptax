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

import optax

from synaptax.neuron_models import SNN_LIF, SNN_rec_LIF, SNN_Sigma_Delta
from synaptax.experiments.shd.bptt import make_bptt_step, make_bptt_rec_step
from synaptax.experiments.shd.eprop import make_eprop_step, make_eprop_rec_step
from synaptax.custom_dataloaders import load_shd_or_ssc


parser = argparse.ArgumentParser()

parser.add_argument("-n", "--neuron_model", default="SNN_LIF", type=str, help="Neuron type for the model.")
parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float, help="Learning rate for the model.")
parser.add_argument("-bs", "--batch_size", default=128, type=int, help="Batch size for the model.")
parser.add_argument("-ts", "--timesteps", default=100, type=int, help="Number of timesteps for the model.")
parser.add_argument("-hd", "--hidden", default=256, type=int, help="Hidden layer size for the model.")
parser.add_argument("-e", "--epochs", default=10, type=int, help="Number of epochs for the model.")
parser.add_argument("-d", "--path", default="./data/shd", type=str, help="path to the dataset.")
parser.add_argument("-s", "--seed", default=0, type=int, help="Random seed.")

args = parser.parse_args()


SEED = args.seed
key = jrand.PRNGKey(SEED)
torch.manual_seed(SEED)

NEURON_MODEL = args.neuron_model
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
NUM_TIMESTEPS = args.timesteps
EPOCHS = args.epochs
NUM_HIDDEN = args.hidden
PATH = args.path
NUM_LABELS = 20
NUM_CHANNELS = 700

# # Initialize wandb:
# wandb.login()

# run = wandb.init(
#     # Set the project where this run will be logged
#     project="synaptax-project",
#     # Track hyperparameters and run metadata
#     config={
#         "learning_rate": args.learning_rate,
#         "epochs": args.epochs,
#     },
# )


train_loader = load_shd_or_ssc("shd", PATH, "train", BATCH_SIZE, 
                                nb_steps=NUM_TIMESTEPS, shuffle=True)
test_loader = load_shd_or_ssc("shd", PATH, "test", BATCH_SIZE, 
                                nb_steps=NUM_TIMESTEPS, shuffle=True)


# Cross-entropy loss
def ce_loss(z, tgt, W_out):
    out = W_out @ z
    probs = jnn.softmax(out) 
    return -jnp.dot(tgt, jnp.log(probs))


def predict(in_seq, model, weights, z0, u0):
    W_out = weights[-1]
    Ws = weights[:-1]
    def loop_fn(carry, x):
        z, u, z_total = carry
        z_next, u_next = model(x, z, u, *Ws)
        z_total += W_out @ z_next
        carry = (z_next, u_next, z_total)
        return carry, None

    final_carry, _ = lax.scan(loop_fn, (z0, u0, jnp.zeros(NUM_LABELS)), in_seq)
    out = final_carry[2]
    # probs = jax.nn.softmax(out) # not necessary to use softmax here
    return jnp.argmax(out, axis=0)


# Test for one batch:
@partial(jax.jit, static_argnums=2)
def eval_step(in_batch, target_batch, model, weights, z0, u0):
    preds_batch = jax.vmap(predict, in_axes=(0, None, None, None, None))(in_batch, model, weights, z0, u0)
    return (preds_batch == target_batch).mean()


# Test loop
def eval_model(data_loader, model, weights, z0, u0):
    accuracy_batch, num_iters = 0, 0
    for data, target_batch, lengths in data_loader:
        in_batch = jnp.array(data.numpy()).squeeze()
        target_batch = jnp.array(target_batch.numpy())
        accuracy_batch += eval_step(in_batch, target_batch, model, weights, z0, u0)
        num_iters += 1
    return accuracy_batch / num_iters


### Making train loop
z0 = jnp.zeros(NUM_HIDDEN)
u0 = jnp.zeros(NUM_HIDDEN)

wkey, vkey, woutkey, G_W_key, G_V_key = jrand.split(key, 5)

def xavier_normal(key, shape):
    # Calculate the standard deviation for Xavier normal initialization
    fan_in, fan_out = shape
    stddev = jnp.sqrt(2.0 / (fan_in + fan_out))
    
    # Generate random numbers from a normal distribution
    return stddev * jrand.normal(key, shape)

init_fn = xavier_normal # jax.nn.initializers.orthogonal() # jax.nn.initializers.he_normal()

W = init_fn(wkey, (NUM_HIDDEN, NUM_CHANNELS))
V = init_fn(vkey, (NUM_HIDDEN, NUM_HIDDEN))
W_out = init_fn(woutkey, (NUM_LABELS, NUM_HIDDEN))

G_W0 = init_fn(G_W_key, (NUM_HIDDEN, NUM_CHANNELS))
G_V0 = init_fn(G_V_key, (NUM_HIDDEN, NUM_HIDDEN))
W_out0 = jnp.zeros((NUM_LABELS, NUM_HIDDEN))

optim = optax.adam(LEARNING_RATE)
# weights = (W, V, W_out)
weights = (W, W_out)
opt_state = optim.init(weights)
model = SNN_LIF
step_fn = make_eprop_step(model, optim, ce_loss, unroll=NUM_TIMESTEPS)


# Training loop
for ep in range(EPOCHS):
    pbar = tqdm(train_loader)
    for data, target_batch, lengths in pbar:
        in_batch = jnp.array(data.numpy()).squeeze()
        target_batch = jnp.array(target_batch.numpy())
        target_batch = jnn.one_hot(target_batch, NUM_LABELS)

        # just comment out "bptt" with "eprop" to switch between the two training methods
        # loss, weights, opt_state = recurrent_eprop_train_step(in_batch, target_batch, opt_state, weights, G_W0, G_V0)
        loss, weights, opt_state = step_fn(in_batch, target_batch, opt_state, weights, z0, u0, G_W0, W_out0)
        # loss, weights, opt_state = bptt_train_step(in_batch, target_batch, opt_state, weights)
        # loss, weights, opt_state = step_fn(in_batch, target_batch, opt_state, weights, z0, u0)
        pbar.set_description(f"Epoch: {ep + 1}, loss: {loss.mean() / NUM_TIMESTEPS}")
    
    train_acc = eval_model(train_loader, model, weights, z0, u0)
    test_acc = eval_model(test_loader, model, weights, z0, u0)
    print(f"Epoch: {ep + 1}, Train Acc: {train_acc}, Test Acc: {test_acc}")
    
