import graphax as gx
import jax
import jax.numpy as jnp
import jax.random as jrand
import equinox as eqx
import tonic
import torch
import optax

from functools import partial
from tqdm import tqdm
import argparse
from neuron_models import SNN_LIF, SNN_Sigma_Delta
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--neuron_model', default='SNN_LIF', type=str, help='Neuron type for the model.')
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='Learning rate for the model.')
parser.add_argument('-bs', '--batch_size', default=128, type=int, help='Batch size for the model.')
parser.add_argument('-ts', '--timesteps', default=100, type=int, help='Number of timesteps for the model.')
parser.add_argument('-hd', '--hidden', default=128, type=int, help='Hidden layer size for the model.')
parser.add_argument('-e', '--epochs', default=10, type=int, help='Number of epochs for the model.')
parser.add_argument('-ds', '--dataset', default='SHD', type=str, help='Dataset used for the model.')

args = parser.parse_args()

NEURON_MODEL = args.neuron_model
LEARNING_RATE = args.learning_rate

# Handling the dataset:
BATCH_SIZE = args.batch_size
NUM_TIMESTEPS = args.timesteps
EPOCHS = args.epochs
NUM_HIDDEN = args.hidden

# Initialize wandb:
wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project="synaptax-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    },
)

if args.dataset == 'SHD':
    NUM_LABELS = 20
    NUM_CHANNELS = 700
    SENSOR_SIZE = tonic.datasets.SHD.sensor_size
    frame_transform = tonic.transforms.ToFrame(sensor_size=SENSOR_SIZE, n_time_bins=NUM_TIMESTEPS)

    transform = tonic.transforms.Compose([frame_transform,])
    train_set = tonic.datasets.SHD(save_to='/Users/kaya/Datasets', train=True, transform=transform)
    test_set = tonic.datasets.SHD(save_to='/Users/kaya/Datasets', train=False, transform=transform)

    torch.manual_seed(42)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

# Use cross-entropy loss
def loss_fn(z, tgt, W_out):
    out = W_out @ z
    probs = jax.nn.softmax(out) 
    loss_ = -jnp.dot(tgt, jnp.log(probs)) # cross-entopy loss
    return loss_
    
'''
G: Accumulated gradient of U (hidden state) w.r.t. parameters W and V.
F: Immediate gradient of U (hidden state) w.r.t. parameters W and V.
H: Gradient of current U w.r.t. previous timestep U.
'''

'''
in_seq: (batch_dim, num_timesteps, sensor_size)
'''

def SNN_eprop_timeloop(in_seq, target, start_state, W, V, W_out, G_W0, G_V0):
    def loop_fn(carry, in_):
        state, G_W_val, G_V_val, W_grad_val, V_grad_val, W_out_grad_val, loss = carry

        if NEURON_MODEL == 'SNN_LIF':
            z, u = state
            outputs, grads = gx.jacve(SNN_LIF, order = 'rev', argnums=(2, 3, 4), has_aux=True, sparse_representation=True)(in_, z, u, W, V)
            next_z, next_u = outputs
            new_state = tuple(outputs)
            hidden_grads = grads[1] # only gradient of u_next w.r.t. u, W and V. IN GENERAL : Gradient of all hidden states (not z), W and V. 
        elif NEURON_MODEL == 'SNN_Sigma_Delta':
            z, e, u_mem, s, i = state
            outputs, grads = outputs, grads = gx.jacve(SNN_Sigma_Delta, order = 'rev', argnums=(3, 6, 7), has_aux=True, sparse_representation=True)(in_, z, e, u_mem, s, i, W, V)
            next_z, next_e, next_u_mem, next_s, next_i = outputs
            new_state = tuple(outputs)
            hidden_grads = grads[2]

        # W is the 
        F_W, F_V = hidden_grads[1], hidden_grads[2] # gradients of u_next w.r.t. W and V respectively.
        G_W = F_W.copy(G_W_val) # G_W_val is the gradient of prev. timestep u w.r.t. W.
        G_V = F_V.copy(G_V_val) # G_V_val is the gradient of prev. timestep u w.r.t. V.

        H_I = hidden_grads[0] # grad. of u_next w.r.t. previous timestep u.
        G_W = H_I * G_W + F_W
        G_V = H_I * G_V + F_V

        # spike output must be in the 0th element of new_state:
        _loss, loss_grads = gx.jacve(loss_fn, order = 'rev', argnums=(0, 2), has_aux=True, sparse_representation=True)(new_state[0], target, W_out)
        loss += _loss
        loss_grad, W_out_grad = loss_grads[0], loss_grads[1]
        W_grad = loss_grad * G_W
        V_grad = loss_grad * G_V

        W_grad_val += W_grad.val
        V_grad_val += V_grad.val
        W_out_grad_val += W_out_grad.val

        new_carry = (new_state, G_W.val, G_V.val, W_grad_val, V_grad_val, W_out_grad_val, loss)
        return new_carry, None
    final_carry, _ = jax.lax.scan(loop_fn, (start_state, G_W0, G_V0, G_W0, G_V0, jnp.zeros((NUM_LABELS, NUM_HIDDEN)), .0), in_seq, length=NUM_TIMESTEPS)
    _, _, _, W_grad, V_grad, W_out_grad, loss = final_carry
    return loss, W_grad, V_grad, W_out_grad

batch_vmap_SNN_eprop_timeloop = jax.vmap(SNN_eprop_timeloop, in_axes=(0, 0, None, None, None, None, None, None))

if NEURON_MODEL == 'SNN_LIF':
    z0 = jnp.zeros(NUM_HIDDEN)
    u0 = jnp.zeros(NUM_HIDDEN)
    start_state = (z0, u0)
elif NEURON_MODEL == 'SNN_Sigma_Delta':
    z0 = jnp.zeros(NUM_HIDDEN)
    e0 = jnp.zeros(NUM_HIDDEN)
    u_mem0 = jnp.zeros(NUM_HIDDEN)
    s0 = jnp.zeros(NUM_HIDDEN)
    i0 = jnp.zeros(NUM_HIDDEN)
    start_state = (z0, e0, u_mem0, s0, i0)

key = jrand.PRNGKey(0)
wkey, vkey, G_W_key, G_V_key, woutkey = jrand.split(key, 5)

init_ = jax.nn.initializers.orthogonal()

W = init_(wkey, (NUM_HIDDEN, NUM_CHANNELS))
V = init_(vkey, (NUM_HIDDEN, NUM_HIDDEN))
W_out = init_(woutkey, (NUM_LABELS, NUM_HIDDEN))

G_W0 = init_(G_W_key, (NUM_HIDDEN, NUM_CHANNELS))
G_V0 = init_(G_V_key, (NUM_HIDDEN, NUM_HIDDEN))

optim = optax.adam(LEARNING_RATE)
weights = (W, V, W_out)
opt_state = optim.init(weights)

def SNN_bptt_timeloop(in_seq, tgt, z0, u0, W, V, W_out):
    def loop_fn(carry, in_seq):
        z, u, loss = carry
        next_z, next_u = SNN_LIF(in_seq, z, u, W, V)
        # By neglecting the gradient wrt. S, we basically compute only the 
        # implicit recurrence, but not the explicit recurrence
        loss += loss_fn(next_z, tgt, W_out)
        new_carry = (next_z, next_u, loss)
        return new_carry, None
    
    # TODO: implement pytree handling for SparseTensor types
    carry, _ = jax.lax.scan(loop_fn, (z0, u0, 0.), in_seq, length=NUM_TIMESTEPS)
    z, v, loss = carry
    return loss 


batch_vmap_SNN_bptt_timeloop = jax.vmap(SNN_bptt_timeloop, in_axes=(0, 0, None, None, None, None, None))

@partial(jax.jacrev, argnums=(4, 5, 6), has_aux=True)
def loss_and_grad(in_seq, target, start_state, _W, _V, _W_out):
    losses = batch_vmap_SNN_eprop_timeloop(in_seq, target, start_state, _W, _V, _W_out)
    #losses = batch_vmap_SNN_bptt_timeloop(in_seq, target, z0,u0, _W, _V, _W_out)
    return jnp.mean(losses), jnp.mean(losses) # has to return this twice so that it returns loss and grad!

# Train for one batch:
@jax.jit
def eprop_train_step(in_batch, target, opt_state, weights, G_W0, G_V0):
    _W, _V, _W_out = weights
    loss, W_grad, V_grad, W_out_grad = batch_vmap_SNN_eprop_timeloop(in_batch, target, start_state, _W, _V, _W_out, G_W0, G_V0)
    grads = (jnp.mean(W_grad, axis=0), jnp.mean(V_grad, axis=0), jnp.mean(W_out_grad, axis=0)) # take the mean across the batch dim for all gradient updates
    updates, opt_state = optim.update(grads, opt_state, weights)
    weights = jax.tree_util.tree_map(lambda x, y: optax.apply_updates(x, y), weights, updates)
    return loss, weights, opt_state


@jax.jit
def bptt_train_step(in_seq, target, opt_state, weights):
    _W, _V, _W_out = weights
    grads, loss = loss_and_grad(in_seq, target, start_state, _W, _V, _W_out)
    updates, opt_state = optim.update(grads, opt_state)
    weights = jax.tree_util.tree_map(lambda x, y: x + y, weights, updates)
    return loss, weights, opt_state


def predict(in_seq, weights):
    W, V, W_out = weights
    def loop_fn(carry, in_):
        if NEURON_MODEL == 'SNN_LIF':
            z, u = carry
            z_next, u_next = SNN_LIF(in_, z, u, W, V)
            carry = (z_next, u_next)
        elif NEURON_MODEL == 'SNN_Sigma_Delta':
            z, e, u_mem, s, i = carry
            z_next, e_next, u_mem_next, s_next, i_next = SNN_Sigma_Delta(in_, z, e, u_mem, s, i, W, V)
            carry = (z_next, e_next, u_mem_next, s_next, i_next)
        return carry, carry

    _, carry_final = jax.lax.scan(loop_fn, start_state, in_seq) # loop over the timesteps
    z_sum = jnp.zeros(carry_final[0][0].shape)
    def sum_spikes(z_sum, carry_step):
        z_sum += carry_step[0]
        return z_sum, _
    z_sum, _ = jax.lax.scan(sum_spikes, z_sum, carry_final) # sum the spike outputs through timesteps.
    out = W_out @ z_sum
    probs = jax.nn.softmax(out)
    pred = jnp.argmax(probs, axis=0)
    return pred

# Test for one batch:
@jax.jit
def accuracy_step(in_batch, target_batch, weights):
    preds_batch = jax.vmap(predict, in_axes=(0, None))(in_batch, weights) # vmap over batch dimension
    return (preds_batch == target_batch).mean()

# Test loop
def test_():
    accuracy_batch = 0.
    num_iters = 0
    for data, target_batch in tqdm(test_loader):
        in_batch = jnp.array(data.numpy()).squeeze()
        target_batch = jnp.array(target_batch.numpy())
        accuracy_batch += accuracy_step(in_batch, target_batch, weights)
        num_iters += 1
    accuracy = accuracy_batch / num_iters
    return accuracy

# Training loop
for ep_ in range(EPOCHS):
    for data, target_batch in tqdm(train_loader):
        in_batch = jnp.array(data.numpy()).squeeze()
        target_batch = jnp.array(target_batch.numpy())
        target_batch_one_hot = jax.nn.one_hot(target_batch, NUM_LABELS)
        # just comment out 'bptt' with 'eprop' to switch between the two training methods
        loss, weights, opt_state = eprop_train_step(in_batch, target_batch_one_hot, opt_state, weights, G_W0, G_V0)
        #loss, weights, opt_state = bptt_train_step(in_batch, target_batch, opt_state, weights)
        print("Epoch: ", ep_ + 1, ", loss: ", loss.mean())
        wandb.log({"train accuracy": accuracy_step(in_batch, target_batch, weights), "loss": loss.mean()})
    wandb.log({"test accuracy": test_()})
