from functools import partial

import jax
import jax.nn as jnn
import jax.lax as lax 
import jax.numpy as jnp
import jax.random as jrand

import graphax as gx

from synaptax.neuron_models import SNN_LIF
from synaptax.experiments.shd.eprop import make_eprop_timeloop
from synaptax.experiments.shd.bptt import make_bptt_timeloop


def test_gradient_equivalence():
    # Define the initial weights:
    key = jrand.PRNGKey(0)
    W = jrand.normal(key, (8, 8))
    W_out = jrand.normal(key, (4, 8))

    # Define the initial hidden state:
    z0 = jnp.zeros(8)
    u0 = jnp.zeros(8)
    G_W0 = jnp.zeros((8, 8))
    W_out0 = jnp.zeros((4, 8))

    # Cross-entropy loss
    def ce_loss(z, tgt, W_out):
        out = W_out @ z
        probs = jnn.softmax(out) 
        return -jnp.dot(tgt, jnp.log(probs))

    # Define the eprop and bptt steps:
    eprop_grads = make_eprop_timeloop(SNN_LIF, ce_loss, unroll=5)
    bptt_timeloop = make_bptt_timeloop(SNN_LIF, ce_loss, unroll=5)

    @partial(jax.jacrev, argnums=(4, 5), has_aux=True)
    def bptt_grads(in_seq, target, z0, u0, _W, _W_out):
        losses = bptt_timeloop(in_seq, target, z0, u0, _W, _W_out)
        loss = jnp.mean(losses)
        return loss, loss

    # Define the input sequence:
    in_seq = jrand.normal(key, (16, 10, 8))
    in_seq = jnp.where(in_seq > 0., 1., 0.)

    # Define the target sequence:
    target = jrand.normal(key, (16, 4))

    # Run the eprop and bptt steps:
    eprop_loss, eprop_W_grad, eprop_W_out_grad = eprop_grads(in_seq, target, z0, u0, W, W_out, G_W0, W_out0)
    eprop_loss = jnp.mean(eprop_loss)
    eprop_W_grad = jnp.mean(eprop_W_grad, axis=0)
    eprop_W_out_grad = jnp.mean(eprop_W_out_grad, axis=0)
    grads, bptt_loss = bptt_grads(in_seq, target, z0, u0, W, W_out)
    bptt_W_grad, bptt_W_out_grad = grads

    print("eprop", eprop_loss)
    print("bptt", bptt_loss)

    print("eprop", eprop_W_grad)
    print("bptt", bptt_W_grad)


    # Check if the gradients are equivalent:
    assert jnp.allclose(eprop_loss, bptt_loss)
    assert jnp.allclose(eprop_W_grad, bptt_W_grad)
    assert jnp.allclose(eprop_W_out_grad, bptt_W_out_grad)


test_gradient_equivalence()

