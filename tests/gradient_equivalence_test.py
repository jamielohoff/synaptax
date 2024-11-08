import unittest
from functools import partial

import jax
import jax.nn as jnn
import jax.lax as lax 
import jax.numpy as jnp
import jax.random as jrand

import graphax as gx

from synaptax.neuron_models import SNN_LIF
from synaptax.experiments.shd.eprop import make_eprop_timeloop, make_stupid_eprop_timeloop
from synaptax.experiments.shd.bptt import make_bptt_timeloop


# jax.config.update("jax_disable_jit", True)


class TestGradientEquivalence(unittest.TestCase):
    def test_gradient_equivalence_loop(self):
        # Define the initial weights:
        key = jrand.PRNGKey(43)
        # Weird difference in gradient is random-seed dependent!
        W = jrand.normal(key, (8, 8))
        W_out = jrand.normal(key, (4, 8))

        # Define the initial hidden state:
        z0 = jnp.zeros(8)
        u0 = jrand.uniform(key, 8)
        G_W0 = jnp.zeros((8, 8))
        W_out0 = jnp.zeros((4, 8))

        # Cross-entropy loss
        def ce_loss(z, tgt, W_out):
            out = jnp.dot(W_out, z)
            probs = jnn.softmax(out) 
            return -jnp.dot(tgt, jnp.log(probs))

        # Define the eprop and bptt steps:
        eprop_grads = make_eprop_timeloop(SNN_LIF, ce_loss, unroll=1, burnin_steps=0)
        bptt_timeloop = make_bptt_timeloop(SNN_LIF, ce_loss, unroll=1, burnin_steps=0)

        @partial(jax.jacrev, argnums=(4, 5), has_aux=True)
        def bptt_grads(in_seq, target, z0, u0, _W, _W_out):
            losses = bptt_timeloop(in_seq, target, z0, u0, _W, _W_out)
            loss = jnp.mean(losses)
            return loss, loss

        # Define the input sequence:
        batch_size = 1
        T = 10
        in_seq = jnp.zeros((batch_size, T, 8)) # jrand.normal(key, (batch_size, T, 8))
        in_seq = jnp.where(in_seq > 0., 1., 0.)
        in_seq = in_seq.at[0, 0, 0].set(1.) # put a single spike in the input sequence

        # Define the target sequence:
        target = jrand.randint(key, batch_size, minval=0, maxval=4)
        target = jnn.one_hot(target, 4)

        # Run the eprop and bptt steps:
        eprop_loss, eprop_W_grad, eprop_W_out_grad = eprop_grads(in_seq, target, z0, u0, W, W_out, G_W0, W_out0)
        eprop_loss = jnp.mean(eprop_loss)
        eprop_W_grad = jnp.mean(eprop_W_grad, axis=0)
        eprop_W_out_grad = jnp.mean(eprop_W_out_grad, axis=0)
        
        grads, bptt_loss = bptt_grads(in_seq, target, z0, u0, W, W_out)
        bptt_W_grad, bptt_W_out_grad = grads

        # print("eprop W_grad", eprop_W_grad)
        # print("bptt W_grad", bptt_W_grad)
        delta = eprop_W_grad - bptt_W_grad
        print("delta", jnp.where(delta < 1e-5, 0., eprop_W_grad - bptt_W_grad))
        
        # Check if the gradients are equivalent:
        self.assertTrue(jnp.allclose(eprop_loss, bptt_loss))
        self.assertTrue(jnp.allclose(eprop_W_grad, bptt_W_grad))
        self.assertTrue(jnp.allclose(eprop_W_out_grad, bptt_W_out_grad))
    
    # def test_gradient_equivalence_single_step(self):
    #     # Define the initial weights:
    #     key = jrand.PRNGKey(0)
    #     W = jrand.normal(key, (8, 8))

    #     # Define the initial hidden state:
    #     z0 = jnp.zeros(8)
    #     u0 = jnp.zeros(8)

    #     graphax_grad_fn = gx.jacve(SNN_LIF, order = "rev", argnums=(2, 3), sparse_representation=False)
    #     jax_grad_fn = jax.jacrev(SNN_LIF, argnums=(2, 3))

    #     # Define the input sequence:
    #     x = jrand.normal(key, 8)
    #     x = jnp.where(x > 0., 1., 0.)

    #     # Run the eprop and bptt steps:
    #     graphax_grads = graphax_grad_fn(x, z0, u0, W)
    #     jax_grads = jax_grad_fn(x, z0, u0, W)
        
    #     # Check if the gradients are equivalent:
    #     self.assertTrue(gx.tree_allclose(graphax_grads[0], jax_grads[0]))
    #     self.assertTrue(gx.tree_allclose(graphax_grads[1], jax_grads[1]))
    
    # def test_loss_fn_gradient_equivalence(self):
    #     # Define the initial weights:
    #     key = jrand.PRNGKey(0)
    #     W_out = jrand.normal(key, (4, 8))

    #     # Cross-entropy loss
    #     def ce_loss(z, tgt, W_out):
    #         out = jnp.dot(W_out, z)
    #         probs = jnn.softmax(out) 
    #         return -jnp.dot(tgt, jnp.log(probs))
        
    #     z = jrand.normal(key, 8)
    #     z = jnp.where(z > 0., 1., 0.)

    #     # Define the target sequence:
    #     target = jrand.randint(key, (), minval=0, maxval=4)
    #     target = jnn.one_hot(target, 4)

    #     # Define the eprop and bptt steps:
    #     graphax_loss_grad = gx.jacve(ce_loss, order="rev", argnums=(0, 2), sparse_representation=False)(z, target, W_out)
    #     jax_loss_grad = jax.jacrev(ce_loss, argnums=(0, 2))(z, target, W_out)

    #     self.assertTrue(gx.tree_allclose(graphax_loss_grad, jax_loss_grad))


if __name__ == "__main__":
    unittest.main()

