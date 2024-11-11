import unittest
from functools import partial

import jax
import jax.nn as jnn
import jax.lax as lax 
import jax.numpy as jnp
import jax.random as jrand

import graphax as gx

from synaptax.neuron_models import SNN_LIF, SNN_ALIF
from synaptax.experiments.shd.eprop import make_eprop_timeloop, make_stupid_eprop_timeloop, make_eprop_timeloop_ALIF
from synaptax.experiments.shd.bptt import make_bptt_timeloop, make_bptt_timeloop_ALIF


# jax.config.update("jax_disable_jit", True)


class TestGradientEquivalence(unittest.TestCase):
    def test_gradient_equivalence_loop_LIF(self):
        num_labels = 4
        num_inputs = 4
        num_hidden = 4
        # Define the initial weights:
        key = jrand.PRNGKey(123)
        # NOTE: Weird difference in gradient is random-seed dependent!
        init_fn = jnn.initializers.orthogonal(jnp.sqrt(2))
        W = init_fn(key, (num_hidden, num_inputs))
        W_out = init_fn(key, (num_labels, num_hidden))

        # Define the initial hidden state:
        z0 = jrand.uniform(key, (num_hidden,))
        z0 = jnp.where(z0 > .8, 1., 0.)
        u0 = jrand.uniform(key, (num_hidden,))
        G_W0 = jnp.zeros_like(W)
        W_out0 = jnp.zeros_like(W_out)

        # Cross-entropy loss
        def ce_loss(z, tgt, W_out):
            out = jnp.dot(W_out, z)
            probs = jnn.softmax(out) 
            return -jnp.dot(tgt, jnp.log(probs + 1e-8))

        # Define the eprop and bptt steps:
        eprop_grads = make_eprop_timeloop(SNN_LIF, ce_loss, unroll=1, burnin_steps=0)
        # stupid_eprop_grads = make_stupid_eprop_timeloop(SNN_LIF, ce_loss, unroll=1, burnin_steps=0)
        bptt_timeloop = make_bptt_timeloop(SNN_LIF, ce_loss, unroll=1, burnin_steps=0)

        @partial(jax.jacrev, argnums=(4, 5), has_aux=True)
        def bptt_grads(in_seq, target, z0, u0, _W_out, _W):
            losses = bptt_timeloop(in_seq, target, z0, u0, _W_out, _W)
            loss = jnp.mean(losses)
            return loss, loss

        # Define the input sequence:
        batch_size = 1
        T = 3
        in_seq = jrand.normal(key, (batch_size, T, num_inputs))
        in_seq = jnp.where(in_seq > .8, 1., 0.)

        # Define the target sequence:
        target = jrand.randint(key, (batch_size,) , minval=0, maxval=num_labels)
        target = jnn.one_hot(target, num_labels)

        # Run the eprop and bptt steps:
        eprop_loss, eprop_W_out_grad, eprop_W_grad = eprop_grads(in_seq, target, z0, u0, G_W0, W_out0, W_out, W)
        # stupid_eprop_loss, stupid_eprop_W_grad, stupid_eprop_W_out_grad = stupid_eprop_grads(in_seq, target, z0, u0, W, W_out, jnp.zeros((8, 8, 8)), W_out0)

        eprop_loss = jnp.mean(eprop_loss)
        eprop_W_grad = jnp.mean(eprop_W_grad, axis=0)
        # stupid_eprop_W_grad = jnp.mean(stupid_eprop_W_grad, axis=0)
        eprop_W_out_grad = jnp.mean(eprop_W_out_grad, axis=0)
        grads, bptt_loss = bptt_grads(in_seq, target, z0, u0, W_out, W)
        bptt_W_out_grad, bptt_W_grad = grads

        # print("eprop W_grad", eprop_W_grad)
        # print("bptt W_grad", bptt_W_grad)
        delta = eprop_W_grad - bptt_W_grad
        # delta_2 = stupid_eprop_W_grad - bptt_W_grad
        print("delta", jnp.where(delta < 1e-8, 0., delta))
        # print("delta_2", jnp.where(delta_2 < 1e-8, 0., delta_2))
        
        # Check if the gradients are equivalent:
        self.assertTrue(jnp.allclose(eprop_loss, bptt_loss))
        self.assertTrue(jnp.allclose(eprop_W_grad, bptt_W_grad))
        self.assertTrue(jnp.allclose(eprop_W_out_grad, bptt_W_out_grad))
    
    # def test_gradient_equivalence_loop_ALIF(self):
    #     num_labels = 20
    #     num_inputs = 700
    #     num_hidden = 512
    #     # Define the initial weights:
    #     key = jrand.PRNGKey(123)
    #     # NOTE: Weird difference in gradient is random-seed dependent!
    #     init_fn = jnn.initializers.orthogonal(jnp.sqrt(2))
    #     W_out_init_fn = jnn.initializers.xavier_normal()
    #     W = init_fn(key, (num_hidden, num_inputs))
    #     W_out = W_out_init_fn(key, (num_labels, num_hidden))

    #     zkey, ukey, akey = jrand.split(key, 3)
    #     # Define the initial hidden state:
    #     z0 = jrand.uniform(zkey, (num_hidden,))
    #     z0 = jnp.where(z0 > .8, 1., 0.)
    #     u0 = jrand.uniform(ukey, (num_hidden,))
    #     a0 = jrand.uniform(akey, (num_hidden,))
    #     G_W0 = jnp.zeros_like(W)
    #     W_out0 = jnp.zeros_like(W_out)

    #     # Cross-entropy loss
    #     def ce_loss(z, tgt, W_out):
    #         out = jnp.dot(W_out, z)
    #         probs = jnn.softmax(out) 
    #         return -jnp.dot(tgt, jnp.log(probs + 1e-8))

    #     # Define the eprop and bptt steps:
    #     eprop_grads = make_eprop_timeloop_ALIF(SNN_ALIF, ce_loss, unroll=1, burnin_steps=0)
    #     bptt_timeloop = make_bptt_timeloop_ALIF(SNN_ALIF, ce_loss, unroll=1, burnin_steps=0)

    #     @partial(jax.jacrev, argnums=(5, 6), has_aux=True)
    #     def bptt_grads(in_seq, target, z0, u0, a0, _W_out, _W):
    #         losses = bptt_timeloop(in_seq, target, z0, u0, a0, _W_out, _W)
    #         loss = jnp.mean(losses)
    #         return loss, loss

    #     # Define the input sequence:
    #     batch_size = 16
    #     T = 100
    #     in_seq = jrand.normal(key, (batch_size, T, num_inputs))
    #     in_seq = jnp.where(in_seq > .8, 1., 0.)

    #     # Define the target sequence:
    #     target = jrand.randint(key, (batch_size,) , minval=0, maxval=num_labels)
    #     target = jnn.one_hot(target, num_labels)

    #     # Run the eprop and bptt steps:
    #     eprop_loss, eprop_W_out_grad, eprop_W_grad = eprop_grads(in_seq, target, z0, u0, a0, G_W0, G_W0, W_out0, W_out, W)
    #     # stupid_eprop_loss, stupid_eprop_W_grad, stupid_eprop_W_out_grad = stupid_eprop_grads(in_seq, target, z0, u0, W, W_out, jnp.zeros((8, 8, 8)), W_out0)

    #     eprop_loss = jnp.mean(eprop_loss)
    #     eprop_W_grad = jnp.mean(eprop_W_grad, axis=0)
    #     # stupid_eprop_W_grad = jnp.mean(stupid_eprop_W_grad, axis=0)
    #     eprop_W_out_grad = jnp.mean(eprop_W_out_grad, axis=0)
    #     grads, bptt_loss = bptt_grads(in_seq, target, z0, u0, a0, W_out, W)
    #     bptt_W_out_grad, bptt_W_grad = grads

    #     # print("eprop W_grad", eprop_W_grad)
    #     # print("bptt W_grad", bptt_W_grad)
    #     delta = eprop_W_grad - bptt_W_grad
    #     # delta_2 = stupid_eprop_W_grad - bptt_W_grad
    #     print("delta", jnp.where(delta < 1e-4, 0., delta))
    #     # print("delta_2", jnp.where(delta_2 < 1e-8, 0., delta_2))
        
    #     # Check if the gradients are equivalent:
    #     self.assertTrue(jnp.allclose(eprop_loss, bptt_loss))
    #     self.assertTrue(jnp.allclose(eprop_W_grad, bptt_W_grad))
    #     self.assertTrue(jnp.allclose(eprop_W_out_grad, bptt_W_out_grad))
    
    # def test_gradient_equivalence_single_step_LIF(self):
    #     # Define the initial weights:
    #     key = jrand.PRNGKey(0)
    #     W = jrand.normal(key, (8, 8))

    #     # Define the initial hidden state:
    #     z0 = jrand.uniform(key, (8,))
    #     z0 = jnp.where(z0 > .5, 1., 0.)
    #     u0 = jrand.uniform(key, (8,))

    #     graphax_grad_fn = gx.jacve(SNN_LIF, order = "rev", argnums=(2, 3), sparse_representation=False)
    #     jax_grad_fn = jax.jacrev(SNN_LIF, argnums=(2, 3))

    #     # Define the input sequence:
    #     x = jrand.normal(key, (8,))
    #     x = jnp.where(x > 0., 1., 0.)

    #     # Run the eprop and bptt steps:
    #     graphax_grads = graphax_grad_fn(x, z0, u0, W)
    #     jax_grads = jax_grad_fn(x, z0, u0, W)
    #     print(graphax_grads[0], jax_grads[0])
    #     # Check if the gradients are equivalent:
    #     self.assertTrue(gx.tree_allclose(graphax_grads[0], jax_grads[0]))
    #     self.assertTrue(gx.tree_allclose(graphax_grads[1], jax_grads[1]))
        
    # def test_gradient_equivalence_single_step_ALIF(self):
    #     # Define the initial weights:
    #     key = jrand.PRNGKey(0)
    #     W = jrand.normal(key, (8, 8))

    #     # Define the initial hidden state:
    #     z0 = jrand.uniform(key, (8,))
    #     z0 = jnp.where(z0 > .5, 1., 0.)
    #     u0 = jrand.uniform(key, (8, 2))

    #     graphax_grad_fn = jax.jit(gx.jacve(SNN_ALIF, order="rev", argnums=(2, 3), sparse_representation=False))
    #     sparse_graphax_grad_fn = jax.jit(gx.jacve(SNN_ALIF, order="rev", argnums=(2, 3), sparse_representation=True))
    #     jax_grad_fn = jax.jit(jax.jacrev(SNN_ALIF, argnums=(2, 3)))

    #     # Define the input sequence:
    #     x = jrand.normal(key, (8,))
    #     x = jnp.where(x > 0., 1., 0.)

    #     # Run the eprop and bptt steps:
    #     graphax_grads = graphax_grad_fn(x, z0, u0, W)
    #     sparse_graphax_grads = sparse_graphax_grad_fn(x, z0, u0, W)
    #     # print("sparse graphax grads:", sparse_graphax_grads[1])
    #     jax_grads = jax_grad_fn(x, z0, u0, W)
    #     # print(graphax_grads[1], jax_grads[1])
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

