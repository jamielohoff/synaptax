from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu

from .bptt import make_bptt_timeloop, make_bptt_timeloop_ALIF, make_bptt_rec_timeloop


def make_rtrl_step(model, optim, loss_fn, unroll: int = 10, burnin_steps: int = 30):
    # Maps through training examples:
    timeloop_fn = make_bptt_timeloop(model, loss_fn, unroll, burnin_steps)

    @partial(jax.jacfwd, argnums=(4, 5), has_aux=True)
    def rtrl_loss_and_grad(in_seq, target, z0, u0, _W_out, _W):
        losses = timeloop_fn(in_seq, target, z0, u0, _W_out, _W)
        loss = jnp.mean(losses)
        return loss, loss

    @jax.jit
    def rtrl_train_step(in_batch, target_batch, opt_state, weights, z0, u0):
        # _W, _W_out, _beta, _threshold = weights
        grads, loss = rtrl_loss_and_grad(in_batch, target_batch, z0, u0, *weights)
        updates, opt_state = optim.update(grads, opt_state, params=weights)
        weights = jtu.tree_map(lambda x, y: x + y, weights, updates)
        return loss, weights, opt_state
    
    return rtrl_train_step


def make_rtrl_step_ALIF(model, optim, loss_fn, unroll: int = 10, burnin_steps: int = 30):
    # Maps through training examples:
    timeloop_fn = make_bptt_timeloop_ALIF(model, loss_fn, unroll, burnin_steps)

    @partial(jax.jacfwd, argnums=(5, 6), has_aux=True)
    def rtrl_loss_and_grad(data, labels, z0, u0, a0, _W_out, _W):
        losses = timeloop_fn(data, labels, z0, u0, a0, _W_out, _W)
        loss = jnp.mean(losses)
        return loss, loss

    def rtrl_train_step_ALIF(data, labels, opt_state, weights, z0, u0, a0):
        # _W, _W_out, _beta, _threshold = weights
        grads, loss = rtrl_loss_and_grad(data, labels, z0, u0, a0, *weights)
        updates, opt_state = optim.update(grads, opt_state, params=weights)
        weights = jtu.tree_map(lambda x, y: x + y, weights, updates)
        return loss, weights, opt_state
    
    return rtrl_train_step_ALIF

