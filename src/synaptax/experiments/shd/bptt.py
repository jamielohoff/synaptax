from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu


### LIF BPTT
def make_bptt_timeloop(model, loss_fn, unroll: int = 10):
    def SNN_bptt_timeloop(in_seq, tgt, z0, u0, W, W_out):
        def loop_fn(carry, in_seq):
            z, u, loss = carry
            next_z, next_u = model(in_seq, z, u, W)
            # By neglecting the gradient wrt. S, we basically compute only the 
            # implicit recurrence, but not the explicit recurrence
            loss += loss_fn(next_z, tgt, W_out)
            new_carry = (next_z, next_u, loss)
            return new_carry, None

        final_carry, _ = lax.scan(loop_fn, (z0, u0, 0.), in_seq, unroll=unroll)
        _, _, loss = final_carry
        return loss 

    return jax.vmap(SNN_bptt_timeloop, in_axes=(0, 0, None, None, None, None))


def make_bptt_step(model, optim, loss_fn, unroll: int = 10):

    timeloop_fn = make_bptt_timeloop(model, loss_fn, unroll)

    @partial(jax.jacrev, argnums=(4, 5), has_aux=True)
    def bptt_loss_and_grad(in_seq, target, z0, u0, _W, _W_out):
        losses = timeloop_fn(in_seq, target, z0, u0, _W, _W_out)
        loss = jnp.mean(losses)
        return loss, loss

    @jax.jit
    def bptt_train_step(in_seq, target, opt_state, weights, z0, u0):
        _W, _W_out = weights
        grads, loss = bptt_loss_and_grad(in_seq, target, z0, u0, _W, _W_out)
        updates, opt_state = optim.update(grads, opt_state, params=weights)
        weights = jtu.tree_map(lambda x, y: x + y, weights, updates)
        return loss, weights, opt_state
    
    return bptt_train_step



### Recurrent LIF BPTT
def make_bptt_rec_timeloop(model, loss_fn, unroll: int = 10):
    def rec_SNN_bptt_timeloop(in_seq, tgt, z0, u0, W, V, W_out):
        def loop_fn(carry, in_seq):
            z, u, loss = carry
            next_z, next_u = model(in_seq, z, u, W, V)
            # By neglecting the gradient wrt. S, we basically compute only the 
            # implicit recurrence, but not the explicit recurrence
            loss += loss_fn(next_z, tgt, W_out)
            new_carry = (next_z, next_u, loss)
            return new_carry, None

        final_carry, _ = lax.scan(loop_fn, (z0, u0, 0.), in_seq, unroll=unroll)
        _, _, loss = final_carry
        return loss 


    return jax.vmap(rec_SNN_bptt_timeloop, in_axes=(0, 0, None, None, None, None, None))

def make_bptt_rec_step(model, optim, loss_fn, unroll: int = 10):

    timeloop_fn = make_bptt_timeloop(model, loss_fn, unroll)

    @partial(jax.jacrev, argnums=(4, 5, 6), has_aux=True)
    def recurrent_bptt_loss_and_grad(in_seq, target, z0, u0, _W, _V, _W_out):
        losses = timeloop_fn(in_seq, target, z0, u0, _W, _V, _W_out)
        loss = jnp.mean(losses)
        return loss, loss

    @jax.jit
    def recurrent_bptt_train_step(in_seq, target, opt_state, weights, z0, u0):
        _W, _V, _W_out = weights
        grads, loss = recurrent_bptt_loss_and_grad(in_seq, target, z0, u0, _W, _V, _W_out)
        updates, opt_state = optim.update(grads, opt_state, params=weights)
        weights = jtu.tree_map(lambda x, y: x + y, weights, updates)
        return loss, weights, opt_state
    
    return recurrent_bptt_train_step

