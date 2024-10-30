import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu

import graphax as gx


mean_axis0 = lambda x: jnp.mean(x, axis=0)


def make_eprop_timeloop(model, loss_fn, unroll: int = 10):
    """
    G: Accumulated gradient of U (hidden state) w.r.t. parameters W and V.
    F: Immediate gradient of U (hidden state) w.r.t. parameters W and V.
    H: Gradient of current U w.r.t. previous timestep U.
    in_seq: (batch_dim, num_timesteps, sensor_size)
    """
    # TODO: check gradient equivalence!
    def SNN_eprop_timeloop(in_seq, target, z0, u0, W, W_out, G_W0, W_out0):
        # NOTE: we might have a vanishing gradient problem here!
        def loop_fn(carry, in_seq):
            z, u, G_W_val, W_grad_val, W_out_grad_val, loss = carry
            outputs, grads = gx.jacve(model, order = "rev", argnums=(2, 3), has_aux=True, sparse_representation=True)(in_seq, z, u, W)
            next_z, next_u = outputs
            # grads contain the gradients of z_next and u_next w.r.t. u, W and V. But we ignore grads of z (explicit recurrence) 
            u_grads = grads[1] # only gradient of u_next w.r.t. u, W and V.

            F_W = u_grads[1] # gradients of u_next w.r.t. W and V respectively.
            G_W = F_W.copy(G_W_val) # G_W_val is the gradient of prev. timestep w.r.t. W.

            H_I = u_grads[0] # grad. of u_next w.r.t. previous timestep u.
            G_W = H_I * G_W + F_W

            _loss, loss_grads = gx.jacve(loss_fn, order = "rev", argnums=(0, 2), has_aux=True, sparse_representation=True)(next_z, target, W_out)
            loss += _loss
            loss_grad, W_out_grad = loss_grads[0], loss_grads[1]
            W_grad = loss_grad * G_W

            W_grad_val += W_grad.val
            W_out_grad_val += W_out_grad.val

            new_carry = (next_z, next_u, G_W.val, W_grad_val, W_out_grad_val, loss)
            return new_carry, None
        
        init_carry = (z0, u0, G_W0, G_W0, W_out0, .0)
        final_carry, _ = lax.scan(loop_fn, init_carry, in_seq, unroll=unroll)
        _, _, _, W_grad, W_out_grad, loss = final_carry
        return loss, W_grad, W_out_grad

    return jax.vmap(SNN_eprop_timeloop, in_axes=(0, 0, None, None, None, None, None, None))


def make_eprop_step(model, optim, loss_fn, unroll: int = 10):
    timeloop_fn = make_eprop_timeloop(model, loss_fn, unroll)

    @jax.jit
    def eprop_train_step(in_batch, target, opt_state, weights, z0, u0, G_W0, W_out0):
        _W, _W_out = weights
        loss, W_grad, W_out_grad = timeloop_fn(in_batch, target, z0, u0, _W, _W_out, G_W0, W_out0)
        # take the mean across the batch dim for all gradient updates
        grads = tuple(map(mean_axis0, (W_grad, W_out_grad)))
        updates, opt_state = optim.update(grads, opt_state, params=weights)
        weights = jtu.tree_map(lambda x, y: x + y, weights, updates)
        return loss, weights, opt_state
    
    return eprop_train_step


def make_eprop_rec_timeloop(model, loss_fn, unroll: int = 10):
    """
    G: Accumulated gradient of U (hidden state) w.r.t. parameters W and V.
    F: Immediate gradient of U (hidden state) w.r.t. parameters W and V.
    H: Gradient of current U w.r.t. previous timestep U.
    in_seq: (batch_dim, num_timesteps, sensor_size)
    """
    # TODO: check gradient equivalence!
    def SNN_eprop_rec_timeloop(in_seq, target, z0, u0, W, V, W_out, G_W0, G_V0, W_out0):
        def loop_fn(carry, in_seq):
            z, u, G_W_val, G_V_val, W_grad_val, V_grad_val, W_out_grad_val, loss = carry
            outputs, grads = gx.jacve(model, order = "rev", argnums=(2, 3, 4), has_aux=True, sparse_representation=True)(in_seq, z, u, W, V)
            next_z, next_u = outputs
            # grads contain the gradients of z_next and u_next w.r.t. u, W and V. But we ignore grads of z (explicit recurrence) 
            u_grads = grads[1] # only gradient of u_next w.r.t. u, W and V.
            # W is the 
            F_W, F_V = u_grads[1], u_grads[2] # gradients of u_next w.r.t. W and V respectively.
            G_W = F_W.copy(G_W_val) # G_W_val is the gradient of prev. timestep u w.r.t. W.
            G_V = F_V.copy(G_V_val) # G_V_val is the gradient of prev. timestep u w.r.t. V.

            H_I = u_grads[0] # grad. of u_next w.r.t. previous timestep u.
            G_W = H_I * G_W + F_W
            G_V = H_I * G_V + F_V # no recurrent weights

            _loss, loss_grads = gx.jacve(loss_fn, order = "rev", argnums=(0, 2), has_aux=True, sparse_representation=True)(next_z, target, W_out)
            loss += _loss
            loss_grad, W_out_grad = loss_grads[0], loss_grads[1]
            W_grad = loss_grad * G_W
            V_grad = loss_grad * G_V

            W_grad_val += W_grad.val
            V_grad_val += V_grad.val
            W_out_grad_val += W_out_grad.val

            new_carry = (next_z, next_u, G_W.val, G_V.val, W_grad_val, V_grad_val, W_out_grad_val, loss)
            return new_carry, None
        
        init_carry = (z0, u0, G_W0, G_V0, G_W0, G_V0, W_out0, .0)
        final_carry, _ = lax.scan(loop_fn, init_carry, in_seq, unroll=unroll)
        _, _, _, _, W_grad, V_grad, W_out_grad, loss = final_carry
        return loss, W_grad, V_grad, W_out_grad

    return jax.vmap(SNN_eprop_rec_timeloop, in_axes=(0, 0, None, None, None, None, None, None, None, None))


def make_eprop_rec_step(model, optim, loss_fn, unroll: int = 10):
    timeloop_fn = make_eprop_rec_timeloop(model, loss_fn, unroll)

    @jax.jit
    def recurrent_eprop_train_step(in_batch, target, opt_state, weights, z0, u0, G_W0, G_V0, W_out0):
        _W, _V, _W_out = weights
        loss, W_grad, V_grad, W_out_grad = timeloop_fn(in_batch, target, z0, u0, _W, _V, _W_out, G_W0, G_V0, W_out0)
        # take the mean across the batch dim for all gradient updates
        grads = map(mean_axis0, (W_grad, V_grad, W_out_grad)) 
        updates, opt_state = optim.update(grads, opt_state, params=weights)
        weights = jtu.tree_map(lambda x, y: x + y, weights, updates)
        return loss, weights, opt_state
    
    return recurrent_eprop_train_step

