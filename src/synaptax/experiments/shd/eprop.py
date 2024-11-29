import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu

import graphax as gx


mean_axis0 = lambda x: jnp.mean(x, axis=0)
surrogate = lambda x: jnn.sigmoid(10.*x)


def make_eprop_timeloop(model, loss_fn, unroll: int = 10, burnin_steps: int = 30):
    """
    G: Accumulated gradient of U (hidden state) w.r.t. parameters W and V.
    F: Immediate gradient of U (hidden state) w.r.t. parameters W and V.
    H: Gradient of current U w.r.t. previous timestep U.
    in_seq: (batch_dim, num_timesteps, sensor_size)
    """
    # TODO: check gradient equivalence!
    def SNN_eprop_timeloop(in_seq, target, z0, u0, G_W0, W_out0, W_out, W):
        def burnin_loop_fn(carry, in_seq):
            z, u = carry
            outputs = model(in_seq, z, u, W)
            next_z, next_u = outputs
            new_carry = (next_z, next_u)

            return new_carry, None

        def loop_fn(carry, in_seq):
            z, u, G_W_val, W_grad_val, W_out_grad_val, loss = carry
            outputs, grads = gx.jacve(model, order="rev", argnums=(2, 3), has_aux=True, sparse_representation=True)(in_seq, z, u, W)
            next_z, next_u = outputs
            # grads contain the gradients of z_next and u_next w.r.t. u, W and V. But we ignore grads of z (explicit recurrence) 
            H_I, F_W = grads[1] # only gradient of u_next w.r.t. u, W and V.
            
            G_W = F_W.copy(G_W_val) # G_W_val is the gradient of prev. timestep w.r.t. W.
            G_W_next = H_I * G_W + F_W
            # print("G_W_next", G_W_next)

            # This calculates dL/dz
            # We still need dz/du to calculate dL/du
            _loss, loss_grads = gx.jacve(loss_fn, order="rev", argnums=(0, 2), has_aux=True, sparse_representation=True)(next_z, target, W_out)
            loss += _loss
            loss_grad, W_out_grad = loss_grads[0], loss_grads[1]
            # TODO: Put a threshold variable there!
            dzdu = gx.jacve(surrogate, order="rev", sparse_representation=True)(next_u - 1.)
            loss_grad = loss_grad * dzdu
            W_grad = loss_grad * G_W_next

            W_grad_val += W_grad.val
            W_out_grad_val += W_out_grad.val

            new_carry = (next_z, next_u, G_W_next.val, W_grad_val, W_out_grad_val, loss)
            return new_carry, None
        
        burnin_init_carry = (z0, u0)
        burnin_carry, _ = lax.scan(burnin_loop_fn, burnin_init_carry, in_seq[:burnin_steps], unroll=1)
        z_burnin, u_burnin = burnin_carry[0], burnin_carry[1]

        init_carry = (z_burnin, u_burnin, G_W0, G_W0, W_out0, .0)
        final_carry, _ = lax.scan(loop_fn, init_carry, in_seq[burnin_steps:], unroll=unroll)
        _, _, _, W_grad, W_out_grad, loss = final_carry
        return loss, W_out_grad, W_grad

    return jax.vmap(SNN_eprop_timeloop, in_axes=(0, 0, None, None, None, None, None, None))


def make_eprop_step(model, optim, loss_fn, unroll: int = 10, burnin_steps: int = 30):
    timeloop_fn = make_eprop_timeloop(model, loss_fn, unroll, burnin_steps)

    def eprop_train_step(data, labels, opt_state, z0, u0, G_W0, W_out0, weights):
        loss, W_grad, W_out_grad = timeloop_fn(data, labels, z0, u0, G_W0, W_out0, *weights)
        # take the mean across the batch dim for all gradient updates
        grads = tuple(map(mean_axis0, (W_grad, W_out_grad)))
        updates, opt_state = optim.update(grads, opt_state, params=weights)
        weights = jtu.tree_map(lambda x, y: x + y, weights, updates)

        return loss, weights, opt_state
    
    return eprop_train_step


def make_eprop_timeloop_ALIF(model, loss_fn, unroll: int = 10, burnin_steps: int = 30):
    """
    G: Accumulated gradient of U (hidden state) w.r.t. parameters W and V.
    F: Immediate gradient of U (hidden state) w.r.t. parameters W and V.
    H: Gradient of current U w.r.t. previous timestep U.
    in_seq: (batch_dim, num_timesteps, sensor_size)
    """
    # TODO: check gradient equivalence!
    def SNN_eprop_timeloop_ALIF(in_seq, target, z0, u0, a0, G_W_u0, G_W_a0, W_out0, W_out, W):
        # def burnin_loop_fn(carry, in_seq):
        #     z, u, a = carry
        #     outputs = model(in_seq, z, u, a, W)
        #     next_z, next_u, next_a = outputs
        #     new_carry = (next_z, next_u, next_a)

        #     return new_carry, None

        def loop_fn(carry, in_seq):
            z, u, a, G_W_u_val, G_W_a_val, W_grad_val, W_out_grad_val, loss = carry
            outputs, grads = gx.jacve(model, order="rev", argnums=(2, 3, 4), has_aux=True, sparse_representation=True)(in_seq, z, u, a, W)
            next_z, next_u, next_a = outputs
            
            # grads contain the gradients of z_next and u_next w.r.t. u, W and V. But we ignore grads of z (explicit recurrence) 
            H_I_uu, H_I_ua, F_W_u = grads[1] # only gradient of u_next w.r.t. u, W and V.
            H_I_au, H_I_aa, F_W_a = grads[2] # only gradient of u_next w.r.t. u, W and V.
            
            G_W_u = F_W_u.copy(G_W_u_val) # G_W_val is the gradient of prev. timestep w.r.t. W.
            G_W_a = F_W_u.copy(G_W_a_val) # G_W_val is the gradient of prev. timestep w.r.t. W.
            # This should be optimized with a single operation using block-diagonality
            G_W_u_next = H_I_uu * G_W_u + H_I_ua * G_W_a + F_W_u
            G_W_a_next = H_I_aa * G_W_a + H_I_au * G_W_u # + F_W_a

            # This calculates dL/dz
            # We still need dz/du to calculate dL/du
            _loss, loss_grads = gx.jacve(loss_fn, order="rev", argnums=(0, 2), has_aux=True, sparse_representation=True)(next_z, target, W_out)
            loss += _loss
            loss_grad, W_out_grad = loss_grads[0], loss_grads[1]
            # TODO: Put a threshold variable there!
            dzdu = gx.jacve(surrogate, order="rev", sparse_representation=True)(next_u - 1.)
            loss_grad = loss_grad * dzdu
            W_grad = loss_grad * G_W_u_next

            W_grad_val += W_grad.val
            W_out_grad_val += W_out_grad.val

            new_carry = (next_z, next_u, next_a, G_W_u_next.val, G_W_a_next.val, W_grad_val, W_out_grad_val, loss)
            return new_carry, None
        
        # burnin_init_carry = (z0, u0, a0)
        # burnin_carry, _ = lax.scan(burnin_loop_fn, burnin_init_carry, in_seq[:burnin_steps], unroll=unroll)
        z_burnin, u_burnin, a_burnin = (z0, u0, a0) # burnin_carry[0], burnin_carry[1], burnin_carry[2]
        init_carry = (z_burnin, u_burnin, a_burnin, G_W_u0, G_W_a0, G_W_u0, W_out0, 0.)
        final_carry, _ = lax.scan(loop_fn, init_carry, in_seq, unroll=unroll)
        _, _, _, _, _, W_grad, W_out_grad, loss = final_carry
        return loss, W_out_grad, W_grad

    return jax.vmap(SNN_eprop_timeloop_ALIF, in_axes=(0, 0, None, None, None, None, None, None, None, None))


def make_eprop_step_ALIF(model, optim, loss_fn, unroll: int = 10, burnin_steps: int = 30):
    timeloop_fn = make_eprop_timeloop_ALIF(model, loss_fn, unroll, burnin_steps)

    def eprop_train_step(data, labels, opt_state, z0, u0, a0, G_W_u0, G_W_a0, W_out0, weights):
        loss, W_grad, W_out_grad = timeloop_fn(data, labels, z0, u0, a0, G_W_u0, G_W_a0, W_out0, *weights)
        # take the mean across the batch dim for all gradient updates
        grads = tuple(map(mean_axis0, (W_grad, W_out_grad)))
        updates, opt_state = optim.update(grads, opt_state, params=weights)
        weights = jtu.tree_map(lambda x, y: x + y, weights, updates)

        return loss, weights, opt_state
    
    return eprop_train_step


def make_stupid_eprop_timeloop(model, loss_fn, unroll: int = 10, burnin_steps: int = 30):
    """
    G: Accumulated gradient of U (hidden state) w.r.t. parameters W and V.
    F: Immediate gradient of U (hidden state) w.r.t. parameters W and V.
    H: Gradient of current U w.r.t. previous timestep U.
    in_seq: (batch_dim, num_timesteps, sensor_size)
    """
    # TODO: check gradient equivalence!
    def SNN_stupid_eprop_timeloop(in_seq, target, z0, u0, G_W0, W_out0, W_out, W):
        def loop_fn(carry, in_seq):
            z, u, G_W, W_grad, W_out_grad, loss = carry
            next_z, next_u = model(in_seq, z, u, W)
            grads = jax.jacrev(model, argnums=(2, 3))(in_seq, z, u, W)

            # grads contain the gradients of z_next and u_next w.r.t. u, W and V. But we ignore grads of z (explicit recurrence) 
            u_grads = grads[1] # only gradient of u_next w.r.t. u, W and V.
            H_I = u_grads[0] # grad. of u_next w.r.t. previous timestep u.
            F_W = u_grads[1] # gradients of u_next w.r.t. W and V respectively.
            G_W = jnp.einsum("ij,jkl->ikl", H_I, G_W) + F_W

            loss_grads = jax.jacrev(loss_fn, argnums=(0, 2))(next_z, target, W_out)
            loss += loss_fn(next_z, target, W_out)
            loss_grad, _W_out_grad = loss_grads[0], loss_grads[1]
            dzdu = jax.jacrev(surrogate)(next_u - 1.)
            loss_grad = loss_grad @ dzdu
            _W_grad = jnp.einsum("i,ijk->jk", loss_grad, G_W)
            W_grad += _W_grad
            W_out_grad += _W_out_grad

            new_carry = (next_z, next_u, G_W, W_grad, W_out_grad, loss)

            return new_carry, None
        
        # burnin_init_carry = (z0, u0)
        # burnin_carry, _ = lax.scan(burnin_loop_fn, burnin_init_carry, in_seq[:burnin_steps], unroll=unroll)
        z_burnin, u_burnin = z0, u0 # burnin_carry[0], burnin_carry[1]
        init_carry = (z_burnin, u_burnin, jnp.zeros((4, 4, 4)), G_W0, W_out0, 0.)
        final_carry, _ = lax.scan(loop_fn, init_carry, in_seq, unroll=unroll)
        _, _, _, W_grad, W_out_grad, loss = final_carry

        return loss, W_out_grad, W_grad

    return jax.vmap(SNN_stupid_eprop_timeloop, in_axes=(0, 0, None, None, None, None, None, None))


def make_eprop_rec_timeloop(model, loss_fn, unroll: int = 10, burnin_steps: int = 30):
    """
    G: Accumulated gradient of U (hidden state) w.r.t. parameters W and V.
    F: Immediate gradient of U (hidden state) w.r.t. parameters W and V.
    H: Gradient of current U w.r.t. previous timestep U.
    in_seq: (batch_dim, num_timesteps, sensor_size)
    """
    # TODO: check gradient equivalence!
    def SNN_eprop_rec_timeloop(in_seq, target, z0, u0, W, V, W_out, G_W0, G_V0, W_out0):
        def burnin_loop_fn(carry, in_seq):
            z, u = carry
            outputs = model(in_seq, z, u, W, V)
            next_z, next_u = outputs
            new_carry = (next_z, next_u)
            
            return new_carry, None
        
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

            H_I = u_grads[0] # grad of u_next w.r.t. previous timestep u.
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
        
        burnin_init_carry = (z0, u0)
        burnin_carry, _ = lax.scan(burnin_loop_fn, burnin_init_carry, in_seq[:burnin_steps], unroll=1)
        z_burnin, u_burnin = burnin_carry[0], burnin_carry[1]
        init_carry = (z_burnin, u_burnin, G_W0, G_V0, G_W0, G_V0, W_out0, .0)
        final_carry, _ = lax.scan(loop_fn, init_carry, in_seq[burnin_steps:], unroll=unroll)
        _, _, _, _, W_grad, V_grad, W_out_grad, loss = final_carry

        return loss, W_grad, V_grad, W_out_grad

    return jax.vmap(SNN_eprop_rec_timeloop, in_axes=(0, 0, None, None, None, None, None, None, None, None))


def make_eprop_rec_step(model, optim, loss_fn, unroll: int = 10, burnin_steps: int = 30):
    timeloop_fn = make_eprop_rec_timeloop(model, loss_fn, unroll, burnin_steps)

    def recurrent_eprop_train_step(in_batch, target_batch, opt_state, weights, z0, u0, G_W0, G_V0, W_out0):
        _W, _V, _W_out = weights
        loss, W_grad, V_grad, W_out_grad = timeloop_fn(in_batch, target_batch, z0, u0, _W, _V, _W_out, G_W0, G_V0, W_out0)
        # take the mean across the batch dim for all gradient updates
        grads = map(mean_axis0, (W_grad, V_grad, W_out_grad)) 
        updates, opt_state = optim.update(grads, opt_state, params=weights)
        weights = jtu.tree_map(lambda x, y: x + y, weights, updates)

        return loss, weights, opt_state
    
    return recurrent_eprop_train_step

