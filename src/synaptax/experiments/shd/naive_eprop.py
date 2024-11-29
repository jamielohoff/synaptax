import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu


mean_axis0 = lambda x: jnp.mean(x, axis=0)
surrogate = lambda x: jnn.sigmoid(10.*x)


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


def make_stupid_eprop_step(model, optim, loss_fn, unroll: int = 10, burnin_steps: int = 30):
    timeloop_fn = make_stupid_eprop_timeloop(model, loss_fn, unroll, burnin_steps)

    def eprop_train_step(in_batch, target_batch, opt_state, z0, u0, G_W_0, W0, W_out0, weights):
        loss, W_grad, W_out_grad = timeloop_fn(in_batch, target_batch, z0, u0, G_W_0, W0, W_out0, *weights)
        # take the mean across the batch dim for all gradient updates
        grads = tuple(map(mean_axis0, (W_grad, W_out_grad)))
        updates, opt_state = optim.update(grads, opt_state, params=weights)
        weights = jtu.tree_map(lambda x, y: x + y, weights, updates)

        return loss, weights, opt_state
    
    return eprop_train_step


def make_stupid_eprop_ALIF_timeloop(model, loss_fn, unroll: int = 10, burnin_steps: int = 30):
    """
    G: Accumulated gradient of U (hidden state) w.r.t. parameters W and V.
    F: Immediate gradient of U (hidden state) w.r.t. parameters W and V.
    H: Gradient of current U w.r.t. previous timestep U.
    in_seq: (batch_dim, num_timesteps, sensor_size)
    """
    # TODO: check gradient equivalence!
    def SNN_stupid_eprop_ALIF_timeloop(in_seq, target, z0, u0, a0, G_W_u0, G_W_a0, W0, W_out0, W, W_out):
        def burnin_loop_fn(carry, in_seq):
            z, u, a = carry
            outputs = model(in_seq, z, u, a, W)
            next_z, next_u, next_a = outputs
            new_carry = (next_z, next_u, next_a)

            return new_carry, None
        
        def loop_fn(carry, in_seq):
            z, u, a, G_W_u, G_W_a, W_grad, W_out_grad, loss = carry
            next_z, next_u, next_a = model(in_seq, z, u, a, W)
            grads = jax.jacrev(model, argnums=(2, 3, 4))(in_seq, z, u, a, W)

            # grads contain the gradients of z_next and u_next w.r.t. u, W and V. But we ignore grads of z (explicit recurrence) 
            u_grads = grads[1] # only gradient of u_next w.r.t. u, W and V.
            a_grads = grads[2]
            
            H_I_uu = u_grads[0] # grad. of u_next w.r.t. previous timestep u.
            H_I_ua = u_grads[1]
            F_W_u = u_grads[2] # gradients of u_next w.r.t. W and V respectively.

            H_I_au = a_grads[0] # grad. of u_next w.r.t. previous timestep u.
            H_I_aa = a_grads[1]
            F_W_a = a_grads[2] # gradients of u_next w.r.t. W and V respectively.

            G_W_u = jnp.einsum("ij,jkl->ikl", H_I_uu, G_W_u) + jnp.einsum("ij,jkl->ikl", H_I_ua, G_W_a) + F_W_u
            G_W_a = jnp.einsum("ij,jkl->ikl", H_I_aa, G_W_a) + jnp.einsum("ij,jkl->ikl", H_I_au, G_W_u) # + F_W_a

            loss_grads = jax.jacrev(loss_fn, argnums=(0, 2))(next_z, target, W_out)
            loss += loss_fn(next_z, target, W_out)
            loss_grad, _W_out_grad = loss_grads[0], loss_grads[1]
            dzdu = jax.jacrev(surrogate)(next_u - 1.)
            loss_grad = loss_grad @ dzdu
            # print("loss grad", loss_grad.shape, dzdu.shape)
            _W_grad = jnp.einsum("i,ijk->jk", loss_grad, G_W_u)
            W_grad += _W_grad
            W_out_grad += _W_out_grad

            new_carry = (next_z, next_u, next_a, G_W_u, G_W_a, W_grad, W_out_grad, loss)

            return new_carry, None
        
        burnin_init_carry = (z0, u0, a0)
        burnin_carry, _ = lax.scan(burnin_loop_fn, burnin_init_carry, in_seq[:burnin_steps], unroll=unroll)
        z_burnin, u_burnin, a_burnin = burnin_carry[0], burnin_carry[1], burnin_carry[2]
        init_carry = (z_burnin, u_burnin, a_burnin, G_W_u0, G_W_a0, W0, W_out0, 0.)
        final_carry, _ = lax.scan(loop_fn, init_carry, in_seq[burnin_steps:], unroll=unroll)
        _, _, _, _, _, W_grad, W_out_grad, loss = final_carry

        return loss, W_grad, W_out_grad

    return jax.vmap(SNN_stupid_eprop_ALIF_timeloop, in_axes=(0, 0, None, None, None, None, None, None, None, None, None))


def make_stupid_eprop_step_ALIF(model, optim, loss_fn, unroll: int = 10, burnin_steps: int = 30):
    timeloop_fn = make_stupid_eprop_ALIF_timeloop(model, loss_fn, unroll, burnin_steps)

    def eprop_train_step(in_batch, target_batch, opt_state, z0, u0, a0, G_W_u0, G_W_a0, W0, W_out0, weights):
        loss, W_grad, W_out_grad = timeloop_fn(in_batch, target_batch, z0, u0, a0, G_W_u0, G_W_a0, W0, W_out0, *weights)
        # take the mean across the batch dim for all gradient updates
        grads = tuple(map(mean_axis0, (W_grad, W_out_grad)))
        updates, opt_state = optim.update(grads, opt_state, params=weights)
        weights = jtu.tree_map(lambda x, y: x + y, weights, updates)

        return loss, weights, opt_state
    
    return eprop_train_step

