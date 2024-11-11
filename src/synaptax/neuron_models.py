import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp


# surrogate definition:
surrogate = lambda x: jnn.sigmoid(10.*x)


def SNN_LIF(x, z, u, W, beta=.95, threshold=1.):
    """
    Single layer SNN with implicit and explicit recurrence.
    x: layer spike inputs.
    z: previous timestep layer spike outputs.
    u: membrane potential.
    W: forward weights.
    """
    u_next = beta * u + (1.-beta) * W @ x
    surr = surrogate(u_next-threshold)
    # Trick so that in forward pass we get the heaviside spike output and in
    # backward pass we get the derivative of surrogate only without heaviside.
    z_next = lax.stop_gradient(jnp.heaviside(u_next-threshold, 0.) - surr) + surr
    u_next -= z_next*u_next
    return z_next, u_next


def SNN_ALIF(x, z, u, a, W, alpha=.95, beta=.8, gamma=1., threshold=1.):
    a_next = alpha*u + beta*a + gamma*z
    u_next = alpha * (u-z) + (1.-alpha) * (W @ x - a_next)
    surr = surrogate(u_next-threshold)
    z_next = lax.stop_gradient(jnp.heaviside(u_next-threshold, 0.) - surr) + surr
    return z_next, u_next, a_next


def SNN_rec_LIF(x, z, u, W, V, beta=0.95, threshold=1.):
    """
    Single layer SNN with implicit and explicit recurrence.
    x: layer spike inputs.
    z: previous timestep layer spike outputs.
    v: membrane potential.
    W: forward weights.
    V: explicit recurrence weights.
    """
    u_next = beta * u + W @ x + V @ z
    surr = surrogate(u_next-threshold)
    # Trick so that in forward pass we get the heaviside spike output and in
    # backward pass we get the derivative of surrogate only without heaviside.
    z_next = lax.stop_gradient(jnp.heaviside(u_next-threshold, 0.) - surr) + surr
    u_next -= z_next*threshold

    return z_next, u_next

def SNN_rec_LIF_Stopgrad(x, z, u, W, V, beta=0.95, threshold=1.):
    """
    Single layer SNN with implicit and explicit recurrence.
    x: layer spike inputs.
    z: previous timestep layer spike outputs.
    v: membrane potential.
    W: implicit recurrence weights.
    V: explicit recurrence weights.
    """
    u_next = beta * u + (1.-beta) * (W @ x + lax.stop_gradient(V @ z))
    surr = surrogate(u_next-threshold)
    # Trick so that in forward pass we get the heaviside spike output and in
    # backward pass we get the derivative of surrogate only without heaviside.
    z_next = lax.stop_gradient(jnp.heaviside(u_next-threshold, 0.) - surr) + surr
    u_next -= z_next*threshold
    return z_next, u_next


def SNN_Sigma_Delta(x, z, e, u_mem, s, i, W, V, input_decay=.95, 
                    membrane_decay=.95, feedback_decay=.95, threshold=1.):
    act = W @ x + V @ z
    i = i * input_decay + act
    e = i - s
    u_mem = u_mem * membrane_decay + e
    surr = surrogate(u_mem-threshold)
    z_out = lax.stop_gradient(jnp.heaviside(u_mem-threshold, .0) - surr) + surr
    s = s * feedback_decay + z_out
    return z_out, e, u_mem, s, i

