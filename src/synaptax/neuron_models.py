import jax
import jax.lax as lax
import jax.numpy as jnp


# surrogate definition:
surrogate = lambda x: 1. / (1. + 10.*jnp.abs(x))


def SNN_LIF(x, z, u, W):
    """
    Single layer SNN with implicit and explicit recurrence.
    x: layer spike inputs.
    z: previous timestep layer spike outputs.
    v: membrane potential.
    W: forward weights.
    """
    beta = 0.95
    threshold = 1.
    u_next = beta * u + (1. - beta) * jnp.dot(W, x)
    surr = surrogate(u_next - 1.)
    # Trick so that in forward pass we get the heaviside spike output and in
    # backward pass we get the derivative of surrogate only without heaviside.
    z_next = lax.stop_gradient(jnp.heaviside(u_next - threshold, 0.) - surr) + surr
    u_next -= lax.stop_gradient(z_next*u_next)
    return z_next, u_next


def SNN_rec_LIF(x, z, u, W, V):
    """
    Single layer SNN with implicit and explicit recurrence.
    x: layer spike inputs.
    z: previous timestep layer spike outputs.
    v: membrane potential.
    W: forward weights.
    V: explicit recurrence weights.
    """
    beta = 0.95 # seems to work best with this value
    threshold = 1.
    u_next = beta * u + jnp.dot(W, x) + jnp.dot(V, z)
    surr = surrogate(u_next - 1.)
    # Trick so that in forward pass we get the heaviside spike output and in
    # backward pass we get the derivative of surrogate only without heaviside.
    z_next = lax.stop_gradient(jnp.heaviside(u_next - threshold, 0.) - surr) + surr
    u_next -= lax.stop_gradient(z_next * u_next)
    return z_next, u_next

def SNN_rec_LIF_Stopgrad(x, z, u, W, V):
    """
    Single layer SNN with implicit and explicit recurrence.
    x: layer spike inputs.
    z: previous timestep layer spike outputs.
    v: membrane potential.
    W: implicit recurrence weights.
    V: explicit recurrence weights.
    """
    beta = 0.95 # seems to work best with this value
    threshold = 1.
    u_next = beta * u + (1. - beta) * (jnp.dot(W, x)+ lax.stop_gradient(jnp.dot(V, z)))
    surr = surrogate(u_next)
    # Trick so that in forward pass we get the heaviside spike output and in
    # backward pass we get the derivative of surrogate only without heaviside.
    z_next = lax.stop_gradient(jnp.heaviside(u_next - threshold, 0.) - surr) + surr
    u_next -= z_next * threshold
    return z_next, u_next


def SNN_Sigma_Delta(in_, z, e, u_mem, s, i, W, V):
    input_decay = .95
    membrane_decay = .95
    feedback_decay = .95
    threshold_ = .0

    act_ = jnp.dot(W, in_) + jnp.dot(V, z)
    i = i * input_decay + act_
    e = i - s
    u_mem = u_mem * membrane_decay + e
    surr_ = surrogate(u_mem - threshold_)
    z_out = jax.lax.stop_gradient(jnp.heaviside(u_mem - threshold_, .0) - surr_) + surr_
    s = s * feedback_decay + z_out
    return z_out, e, u_mem, s, i

