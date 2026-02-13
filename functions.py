import jax
import jax.numpy as jnp
import equinox as eqx

@eqx.filter_jit
def compute_accuracy(model, x, y, key=None):
    pred_y = jax.vmap(lambda x: model(x, enable_dropout=key!=None, key=key))(x)
    results = jax.vmap(catAccuracy)(y, pred_y)
    return jnp.mean(results)

@eqx.filter_jit
def catAccuracy(y, pred_y):
    mask = jnp.asarray(y != 0, dtype=jnp.int32)
    # return mean(piecewise(1.0 if y==pred_y, else 0.0)) not including mask == 1.0
    # /\ but differentiable and jit-able = \/
    #jax is a hellhole.
    return jnp.average(jax.vmap(lambda a, b: jax.lax.cond(a == jnp.argmax(b), lambda:1.0, lambda:0.0))(y, pred_y), weights=mask)

@eqx.filter_jit
def evaluate(model, testloader, loss, compute_accuracy, num_batches=10):
    avg_loss = 0
    avg_acc = 0
    batch_count = 0
    for i, (x, y) in enumerate(testloader):
        if i >= num_batches:
            break
        else:
            x = jnp.array(x)
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)
        batch_count += 1
    if batch_count > 0:
        return avg_loss / batch_count, avg_acc / batch_count
    return 0.0, 0.0

@eqx.filter_jit
def fullCrossEntropy(model, x, y, key=None):
    pred_y = jax.vmap(lambda x: model(x, enable_dropout=key!=None, key=key))(x)
    return jnp.mean(jax.vmap(catCrossEntropy)(y, pred_y))

@eqx.filter_jit
def catCrossEntropy(y, pred_y):
    mask = jnp.asarray(y != 0, dtype=jnp.int32)
    mask = jnp.expand_dims(mask, axis=1)
    mask = jnp.repeat(mask, repeats=pred_y.shape[1], axis=1)
    y = jax.nn.one_hot(y, pred_y.shape[1])
    return -jnp.sum(y * jnp.log(pred_y) * mask)