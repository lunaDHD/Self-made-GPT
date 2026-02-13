import jax
import equinox as eqx
import optax
import random
from GPT import GPT, Tokenizer
from GPTdatasetLoader import getDataLoaders
from functions import *

BATCH_SIZE = 256
LEARNING_RATE = 3e-5
PRINT_EVERY = 10
SEED = 7777 + random.randint(0, 100)
max_length = 64
embedding_size = 240
intermediate_size = 6 * embedding_size
num_layers = 6
num_heads = 5
dropout_rate = 0.01
attention_dropout_rate = 0.01
key = jax.random.PRNGKey(SEED)


print("Loading Tokenizer ...")

tokenizer = Tokenizer()

print("Loading Dataset ...")

train_dataloader, test_dataloader = getDataLoaders(BATCH_SIZE, tokenizer, max_length)

#TODO:
# ✓ make a todo list
# ✓ fix the dataset
# ✓ create loss function
# ✓ create compute_accuracy function
# ✓ create evaluate function
# ✓ create make_step function
# ✓ create train function
# ✓ debug
# ✓ run it on the laptop
# ✓ make it run on GPU - MFW it already does - MFW IT DOES NOT ????
# - run it on a real computer
# ? PROFIT ?????

print("Initializing Functions ...")

@eqx.filter_jit
def make_step(model, opt_state, x, y, optim, key):
    loss_value, grads = eqx.filter_value_and_grad(fullCrossEntropy)(model, x, y, key)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

def train(model, trainloader, testloader, optim, print_every, loss, compute_accuracy, *, key):
    try:
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        step = 0
        callKey, key = jax.random.split(key)
        for (x, y) in trainloader:
            callKey, currentKey = jax.random.split(callKey)
            model, opt_state, train_loss = make_step(model, opt_state, x, y, optim, currentKey)
            if (step % print_every) == 0:
                test_loss, test_accuracy = evaluate(model, testloader, loss, compute_accuracy, num_batches = 1)
                print(f"{step=:03d}, train_loss={train_loss.item():8.5f}, ", f"test_loss={test_loss.item():8.5f}, test_accuracy={test_accuracy.item():.2%}")
            step += 1
    except KeyboardInterrupt:
        return model
    except Exception as e:
        print("Error occured:")
        raise e

print("Initializing model ...")
datakey, modelkey = jax.random.split(key, 2)
model = GPT(tokenizer.getVocabSize(), max_length, embedding_size, embedding_size, intermediate_size, num_layers, num_heads, dropout_rate, attention_dropout_rate, modelkey)
model = eqx.tree_deserialise_leaves("models/GPT_1.4.eqx", model)
optim = optax.rmsprop(LEARNING_RATE)

print("Launching!")
model = train(model, train_dataloader, test_dataloader, optim, PRINT_EVERY, fullCrossEntropy, compute_accuracy, key=key)
print("Train Successful!")
eqx.tree_serialise_leaves(f"models/GPT_1.4.eqx", model)
print("Model saved!")