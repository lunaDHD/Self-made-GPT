from GPT import GPT, Tokenizer
import jax
import equinox as eqx
import random

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STEPS = 600
PRINT_EVERY = 5
SEED = 7777 + random.randint(0, 100)
max_length = 64
embedding_size = 240
intermediate_size = 6 * embedding_size
num_layers = 6
num_heads = 5
dropout_rate = 0.0
attention_dropout_rate = 0.0
key = jax.random.PRNGKey(SEED)
randkey = jax.random.PRNGKey(SEED + random.randint(0, 50))

print("Loading Tokenizer ...")

tokenizer = Tokenizer()
history = " [INST] "

print("Loading Model ...")

model = GPT(tokenizer.getVocabSize(), max_length, embedding_size, embedding_size, intermediate_size, num_layers, num_heads, dropout_rate, attention_dropout_rate, key)
model = eqx.tree_deserialise_leaves("models/GPT_1.4.eqx", model)

print("Loading Functions ...")

@eqx.filter_jit(donate="all")
def callModel(model, input, key):
    newToken = model(jax.numpy.array(input, dtype=jax.numpy.int32))
    return sample_with_temperature(key, newToken[-1])

def generateLine(history):
    global randkey
    history = tokenizer.tokenize(history)
    out = []
    while len(out) == 0 or out[-1] not in [0, 1, 2, 3]:
        newkey, randkey = jax.random.split(randkey)
        out.append(int(callModel(model, history + out, newkey)))
    if out[-1] != 1:
        out[-1] = 1
    result = ""
    for token_id in out:
        if token_id == 0:
            break
        result += tokenizer.getToken(token_id).replace("_", " ")
    return result

def generateToken(history):
    global randkey
    history = tokenizer.tokenize(history)
    out = []
    newToken = model(jax.numpy.array(history + out, dtype=jax.numpy.int32))
    newkey, randkey = jax.random.split(randkey)
    out.append(int(sample_with_temperature(newkey, newToken[-1])))
    result = ""
    for token_id in out[1:]:
        if token_id == 0:
            break
        result += tokenizer.getToken(token_id).replace("_", " ")
    return result

def softmax_temperature(logits, temperature):
    logits = logits / temperature
    logits = logits - jax.numpy.max(logits)
    return jax.nn.softmax(logits)

def sample_with_temperature(key, logits, temperature=0.00000000001):
    probs = softmax_temperature(logits, temperature)
    return jax.random.categorical(key, jax.numpy.log(probs))

while True:
    prompt = input("> ") + " [/INST]"
    history += prompt
    result = generateLine(history)
    history += result
    print(history)