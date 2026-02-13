import jax
import jax.numpy as jnp
import equinox as eqx

class Tokenizer():
    tokens: list
    def __init__(self):
        self.tokens = []
        with open("tokens.txt", "r") as f:
            for line in f.readlines():
                self.tokens.append(line.strip())
        self.tokens.append("\n")
    def getVocabSize(self):
        return len(self.tokens) + 1
    def getToken(self, index):
        return self.tokens[index]
    def tokenize(self, string):
        string = " " + (string.encode("ascii", "ignore").decode()).lower()
        ans = []
        while string:
            maxlength = 0
            best = -1
            for i,token in enumerate(self.tokens):
                if (len(token) <= len(string) and token.replace("_", " ") == string[:len(token)] and len(token) > maxlength):
                    maxlength = len(token)
                    best = i
            #if string[0] == "_":
            #   maxlength = 1
            #   best = len(self.tokens)
            if best == -1:
                print(f"Could not parse prompt: \'{string}\'")
                maxlength = 1
                best = len(self.tokens) + 1
            ans.append(best)
            string = string[maxlength:]
        return ans

class Embedding(eqx.Module):
    token_embedder: any
    position_embedder: any
    layernorm: any
    dropout: any
    def __init__(self, vocab_size, max_length, embedding_size, hidden_size, dropout_rate, key):
        token_key, position_key = jax.random.split(key, 2)

        self.token_embedder = eqx.nn.Embedding(num_embeddings=vocab_size, embedding_size=embedding_size, key=token_key)
        self.position_embedder = eqx.nn.Embedding(num_embeddings=max_length, embedding_size=embedding_size, key=position_key)
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)
    def __call__(self, token_ids, position_ids, enable_dropout = False, key = None,):
        tokens = jax.vmap(self.token_embedder)(token_ids)
        positions = jax.vmap(self.position_embedder)(position_ids)
        embedded_inputs = tokens + positions
        embedded_inputs = jax.vmap(self.layernorm)(embedded_inputs)
        embedded_inputs = self.dropout(embedded_inputs, inference=not enable_dropout, key=key)
        return embedded_inputs

class AttentionBlock(eqx.Module):
    num_heads: any
    attention: any
    layernorm: any
    dropout: any
    def __init__(self, hidden_size, num_heads, dropout_rate, attention_dropout_rate, key):
        self.num_heads = num_heads
        self.attention = eqx.nn.MultiheadAttention(num_heads=num_heads, query_size=hidden_size, use_query_bias=True, use_key_bias=True, use_value_bias=True, use_output_bias=True, dropout_p=attention_dropout_rate, key=key,)
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, inputs, mask, enable_dropout, key):
        if mask is not None:
            mask = self.make_self_attention_mask(mask)
        attention_key, dropout_key = ((None, None) if key is None else jax.random.split(key))

        attention_output = self.attention(query=inputs, key_=inputs, value=inputs, mask=mask, inference=not enable_dropout, key=attention_key)

        result = attention_output
        result = self.dropout(result, inference=not enable_dropout, key=dropout_key)
        result = result + inputs
        result = jax.vmap(self.layernorm)(result)
        return result

    def make_self_attention_mask(self, mask):
        mask = jnp.multiply(jnp.expand_dims(mask, axis=-1), jnp.expand_dims(mask, axis=-2))
        mask = jnp.expand_dims(mask, axis=-3)
        mask = jnp.repeat(mask, repeats=self.num_heads, axis=-3)
        return mask.astype(jnp.float32)

class FeedForwardBlock(eqx.Module):
    mlp: any
    output: any
    layernorm: any
    dropout: any
    def __init__(self, hidden_size, intermediate_size, dropout_rate, key):
        mlp_key, output_key = jax.random.split(key)
        self.mlp = eqx.nn.Linear(
            in_features=hidden_size, out_features=intermediate_size, key=mlp_key
        )
        self.output = eqx.nn.Linear(
            in_features=intermediate_size, out_features=hidden_size, key=output_key
        )

        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, inputs, enable_dropout, key):
        hidden = self.mlp(inputs)
        hidden = jax.nn.gelu(hidden)

        output = self.output(hidden)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        output += inputs
        output = self.layernorm(output)

        return output

class TransformerLayer(eqx.Module):
    attention_block:any
    ff_block:any
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout_rate, attention_dropout_rate, key):
        attention_key, ff_key = jax.random.split(key)

        self.attention_block = AttentionBlock(hidden_size=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate, attention_dropout_rate=attention_dropout_rate, key=attention_key)
        self.ff_block = FeedForwardBlock(hidden_size=hidden_size, intermediate_size=intermediate_size, dropout_rate=dropout_rate, key=ff_key)

    def __call__(self, inputs, mask, *, enable_dropout, key):
        attn_key, ff_key = (None, None) if key is None else jax.random.split(key)
        attention_output = self.attention_block(inputs, mask, enable_dropout=enable_dropout, key=attn_key)
        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        output = jax.vmap(self.ff_block, in_axes=(0, None, 0))(attention_output, enable_dropout, ff_keys)
        return output

class Unembedding(eqx.Module):
    embedding: any
    dropout: any
    def __init__(self, vocab_size, embedding_size, dropout_rate, key):
        self.embedding = eqx.nn.Linear(embedding_size, vocab_size, key=key)
        self.dropout = eqx.nn.Dropout(dropout_rate)
    def __call__(self, inputs, *, enable_dropout, dropout_key = None):
        return jax.vmap(jax.nn.softmax)(self.dropout(jax.vmap(self.embedding)(inputs), inference=not enable_dropout, key=dropout_key))

class GPT(eqx.Module):
    embedding_block: any
    layers: any
    unembedding_block: any
    max_length: int
    def __init__(self, vocab_size, max_length, embedding_size, hidden_size, intermediate_size, num_layers, num_heads, dropout_rate, attention_dropout_rate, key):
        embedder_key, layer_key, unembedder_key = jax.random.split(key, num=3)
        self.embedding_block = Embedding(
            vocab_size,
            max_length,
            embedding_size,
            hidden_size,
            dropout_rate,
            embedder_key
        )
        layer_keys = jax.random.split(key, num=num_layers)
        self.layers = []
        for layer_key in layer_keys:
            self.layers.append(
                TransformerLayer(hidden_size, intermediate_size, num_heads, dropout_rate, attention_dropout_rate, layer_key)
            )
        self.unembedding_block = Unembedding(
            vocab_size,
            embedding_size,
            dropout_rate,
            unembedder_key
        )
        self.max_length = max_length
    def __call__(self, token_ids, position_ids=None, enable_dropout=False, key=None):
        emb_key, l_key, unemb_key = (None, None, None) if key is None else jax.random.split(key, num=3)
        if (position_ids == None):
            position_ids = jnp.array(range(0, len(token_ids)))

        embeddings = self.embedding_block(
            token_ids=token_ids,
            position_ids=position_ids,
            enable_dropout=enable_dropout,
            key=emb_key,
        )

        mask = jnp.asarray(token_ids != 0, dtype=jnp.int32)

        x = embeddings

        n = 0
        for layer in self.layers:
            cl_key, l_key = (None, None) if l_key is None else jax.random.split(l_key)
            x = layer(x, mask, enable_dropout=enable_dropout, key=cl_key)
            n += 1
        return self.unembedding_block(x, enable_dropout=enable_dropout, dropout_key=unemb_key)
