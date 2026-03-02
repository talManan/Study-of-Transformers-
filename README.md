# Study-of-Transformers-
Day 1  
---

##  Transformers in Generative AI
>  Source: Krish Naik — Generative AI Playlist (Videos 101, 102)

---

###  Plan of Action
```
1. RNN / LSTM / GRU  →  2. Encoder-Decoder Architecture  →  3. Attention Mechanism  →  4. Transformers
```

**Topics Covered Under Transformers:**
1. Why Transformers?
2. Architecture of Transformers
3. Self Attention → Q, K, V
4. Positional Encoding
5. Multi-Head Attention
6. Combining the Working of Transformers

---

###  101. What and Why — Transformers

> Transformers in NLP are a type of deep learning model that use **self-attention mechanisms** to analyze and process natural language data. They are **encoder-decoder models** used for **Seq2Seq tasks** like machine translation.

**Example — Language Translation:**
- English → French  *(Google Translate)*
- Input: many words → Output: many words `{length of the sentence}`
- Longer sentences → BLEU Score ↓ *(quality metric for translation)*

**Why NOT RNN/LSTM?**
- Cannot send all words in a sentence **in parallel** → not scalable
- Struggles with **long sentences** → accuracy drops
- Huge datasets → needs scalable training

**Why Transformers?**

| Feature | Benefit |
|---|---|
| **Self-Attention Module** | All words sent to encoder in **parallel** |
| **Scalable** | Handles massive datasets efficiently |
| **Transfer Learning** | BERT, GPT → SOTA Models → DALL·E → LLMs |
| **Multimodal** | Works on NLP + Image tasks |

**Transformer's Place in the AI Ecosystem:**
```
Transformers (BERT, GPT)
       ↓ trained on huge data
  Transfer Learning
       ↓
  SOTA Models → DALL·E
       ↓
  LLMs → Generative AI
       ↓
OpenAI → ChatGPT → GPT-4o
```

---

###  102. Understanding the Basic Architecture of the Encoder

####  Contextual Embedding → Self Attention

- **Word2Vec / Embedding Layer** → converts each word into a static vector
- **Self-Attention** → produces **Contextual Vector Embeddings** (dynamic, context-aware)

**Example:**
```
"My name is Krish and I play CRICKET"
                              ↑
         Self-Attention links CRICKET back to Krish (sports context)
         → Contextual meaning changes based on surrounding words
```

- Word2Vec gives **same vector** for a word regardless of context
- Self-Attention gives **different vectors** based on the sentence context ✅

---

####  Basic Transformer Architecture (Seq2Seq Task)
```
Input: "How are you?"
     ↓
[Embedding Layer]       ← words → vectors
     ↓
[Positional Encoding]   ← adds order/position info (since all words processed in parallel)
     ↓
╔══════════════════════════════╗
║       ENCODER  × 6          ║
║  ┌──────────────────────┐   ║
║  │    Self-Attention     │   ║  ← all words processed in parallel
║  └──────────────────────┘   ║
║            ↓                ║
║  ┌──────────────────────┐   ║
║  │  Feed Forward Neural  │   ║
║  │      Network          │   ║
║  └──────────────────────┘   ║
╚══════════════════════════════╝
     ↓ Contextual Vectors: Z1, Z2, Z3...
╔══════════════════════════════╗
║       DECODER  × 6          ║
║  ┌──────────────────────┐   ║
║  │   Self-Attention      │   ║  ← Masked (looks at previous tokens only)
║  └──────────────────────┘   ║
║            ↓                ║
║  ┌──────────────────────┐   ║
║  │ Encoder-Decoder Attn  │   ║  ← Cross-attention with encoder output
║  └──────────────────────┘   ║
║            ↓                ║
║  ┌──────────────────────┐   ║
║  │    Feed Forward       │   ║
║  └──────────────────────┘   ║
╚══════════════════════════════╝
     ↓
Output: "Comment vas-tu?"
```

---

####  Encoder — Deep Dive

**Flow inside one Encoder block:**
```
Input Words: [How]   [Are]   [You]
                ↓       ↓       ↓
         [Embedding Layer] — vectors for each word
                ↓       ↓       ↓
         ┌─────────────────────────┐
         │      Self-Attention     │  ← All words passed PARALLELLY
         └─────────────────────────┘
                ↓       ↓       ↓
             Z1      Z2       Z3     ← Contextual Vectors
                ↓       ↓       ↓
         ┌─────────────────────────┐
         │   Feed Forward NN       │
         └─────────────────────────┘
                ↓       ↓       ↓
           Output to Encoder 2 ...
```

**Stacked Encoders (×6):**
```
Encoder 1 output → Encoder 2 → ... → Encoder 6
                                          ↓
                                   Passed to Decoder
```

---

####  Key Concepts Summary

| Concept | Description |
|---|---|
| **Self-Attention** | Each token attends to all other tokens simultaneously |
| **Contextual Embedding** | Unlike Word2Vec (static), vectors change based on context |
| **Positional Encoding** | Adds order info since all words processed in parallel |
| **Multi-Head Attention** | Multiple self-attention ops run in parallel → richer representations |
| **Residual Connections** | Skip connections + layer normalization for stable training |
| **Feed Forward NN** | Processes each token's contextual representation independently |

---

####  Self Attention — Q, K, V

Each word generates 3 vectors:

| Vector | Full Name | Role |
|---|---|---|
| **Q** | Query | What am I looking for? |
| **K** | Key | What information do I contain? |
| **V** | Value | What do I actually pass forward? |

**Attention Formula:**
```
Attention(Q, K, V) = softmax( QKᵀ / √dk ) × V
```

**Attention Weights:**
```
[a11, a12, a13]  ← attention weights for word 1 w.r.t all words
       ↓
Context Vector: Ct = Σ aij * hj
```

---

#### 📐 Decoder — Structure
```
DECODER block contains:
  1. Self-Attention (Masked)          ← only looks at previous tokens
  2. Encoder-Decoder Attention        ← attends to encoder's output
  3. Feed Forward Neural Network
```

- **Additional Context → Decoder** improves accuracy for long sentences
- Decoder generates output **one token at a time** (auto-regressive)

---

####  Big Picture — Why Transformers Dominate AI
```
Transformers
  ├── BERT  (Encoder-only)      → NLP Understanding tasks
  ├── GPT   (Decoder-only)      → Text Generation
  └── T5    (Encoder-Decoder)   → Seq2Seq (Translation, Summarization)
       ↓
  Transfer Learning on Huge Data
       ↓
  SOTA Models → DALL·E → LLMs → Generative AI
```

>  Transformers with large datasets → Amazing SOTA results in NLP
>  Transfer Learning → Multimodal Tasks (NLP + Image)
>  Foundation for all modern LLMs and Generative AI systems

---
