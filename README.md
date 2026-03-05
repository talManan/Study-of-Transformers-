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



Day 2 

---

## 🧠 103. Self Attention Layer — Working

### 📌 Self Attention At A Higher Level

> **Self-attention (Scaled Dot-Product Attention)** allows the model to weigh the importance of different tokens in the input sequence relative to each other → produces **Contextual Embeddings**

**Example:**
```
"The cat sat on the mat, the cat lay on the rug"
        ↓ Word Embedding for each token
        ↓ Self Attention
→ Each word gets a Contextual Embedding vector
   "cat" → attends to "sat", "lay", "mat", "rug" → rich context
```

**Real-world analogy (YouTube Search):**
```
Query   → Search keywords (what you're looking for)
Keys    → Tags, Title, Description (what each video contains)
Values  → Output Video (what gets returned)
```

---

### 📌 Self Attention In Detail — Step by Step

**Setup:**
```
Input Sequence = ["The", "CAT", "SAT"]
Embedding size = 4
Q, K, V dimension = 4
```

#### Step 1️⃣ — Token Embedding
```
E_The = [1, 0, 1, 0]
E_CAT = [0, 1, 0, 1]
E_SAT = [1, 1, 1, 1]
```

#### Step 2️⃣ — Linear Transformation (Create Q, K, V)
> We create Q, K, V by multiplying embeddings by **learned weight matrices** Wq, Wk, Wv
```
Q = Embedding × Wq    (dependencies of each word)
K = Embedding × Wk    (context of words)
V = Embedding × Wv

# If Wq = Wk = Wv = Identity Matrix (I):
Q_The = K_The = V_The = [1, 0, 1, 0]
Q_CAT = K_CAT = V_CAT = [0, 1, 0, 1]
Q_SAT = K_SAT = V_SAT = [1, 1, 1, 1]
```

#### Step 3️⃣ — Compute Attention Scores
> Score determines **how much focus to place on other parts** of the input sentence as we encode a word at a certain position
```
Score(Q_The, K_The) = [1,0,1,0]·[1,0,1,0]ᵀ = 2
Score(Q_The, K_CAT) = [1,0,1,0]·[0,1,0,1]ᵀ = 0
Score(Q_The, K_SAT) = [1,0,1,0]·[1,1,1,1]ᵀ = 2

Score(Q_CAT, K_The) = 0,  Score(Q_CAT, K_CAT) = 2,  Score(Q_CAT, K_SAT) = 2
Score(Q_SAT, K_The) = 2,  Score(Q_SAT, K_CAT) = 2,  Score(Q_SAT, K_SAT) = 4
```

#### Step 4️⃣ — Scaling (Divide by √dk)
> Scaling is **crucial** to prevent dot products from growing too large → ensures stable gradients
```
dk = 4  →  √dk = 2

Why scale?
  ① Without scaling: dot products grow large → Gradient Exploding
  ② Large values → Softmax Saturation → Vanishing Gradient Problem

# Without Scaling example:
  Scores [6, 4] → Softmax ≈ [0.88, 0.12]  (too skewed)

# With Scaling:
  [6,4] → [6/2, 4/2] = [3,2] → Softmax ≈ [0.73, 0.27]  (more balanced ✅)

Scaled Scores for "The":
  Score(Q_The, K_The) = 2/2 = 1
  Score(Q_The, K_CAT) = 0/2 = 0
  Score(Q_The, K_SAT) = 2/2 = 1
```

#### Step 5️⃣ — Apply Softmax
> Softmax score determines **how much each word will be expressed** at this position
```
Attention Weights "The" = Softmax([1, 0, 1]) = [0.4223, 0.1554, 0.4223]
Attention Weights "CAT" = Softmax([0, 2, 2]) = [0.1554, 0.4223, 0.4223]
Attention Weights "SAT" = Softmax([2, 2, 4]) = [0.2119, 0.2119, 0.5762]
```

#### Step 6️⃣ — Weighted Sum of Values
> Multiply attention weights by corresponding Value vectors → **Contextual Vector**
```
Output(The) = 0.4223 × V_The + 0.1554 × V_CAT + 0.4223 × V_SAT
            = 0.4223[1,0,1,0] + 0.1554[0,1,0,1] + 0.4223[1,1,1,1]
            = [1.2669, 0.9999, 1.2669, 0.9999]   ← Contextual Vector Z1
```

#### 🔁 Complete Self-Attention Pipeline
```
Input Token
    ↓
① Q, K, V  [Wq, Wk, Wv]
    ↓
② Attention Score  (Q·Kᵀ)
    ↓
③ Scale  (÷ √dk)
    ↓
④ Softmax
    ↓
⑤ Weighted Sum of Values  (Softmax × V)
    ↓
Contextual Vector Z
```

**Formula:**
```
Attention(Q, K, V) = softmax( Q·Kᵀ / √dk ) × V
```

---

## 🔀 104. Multi-Head Attention

### 📌 What is Multi-Head Attention?

> Instead of running **one** self-attention, we run **multiple attention heads in parallel**, each with its own Wq, Wk, Wv weight matrices → expands the model's ability to focus on **different positions and relationships** simultaneously
```
Single Attention Head:
  Z = softmax(QKᵀ / √dk) × V  → 1 output vector

Multi-Head Attention (8 heads):
  Head 0:  Q0,K0,V0  [W0q, W0k, W0v]  → Z0
  Head 1:  Q1,K1,V1  [W1q, W1k, W1v]  → Z1
  ...
  Head 7:  Q7,K7,V7  [W7q, W7k, W7v]  → Z7
```

### 📌 Multi-Head Attention — Step by Step

#### Step 1️⃣ — Input Embedding + Positional Encoding
```
Input → Embedding Layer → + Positional Encoding → X
```

#### Step 2️⃣ — Split into 8 Heads, create Q/K/V per head
```
X × W0q → Q0,  X × W0k → K0,  X × W0v → V0
X × W1q → Q1,  X × W1k → K1,  X × W1v → V1
...
X × W7q → Q7,  X × W7k → K7,  X × W7v → V7
```

#### Step 3️⃣ — Scaled Dot-Product Attention per head
```
Z0 = softmax(Q0·K0ᵀ / √dk) × V0
Z1 = softmax(Q1·K1ᵀ / √dk) × V1
...
Z7 = softmax(Q7·K7ᵀ / √dk) × V7
```

#### Step 4️⃣ — Concatenate all heads
```
[Z0 | Z1 | Z2 | Z3 | Z4 | Z5 | Z6 | Z7]  ← 7 Head Attention
```

#### Step 5️⃣ — Multiply with weight matrix W⁰
```
Z = Concat(Z0...Z7) × W⁰  → Final output matrix
                               ↓
                         Feed Forward NN
```

**Research Paper specs:**
```
Embedding Vector = 512
Q, K, V          = 64
Head Attention   = 8
√64              = 8
```

### 📌 Why Multi-Head Attention?

| Benefit | Explanation |
|---|---|
| **Multiple perspectives** | Each head can learn different relationships between tokens |
| **Different positions** | Head 0 might focus on syntax, Head 1 on semantics, etc. |
| **Richer representations** | Concatenated output captures far more context than single head |

---

### 📌 Positional Encoding

> **Problem:** Self-Attention processes all words in parallel → loses **sequential order** of words
```
"Lion kills Tiger"  ≠  "Tiger kills Lion"
       ↓ Without positional encoding, Self-Attention treats them the SAME ❌
```

**Advantage of parallel processing:** Word tokens processed simultaneously ✅
**Drawback:** Lacks sequential structure of words (order) ❌
**Solution:** Positional Encoding → add position info to embedding vectors

#### Types of Positional Encoding

1. **Sinusoidal Positional Encoding** → uses Sine and Cosine functions of different frequencies
2. **Learned Positional Encoding** → positional encodings are learned during training

#### Sinusoidal Formula
```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )

Where:
  pos     = position of the word in sequence
  i       = dimension index
  d_model = dimensionality of embeddings
```

**Example (d_model = 4):**
```
"The" → Embedding [0.1, 0.2, 0.3, 0.4] + PE[0, 1, 0, 1]   → [0.1, 1.2, 0.3, 1.4]
"CAT" → Embedding [0.5, 0.6, 0.7, 0.8] + PE[0.84, 0.54, 0.01, 0.999]
"SAT" → Embedding [0.9, 1.0, 1.1, 1.2] + PE[...]
```

**Flow after Positional Encoding:**
```
Token Embeddings + Positional Encoding
         ↓
    Self Attention
         ↓
    Multi-Head Attention
```

---

### 📌 Layer Normalization in Transformers

> All 4 key components of encoder: Self-Attention → Multi-Head Attention → Positional Encoding → **Layer Normalization (Add & Norm)**

**Normalization types:**
```
Normalization
  ├── Batch Normalization  → normalizes across the batch (column-wise)
  └── Layer Normalization  → normalizes across the layer (row-wise) ← used in Transformers
```

**Formula:**
```
z_score = (xi - μ) / σ    →   μ=0, σ=1

Layer Norm: y = γ × [(z - μ) / σ] + β
  γ (gamma) = learned scale parameter
  β (beta)  = learned shift parameter
```

**Advantages of Layer Normalization:**
```
① Improved Training Stability
       ↓
   Solves Vanishing and Exploding Gradient problems

② Faster Convergence
       ↓
   Back Propagation → Stable updates
```

**Example:**
```
"CAT" = [2.0, 4.0, 6.0, 8.0]
  μ  = (2+4+6+8)/4 = 5.0
  σ² = 5.0,  σ = 2.236

Normalized:
  x̂1 = (2.0-5.0)/2.236 ≈ -1.34
  x̂2 = (4.0-5.0)/2.236 ≈ -0.45
  x̂3 = (6.0-5.0)/2.236 ≈  0.45
  x̂4 = (8.0-5.0)/2.236 ≈  1.34

x̂ = [-1.34, -0.45, 0.45, 1.34]

Scale & Shift:
  γ = [1.0, 1.0, 1.0, 1.0],  β = [0.0, 0.0, 0.0, 0.0]
  y = γ × x̂ + β = [-1.34, -0.45, 0.45, 1.34]
```

---

### 📌 Residual Connections (Skip Connections)

> Residual connections create **short paths for gradients** to flow directly through the network
```
Output = LayerNorm(X + SubLayer(X))
              ↑
         Residuals = additional signal to Layer Normalization
```

**3 Key Benefits:**
```
① Addressing Vanishing Gradient Problem
     → Gradients remain sufficiently large through shortcut path

② Improve Gradient Flow
     → Convergence will be faster

③ Enables Training of Deeper Networks
     → More layers = more learning capacity
```

---

### 📌 Feed Forward Neural Network (FFN)

**2 roles in Transformer:**
```
① Adding Non-Linearity  →  solves linear function problem
② Processing Each Position Independently
     Self-Attention  → captures relationships between tokens
     FFN             → each token representation processed independently
                             ↓
                       Transforming representations further
                       → allows model to learn Richer Representations (ANN)

③ FFN → Deeper → Adds Depth to the Model
     Depth ↑ → More learnings → More DATA
```

---

### 📌 Complete Encoder Architecture (Research Paper)
```
Input Sequence  →  Text Embeddings + Positional Encoding  →  512 dim (every word)
                                    ↓
                          Multi-Head Attention  →  Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8
                                    ↓
                            Add And Norm  (Residuals)
                                    ↓
                          Feed Forward NN
                                    ↓
                            Add And Norm
                                    ↓
                         Encoder Output (×6 stacked)

Research Paper:
  Q, K = 64     Head Attention = 8
  V    = 64     √64 = 8
```

---

> 📚 **Source:** Krish Naik — Generative AI Playlist 


----
Day - 3 

