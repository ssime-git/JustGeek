---
layout: post-interactive
title: "A Neuron Is All You Need: Building Neural Networks from Scratch"
date: 2025-12-06
author: "Sébastien Sime"
categories: [Machine Learning, Deep Learning]
tags: [neural-networks, deep-learning, python, pyodide, interactive]
---

## Let's Build a Brain (Starting Embarrassingly Simple)

Every transformer, every GPT, every image classifier—at their core—is just neurons doing math. Simple math. We're going to build that intuition from scratch, and you'll run real code as we go.

No prerequisites except basic Python and NumPy comfort. If you know what `np.dot` does, you're ready.

Let's confirm your environment is ready:

<div class="pyodide-cell" id="cell-setup">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">import numpy as np
print("Environment ready.")
print("Let's build something.")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

---

## A Tiny Decision Maker

Start with a relatable framing—a neuron is just a function that takes numbers in and spits a number out. That's it. The "intelligence" comes from *which* numbers matter and *how much*.

Let's build a concrete scenario: deciding whether to go for a run based on weather, energy, and time.

**Can we build this decision-maker in 4 lines of NumPy?**

<div class="pyodide-cell" id="cell-linear">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false"># Three inputs: weather_score, energy_level, free_time (all 0-1)
inputs = np.array([0.8, 0.6, 0.9])

# How much each factor matters (we'll learn to find these later)
weights = np.array([0.5, 0.7, 0.3])

# A personal bias (maybe you just love running)
bias = 0.1

# The neuron's "opinion"
score = np.dot(inputs, weights) + bias
print(f"Decision score: {score:.2f}")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

A score of 1.17. But what does that mean? Is that a "yes go run"? What if the score was 47.3? Or -2.8?

**We need to squash this into something interpretable. Something bounded.**

### Enter Sigmoid

<div class="pyodide-cell" id="cell-sigmoid">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

inputs = np.array([0.8, 0.6, 0.9])
weights = np.array([0.5, 0.7, 0.3])
bias = 0.1
score = np.dot(inputs, weights) + bias

confidence = sigmoid(score)
print(f"Confidence to go run: {confidence:.1%}")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

Now we have a probability. 76% confident—sounds like a "yes, probably". This squashing function is called an **activation function**, and sigmoid is just one flavor.

### Visualize Sigmoid

<div class="pyodide-cell" id="cell-sigmoid-viz">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

inputs = np.array([0.8, 0.6, 0.9])
weights = np.array([0.5, 0.7, 0.3])
bias = 0.1
score = np.dot(inputs, weights) + bias
confidence = sigmoid(score)

z_range = np.linspace(-8, 8, 200)
plt.figure(figsize=(8, 4))
plt.plot(z_range, sigmoid(z_range), 'b-', linewidth=2)
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.scatter([score], [confidence], color='red', s=100, zorder=5)
plt.annotate(f'Our score: {score:.2f}', (score, confidence),
             xytext=(score+1, confidence-0.15), fontsize=10)
plt.title("Sigmoid: Any number → probability")
plt.xlabel("Raw score")
plt.ylabel("Output (0 to 1)")
plt.grid(True, alpha=0.3)
plt.show()
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### The Complete Neuron

<div class="pyodide-cell" id="cell-complete-neuron">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def neuron(inputs, weights, bias):
    z = np.dot(inputs, weights) + bias
    return sigmoid(z)

weights = np.array([0.5, 0.7, 0.3])
bias = 0.1

# Try different inputs
rainy_day = np.array([0.2, 0.6, 0.9])
print(f"Rainy day confidence: {neuron(rainy_day, weights, bias):.1%}")

tired_day = np.array([0.8, 0.2, 0.9])
print(f"Tired day confidence: {neuron(tired_day, weights, bias):.1%}")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

One neuron. Four lines of real computation. It can make decisions, sort of. **But here's the thing—it has a fatal flaw. Let's expose it.**

---

## The Problem No Single Neuron Can Solve

We're going to give our neuron a simple task. Four data points, two categories. It seems trivial. Watch what happens.

### Meet the XOR Problem

<div class="pyodide-cell" id="cell-xor-data">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">import matplotlib.pyplot as plt

# Four points, two labels
# Label 1 (blue): when inputs DIFFER
# Label 0 (red): when inputs are THE SAME

X = np.array([
    [0, 0],  # same → 0
    [0, 1],  # different → 1
    [1, 0],  # different → 1
    [1, 1]   # same → 0
])
y = np.array([0, 1, 1, 0])

# Visualize
plt.figure(figsize=(6, 6))
colors = ['#e74c3c' if label == 0 else '#3498db' for label in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=300, edgecolors='black', linewidths=2)
for i, (x, label) in enumerate(zip(X, y)):
    plt.annotate(f'{label}', (x[0], x[1]), ha='center', va='center',
                 fontsize=14, fontweight='bold', color='white')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.title("Classify: Blue (1) vs Red (0)")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.grid(True, alpha=0.3)
plt.show()
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

**Your mission: draw ONE straight line that separates blue from red.**

Pause. Look at it. Can you?

### Let's Try Anyway

<div class="pyodide-cell" id="cell-single-neuron-xor">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Our trusty single neuron
def single_neuron(x, weights, bias):
    z = np.dot(x, weights) + bias
    return sigmoid(z)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Try these weights (or change them!)
weights = np.array([1.0, -1.0])
bias = 0.0

print("Single neuron's attempt:\n")
for i, x in enumerate(X):
    output = single_neuron(x, weights, bias)
    prediction = 1 if output > 0.5 else 0
    status = "✓" if prediction == y[i] else "✗"
    print(f"  {x} → {output:.3f} → predict {prediction}, actual {y[i]}  {status}")

correct = sum(1 for i in range(4) if (1 if single_neuron(X[i], weights, bias) > 0.5 else 0) == y[i])
print(f"\nAccuracy: {correct}/4")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

**Try changing `weights` and `bias`. Any values you want. Go ahead, this cell is yours.**

*Spoiler: you'll always get at most 3 out of 4. The math doesn't allow 4/4.*

### Why? Visualize the Decision Boundary

<div class="pyodide-cell" id="cell-decision-boundary">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
weights = np.array([1.0, -1.0])
bias = 0.0

# What line is our neuron drawing?
plt.figure(figsize=(7, 6))

# Plot decision regions
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
Z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
plt.colorbar(label='Neuron output')

# Decision boundary line (where output = 0.5)
# w1*x1 + w2*x2 + b = 0  →  x2 = -(w1*x1 + b)/w2
if weights[1] != 0:
    x1_line = np.linspace(-0.5, 1.5, 100)
    x2_line = -(weights[0] * x1_line + bias) / weights[1]
    valid = (x2_line >= -0.5) & (x2_line <= 1.5)
    plt.plot(x1_line[valid], x2_line[valid], 'k--', linewidth=2, label='Decision boundary')

# Data points
colors = ['#e74c3c' if label == 0 else '#3498db' for label in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=300, edgecolors='black', linewidths=2, zorder=5)

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.title(f"weights={weights}, bias={bias}")
plt.legend()
plt.show()
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

See that dashed line? That's all a single neuron can do—draw one straight line. No matter how you tilt or shift it, you cannot separate the diagonal corners.

This is called a **linearly non-separable problem**. And it's everywhere in real data.

**So what's the fix? What if we had… two lines? Two neurons working together? Let's find out.**

---

## The Power of Teamwork

One neuron, one line. What if two neurons each draw a line, and then a third neuron combines their opinions?

### Architecture Sketch

<pre class="ascii-art"><code>
    INPUT           HIDDEN LAYER        OUTPUT

    x₁ ─────┬──────→ [Neuron A] ───┬
            │                      │
            ├──────→              ├──→ [Output] ──→ prediction
            │                      │
    x₂ ─────┴──────→ [Neuron B] ───┘

    Each hidden neuron draws its own line.
    The output neuron combines them.
</code></pre>

### Build It

<div class="pyodide-cell" id="cell-two-layer">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def two_layer_network(x, W_hidden, b_hidden, W_output, b_output):
    """
    x: single input (2,)
    W_hidden: (2, 2) - two inputs to two hidden neurons
    b_hidden: (2,) - bias for each hidden neuron
    W_output: (2,) - two hidden outputs to one output
    b_output: scalar
    """
    # Hidden layer: two neurons, each sees both inputs
    hidden = sigmoid(np.dot(x, W_hidden) + b_hidden)

    # Output layer: combines the two hidden neurons
    output = sigmoid(np.dot(hidden, W_output) + b_output)

    return output, hidden

# Weights that solve XOR (discovered by training—we'll do that soon)
W_hidden = np.array([[ 5.0,  5.0],
                     [ 5.0,  5.0]])
b_hidden = np.array([-2.5, -7.5])

W_output = np.array([5.0, -5.0])
b_output = -2.5

print("Network defined! These weights look arbitrary.")
print("They're not—they encode the solution.")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

These weights look arbitrary. They're not—they encode the solution. Let's see what each hidden neuron learned.

### What Did Each Neuron Learn?

<div class="pyodide-cell" id="cell-hidden-analysis">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def two_layer_network(x, W_hidden, b_hidden, W_output, b_output):
    hidden = sigmoid(np.dot(x, W_hidden) + b_hidden)
    output = sigmoid(np.dot(hidden, W_output) + b_output)
    return output, hidden

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
W_hidden = np.array([[ 5.0,  5.0], [ 5.0,  5.0]])
b_hidden = np.array([-2.5, -7.5])
W_output = np.array([5.0, -5.0])
b_output = -2.5

print("What each hidden neuron outputs:\n")
print("Input    | Neuron A | Neuron B | Interpretation")
print("-" * 60)

for x in X:
    _, hidden = two_layer_network(x, W_hidden, b_hidden, W_output, b_output)
    A, B = hidden

    # Interpret what each neuron learned
    if x[0] == 0 and x[1] == 0:
        interp = "Neither input active"
    elif x[0] == 1 and x[1] == 1:
        interp = "Both inputs active"
    else:
        interp = "One input active"

    print(f"  {x}   |   {A:.2f}   |   {B:.2f}   | {interp}")

print("\n💡 Insight:")
print("  Neuron A: Fires when AT LEAST ONE input is 1 → Learned OR")
print("  Neuron B: Fires only when BOTH inputs are 1 → Learned AND")
print("  Output: 'OR is true, but AND is false' → That's XOR!")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

Look at the pattern:

- **Neuron A** fires (~1.0) whenever AT LEAST ONE input is 1 → It learned **OR**
- **Neuron B** fires only when BOTH inputs are 1 → It learned **AND**

The output neuron then says: 'OR is true, but AND is false' → **That's XOR!**

### The Full Test

<div class="pyodide-cell" id="cell-xor-solution">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def two_layer_network(x, W_hidden, b_hidden, W_output, b_output):
    hidden = sigmoid(np.dot(x, W_hidden) + b_hidden)
    output = sigmoid(np.dot(hidden, W_output) + b_output)
    return output, hidden

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
W_hidden = np.array([[ 5.0,  5.0], [ 5.0,  5.0]])
b_hidden = np.array([-2.5, -7.5])
W_output = np.array([5.0, -5.0])
b_output = -2.5

print("Complete XOR solution:\n")
for i, x in enumerate(X):
    output, hidden = two_layer_network(x, W_hidden, b_hidden, W_output, b_output)
    prediction = 1 if output > 0.5 else 0
    status = "✓" if prediction == y[i] else "✗"
    print(f"  {x} → hidden {hidden.round(2)} → output {output:.3f} → {prediction}  {status}")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Visualize the Non-Linear Boundary

<div class="pyodide-cell" id="cell-nonlinear-boundary">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def two_layer_network(x, W_hidden, b_hidden, W_output, b_output):
    hidden = sigmoid(np.dot(x, W_hidden) + b_hidden)
    output = sigmoid(np.dot(hidden, W_output) + b_output)
    return output, hidden

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
W_hidden = np.array([[ 5.0,  5.0], [ 5.0,  5.0]])
b_hidden = np.array([-2.5, -7.5])
W_output = np.array([5.0, -5.0])
b_output = -2.5

plt.figure(figsize=(8, 6))

# Decision surface
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
Z = np.array([two_layer_network(np.array([a, b]), W_hidden, b_hidden, W_output, b_output)[0]
              for a, b in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.7)
plt.colorbar(label='Network output')
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

# Data points
colors = ['#e74c3c' if label == 0 else '#3498db' for label in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=300, edgecolors='black', linewidths=2, zorder=5)

plt.title("Two neurons → Non-linear decision boundary")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

That curved boundary couldn't exist with one neuron. Two neurons created it by combining their linear boundaries. **This is the core insight of neural networks: simple units composing into complex decisions.**

But we cheated. We hand-picked those magic weights. How do we find them automatically? That's where learning comes in.

---

## Thinking in Parallel

Before we learn to train, let's clean up our code. All those loops and individual neuron calls? They're hiding something beautiful: **it's all matrix multiplication.**

### Single Sample, Matrix Form

<div class="pyodide-cell" id="cell-matrix-form">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def layer(inputs, weights, biases):
    """
    One layer of neurons, computed all at once.

    inputs: (n_features,) or (n_samples, n_features)
    weights: (n_features, n_neurons)
    biases: (n_neurons,)
    """
    return sigmoid(np.dot(inputs, weights) + biases)

W_hidden = np.array([[ 5.0,  5.0], [ 5.0,  5.0]])
b_hidden = np.array([-2.5, -7.5])
W_output = np.array([5.0, -5.0])
b_output = -2.5

# Same computation, cleaner code
x = np.array([1, 0])

hidden = layer(x, W_hidden, b_hidden)
output = layer(hidden, W_output.reshape(2, 1), np.array([b_output]))

print(f"Input: {x}")
print(f"Hidden: {hidden.round(3)}")
print(f"Output: {output[0]:.3f}")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Batch Processing—All Samples at Once

<div class="pyodide-cell" id="cell-batch">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def layer(inputs, weights, biases):
    return sigmoid(np.dot(inputs, weights) + biases)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
W_hidden = np.array([[ 5.0,  5.0], [ 5.0,  5.0]])
b_hidden = np.array([-2.5, -7.5])
W_output = np.array([5.0, -5.0])
b_output = -2.5

# Process ALL four XOR samples simultaneously
hidden_all = layer(X, W_hidden, b_hidden)
output_all = layer(hidden_all, W_output.reshape(2, 1), np.array([b_output]))

print("All samples at once:\n")
print("Hidden layer outputs:")
print(hidden_all.round(3))
print("\nFinal outputs:")
print(output_all.round(3).flatten())
print("\nTargets:")
print(y)
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

Four samples, processed in parallel. On a GPU, this could be four million. The math is identical—just bigger matrices.

---

## How Bad Are We?

Training means adjusting weights to make better predictions. But "better" needs a number. We need to quantify wrongness.

### Mean Squared Error

<div class="pyodide-cell" id="cell-mse">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def layer(inputs, weights, biases):
    return sigmoid(np.dot(inputs, weights) + biases)

def mse_loss(predictions, targets):
    """Average squared difference between predictions and targets."""
    return np.mean((predictions - targets) ** 2)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
W_hidden = np.array([[ 5.0,  5.0], [ 5.0,  5.0]])
b_hidden = np.array([-2.5, -7.5])
W_output = np.array([5.0, -5.0])
b_output = -2.5

# Our network's predictions
hidden_all = layer(X, W_hidden, b_hidden)
output_all = layer(hidden_all, W_output.reshape(2, 1), np.array([b_output]))
predictions = output_all.flatten()

loss = mse_loss(predictions, y)
print(f"Predictions: {predictions.round(3)}")
print(f"Targets:     {y}")
print(f"Loss:        {loss:.4f}")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

Loss = 0.002. Very small, because our hand-picked weights are good. What if we had random weights?

### Random Weights = High Loss

<div class="pyodide-cell" id="cell-random-loss">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def layer(inputs, weights, biases):
    return sigmoid(np.dot(inputs, weights) + biases)

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

np.random.seed(42)
W_random = np.random.randn(2, 2) * 0.5
b_random = np.random.randn(2) * 0.5
W_out_random = np.random.randn(2, 1) * 0.5
b_out_random = np.random.randn(1) * 0.5

hidden_random = layer(X, W_random, b_random)
output_random = layer(hidden_random, W_out_random, b_out_random)
loss_random = mse_loss(output_random.flatten(), y)

print(f"Random predictions: {output_random.flatten().round(3)}")
print(f"Targets:            {y}")
print(f"Loss:               {loss_random:.4f}")
print(f"\nThe goal of training: start with random weights,")
print(f"adjust them until loss approaches zero.")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

Loss went from 0.002 to ~0.25. **The goal of training: start with random weights, adjust them until loss approaches zero.**

But how do we know which way to adjust? That's the genius of backpropagation.

---

## Following the Gradient Downhill

Imagine loss as a landscape. Random weights drop you somewhere on a mountain. Training is walking downhill. The gradient tells you which direction is down.

For each weight, we ask: *"If I nudge this weight slightly, does the loss go up or down?"*

That's a derivative. And because our network is a chain of functions (input → hidden → output → loss), we use the chain rule to trace the effect backward.

We won't derive the math—but we'll implement it.

### The Backward Pass, One Layer

<div class="pyodide-cell" id="cell-backward">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """Derivative of sigmoid, given its output a."""
    return a * (1 - a)

def layer(inputs, weights, biases):
    return sigmoid(np.dot(inputs, weights) + biases)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

np.random.seed(42)
W_random = np.random.randn(2, 2) * 0.5
b_random = np.random.randn(2) * 0.5
W_out_random = np.random.randn(2, 1) * 0.5
b_out_random = np.random.randn(1) * 0.5

# Forward pass (store activations for backward)
hidden = layer(X, W_random, b_random)
output = layer(hidden, W_out_random, b_out_random)
predictions = output.flatten()

# Backward pass
# Step 1: Error at output
error = predictions - y

# Step 2: Gradient at output layer
d_output = error.reshape(-1, 1) * sigmoid_derivative(output)

# Step 3: Gradients for output weights and bias
d_W_out = np.dot(hidden.T, d_output) / len(X)
d_b_out = np.mean(d_output)

print("Gradient for output weights:")
print(d_W_out.flatten().round(4))
print(f"Gradient for output bias: {d_b_out:.4f}")
print("\nThese gradients say: 'output weight 0 should decrease slightly,")
print("output weight 1 should increase.' Follow them, and loss goes down.")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

These gradients say: 'output weight 0 should decrease slightly, output weight 1 should increase.' Follow them, and loss goes down.

### Propagate Back to Hidden Layer

<div class="pyodide-cell" id="cell-backprop-full">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def layer(inputs, weights, biases):
    return sigmoid(np.dot(inputs, weights) + biases)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

np.random.seed(42)
W_random = np.random.randn(2, 2) * 0.5
b_random = np.random.randn(2) * 0.5
W_out_random = np.random.randn(2, 1) * 0.5
b_out_random = np.random.randn(1) * 0.5

hidden = layer(X, W_random, b_random)
output = layer(hidden, W_out_random, b_out_random)
predictions = output.flatten()

error = predictions - y
d_output = error.reshape(-1, 1) * sigmoid_derivative(output)

# Step 4: Error propagated to hidden layer
error_hidden = np.dot(d_output, W_out_random.T) * sigmoid_derivative(hidden)

# Step 5: Gradients for hidden weights and biases
d_W_hidden = np.dot(X.T, error_hidden) / len(X)
d_b_hidden = np.mean(error_hidden, axis=0)

print("Gradient for hidden weights:")
print(d_W_hidden.round(4))
print(f"Gradient for hidden biases: {d_b_hidden.round(4)}")
print("\nThe error flows backward through the network—hence 'backpropagation.'")
print("Each weight gets blamed in proportion to how much it contributed to the mistake.")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

The error flows backward through the network—hence 'backpropagation.' Each weight gets blamed in proportion to how much it contributed to the mistake.

---

## Watch It Learn

Let's wrap everything into a class and train it from scratch on XOR. Watch the loss drop.

### The Complete Network

<div class="pyodide-cell" id="cell-network-class">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """Initialize with random weights."""
        self.weights = []
        self.biases = []

        np.random.seed(42)  # For reproducibility
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """Forward pass, storing activations."""
        self.activations = [X]
        current = X

        for W, b in zip(self.weights, self.biases):
            current = sigmoid(np.dot(current, W) + b)
            self.activations.append(current)

        return current

    def backward(self, y, learning_rate=1.0):
        """Backward pass, updating weights."""
        m = len(y)
        y = y.reshape(-1, 1)

        # Start from output
        delta = (self.activations[-1] - y) * sigmoid_derivative(self.activations[-1])

        # Go backward through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.mean(delta, axis=0)

            # Propagate error to previous layer (if not at input)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.activations[i])

            # Update weights
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs, learning_rate=1.0, verbose_every=500):
        """Full training loop."""
        history = []

        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = mse_loss(predictions.flatten(), y)
            history.append(loss)
            self.backward(y, learning_rate)

            if epoch % verbose_every == 0:
                print(f"Epoch {epoch:4d}: Loss = {loss:.4f}")

        return history

print("NeuralNetwork class defined!")
print("Ready to train on XOR problem.")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Train and Watch

<div class="pyodide-cell" id="cell-train">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        self.activations = [X]
        current = X
        for W, b in zip(self.weights, self.biases):
            current = sigmoid(np.dot(current, W) + b)
            self.activations.append(current)
        return current

    def backward(self, y, learning_rate=1.0):
        m = len(y)
        y = y.reshape(-1, 1)
        delta = (self.activations[-1] - y) * sigmoid_derivative(self.activations[-1])
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.mean(delta, axis=0)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.activations[i])
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs, learning_rate=1.0, verbose_every=500):
        history = []
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = mse_loss(predictions.flatten(), y)
            history.append(loss)
            self.backward(y, learning_rate)
            if epoch % verbose_every == 0:
                print(f"Epoch {epoch:4d}: Loss = {loss:.4f}")
        return history

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Create network: 2 inputs → 4 hidden → 1 output
nn = NeuralNetwork([2, 4, 1])

print("Training on XOR...\n")
history = nn.train(X, y, epochs=3000, learning_rate=2.0, verbose_every=500)

print("\nFinal predictions:")
predictions = nn.forward(X)
for i in range(len(X)):
    pred = predictions[i, 0]
    status = "✓" if (pred > 0.5) == y[i] else "✗"
    print(f"  {X[i]} → {pred:.3f} (target: {y[i]})  {status}")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Visualize Learning

<div class="pyodide-cell" id="cell-visualize-learning">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        self.activations = [X]
        current = X
        for W, b in zip(self.weights, self.biases):
            current = sigmoid(np.dot(current, W) + b)
            self.activations.append(current)
        return current

    def backward(self, y, learning_rate=1.0):
        m = len(y)
        y = y.reshape(-1, 1)
        delta = (self.activations[-1] - y) * sigmoid_derivative(self.activations[-1])
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.mean(delta, axis=0)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.activations[i])
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs, learning_rate=1.0, verbose_every=500):
        history = []
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = mse_loss(predictions.flatten(), y)
            history.append(loss)
            self.backward(y, learning_rate)
        return history

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

nn = NeuralNetwork([2, 4, 1])
history = nn.train(X, y, epochs=3000, learning_rate=2.0)

plt.figure(figsize=(10, 4))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history)
plt.title("Loss Over Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

# Final decision boundary
plt.subplot(1, 2, 2)
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
Z = nn.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.7)
plt.colorbar()
colors = ['#e74c3c' if label == 0 else '#3498db' for label in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black')
plt.title("Learned Decision Boundary")

plt.tight_layout()
plt.show()
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

From random guessing to solving XOR in 3000 tiny steps. Each step, the gradients pointed downhill, and the network followed.

**Try changing the hidden layer size from 4 to 2. Still works? What about learning_rate—can you make it learn faster? Slower? Unstable?**

---

## From 50 Lines to Frameworks

Everything we built—forward pass, backward pass, gradient descent—is exactly what TensorFlow, PyTorch, and Keras do. They just do it faster, on GPUs, with more bells and whistles.

### The Keras Equivalent

What we built in ~60 lines of NumPy is equivalent to:

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(4, activation='sigmoid', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='sgd', loss='mse')
model.fit(X, y, epochs=3000, verbose=0)
```

---

## What You Now Understand

You now understand:

- ✅ **What a neuron computes** (weighted sum + activation)
- ✅ **Why activation functions matter** (non-linearity enables complex boundaries)
- ✅ **Why we need multiple neurons** (each learns a different pattern)
- ✅ **Why we need depth** (compose patterns into hierarchies)
- ✅ **How training works** (gradient descent via backpropagation)

Everything else—CNNs, RNNs, Transformers, attention—builds on these foundations. The attention mechanism in GPT? It's just a clever way to compute weights dynamically. But at the end of the day, it's still neurons, still forward passes, still gradients.

**A neuron really is all you need—to start.**

---

<!-- Turnstile Script (invisible - l'utilisateur ne voit rien) -->
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>

<div class="question-block" data-worker-url="https://rag-blog-worker.seb-sime.workers.dev/api/ask">
  <h3>💬 Une question sur l'article ?</h3>
  <p>Posez votre question et obtenez une réponse basée sur le contenu de cet article grâce au système RAG local.</p>

  <div id="rag-status">⏳ Initialisation du système RAG local...</div>

  <div class="question-input-wrapper">
    <input
      type="text"
      id="user-question"
      placeholder="Ex: What's the difference between a single neuron and a two-layer network?"
      disabled
    />
    <button id="ask-button" disabled>⏳ Chargement...</button>
  </div>

  <div id="answer-container"></div>
</div>

<!-- Hidden container for Turnstile (invisible mode) -->
<div id="turnstile-container" style="display:none;"></div>

---

## References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors"
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning" (Nature)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" (MIT Press)
- Nielsen, M. A. (2015). "Neural Networks and Deep Learning"
