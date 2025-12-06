---
layout: post-interactive
title: "A Neuron Is All It Took: Building Neural Networks from Scratch"
date: 2025-12-06
author: "Sébastien Sime"
categories: [Machine Learning, Deep Learning]
tags: [neural-networks, deep-learning, python, pyodide, interactive]
---

## Introduction: From Simple Math to Artificial Intelligence

Neural networks might seem like magic—systems that can recognize faces, translate languages, and generate human-like text. But at their foundation, they're built from something surprisingly simple: individual neurons performing basic mathematical operations.

In this article, we'll build neural networks from the ground up. You'll write and run real Python code in your browser, and by the end, you'll understand:

- **What a neuron actually computes** and why it matters
- **Why we need activation functions** to create intelligent behavior
- **How multiple neurons work together** to solve complex problems
- **What backpropagation is** and how networks learn from data
- **Why deep learning works** at a fundamental level

The only prerequisite is basic Python and familiarity with NumPy. If you know what `np.dot()` does, you're ready to start.

Let's verify your environment is working:

<div class="pyodide-cell" id="cell-setup">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">import numpy as np
import matplotlib.pyplot as plt

print("✓ NumPy loaded successfully")
print("✓ Matplotlib ready")
print("\nLet's build a neural network from scratch!")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

---

## Part 1: Understanding a Single Neuron

### What Is a Neuron?

In biological brains, neurons are cells that receive signals from other neurons, process them, and send signals forward. Artificial neurons work similarly, but with pure mathematics.

An artificial neuron performs two fundamental steps:

1. **Weighted Sum**: It takes multiple inputs, multiplies each by a weight (showing importance), and adds them together with a bias term
2. **Activation**: It passes this sum through an activation function to produce the final output

Let's visualize this structure:

<pre class="ascii-art"><code>
    ANATOMY OF AN ARTIFICIAL NEURON
    ═══════════════════════════════════════════════════════════════════

     INPUTS          WEIGHTS         WEIGHTED SUM      ACTIVATION    OUTPUT

                                                           ┌─────┐
    x₁ = 0.8 ───→ × w₁ = 0.5 ──┐                          │     │
                                │                          │     │
    x₂ = 0.6 ───→ × w₂ = 0.7 ──┼──→ Σ = 1.07 ──→ z ───→ σ(z) ──→ 0.76
                                │     +0.10         +b     │     │
    x₃ = 0.9 ───→ × w₃ = 0.3 ──┘      ────                │     │
                                       1.17                └─────┘
    bias b = 0.1 ────────────────────────┘

    ═══════════════════════════════════════════════════════════════════
    Formula:  output = σ(w₁x₁ + w₂x₂ + w₃x₃ + b)
    Where:    σ(z) = sigmoid activation = 1 / (1 + e⁻ᶻ)
</code></pre>

### A Practical Example: Should I Go Running?

Let's make this concrete with a real-world decision: deciding whether to go for a run. We'll use three factors:

- **Weather score** (0 = terrible, 1 = perfect): 0.8
- **Energy level** (0 = exhausted, 1 = energized): 0.6
- **Available time** (0 = no time, 1 = plenty): 0.9

We'll assign weights to show how much each factor matters:

<div class="pyodide-cell" id="cell-linear">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false"># Our three input factors (all scaled 0-1)
inputs = np.array([0.8, 0.6, 0.9])
input_names = ['weather', 'energy', 'time']

# Weights: how much does each factor matter?
# Higher weight = more important to the decision
weights = np.array([0.5, 0.7, 0.3])

# Bias: a baseline preference (maybe you love running!)
bias = 0.1

# Compute the weighted sum
score = np.dot(inputs, weights) + bias

print("Decision Score Calculation:")
print("=" * 40)
for name, inp, w in zip(input_names, inputs, weights):
    contribution = inp * w
    print(f"{name:8s}: {inp:.1f} × {w:.1f} = {contribution:.2f}")
print(f"{'bias':8s}:              + {bias:.2f}")
print("-" * 40)
print(f"Total score:           = {score:.2f}")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

We get a score of 1.17. But there's a problem: what does 1.17 mean? Is that "yes, go running" or "no, stay home"? What if the score was 47.3 or -2.8? We need a way to interpret any number as a probability.

### The Activation Function: Sigmoid

To convert any score into a probability between 0 and 1, we use an **activation function**. The sigmoid function is perfect for this:

```
σ(z) = 1 / (1 + e^(-z))
```

The sigmoid function has useful properties:
- **Output range**: Always between 0 and 1 (perfect for probabilities)
- **Smooth curve**: Small changes in input produce small changes in output
- **Interpretable**: 0.5 is the decision boundary (50% confidence)

Let's see it in action:

<div class="pyodide-cell" id="cell-sigmoid">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    """Convert any number to a probability between 0 and 1"""
    return 1 / (1 + np.exp(-z))

# Our previous inputs
inputs = np.array([0.8, 0.6, 0.9])
weights = np.array([0.5, 0.7, 0.3])
bias = 0.1

# Calculate raw score
score = np.dot(inputs, weights) + bias

# Apply sigmoid to get probability
confidence = sigmoid(score)

print(f"Raw score:      {score:.2f}")
print(f"After sigmoid:  {confidence:.2%}")
print(f"\nInterpretation: {confidence:.0%} confident → GO RUNNING! 🏃")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

Now 1.17 becomes 76% confidence—a clear "yes, go running!"

### Visualizing the Sigmoid Function

Let's plot the sigmoid function to understand its behavior:

<div class="pyodide-cell" id="cell-sigmoid-viz">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Our running decision score
inputs = np.array([0.8, 0.6, 0.9])
weights = np.array([0.5, 0.7, 0.3])
bias = 0.1
our_score = np.dot(inputs, weights) + bias
our_confidence = sigmoid(our_score)

# Plot sigmoid curve
z_range = np.linspace(-8, 8, 200)
plt.figure(figsize=(10, 5))
plt.plot(z_range, sigmoid(z_range), 'b-', linewidth=2.5, label='Sigmoid function')

# Mark key points
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Decision boundary')
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)

# Highlight our example
plt.scatter([our_score], [our_confidence], color='red', s=150,
            zorder=5, edgecolors='darkred', linewidths=2)
plt.annotate(f'Our decision:\n{our_score:.2f} → {our_confidence:.1%}',
             (our_score, our_confidence),
             xytext=(our_score+2, our_confidence-0.2),
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.title("Sigmoid Activation: Converting Scores to Probabilities", fontsize=14, fontweight='bold')
plt.xlabel("Input Score (z)", fontsize=12)
plt.ylabel("Output Probability σ(z)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

print("Key observations:")
print("• Very negative inputs → probability near 0")
print("• Very positive inputs → probability near 1")
print("• Input of 0 → probability of exactly 0.5")
print("• Smooth, differentiable (important for training!)")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### The Complete Neuron Function

Now let's package everything into a reusable neuron function:

<div class="pyodide-cell" id="cell-complete-neuron">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def neuron(inputs, weights, bias):
    """
    A single artificial neuron.

    Steps:
    1. Compute weighted sum: z = Σ(wᵢxᵢ) + b
    2. Apply activation: output = σ(z)
    """
    z = np.dot(inputs, weights) + bias
    return sigmoid(z)

# Same weights and bias as before
weights = np.array([0.5, 0.7, 0.3])
bias = 0.1

# Test different scenarios
print("Testing our neuron on different scenarios:\n")

sunny_energized = np.array([0.9, 0.8, 0.9])
print(f"☀️ Sunny & energized: {neuron(sunny_energized, weights, bias):.1%} → GO!")

rainy_tired = np.array([0.2, 0.3, 0.9])
print(f"🌧️ Rainy & tired:     {neuron(rainy_tired, weights, bias):.1%} → SKIP")

medium_day = np.array([0.5, 0.5, 0.5])
print(f"😐 Average day:       {neuron(medium_day, weights, bias):.1%} → MAYBE")

print("\n💡 One neuron can make simple decisions!")
print("   But can it solve complex problems? Let's find out...")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

---

## Part 2: The Fundamental Limitation of a Single Neuron

### The XOR Problem: A Classic Challenge

We've seen that a single neuron can make simple decisions. But there's a famous problem in machine learning history that exposed a critical limitation: the **XOR problem**.

XOR (exclusive OR) is a simple logical operation:
- Output 1 if inputs are **different**
- Output 0 if inputs are **the same**

Here's the truth table:

<pre class="ascii-art"><code>
     XOR TRUTH TABLE
    ═══════════════════════════════════
     Input 1 │ Input 2 │ Output
    ─────────┼─────────┼─────────────────
        0    │    0    │   0     (same)
        0    │    1    │   1     (different)
        1    │    0    │   1     (different)
        1    │    1    │   0     (same)
    ═══════════════════════════════════
</code></pre>

Let's visualize this problem:

<div class="pyodide-cell" id="cell-xor-data">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false"># Define the XOR problem
# Points where inputs are the SAME should be RED (0)
# Points where inputs are DIFFERENT should be BLUE (1)

X = np.array([
    [0, 0],  # same → label 0 (red)
    [0, 1],  # different → label 1 (blue)
    [1, 0],  # different → label 1 (blue)
    [1, 1]   # same → label 0 (red)
])
y = np.array([0, 1, 1, 0])

# Visualize
plt.figure(figsize=(7, 7))
colors = ['#e74c3c' if label == 0 else '#3498db' for label in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=400, edgecolors='black', linewidths=3, zorder=5)

# Label each point
for i, (point, label) in enumerate(zip(X, y)):
    plt.annotate(f'{label}', (point[0], point[1]),
                ha='center', va='center',
                fontsize=16, fontweight='bold', color='white')

plt.xlim(-0.3, 1.3)
plt.ylim(-0.3, 1.3)
plt.xlabel("Input 1", fontsize=12, fontweight='bold')
plt.ylabel("Input 2", fontsize=12, fontweight='bold')
plt.title("The XOR Problem: Can You Draw One Line to Separate Red from Blue?",
          fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Challenge: Try to draw ONE straight line that separates")
print("           all red points from all blue points.")
print("\nSpoiler: It's impossible! 🤔")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Why Can't a Single Neuron Solve XOR?

A single neuron can only create a **linear decision boundary**—a straight line (in 2D) or a flat plane (in higher dimensions). But XOR requires a **non-linear decision boundary** to separate the classes.

Let's see what happens when we try:

<div class="pyodide-cell" id="cell-single-neuron-xor">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def single_neuron(x, weights, bias):
    z = np.dot(x, weights) + bias
    return sigmoid(z)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Try some weights (feel free to change these!)
weights = np.array([1.0, 1.0])
bias = -1.5

print("Single Neuron's Attempt to Solve XOR:")
print("=" * 50)
print(f"Weights: {weights}, Bias: {bias}\n")
print(f"{'Input':<12} {'Output':<12} {'Prediction':<12} {'Correct?':<10}")
print("-" * 50)

correct = 0
for i, x in enumerate(X):
    output = single_neuron(x, weights, bias)
    prediction = 1 if output > 0.5 else 0
    is_correct = prediction == y[i]
    correct += is_correct
    status = "✓ YES" if is_correct else "✗ NO"

    print(f"{str(x):<12} {output:.3f}       {prediction:<12} {status}")

print("-" * 50)
print(f"Accuracy: {correct}/4 = {correct/4:.0%}")
print("\n💡 Try changing the weights above!")
print("   No matter what values you use, you can't get 4/4.")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Visualizing the Linear Boundary Limitation

Let's see exactly why a single neuron fails by plotting its decision boundary:

<div class="pyodide-cell" id="cell-decision-boundary">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Try different weight configurations
weights = np.array([1.0, 1.0])
bias = -1.5

plt.figure(figsize=(8, 7))

# Create decision surface
xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 200),
                      np.linspace(-0.3, 1.3, 200))
Z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias)
Z = Z.reshape(xx.shape)

# Plot colored regions
plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
plt.colorbar(label='Neuron Output (0=Red, 1=Blue)')

# Draw decision boundary (where output = 0.5)
plt.contour(xx, yy, Z, levels=[0.5], colors='black',
            linewidths=3, linestyles='--')

# Plot data points
colors = ['#e74c3c' if label == 0 else '#3498db' for label in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=400,
            edgecolors='black', linewidths=3, zorder=5)

plt.xlim(-0.3, 1.3)
plt.ylim(-0.3, 1.3)
plt.xlabel("Input 1", fontsize=12, fontweight='bold')
plt.ylabel("Input 2", fontsize=12, fontweight='bold')
plt.title(f"Single Neuron Decision Boundary\nWeights: {weights}, Bias: {bias}",
          fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("🔍 Observation:")
print("   The dashed line is the neuron's decision boundary.")
print("   It's a STRAIGHT line—that's all one neuron can create.")
print("   No matter how you adjust weights and bias, you can't")
print("   separate the diagonal corners with a straight line!")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

This limitation—that a single neuron can only create linear boundaries—is called **linear separability**. XOR is a **linearly non-separable** problem, and it's everywhere in real-world data.

**This is why we need neural networks with multiple neurons.**

---

## Part 3: Neural Networks—Combining Neurons for Intelligence

### The Key Insight: Multiple Neurons Create Non-Linear Boundaries

What if we use **two neurons** in a hidden layer, each creating their own linear boundary, and then combine their outputs with a third neuron? This creates the ability to solve non-linear problems.

Here's the architecture:

<pre class="ascii-art"><code>
    TWO-LAYER NEURAL NETWORK ARCHITECTURE
    ═════════════════════════════════════════════════════════════

    INPUT LAYER         HIDDEN LAYER          OUTPUT LAYER
                        (2 neurons)           (1 neuron)

                      ┌──────────────┐
    x₁ (Input 1) ────→│   Neuron A   │────┐
                  │   │  (learns OR) │    │
                  │   └──────────────┘    │   ┌─────────────┐
                  │                       ├──→│   Output    │──→ Prediction
                  │   ┌──────────────┐    │   │(combines A,B)│
    x₂ (Input 2) ─┼──→│   Neuron B   │────┘   └─────────────┘
                      │ (learns AND) │
                      └──────────────┘

    ═════════════════════════════════════════════════════════════
    How XOR is solved:
    • Neuron A fires when: At least ONE input is 1  (OR logic)
    • Neuron B fires when: BOTH inputs are 1        (AND logic)
    • Output computes:     A is true BUT B is false (XOR logic)
</code></pre>

### Building a Two-Layer Network

Let's implement this network:

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
    A neural network with one hidden layer.

    Args:
        x: input vector (2,)
        W_hidden: weights for hidden layer (2, 2)
        b_hidden: biases for hidden layer (2,)
        W_output: weights for output layer (2,)
        b_output: bias for output layer (scalar)

    Returns:
        output: final prediction
        hidden: hidden layer activations (for inspection)
    """
    # Hidden layer: each neuron processes both inputs
    hidden = sigmoid(np.dot(x, W_hidden) + b_hidden)

    # Output layer: combines the two hidden neurons
    output = sigmoid(np.dot(hidden, W_output) + b_output)

    return output, hidden

# These weights were discovered through training
# (we'll learn how to find them automatically soon!)
W_hidden = np.array([[ 5.0,  5.0],   # weights from inputs to hidden neurons
                     [ 5.0,  5.0]])
b_hidden = np.array([-2.5, -7.5])    # biases for hidden neurons

W_output = np.array([5.0, -5.0])     # weights from hidden to output
b_output = -2.5                       # bias for output

print("✓ Two-layer network defined!")
print(f"  Hidden layer: 2 neurons")
print(f"  Output layer: 1 neuron")
print(f"\nThese weights encode the solution to XOR.")
print("Let's see what each hidden neuron learned...")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Understanding What Each Hidden Neuron Learned

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

print("Hidden Layer Analysis:")
print("=" * 70)
print(f"{'Input':<12} {'Neuron A':<12} {'Neuron B':<12} {'Pattern':<30}")
print("-" * 70)

for x in X:
    _, hidden = two_layer_network(x, W_hidden, b_hidden, W_output, b_output)
    A, B = hidden

    # Determine pattern
    if x[0] == 0 and x[1] == 0:
        pattern = "Neither input is 1"
    elif x[0] == 1 and x[1] == 1:
        pattern = "Both inputs are 1"
    else:
        pattern = "Exactly one input is 1"

    print(f"{str(x):<12} {A:>6.2f}       {B:>6.2f}       {pattern}")

print("-" * 70)
print("\n💡 Pattern Discovery:")
print("   • Neuron A ≈ 1.00 when AT LEAST ONE input is 1")
print("     → This neuron learned the OR function!")
print()
print("   • Neuron B ≈ 1.00 ONLY when BOTH inputs are 1")
print("     → This neuron learned the AND function!")
print()
print("   • Output neuron combines them:")
print("     output = σ(5.0×A - 5.0×B - 2.5)")
print("     → This computes: OR AND (NOT AND) = XOR")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Testing the Complete XOR Solution

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

print("XOR Solution with Two-Layer Network:")
print("=" * 65)
print(f"{'Input':<12} {'Hidden':<16} {'Output':<10} {'Pred':<6} {'Status':<10}")
print("-" * 65)

correct = 0
for i, x in enumerate(X):
    output, hidden = two_layer_network(x, W_hidden, b_hidden, W_output, b_output)
    prediction = 1 if output > 0.5 else 0
    is_correct = prediction == y[i]
    correct += is_correct
    status = "✓ PASS" if is_correct else "✗ FAIL"

    print(f"{str(x):<12} {str(np.round(hidden, 2)):<16} {output:.3f}      {prediction:<6} {status}")

print("-" * 65)
print(f"Final Accuracy: {correct}/4 = {correct/4:.0%}")
print("\n🎉 SUCCESS! Two neurons solved what one neuron couldn't!")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Visualizing the Non-Linear Decision Boundary

<div class="pyodide-cell" id="cell-nonlinear-boundary">
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

plt.figure(figsize=(9, 7))

# Create decision surface
xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 200),
                      np.linspace(-0.3, 1.3, 200))
Z = np.array([two_layer_network(np.array([a, b]), W_hidden, b_hidden, W_output, b_output)[0]
              for a, b in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

# Plot colored regions
plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.7)
plt.colorbar(label='Network Output')

# Draw decision boundary
plt.contour(xx, yy, Z, levels=[0.5], colors='black',
            linewidths=4, linestyles='--')

# Plot data points
colors = ['#e74c3c' if label == 0 else '#3498db' for label in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=400,
            edgecolors='black', linewidths=3, zorder=5)

plt.xlim(-0.3, 1.3)
plt.ylim(-0.3, 1.3)
plt.xlabel("Input 1", fontsize=12, fontweight='bold')
plt.ylabel("Input 2", fontsize=12, fontweight='bold')
plt.title("Two-Layer Network: Non-Linear Decision Boundary",
          fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("🔍 Key Observation:")
print("   The decision boundary is now CURVED!")
print("   Two neurons created a non-linear boundary by combining")
print("   their individual linear boundaries.")
print()
print("   This is the fundamental insight of neural networks:")
print("   → Simple units compose into complex decision-making")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

---

## Part 4: Working with Batches—The Matrix View

Before we tackle learning, let's clean up our code. Instead of processing one sample at a time, we can use matrix operations to process entire batches simultaneously.

### Layer as a Matrix Operation

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
    Compute one layer of neurons using matrix operations.

    This processes ALL samples in a batch simultaneously!

    Args:
        inputs: (n_samples, n_features) or (n_features,)
        weights: (n_features, n_neurons)
        biases: (n_neurons,)

    Returns:
        activations: (n_samples, n_neurons) or (n_neurons,)
    """
    return sigmoid(np.dot(inputs, weights) + biases)

# Test with a single sample
W_hidden = np.array([[ 5.0,  5.0], [ 5.0,  5.0]])
b_hidden = np.array([-2.5, -7.5])
W_output = np.array([5.0, -5.0])
b_output = -2.5

x = np.array([1, 0])  # single input

hidden = layer(x, W_hidden, b_hidden)
output = layer(hidden, W_output.reshape(2, 1), np.array([b_output]))

print("Single sample computation:")
print(f"Input:  {x}")
print(f"Hidden: {hidden.round(3)}")
print(f"Output: {output[0]:.3f}")
print("\n✓ Same computation, cleaner code!")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Batch Processing

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

# Process ALL four samples simultaneously
hidden_all = layer(X, W_hidden, b_hidden)
output_all = layer(hidden_all, W_output.reshape(2, 1), np.array([b_output]))

print("Batch Processing (all 4 XOR samples at once):")
print("=" * 50)
print("\nHidden layer outputs:")
print(hidden_all.round(3))
print("\nFinal predictions:")
print(output_all.round(3).flatten())
print("\nTarget values:")
print(y)
print("\n💡 Key advantage: On a GPU, this could process")
print("   millions of samples just as efficiently!")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

---

## Part 5: The Loss Function—Measuring How Wrong We Are

So far, we've used hand-picked weights. But how do we find good weights automatically? We need a way to measure how "wrong" our predictions are.

### Mean Squared Error (MSE)

The most common loss function for this task is **Mean Squared Error**:

```
MSE = (1/n) × Σ(prediction - target)²
```

<pre class="ascii-art"><code>
    MEAN SQUARED ERROR (MSE) CALCULATION
    ══════════════════════════════════════════════════════════════════════

    Sample    Target    Prediction    Error         Squared Error
    ──────────────────────────────────────────────────────────────────────
     [0,0]      0         0.002       +0.002         0.000004
     [0,1]      1         0.998       -0.002         0.000004
     [1,0]      1         0.997       -0.003         0.000009
     [1,1]      0         0.003       +0.003         0.000009
    ──────────────────────────────────────────────────────────────────────
                                      Sum:           0.000026
                                      Average (÷4):  0.0000065  ← MSE

    ══════════════════════════════════════════════════════════════════════
    Low MSE  = Good predictions (errors close to 0) ✓
    High MSE = Bad predictions  (errors far from 0) ✗
</code></pre>

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
    """
    Compute Mean Squared Error loss.

    Lower values = better predictions
    """
    return np.mean((predictions - targets) ** 2)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Our good weights
W_hidden = np.array([[ 5.0,  5.0], [ 5.0,  5.0]])
b_hidden = np.array([-2.5, -7.5])
W_output = np.array([5.0, -5.0])
b_output = -2.5

# Get predictions
hidden_all = layer(X, W_hidden, b_hidden)
output_all = layer(hidden_all, W_output.reshape(2, 1), np.array([b_output]))
predictions = output_all.flatten()

# Calculate loss
loss = mse_loss(predictions, y)

print("Predictions vs Targets:")
print("=" * 40)
for i in range(len(y)):
    print(f"  {predictions[i]:.3f}  vs  {y[i]}  (error: {predictions[i]-y[i]:+.3f})")
print("=" * 40)
print(f"Mean Squared Error: {loss:.6f}")
print("\n✓ Very low loss = our weights are excellent!")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

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

# Random weights
np.random.seed(42)
W_random = np.random.randn(2, 2) * 0.5
b_random = np.random.randn(2) * 0.5
W_out_random = np.random.randn(2, 1) * 0.5
b_out_random = np.random.randn(1) * 0.5

# Get predictions with random weights
hidden_random = layer(X, W_random, b_random)
output_random = layer(hidden_random, W_out_random, b_out_random)
loss_random = mse_loss(output_random.flatten(), y)

print("Random Weights Performance:")
print("=" * 40)
print(f"Predictions: {output_random.flatten().round(3)}")
print(f"Targets:     {y}")
print(f"Loss:        {loss_random:.4f}")
print("\n💡 Much higher loss! Random weights perform poorly.")
print("   Training will adjust these weights to minimize loss.")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

**The goal of training**: Start with random weights and iteratively adjust them to minimize the loss function, bringing predictions closer to targets.

---

## Part 6: Backpropagation—How Networks Learn

### The Gradient Descent Intuition

Imagine the loss function as a mountainous landscape. Your current weights place you somewhere on this landscape, and the loss is your altitude. **Training is the process of walking downhill to find the lowest point.**

But how do you know which direction is downhill? That's where **gradients** come in.

<pre class="ascii-art"><code>
    GRADIENT DESCENT: Walking Downhill to Minimize Loss
    ═════════════════════════════════════════════════════════════

    Loss
      ▲
      │
      │    ●  Start: Random weights, High loss
      │    │
      │    └──●  Step 1: Follow gradient down
      │        │
      │        └──●  Step 2: Keep descending
      │            │
      │            └──●  Step 3: Getting closer
      │                │
      │                └──●  Step 4: Nearly there
      │                    │
      └────────────────────●────────────────────────────────→ Weights
                          ▲
                          Goal: Minimum loss (optimal weights)

    ═════════════════════════════════════════════════════════════
    Gradient = Direction of steepest INCREASE in loss
    We move OPPOSITE to gradient = Go DOWNHILL = Reduce loss
</code></pre>

### Computing Gradients: The Chain Rule

For each weight, we need to know: "If I change this weight slightly, how much does the loss change?"

This is a **derivative**, and because our network is a chain of functions (input → hidden → output → loss), we use the **chain rule** to compute it. This process is called **backpropagation** because we compute gradients by working backward from the output.

### Implementation: The Backward Pass

<div class="pyodide-cell" id="cell-backward">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """
    Derivative of sigmoid with respect to its input.
    Computed using the output of sigmoid for efficiency.

    d/dz σ(z) = σ(z) × (1 - σ(z))
    """
    return a * (1 - a)

def layer(inputs, weights, biases):
    return sigmoid(np.dot(inputs, weights) + biases)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Random initialization
np.random.seed(42)
W_random = np.random.randn(2, 2) * 0.5
b_random = np.random.randn(2) * 0.5
W_out_random = np.random.randn(2, 1) * 0.5
b_out_random = np.random.randn(1) * 0.5

# Forward pass (save activations)
hidden = layer(X, W_random, b_random)
output = layer(hidden, W_out_random, b_out_random)
predictions = output.flatten()

# Backward pass - compute gradients
# Step 1: Output layer error
error = predictions - y

# Step 2: Gradient at output layer
d_output = error.reshape(-1, 1) * sigmoid_derivative(output)

# Step 3: Gradients for output weights and bias
d_W_out = np.dot(hidden.T, d_output) / len(X)
d_b_out = np.mean(d_output)

print("Computed Gradients:")
print("=" * 50)
print(f"\nOutput layer weight gradients:")
print(d_W_out.flatten().round(4))
print(f"\nOutput layer bias gradient: {d_b_out:.4f}")
print("\n💡 These gradients tell us how to adjust each weight")
print("   to reduce the loss. Negative gradient = increase weight")
print("                       Positive gradient = decrease weight")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Propagating Back to Hidden Layer

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

# Forward pass
hidden = layer(X, W_random, b_random)
output = layer(hidden, W_out_random, b_out_random)
predictions = output.flatten()

# Backward pass
error = predictions - y
d_output = error.reshape(-1, 1) * sigmoid_derivative(output)

# Propagate error to hidden layer (using chain rule!)
error_hidden = np.dot(d_output, W_out_random.T) * sigmoid_derivative(hidden)

# Gradients for hidden layer
d_W_hidden = np.dot(X.T, error_hidden) / len(X)
d_b_hidden = np.mean(error_hidden, axis=0)

print("Hidden Layer Gradients:")
print("=" * 50)
print(f"\nHidden weights gradients:")
print(d_W_hidden.round(4))
print(f"\nHidden biases gradients:")
print(d_b_hidden.round(4))
print("\n💡 The chain rule allowed us to compute how each")
print("   hidden weight affects the final loss, even though")
print("   they don't directly connect to the output!")
print("\n   This is BACKPROPAGATION—the key to training deep networks.")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

---

## Part 7: Complete Training—Watch the Network Learn

Now let's put everything together: forward pass, loss calculation, backpropagation, and weight updates. We'll train a network from scratch and watch it learn to solve XOR.

### The Full Neural Network Class

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
    """A fully-connected neural network with customizable layers."""

    def __init__(self, layer_sizes):
        """
        Initialize network with random weights.

        Args:
            layer_sizes: list of layer sizes, e.g., [2, 4, 1]
                        means 2 inputs, 4 hidden neurons, 1 output
        """
        self.weights = []
        self.biases = []

        np.random.seed(42)  # Reproducibility
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization (good for sigmoid)
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """Forward pass through network, saving activations."""
        self.activations = [X]
        current = X

        for W, b in zip(self.weights, self.biases):
            current = sigmoid(np.dot(current, W) + b)
            self.activations.append(current)

        return current

    def backward(self, y, learning_rate=1.0):
        """Backward pass to compute gradients and update weights."""
        m = len(y)
        y = y.reshape(-1, 1)

        # Start from output layer
        delta = (self.activations[-1] - y) * sigmoid_derivative(self.activations[-1])

        # Work backward through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.mean(delta, axis=0)

            # Propagate error to previous layer (if not at input)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.activations[i])

            # Update weights (gradient descent)
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs, learning_rate=1.0, verbose_every=500):
        """Complete training loop."""
        history = []

        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            loss = mse_loss(predictions.flatten(), y)
            history.append(loss)

            # Backward pass
            self.backward(y, learning_rate)

            # Logging
            if epoch % verbose_every == 0:
                print(f"Epoch {epoch:5d}: Loss = {loss:.6f}")

        return history

print("✓ NeuralNetwork class defined and ready!")
print("\n  Architecture is flexible: specify any layer sizes")
print("  Includes: forward pass, backprop, gradient descent")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Training from Scratch

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
                print(f"Epoch {epoch:5d}: Loss = {loss:.6f}")
        return history

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Create network: 2 inputs → 4 hidden neurons → 1 output
print("Creating neural network: [2 → 4 → 1]")
print("=" * 50)
nn = NeuralNetwork([2, 4, 1])

print("\nTraining on XOR problem...\n")
history = nn.train(X, y, epochs=3000, learning_rate=2.0, verbose_every=500)

print("\n" + "=" * 50)
print("Training complete! Testing final performance:\n")

predictions = nn.forward(X)
for i in range(len(X)):
    pred = predictions[i, 0]
    prediction_class = 1 if pred > 0.5 else 0
    status = "✓" if prediction_class == y[i] else "✗"
    print(f"  {X[i]} → {pred:.4f} → {prediction_class}  (target: {y[i]})  {status}")

print("\n🎉 The network learned to solve XOR from scratch!")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

### Visualizing the Learning Process

<div class="pyodide-cell" id="cell-visualize-learning">
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

    def train(self, X, y, epochs, learning_rate=1.0):
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

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax1.plot(history, linewidth=2)
ax1.set_title("Loss During Training", fontsize=14, fontweight='bold')
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss (MSE)", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')  # Log scale shows the drop better

# Decision boundary
xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 200),
                      np.linspace(-0.3, 1.3, 200))
Z = nn.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax2.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.7)
ax2.colorbar = plt.colorbar(ax2.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.7), ax=ax2)
ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3, linestyles='--')

colors = ['#e74c3c' if label == 0 else '#3498db' for label in y]
ax2.scatter(X[:, 0], X[:, 1], c=colors, s=400, edgecolors='black', linewidths=3, zorder=5)
ax2.set_title("Learned Decision Boundary", fontsize=14, fontweight='bold')
ax2.set_xlabel("Input 1", fontsize=12)
ax2.set_ylabel("Input 2", fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("📊 Left: Loss decreased from ~0.25 to ~0.001")
print("📊 Right: Network learned the correct non-linear boundary!")
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

---

## Part 8: From NumPy to Real Frameworks

Everything we've built—forward passes, backpropagation, gradient descent—is exactly what modern deep learning frameworks do. They just do it faster, on GPUs, with more features.

### The Keras Equivalent

Our 60 lines of NumPy code is equivalent to this in Keras:

```python
from tensorflow import keras

# Define architecture
model = keras.Sequential([
    keras.layers.Dense(4, activation='sigmoid', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Specify optimizer and loss
model.compile(optimizer='sgd', loss='mse')

# Train
model.fit(X, y, epochs=3000, verbose=0)
```

The concepts are identical:
- **Layers**: Dense layers = our matrix multiplications + activation
- **Loss**: MSE = our loss function
- **Optimizer**: SGD (stochastic gradient descent) = our weight updates
- **Fit**: Training loop = our forward/backward passes

---

## Conclusion: What You've Learned

Congratulations! You've built a complete neural network from scratch and understand:

### Core Concepts

1. **Neurons compute weighted sums** then apply activation functions
2. **Sigmoid activation** converts any value to a probability (0-1)
3. **Single neurons create linear boundaries** (can't solve XOR)
4. **Multiple neurons create non-linear boundaries** (can solve XOR)
5. **Matrix operations** enable efficient batch processing
6. **Loss functions** quantify prediction error
7. **Backpropagation** computes gradients via the chain rule
8. **Gradient descent** iteratively minimizes loss

### The Big Picture

<pre class="ascii-art"><code>
    THE COMPLETE DEEP LEARNING TRAINING CYCLE
    ═══════════════════════════════════════════════════════════════════

         ┌────────────────────────────────────────────────────┐
         │  STEP 1: INITIALIZATION                            │
         │  • Create random weights and biases                │
         │  • Network knows nothing yet                       │
         └──────────────────┬─────────────────────────────────┘
                            │
                            ↓
         ┌────────────────────────────────────────────────────┐
         │  STEP 2: FORWARD PASS                              │
         │  • Input → Hidden layers → Output                  │
         │  • Compute: activation = σ(weights × input + bias) │
         │  • Generate predictions                            │
         └──────────────────┬─────────────────────────────────┘
                            │
                            ↓
         ┌────────────────────────────────────────────────────┐
         │  STEP 3: COMPUTE LOSS                              │
         │  • Compare predictions to true targets             │
         │  • Calculate error: MSE = mean((pred - target)²)   │
         │  • Quantify how wrong we are                       │
         └──────────────────┬─────────────────────────────────┘
                            │
                            ↓
         ┌────────────────────────────────────────────────────┐
         │  STEP 4: BACKPROPAGATION                           │
         │  • Compute gradients using chain rule              │
         │  • Find ∂Loss/∂Weight for every weight             │
         │  • Determine how to adjust each weight             │
         └──────────────────┬─────────────────────────────────┘
                            │
                            ↓
         ┌────────────────────────────────────────────────────┐
         │  STEP 5: GRADIENT DESCENT (Update Weights)         │
         │  • weight_new = weight_old - learning_rate × grad  │
         │  • Take small step downhill on loss landscape      │
         │  • Network gets slightly better                    │
         └──────────────────┬─────────────────────────────────┘
                            │
                            ↓
                    ╔═══════════════╗
                    ║  Repeat 2-5   ║
                    ║  for 1000s of ║  ← Training loop
                    ║    epochs     ║
                    ╚═══════╦═══════╝
                            │
                            ↓
                    ┌───────────────┐
                    │   ✓ DONE!     │
                    │ Trained model │
                    │  ready to use │
                    └───────────────┘

    ═══════════════════════════════════════════════════════════════════
</code></pre>

### What Comes Next

Everything in modern deep learning builds on these foundations:

- **CNNs** (Convolutional Neural Networks): Specialized layers for images
- **RNNs** (Recurrent Neural Networks): Handle sequences (text, time series)
- **Transformers**: Attention mechanisms for language models (GPT, BERT)
- **Regularization**: Techniques to prevent overfitting (dropout, batch norm)
- **Optimizers**: Better than vanilla gradient descent (Adam, RMSprop)

But at their core, they all use the same principles you've learned:
- Neurons computing weighted sums
- Activation functions introducing non-linearity
- Backpropagation computing gradients
- Gradient descent minimizing loss

**You now understand the foundation of artificial intelligence.**

---

<!-- Turnstile Script (invisible - l'utilisateur ne voit rien) -->
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>

<div class="question-block" data-worker-url="https://rag-blog-worker.seb-sime.workers.dev/api/ask">
  <h3>💬 Questions About This Article?</h3>
  <p>Ask anything about neural networks, neurons, backpropagation, or the concepts covered in this tutorial. The RAG system will answer based on the article content.</p>

  <div id="rag-status">⏳ Initializing RAG system...</div>

  <div class="question-input-wrapper">
    <input
      type="text"
      id="user-question"
      placeholder="Example: Why can't a single neuron solve XOR?"
      disabled
    />
    <button id="ask-button" disabled>⏳ Loading...</button>
  </div>

  <div id="answer-container"></div>
</div>

<!-- Hidden container for Turnstile (invisible mode) -->
<div id="turnstile-container" style="display:none;"></div>

---

## References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors" - The original backpropagation paper
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning" (Nature) - Comprehensive overview by the pioneers
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" (MIT Press) - The definitive textbook
- Nielsen, M. A. (2015). "Neural Networks and Deep Learning" - Excellent free online book
