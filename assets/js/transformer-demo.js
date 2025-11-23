/**
 * Transformer Generation Demo with Pyodide
 * Complete step-by-step visualization of text generation process
 */

class TransformerDemo {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        this.pyodide = null;
        this.isLoading = false;
        this.isReady = false;

        this.init();
    }

    async init() {
        // Find elements
        this.inputEl = this.container.querySelector('.transformer-input');
        this.runButton = this.container.querySelector('.transformer-run-btn');
        this.statusEl = this.container.querySelector('.transformer-status');
        this.resultEl = this.container.querySelector('.transformer-result');

        if (!this.runButton) return;

        // Add event listener
        this.runButton.addEventListener('click', () => this.runDemo());

        // Load Pyodide
        await this.loadPyodide();
    }

    async loadPyodide() {
        if (this.isLoading || this.isReady) return;

        this.isLoading = true;
        this.updateStatus('⏳ Chargement de Pyodide (10MB)...', 'loading');

        try {
            // Load Pyodide script
            if (!window.loadPyodide) {
                await this.loadPyodideScript();
            }

            // Initialize Pyodide
            this.pyodide = await window.loadPyodide({
                indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/'
            });

            // Load NumPy
            this.updateStatus('⏳ Chargement de NumPy...', 'loading');
            await this.pyodide.loadPackage('numpy');

            this.isReady = true;
            this.isLoading = false;
            this.updateStatus('✅ Prêt ! Entrez une phrase et cliquez sur "Générer"', 'ready');

            if (this.runButton) {
                this.runButton.disabled = false;
                this.runButton.textContent = '🚀 Générer le prochain mot';
            }

        } catch (error) {
            console.error('Erreur chargement Pyodide:', error);
            this.updateStatus('❌ Erreur de chargement: ' + error.message, 'error');
            this.isLoading = false;
        }
    }

    async loadPyodideScript() {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js';
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    async runDemo() {
        if (!this.isReady) {
            this.updateStatus('⏳ Pyodide est en cours de chargement...', 'loading');
            return;
        }

        const inputText = this.inputEl ? this.inputEl.value.trim() : 'le chat mange';

        if (!inputText) {
            this.updateStatus('❌ Veuillez entrer une phrase', 'error');
            return;
        }

        this.updateStatus('🧮 Simulation du processus de génération...', 'loading');
        if (this.runButton) this.runButton.disabled = true;

        try {
            // Python code for complete transformer generation process
            const pythonCode = `
import numpy as np
import json

def softmax(x):
    """Compute softmax values"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def layer_norm(x):
    """Simple layer normalization"""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + 1e-5)

def positional_encoding(seq_len, d_model):
    """Generate sinusoidal positional encodings"""
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def self_attention(Q, K, V):
    """Compute scaled dot-product attention"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    attention_weights = softmax(scores)
    output = attention_weights @ V
    return output, attention_weights, scores

def feed_forward(x, W1, W2):
    """Simple feed-forward network with ReLU"""
    hidden = np.maximum(0, x @ W1)  # ReLU activation
    output = hidden @ W2
    return output, hidden

def simulate_transformer_generation(input_text, vocab):
    """
    Simulate the complete transformer generation process
    Returns all intermediate steps for visualization
    """
    # Step 1: Tokenization
    tokens = input_text.lower().split()
    token_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]

    seq_len = len(tokens)
    d_model = 16  # Small for visualization
    vocab_size = len(vocab)

    # Step 2: Embeddings
    np.random.seed(42)
    embedding_matrix = np.random.randn(vocab_size, d_model) * 0.1
    token_embeddings = np.array([embedding_matrix[tid] for tid in token_ids])

    # Step 3: Positional Encoding
    pos_encoding = positional_encoding(seq_len, d_model)
    embeddings_with_pos = token_embeddings + pos_encoding

    # Step 4: Self-Attention
    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1

    Q = embeddings_with_pos @ W_q
    K = embeddings_with_pos @ W_k
    V = embeddings_with_pos @ W_v

    attention_output, attention_weights, attention_scores = self_attention(Q, K, V)

    # Step 5: Add & Norm (Residual connection + Layer Normalization)
    after_attention = layer_norm(embeddings_with_pos + attention_output)

    # Step 6: Feed-Forward Network
    W_ff1 = np.random.randn(d_model, d_model * 4) * 0.1
    W_ff2 = np.random.randn(d_model * 4, d_model) * 0.1
    ff_output, ff_hidden = feed_forward(after_attention, W_ff1, W_ff2)

    # Step 7: Add & Norm again
    transformer_output = layer_norm(after_attention + ff_output)

    # Step 8: Output projection to vocabulary
    W_output = np.random.randn(d_model, vocab_size) * 0.1
    logits = transformer_output @ W_output

    # Step 9: Get probabilities for next token (using last position)
    next_token_logits = logits[-1]
    next_token_probs = softmax(next_token_logits)

    # Get top 5 predictions
    top_k = 5
    top_indices = np.argsort(next_token_probs)[-top_k:][::-1]
    inv_vocab = {v: k for k, v in vocab.items()}
    predictions = [(inv_vocab.get(idx, '<unk>'), float(next_token_probs[idx])) for idx in top_indices]

    return {
        'input_text': input_text,
        'tokens': tokens,
        'token_ids': token_ids,
        'seq_len': seq_len,
        'd_model': d_model,
        'vocab_size': vocab_size,
        'token_embeddings': token_embeddings.tolist(),
        'pos_encoding': pos_encoding.tolist(),
        'embeddings_with_pos': embeddings_with_pos.tolist(),
        'Q': Q.tolist(),
        'K': K.tolist(),
        'V': V.tolist(),
        'attention_scores': attention_scores.tolist(),
        'attention_weights': attention_weights.tolist(),
        'attention_output': attention_output.tolist(),
        'after_attention': after_attention.tolist(),
        'ff_hidden': ff_hidden.tolist(),
        'ff_output': ff_output.tolist(),
        'transformer_output': transformer_output.tolist(),
        'logits': logits.tolist(),
        'next_token_probs': next_token_probs.tolist(),
        'predictions': predictions
    }

# Simple vocabulary (in practice, this would be much larger)
vocab = {
    '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3,
    'le': 4, 'la': 5, 'un': 6, 'une': 7,
    'chat': 8, 'chien': 9, 'souris': 10, 'oiseau': 11,
    'mange': 12, 'court': 13, 'dort': 14, 'joue': 15,
    'petit': 16, 'grand': 17, 'rapide': 18, 'lent': 19
}

input_text = """${inputText}"""
result = simulate_transformer_generation(input_text, vocab)
json.dumps(result)
`;

            // Execute Python code
            const resultJson = await this.pyodide.runPythonAsync(pythonCode);
            const result = JSON.parse(resultJson);

            // Display results
            this.displayResults(result);

            this.updateStatus('✅ Génération terminée !', 'success');

        } catch (error) {
            console.error('Erreur simulation:', error);
            this.updateStatus('❌ Erreur: ' + error.message, 'error');
            if (this.resultEl) {
                this.resultEl.innerHTML = `<div class="error-message">Erreur: ${error.message}</div>`;
            }
        } finally {
            if (this.runButton) this.runButton.disabled = false;
        }
    }

    displayResults(result) {
        if (!this.resultEl) return;

        const { tokens, attention_weights, predictions } = result;

        let html = `
            <div class="transformer-visualization">
                <h3>🎯 Processus Complet de Génération Transformer</h3>
                <p class="viz-intro">
                    Voici toutes les étapes pour prédire le prochain mot après "<strong>${result.input_text}</strong>"
                </p>

                <!-- Step 1: Tokenization -->
                <div class="generation-step">
                    <div class="step-header">
                        <span class="step-badge">Étape 1</span>
                        <h4>🔤 Tokenization</h4>
                    </div>
                    <div class="step-body">
                        <p>Division du texte en tokens (mots individuels)</p>
                        <div class="token-list">
                            ${tokens.map((token, i) => `
                                <div class="token-item">
                                    <div class="token-text">${token}</div>
                                    <div class="token-id">ID: ${result.token_ids[i]}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>

                <!-- Step 2: Token Embeddings -->
                <div class="generation-step">
                    <div class="step-header">
                        <span class="step-badge">Étape 2</span>
                        <h4>📊 Embeddings de Tokens</h4>
                    </div>
                    <div class="step-body">
                        <p>Chaque token est converti en un vecteur dense de dimension ${result.d_model}</p>
                        <div class="embedding-viz">
                            ${tokens.map((token, i) => `
                                <div class="embedding-row">
                                    <span class="emb-label">${token}:</span>
                                    <div class="emb-vector">
                                        [${result.token_embeddings[i].slice(0, 4).map(v => v.toFixed(2)).join(', ')}...]
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>

                <!-- Step 3: Positional Encoding -->
                <div class="generation-step">
                    <div class="step-header">
                        <span class="step-badge">Étape 3</span>
                        <h4>📍 Encodage Positionnel</h4>
                    </div>
                    <div class="step-body">
                        <p>Ajout d'informations sur la position de chaque token dans la séquence</p>
                        <div class="formula">
                            PE(pos, 2i) = sin(pos / 10000<sup>2i/d</sup>)<br>
                            PE(pos, 2i+1) = cos(pos / 10000<sup>2i/d</sup>)
                        </div>
                        <p class="step-note">
                            ℹ️ Sans cela, le modèle ne pourrait pas distinguer "chat mange souris" de "souris mange chat"
                        </p>
                    </div>
                </div>

                <!-- Step 4: Self-Attention -->
                <div class="generation-step">
                    <div class="step-header">
                        <span class="step-badge">Étape 4</span>
                        <h4>🎯 Auto-Attention (Self-Attention)</h4>
                    </div>
                    <div class="step-body">
                        <p>Calcul de l'attention entre tous les tokens</p>

                        <div class="attention-substeps">
                            <div class="substep">
                                <strong>4a. Calcul de Q, K, V</strong>
                                <div class="formula">
                                    Q = Embeddings × W<sub>Q</sub><br>
                                    K = Embeddings × W<sub>K</sub><br>
                                    V = Embeddings × W<sub>V</sub>
                                </div>
                            </div>

                            <div class="substep">
                                <strong>4b. Scores d'attention</strong>
                                <div class="formula">
                                    scores = (Q × K<sup>T</sup>) / √${result.d_model}
                                </div>
                            </div>

                            <div class="substep">
                                <strong>4c. Poids d'attention (softmax)</strong>
                                <div class="attention-matrix">
                                    <div class="matrix-labels">
                                        ${tokens.map(t => `<div class="label">${t}</div>`).join('')}
                                    </div>
                                    <div class="matrix-grid">
                                        ${attention_weights.map((row, i) => `
                                            <div class="matrix-row">
                                                ${row.map((weight, j) => {
                                                    const intensity = Math.round(weight * 255);
                                                    const bgColor = 'rgb(' + (255 - intensity) + ', ' + (255 - intensity) + ', 255)';
                                                    return '<div class="matrix-cell" style="background-color: ' + bgColor + '" ' +
                                                           'title="' + tokens[i] + ' → ' + tokens[j] + ': ' + (weight * 100).toFixed(1) + '%">' +
                                                           '<span class="cell-value">' + (weight * 100).toFixed(0) + '%</span>' +
                                                           '</div>';
                                                }).join('')}
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Step 5: Add & Norm -->
                <div class="generation-step">
                    <div class="step-header">
                        <span class="step-badge">Étape 5</span>
                        <h4>➕ Connexion Résiduelle + Normalisation</h4>
                    </div>
                    <div class="step-body">
                        <p>Ajout de la connexion résiduelle et normalisation de couche</p>
                        <div class="formula">
                            output = LayerNorm(embeddings + attention_output)
                        </div>
                        <p class="step-note">
                            ℹ️ Les connexions résiduelles aident à entraîner des réseaux profonds
                        </p>
                    </div>
                </div>

                <!-- Step 6: Feed-Forward -->
                <div class="generation-step">
                    <div class="step-header">
                        <span class="step-badge">Étape 6</span>
                        <h4>🔄 Réseau Feed-Forward</h4>
                    </div>
                    <div class="step-body">
                        <p>Application d'un réseau de neurones à chaque position</p>
                        <div class="formula">
                            FFN(x) = max(0, x × W<sub>1</sub>) × W<sub>2</sub>
                        </div>
                        <p class="step-note">
                            ℹ️ Dimension cachée: ${result.d_model} → ${result.d_model * 4} → ${result.d_model}
                        </p>
                    </div>
                </div>

                <!-- Step 7: Second Add & Norm -->
                <div class="generation-step">
                    <div class="step-header">
                        <span class="step-badge">Étape 7</span>
                        <h4>➕ Connexion Résiduelle + Normalisation (2)</h4>
                    </div>
                    <div class="step-body">
                        <p>Seconde connexion résiduelle et normalisation</p>
                        <div class="formula">
                            output = LayerNorm(input + FFN_output)
                        </div>
                    </div>
                </div>

                <!-- Step 8: Output Projection -->
                <div class="generation-step">
                    <div class="step-header">
                        <span class="step-badge">Étape 8</span>
                        <h4>🎲 Projection vers le Vocabulaire</h4>
                    </div>
                    <div class="step-body">
                        <p>Projection du vecteur de sortie vers l'espace du vocabulaire (${result.vocab_size} mots)</p>
                        <div class="formula">
                            logits = transformer_output × W<sub>output</sub>
                        </div>
                    </div>
                </div>

                <!-- Step 9: Softmax & Prediction -->
                <div class="generation-step highlighted">
                    <div class="step-header">
                        <span class="step-badge">Étape 9</span>
                        <h4>✨ Softmax & Prédiction</h4>
                    </div>
                    <div class="step-body">
                        <p>Application du softmax pour obtenir des probabilités</p>
                        <div class="formula">
                            P(mot) = softmax(logits)
                        </div>

                        <div class="predictions">
                            <h5>🏆 Top 5 Prédictions :</h5>
                            ${predictions.map((pred, i) => `
                                <div class="prediction-item rank-${i + 1}">
                                    <div class="pred-rank">#${i + 1}</div>
                                    <div class="pred-word">${pred[0]}</div>
                                    <div class="pred-prob-bar">
                                        <div class="prob-fill" style="width: ${pred[1] * 100}%"></div>
                                    </div>
                                    <div class="pred-prob">${(pred[1] * 100).toFixed(1)}%</div>
                                </div>
                            `).join('')}
                        </div>

                        <div class="prediction-result">
                            <p>
                                <strong>Prochain mot prédit :</strong>
                                <span class="predicted-word">"${predictions[0][0]}"</span>
                            </p>
                            <p class="result-sentence">
                                "${result.input_text} <strong>${predictions[0][0]}</strong>"
                            </p>
                        </div>
                    </div>
                </div>

                <!-- Summary -->
                <div class="generation-summary">
                    <h4>📝 Résumé du Processus</h4>
                    <ol>
                        <li>Le texte est <strong>tokenizé</strong> en mots individuels</li>
                        <li>Chaque token devient un <strong>vecteur dense</strong> (embedding)</li>
                        <li>On ajoute l'<strong>encodage positionnel</strong> pour garder l'ordre</li>
                        <li>Le mécanisme d'<strong>auto-attention</strong> permet à chaque mot de "regarder" les autres</li>
                        <li>Les <strong>connexions résiduelles</strong> et la <strong>normalisation</strong> stabilisent l'apprentissage</li>
                        <li>Le <strong>feed-forward</strong> transforme les représentations</li>
                        <li>La projection finale donne un score pour chaque mot du vocabulaire</li>
                        <li>Le <strong>softmax</strong> convertit les scores en probabilités</li>
                        <li>Le mot avec la plus haute probabilité est choisi</li>
                    </ol>
                </div>
            </div>
        `;

        this.resultEl.innerHTML = html;
    }

    updateStatus(message, type = 'info') {
        if (!this.statusEl) return;
        this.statusEl.textContent = message;
        this.statusEl.className = `transformer-status ${type}`;
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    const demoContainer = document.getElementById('transformer-demo');
    if (demoContainer) {
        new TransformerDemo('transformer-demo');
    }
});
