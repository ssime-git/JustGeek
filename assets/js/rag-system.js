/**
 * RAG System for Interactive Blog Posts
 * Uses Transformers.js for local embeddings, Gemini API for generation,
 * and Cloudflare Turnstile for anti-bot protection
 */

const WORKER_URL = 'https://rag-blog-worker.seb-sime.workers.dev';
const TURNSTILE_SITE_KEY = '0x4AAAAAACEY6vkwXOKRVPHk';

class RAGSystem {
    constructor(options = {}) {
        this.workerUrl = options.workerUrl || WORKER_URL + '/api/ask';
        this.statusElement = options.statusElement;
        this.answerElement = options.answerElement;
        this.questionInput = options.questionInput;
        this.askButton = options.askButton;
        this.turnstileContainer = options.turnstileContainer;

        this.chunks = [];
        this.embeddings = [];
        this.pipeline = null;
        this.isReady = false;
        this.isProcessing = false;
        this.turnstileReady = false;
    }

    /**
     * Initialize Turnstile
     */
    async initTurnstile() {
        return new Promise((resolve) => {
            // Check if Turnstile is already loaded
            if (typeof turnstile !== 'undefined') {
                this.turnstileReady = true;
                resolve();
                return;
            }

            // Wait for Turnstile to load
            const checkTurnstile = setInterval(() => {
                if (typeof turnstile !== 'undefined') {
                    clearInterval(checkTurnstile);
                    this.turnstileReady = true;
                    resolve();
                }
            }, 100);

            // Timeout after 10 seconds
            setTimeout(() => {
                clearInterval(checkTurnstile);
                console.warn('Turnstile not loaded, continuing without it');
                resolve();
            }, 10000);
        });
    }

    /**
     * Get Turnstile token (invisible to user)
     */
    async getTurnstileToken() {
        if (!this.turnstileReady || typeof turnstile === 'undefined') {
            throw new Error('Turnstile not available');
        }

        return new Promise((resolve, reject) => {
            // Reset any existing widget first
            if (this.turnstileContainer) {
                this.turnstileContainer.innerHTML = '';
            }

            turnstile.render(this.turnstileContainer || '#turnstile-container', {
                sitekey: TURNSTILE_SITE_KEY,
                callback: (token) => resolve(token),
                'error-callback': () => reject(new Error('Turnstile verification failed')),
                size: 'invisible'
            });
        });
    }

    /**
     * Initialize the RAG system
     */
    async initialize() {
        try {
            this.updateStatus('⏳ Initialisation de la protection anti-bot...', 'loading');
            await this.initTurnstile();

            this.updateStatus('⏳ Chargement de Transformers.js (25MB)...', 'loading');

            // Load the embedding model directly
            const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
            this.pipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

            this.updateStatus('⏳ Analyse et chunking de l\'article...', 'loading');

            // Chunk the article content
            await this.chunkArticle();

            this.updateStatus('⏳ Génération des embeddings locaux...', 'loading');

            // Generate embeddings for all chunks
            await this.generateEmbeddings();

            this.isReady = true;
            this.updateStatus('✅ Système RAG prêt ! (Recherche locale + Gemini API)', 'ready');

            // Enable the question input and button
            if (this.questionInput) this.questionInput.disabled = false;
            if (this.askButton) {
                this.askButton.disabled = false;
                this.askButton.textContent = '🚀 Poser la question';
            }

        } catch (error) {
            console.error('Erreur d\'initialisation:', error);
            this.updateStatus('❌ Erreur d\'initialisation: ' + error.message, 'error');
        }
    }

    /**
     * Chunk the article content
     */
    async chunkArticle() {
        // Get the article content
        const articleContent = document.querySelector('.post-content');
        if (!articleContent) {
            throw new Error('Article content not found');
        }

        // Extract text from paragraphs and headings
        const elements = articleContent.querySelectorAll('p, h2, h3, h4, li');
        let currentChunk = '';
        const maxChunkSize = 200; // words

        elements.forEach(element => {
            const text = element.textContent.trim();
            if (!text) return;

            // Skip if this is a demo block or question block
            if (element.closest('.demo-block') || element.closest('.question-block')) {
                return;
            }

            const words = text.split(/\s+/);

            if (currentChunk.split(/\s+/).length + words.length > maxChunkSize) {
                if (currentChunk) {
                    this.chunks.push(currentChunk.trim());
                }
                currentChunk = text;
            } else {
                currentChunk += (currentChunk ? ' ' : '') + text;
            }
        });

        if (currentChunk) {
            this.chunks.push(currentChunk.trim());
        }

        console.log(`Article divisé en ${this.chunks.length} chunks`);
    }

    /**
     * Generate embeddings for all chunks
     */
    async generateEmbeddings() {
        for (let i = 0; i < this.chunks.length; i++) {
            const embedding = await this.generateEmbedding(this.chunks[i]);
            this.embeddings.push(embedding);

            // Update progress
            if (this.statusElement) {
                const progress = Math.round(((i + 1) / this.chunks.length) * 100);
                this.updateStatus(`⏳ Génération des embeddings... ${progress}%`, 'loading');
            }
        }
    }

    /**
     * Generate embedding for a text
     */
    async generateEmbedding(text) {
        const output = await this.pipeline(text, { pooling: 'mean', normalize: true });
        return Array.from(output.data);
    }

    /**
     * Calculate cosine similarity between two vectors
     */
    cosineSimilarity(a, b) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Find the most relevant chunks for a question
     */
    async findRelevantChunks(question, topK = 3) {
        const questionEmbedding = await this.generateEmbedding(question);

        const similarities = this.embeddings.map((embedding, index) => ({
            index,
            similarity: this.cosineSimilarity(questionEmbedding, embedding),
            chunk: this.chunks[index]
        }));

        similarities.sort((a, b) => b.similarity - a.similarity);

        return similarities.slice(0, topK);
    }

    /**
     * Ask a question
     */
    async askQuestion(question) {
        if (!this.isReady) {
            this.updateStatus('❌ Le système RAG n\'est pas encore prêt', 'error');
            return;
        }

        if (this.isProcessing) {
            return;
        }

        if (!question || question.trim().length === 0) {
            this.updateStatus('❌ Veuillez entrer une question', 'error');
            return;
        }

        try {
            this.isProcessing = true;
            if (this.askButton) {
                this.askButton.disabled = true;
                this.askButton.textContent = '⏳ Vérification...';
            }

            this.updateStatus('🔐 Vérification anti-bot...', 'loading');

            // Get Turnstile token
            let turnstileToken = null;
            try {
                turnstileToken = await this.getTurnstileToken();
            } catch (e) {
                console.warn('Turnstile verification failed:', e);
                this.updateStatus('⚠️ Vérification échouée, nouvelle tentative...', 'error');
                // Reset and retry once
                if (typeof turnstile !== 'undefined') {
                    turnstile.reset();
                }
                throw new Error('Vérification anti-bot échouée. Veuillez réessayer.');
            }

            if (this.askButton) {
                this.askButton.textContent = '⏳ Recherche...';
            }

            this.updateStatus('🔍 Recherche sémantique locale...', 'searching');

            // Find relevant chunks
            const relevantChunks = await this.findRelevantChunks(question);

            console.log('Chunks pertinents:', relevantChunks);

            this.updateStatus('🤖 Génération de la réponse (Gemini API)...', 'generating');

            // Call the Cloudflare Worker with Turnstile token
            const response = await fetch(this.workerUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    context: relevantChunks.map(c => c.chunk),
                    turnstileToken: turnstileToken
                })
            });

            if (!response.ok) {
                throw new Error(`Erreur API: ${response.status}`);
            }

            const data = await response.json();

            // Display the answer
            this.displayAnswer(data.answer, relevantChunks, data.tokensUsed);

            this.updateStatus('✅ Système RAG prêt ! (Recherche locale + Gemini API)', 'ready');

        } catch (error) {
            console.error('Erreur:', error);
            this.updateStatus('❌ Erreur: ' + error.message, 'error');

            // Show fallback answer
            this.displayError(error.message);

        } finally {
            this.isProcessing = false;
            if (this.askButton) {
                this.askButton.disabled = false;
                this.askButton.textContent = '🚀 Poser la question';
            }
            // Reset Turnstile for next call
            if (typeof turnstile !== 'undefined') {
                turnstile.reset();
            }
        }
    }

    /**
     * Display the answer
     */
    displayAnswer(answer, relevantChunks, tokensUsed = null) {
        if (!this.answerElement) return;

        const tokensInfo = tokensUsed ? `<div class="tokens-info">🔢 Tokens utilisés: ${tokensUsed}</div>` : '';

        const html = `
            <div class="answer-box">
                <div class="answer-content">
                    ${this.formatAnswer(answer)}
                </div>
                ${tokensInfo}
                <details class="answer-sources">
                    <summary>📚 Passages utilisés (${relevantChunks.length})</summary>
                    <div class="sources-list">
                        ${relevantChunks.map((chunk, i) => `
                            <div class="source-item">
                                <div class="source-header">
                                    <span class="source-number">Passage ${i + 1}</span>
                                    <span class="source-score">Score: ${(chunk.similarity * 100).toFixed(1)}%</span>
                                </div>
                                <div class="source-text">${chunk.chunk}</div>
                            </div>
                        `).join('')}
                    </div>
                </details>
            </div>
        `;

        this.answerElement.innerHTML = html;
    }

    /**
     * Format answer text (basic markdown-like formatting)
     */
    formatAnswer(text) {
        return text
            .replace(/\n\n/g, '</p><p>')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`(.+?)`/g, '<code>$1</code>');
    }

    /**
     * Display error message
     */
    displayError(message) {
        if (!this.answerElement) return;

        this.answerElement.innerHTML = `
            <div class="answer-box error">
                <p><strong>❌ Une erreur s'est produite</strong></p>
                <p>${message}</p>
                <p><small>Si l'erreur persiste, veuillez vérifier que le Cloudflare Worker est bien configuré.</small></p>
            </div>
        `;
    }

    /**
     * Update status message
     */
    updateStatus(message, type = 'info') {
        if (!this.statusElement) return;

        this.statusElement.textContent = message;
        this.statusElement.className = `rag-status ${type}`;
    }
}

// Global instance
let ragSystem = null;

/**
 * Initialize RAG system when DOM is ready
 */
document.addEventListener('DOMContentLoaded', () => {
    const questionBlock = document.querySelector('.question-block');
    if (!questionBlock) return;

    const statusElement = document.getElementById('rag-status');
    const answerElement = document.getElementById('answer-container');
    const questionInput = document.getElementById('user-question');
    const askButton = document.getElementById('ask-button');
    const turnstileContainer = document.getElementById('turnstile-container');

    ragSystem = new RAGSystem({
        workerUrl: questionBlock.dataset.workerUrl || WORKER_URL + '/api/ask',
        statusElement,
        answerElement,
        questionInput,
        askButton,
        turnstileContainer
    });

    // Initialize the system
    ragSystem.initialize();

    // Add event listeners
    if (askButton) {
        askButton.addEventListener('click', () => {
            const question = questionInput.value;
            ragSystem.askQuestion(question);
        });
    }

    if (questionInput) {
        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const question = questionInput.value;
                ragSystem.askQuestion(question);
            }
        });
    }
});
