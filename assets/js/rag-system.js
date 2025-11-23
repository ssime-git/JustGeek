/**
 * RAG System for Interactive Blog Posts
 * Uses Transformers.js for local embeddings and Gemini API for generation
 */

class RAGSystem {
    constructor(options = {}) {
        this.workerUrl = options.workerUrl || 'https://rag-blog-worker.YOUR-SUBDOMAIN.workers.dev';
        this.statusElement = options.statusElement;
        this.answerElement = options.answerElement;
        this.questionInput = options.questionInput;
        this.askButton = options.askButton;

        this.chunks = [];
        this.embeddings = [];
        this.pipeline = null;
        this.isReady = false;
        this.isProcessing = false;
    }

    /**
     * Initialize the RAG system
     */
    async initialize() {
        try {
            this.updateStatus('⏳ Chargement de Transformers.js...', 'loading');

            // Load Transformers.js dynamically
            if (typeof pipeline === 'undefined') {
                await this.loadTransformersJS();
            }

            this.updateStatus('⏳ Chargement du modèle d\'embeddings (25MB)...', 'loading');

            // Load the embedding model
            const { pipeline: pipelineFunc } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
            this.pipeline = await pipelineFunc('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

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
                this.askButton.textContent = 'Poser la question';
            }

        } catch (error) {
            console.error('Erreur d\'initialisation:', error);
            this.updateStatus('❌ Erreur d\'initialisation: ' + error.message, 'error');
        }
    }

    /**
     * Load Transformers.js library
     */
    async loadTransformersJS() {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.type = 'module';
            script.textContent = `
                import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';
                window.transformersPipeline = pipeline;
            `;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
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
                this.askButton.textContent = 'Traitement...';
            }

            this.updateStatus('🔍 Recherche sémantique locale...', 'searching');

            // Find relevant chunks
            const relevantChunks = await this.findRelevantChunks(question);

            console.log('Chunks pertinents:', relevantChunks);

            this.updateStatus('🤖 Génération de la réponse (Gemini API)...', 'generating');

            // Call the Cloudflare Worker
            const response = await fetch(this.workerUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    context: relevantChunks.map(c => c.chunk)
                })
            });

            if (!response.ok) {
                throw new Error(`Erreur API: ${response.status}`);
            }

            const data = await response.json();

            // Display the answer
            this.displayAnswer(data.answer, relevantChunks);

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
                this.askButton.textContent = 'Poser la question';
            }
        }
    }

    /**
     * Display the answer
     */
    displayAnswer(answer, relevantChunks) {
        if (!this.answerElement) return;

        const html = `
            <div class="answer-box">
                <div class="answer-content">
                    ${this.formatAnswer(answer)}
                </div>
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

    ragSystem = new RAGSystem({
        workerUrl: questionBlock.dataset.workerUrl || 'https://rag-blog-worker.YOUR-SUBDOMAIN.workers.dev',
        statusElement,
        answerElement,
        questionInput,
        askButton
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
