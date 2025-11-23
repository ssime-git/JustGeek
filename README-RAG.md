# Système RAG Local pour Blog Technique Interactif

Ce blog implémente un système RAG (Retrieval-Augmented Generation) permettant aux lecteurs de poser des questions sur le contenu des articles.

## 📋 Fonctionnalités

- **Questions contextuelles** : Posez des questions sur l'article en cours de lecture
- **Recherche sémantique locale** : Utilise Transformers.js dans le navigateur (pas de serveur backend)
- **Réponses intelligentes** : Génération de réponses via Gemini API
- **100% statique** : Compatible avec GitHub Pages
- **Responsive** : Fonctionne sur tous les appareils
- **Gratuit** : Utilise des tiers gratuits (Cloudflare Workers, Gemini API)

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│      Blog Static (GitHub Pages)      │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Transformers.js (local)       │ │
│  │  - Chunking de l'article       │ │
│  │  - Génération d'embeddings     │ │
│  │  - Recherche sémantique        │ │
│  └───────────┬────────────────────┘ │
└──────────────┼──────────────────────┘
               │
               │ POST {question, context}
               ↓
┌─────────────────────────────────────┐
│   Cloudflare Worker (proxy)         │
│   - Protège la clé API Gemini       │
│   - Construit le prompt             │
│   - Appelle Gemini API              │
└───────────┬─────────────────────────┘
            │
            │ POST avec API key
            ↓
┌─────────────────────────────────────┐
│      Gemini 1.5 Flash API           │
│      (Google AI Studio)             │
└─────────────────────────────────────┘
```

## 🚀 Guide de Démarrage Rapide

### Étape 1 : Déployer le Cloudflare Worker

1. Allez dans le dossier `cloudflare-worker/`
2. Suivez les instructions du [README Cloudflare Worker](./cloudflare-worker/README.md)
3. Notez l'URL de votre worker déployé

### Étape 2 : Configurer l'URL du Worker

Dans vos articles qui utilisent le système RAG, mettez à jour l'attribut `data-worker-url` :

```html
<div class="question-block" data-worker-url="https://rag-blog-worker.YOUR-SUBDOMAIN.workers.dev">
```

### Étape 3 : Publier votre article

Créez ou modifiez un article avec :

```markdown
---
layout: post-interactive
title: "Votre titre"
---

Votre contenu...

<div class="question-block" data-worker-url="https://rag-blog-worker.YOUR-SUBDOMAIN.workers.dev">
  <h3>💬 Une question sur l'article ?</h3>
  <p>Posez votre question et obtenez une réponse basée sur le contenu de cet article.</p>

  <div id="rag-status">⏳ Initialisation du système RAG local...</div>

  <div class="question-input-wrapper">
    <input
      type="text"
      id="user-question"
      placeholder="Ex: Quelle est la différence entre..."
      disabled
    />
    <button id="ask-button" disabled>⏳ Chargement...</button>
  </div>

  <div id="answer-container"></div>
</div>
```

## 📁 Structure des Fichiers

```
JustGeek/
├── _layouts/
│   └── post-interactive.html    # Layout pour articles avec RAG
├── _posts/
│   └── 2025-11-23-transformers-expliques.md  # Article de démo
├── assets/
│   ├── css/
│   │   └── rag-interactive.css  # Styles pour le RAG
│   └── js/
│       └── rag-system.js        # Système RAG client-side
├── cloudflare-worker/
│   ├── src/
│   │   └── index.js            # Code du worker
│   ├── wrangler.toml           # Config Cloudflare
│   ├── package.json
│   └── README.md               # Guide de déploiement
└── README-RAG.md               # Ce fichier
```

## 🎨 Personnalisation

### Modifier les Styles

Éditez `assets/css/rag-interactive.css` pour personnaliser :
- Les couleurs du bloc de questions
- Les animations
- Le style des réponses
- Le responsive design

### Modifier le Prompt Gemini

Éditez la fonction `buildPrompt()` dans `cloudflare-worker/src/index.js` :

```javascript
function buildPrompt(question, context) {
  // Personnalisez votre prompt ici
}
```

### Modifier le Modèle d'Embeddings

Par défaut, le système utilise `Xenova/all-MiniLM-L6-v2` (25MB).

Pour changer de modèle, éditez `assets/js/rag-system.js` :

```javascript
this.pipeline = await pipelineFunc('feature-extraction', 'Xenova/VOTRE-MODELE');
```

Modèles alternatifs :
- `Xenova/multilingual-MiniLM-L12-v2` (50MB, meilleur multilingue)
- `Xenova/paraphrase-multilingual-mpnet-base-v2` (120MB, qualité supérieure)

## 🔧 Fonctionnement Technique

### Initialisation (au chargement de la page)

1. **Chargement de Transformers.js** (~25MB)
2. **Chunking de l'article** en passages de ~200 mots
3. **Génération des embeddings** pour chaque chunk (local dans le navigateur)
4. **Temps total** : 10-15 secondes

### Question/Réponse (à chaque question)

1. **Génération de l'embedding** de la question (local)
2. **Recherche sémantique** : Calcul de similarité cosine avec tous les chunks
3. **Sélection des top 3 chunks** les plus pertinents
4. **Envoi au Worker** : Question + 3 chunks
5. **Appel Gemini API** : Génération de la réponse
6. **Affichage** : Réponse + passages utilisés (accordéon)
7. **Temps total** : 3-5 secondes

## 📊 Métriques et Performances

### Taille des Ressources

- Transformers.js : ~25MB (CDN)
- rag-system.js : ~10KB
- rag-interactive.css : ~5KB

### Temps de Chargement

- Initialisation RAG : 10-15s
- Réponse à une question : 3-5s
- Recherche locale : 1-2s
- Appel API Gemini : 2-3s

### Limites

- **Cloudflare Workers** : 100,000 requêtes/jour (gratuit)
- **Gemini API** : 15 requêtes/minute (gratuit)
- **Chunks maximum** : ~10 par article (recommandé)

## 🔒 Sécurité et Confidentialité

### Données Envoyées au Worker

- Question de l'utilisateur
- 3 passages de l'article (max ~600 mots)
- Pas d'identifiants, pas de cookies, pas de tracking

### Protection de la Clé API

- Clé Gemini stockée comme secret Cloudflare
- Jamais exposée côté client
- Accessible uniquement par le worker

### Données Stockées

- **Aucune donnée persistée**
- Pas de logs des questions
- Cache navigateur : Modèle embeddings (peut être vidé)

## 📱 Compatibilité Navigateurs

- ✅ Chrome/Edge 90+
- ✅ Firefox 90+
- ✅ Safari 15+
- ✅ Mobile (iOS Safari, Chrome Android)

## ❓ FAQ

### Le système RAG fonctionne-t-il hors ligne ?

Non, il nécessite une connexion internet pour :
- Charger Transformers.js depuis le CDN (première fois)
- Appeler l'API Gemini pour générer les réponses

Cependant, la recherche sémantique se fait localement dans le navigateur.

### Combien coûte le système RAG ?

**Gratuit !** Avec les tiers suivants :
- GitHub Pages : gratuit
- Cloudflare Workers : 100k requêtes/jour (gratuit)
- Gemini 1.5 Flash : 15 requêtes/minute (gratuit)

Pour un blog personnel, ces limites sont largement suffisantes.

### Puis-je utiliser un autre LLM que Gemini ?

Oui ! Vous pouvez modifier le worker pour utiliser :
- Claude API (Anthropic)
- OpenAI GPT-4
- Mistral API
- Ollama (auto-hébergé)

Modifiez la fonction `callGeminiAPI()` dans `cloudflare-worker/src/index.js`.

### Comment désactiver le RAG sur certains articles ?

Utilisez simplement le layout `post` au lieu de `post-interactive` :

```markdown
---
layout: post
title: "Article sans RAG"
---
```

### Les questions sont-elles sauvegardées ?

Non, aucune donnée n'est stockée. Chaque question est traitée de manière indépendante et aucun historique n'est conservé.

## 🐛 Dépannage

### Le système RAG ne s'initialise pas

1. Vérifiez la console du navigateur (F12)
2. Vérifiez que Transformers.js se charge correctement
3. Désactivez les bloqueurs de publicité

### Erreur "Erreur API: 500"

Le Cloudflare Worker n'est pas correctement configuré :
1. Vérifiez que le worker est déployé
2. Vérifiez que `GEMINI_API_KEY` est configurée : `wrangler secret list`

### Les réponses sont de mauvaise qualité

1. Vérifiez que le chunking fonctionne bien (console)
2. Augmentez le nombre de chunks retournés (de 3 à 5)
3. Modifiez le prompt dans le worker

### Le chargement est trop lent

1. Utilisez un modèle d'embeddings plus léger
2. Réduisez le nombre de chunks
3. Préchargez Transformers.js en arrière-plan

## 📚 Ressources

- [Documentation Transformers.js](https://huggingface.co/docs/transformers.js)
- [Documentation Cloudflare Workers](https://developers.cloudflare.com/workers/)
- [Documentation Gemini API](https://ai.google.dev/docs)
- [Article de référence : Attention is All You Need](https://arxiv.org/abs/1706.03762)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer des améliorations
- Partager vos articles utilisant ce système

## 📄 Licence

MIT License - Libre d'utilisation et de modification

---

**Version** : 1.0
**Dernière mise à jour** : 23 Novembre 2025
**Auteur** : Sébastien Sime
