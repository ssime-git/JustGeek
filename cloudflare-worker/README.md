# RAG Blog Worker - Cloudflare Worker

Ce worker Cloudflare sert de proxy sécurisé entre votre blog statique et l'API Gemini. Il protège votre clé API Gemini et gère les appels à l'API.

## Prérequis

1. **Compte Cloudflare** (gratuit)
2. **Clé API Google AI Studio** (gratuit)
   - Rendez-vous sur [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Créez une clé API pour Gemini
3. **Node.js** et **npm** installés
4. **Wrangler CLI** installé

## Installation

### 1. Installer Wrangler CLI

```bash
npm install -g wrangler
```

### 2. Se connecter à Cloudflare

```bash
wrangler login
```

Cela ouvrira votre navigateur pour vous connecter à votre compte Cloudflare.

### 3. Obtenir votre clé API Gemini

1. Allez sur [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Créez une nouvelle clé API
3. Copiez la clé (vous en aurez besoin à l'étape suivante)

## Déploiement

### 1. Déployer le worker

Depuis le dossier `cloudflare-worker/` :

```bash
wrangler deploy
```

### 2. Configurer la clé API Gemini (secret)

```bash
wrangler secret put GEMINI_API_KEY
```

Collez votre clé API Gemini quand demandé.

### 3. Obtenir l'URL du worker

Après le déploiement, Wrangler affichera l'URL de votre worker :

```
https://rag-blog-worker.YOUR-SUBDOMAIN.workers.dev
```

Copiez cette URL.

## Configuration du Blog

### 1. Mettre à jour l'article

Dans vos articles qui utilisent le système RAG (ex: `_posts/2025-11-23-transformers-expliques.md`), mettez à jour l'attribut `data-worker-url` :

```html
<div class="question-block" data-worker-url="https://rag-blog-worker.YOUR-SUBDOMAIN.workers.dev">
```

Remplacez `YOUR-SUBDOMAIN` par votre vrai sous-domaine Cloudflare.

### 2. Mettre à jour le fichier JavaScript

Si vous n'utilisez pas `data-worker-url`, vous pouvez aussi mettre à jour l'URL directement dans `assets/js/rag-system.js` :

```javascript
this.workerUrl = options.workerUrl || 'https://rag-blog-worker.YOUR-SUBDOMAIN.workers.dev';
```

## Test

Pour tester que le worker fonctionne correctement :

```bash
curl -X POST https://rag-blog-worker.YOUR-SUBDOMAIN.workers.dev \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Qu'\''est-ce qu'\''un Transformer ?",
    "context": ["Les Transformers sont une architecture de réseau de neurones."]
  }'
```

Vous devriez recevoir une réponse JSON avec une réponse générée par Gemini.

## Monitoring

### Voir les logs

```bash
wrangler tail
```

### Voir les métriques

Allez sur votre [dashboard Cloudflare](https://dash.cloudflare.com/) :
- Workers & Pages
- Sélectionnez `rag-blog-worker`
- Onglet "Metrics"

## Limites (Tier Gratuit)

- **100,000 requêtes/jour**
- **10ms CPU time par requête**
- Largement suffisant pour un blog personnel !

## Coûts API Gemini

- **Gemini 1.5 Flash** est gratuit pour jusqu'à 15 requêtes par minute
- Largement suffisant pour un blog
- Voir [Google AI Pricing](https://ai.google.dev/pricing) pour les détails

## Sécurité

✅ **Ce qui est sécurisé** :
- La clé API Gemini est stockée comme secret Cloudflare
- Jamais exposée côté client
- CORS configuré pour accepter toutes les origines (vous pouvez le restreindre si besoin)

⚠️ **À considérer** :
- Pas de rate limiting par défaut (vous pouvez en ajouter si nécessaire)
- Pas d'authentification des utilisateurs (public)

## Personnalisation

### Modifier le prompt

Éditez la fonction `buildPrompt()` dans `src/index.js` :

```javascript
function buildPrompt(question, context) {
  // Votre prompt personnalisé ici
}
```

### Ajouter du rate limiting

Vous pouvez ajouter du rate limiting en utilisant Cloudflare KV :

```javascript
// Exemple basique
const rateLimitKey = `ratelimit:${clientIP}`;
const count = await env.RATE_LIMIT.get(rateLimitKey);

if (count && parseInt(count) > 10) {
  return new Response('Too many requests', { status: 429 });
}

await env.RATE_LIMIT.put(rateLimitKey, (parseInt(count || 0) + 1).toString(), {
  expirationTtl: 60, // 1 minute
});
```

### Restreindre CORS

Pour n'autoriser que votre domaine :

```javascript
const corsHeaders = {
  'Access-Control-Allow-Origin': 'https://ssime-git.github.io',
  // ...
};
```

## Dépannage

### Erreur 500 : Service configuration error

➡️ La clé API Gemini n'est pas configurée. Exécutez :

```bash
wrangler secret put GEMINI_API_KEY
```

### Erreur 403 : Access Denied

➡️ Vérifiez que votre clé API Gemini est valide sur [Google AI Studio](https://makersuite.google.com/app/apikey).

### Erreur CORS

➡️ Vérifiez que les headers CORS sont bien définis dans le worker.

## Commandes Utiles

```bash
# Déployer
wrangler deploy

# Voir les logs en temps réel
wrangler tail

# Lister les secrets
wrangler secret list

# Supprimer un secret
wrangler secret delete GEMINI_API_KEY

# Tester localement
wrangler dev
```

## Support

Pour toute question ou problème :
- Consultez la [documentation Wrangler](https://developers.cloudflare.com/workers/wrangler/)
- Consultez la [documentation Gemini API](https://ai.google.dev/docs)

---

**Version** : 1.0
**Dernière mise à jour** : 23 Novembre 2025
