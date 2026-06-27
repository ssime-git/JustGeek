---
layout: post-interactive
title: "Apprendre à coder en 2026 : arrête de demander la réponse, construis ton coach"
date: 2026-06-27
author: "Sébastien Sime"
categories: [IA, Cloudflare, WebAssembly]
tags: [agents, cloudflare-workers, cloudflare-think, wasm, apprentissage, code, rag, turnstile]
---

<link rel="stylesheet" href="{{ '/assets/css/codecoach-2026.css' | relative_url }}">
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
<script src="{{ '/assets/js/codecoach-2026.js' | relative_url }}" defer></script>

<div class="codecoach-hero">
  <span class="codecoach-kicker">Article interactif · Worker réel · Agent réel</span>
  <p><strong>Opinion un peu brutale :</strong> si tu apprends à coder en 2026 en demandant à une IA de te pondre la solution complète, tu n'apprends pas à coder. Tu apprends à être spectateur d'un autocomplete géant.</p>
</div>

<div class="codecoach-journey">

<section class="codecoach-step" markdown="1" data-next-label="Ok, montre-moi le piège →">

Tu connais la scène.

Tu bloques sur une fonction. Tu ouvres ton agent de code préféré. Tu écris :

> "Corrige-moi ça."

Il te sort 80 lignes propres, typées, probablement meilleures que les tiennes. Tu copies. Ça marche. Tu ressens une micro-dose de dopamine. Et cinq minutes plus tard, si je te demande **pourquoi** ça marche, ton cerveau affiche :

<div class="codecoach-meme">MOI DEVANT LE CODE GÉNÉRÉ

cerveau.exe has stopped working

[ Copier ] [ Coller ] [ Espérer que personne ne pose de question ]</div>

Le problème n'est pas l'IA. Le problème, c'est le contrat pédagogique qu'on signe avec elle.

Un mauvais contrat :

```text
Moi : fais-le à ma place.
IA  : ok.
Moi : je n'ai rien appris.
```

Un bon contrat :

```text
Moi : aide-moi à raisonner.
IA  : explique ton hypothèse, testons-la, puis corrigeons le modèle mental.
Moi : je comprends.
```

C'est cette deuxième version que je veux explorer ici.

</section>

<section class="codecoach-step" markdown="1" data-next-label="Je veux le prototype réel →">

## La thèse

En 2026, apprendre à coder devrait ressembler moins à :

> "regarder un tuto de 47 minutes en x1.5"

et plus à :

> "voyager dans un mini-lab où chaque hypothèse peut être testée, discutée, corrigée."

Un article de blog ne devrait pas seulement être une page à lire. Il peut devenir un **environnement d'apprentissage**.

Pas besoin de construire un clone de VS Code dans le navigateur pour commencer. Une bonne première version peut être très simple :

<div class="codecoach-mini-list">
  <div><strong>1. Un exercice court</strong><br>Le lecteur doit prédire, écrire ou corriger quelque chose.</div>
  <div><strong>2. Une validation déterministe</strong><br>Une brique non-LLM vérifie la réponse. Aujourd'hui TypeScript, demain WebAssembly.</div>
  <div><strong>3. Un coach IA</strong><br>L'agent ne juge pas seulement vrai/faux. Il explique l'intuition manquante.</div>
  <div><strong>4. Une mémoire pédagogique</strong><br>Le système peut progressivement savoir ce que tu comprends déjà.</div>
</div>

L'idée n'est pas de remplacer l'effort. L'idée est de mieux placer l'effort.

</section>

<section class="codecoach-step" markdown="1" data-next-label="Montre-moi l'architecture →">

## Le prototype

Pour tester cette idée, j'ai construit un petit Worker Cloudflare réel :

```text
https://blog-codecoach-2026.seb-sime.workers.dev
```

Il expose trois routes :

```text
GET  /health
POST /api/check/closure
POST /api/coach/closure
```

La dernière route appelle un agent Cloudflare Think, qui appelle lui-même un tool de validation.

<figure class="codecoach-figure">
  <img src="https://images.unsplash.com/photo-1515879218367-8466d910aaa4?auto=format&fit=crop&w=1400&q=80" alt="Code sur un écran dans une ambiance sombre">
  <figcaption>Le but n'est pas d'avoir plus de code. Le but est d'avoir une meilleure boucle de feedback.</figcaption>
</figure>

<div class="codecoach-architecture">
<pre><code>Lecteur / Article interactif
  │
  │ répond à un exercice
  ▼
Frontend du blog
  │
  │ fetch + Turnstile token
  ▼
Cloudflare Worker
  │
  ├─ Route directe : /api/check/closure
  │    └─ validation déterministe
  │
  └─ CodeCoachAgent via Cloudflare Think
       │
       ├─ Workers AI
       ├─ Prompt de coach
       ├─ Tool checkClosureAnswer()
       ├─ Mini-RAG local
       └─ Durable Object SQLite</code></pre>
</div>

Et surtout : les appels qui consomment le modèle sont protégés par une whitelist d'origine + Cloudflare Turnstile. Parce que laisser un endpoint IA public sans protection, c'est littéralement poser une carte bancaire sur une table de cantine.

</section>

<section class="codecoach-step" markdown="1" data-next-label="Je tente l'exercice →">

## Mini-exercice : les closures

Voici le code. Ne scrolle pas mentalement vers la réponse. Joue le jeu.

```js
function createCounter() {
  let count = 0;

  return function () {
    count++;
    return count;
  };
}

const counter = createCounter();
counter();
counter();
```

Question : **quelle valeur retourne le deuxième appel à `counter()` ?**

<div class="codecoach-callout">
<strong>Ne cherche pas à avoir raison.</strong> Cherche à voir ton intuition. C'est là que l'apprentissage commence.
</div>

<div class="codecoach-exercise" data-exercise="closure-counter">
  <label for="closure-answer">Ta réponse</label>
  <input id="closure-answer" name="answer" type="text" inputmode="text" placeholder="Ex: 1, 2, undefined...">
  <div class="codecoach-actions">
    <button type="button" data-action="check">Vérifier sans IA</button>
    <button type="button" data-action="ask-agent">Demander au coach IA</button>
  </div>
  <div class="codecoach-turnstile" style="display:none"></div>
  <div class="codecoach-result" aria-live="polite">Écris une réponse, puis vérifie. Le bouton "coach" appelle le vrai Worker Cloudflare.</div>
</div>

</section>

<section class="codecoach-step" markdown="1" data-next-label="Pourquoi WASM dans l'histoire ? →">

## Ce qui vient de se passer

Si tu as cliqué sur "Vérifier sans IA", tu as parlé à une fonction déterministe. Pas à un modèle. Pas à une boule de cristal. Une fonction.

Si tu as cliqué sur "Demander au coach IA", le flow était différent :

```text
Ta réponse
  ↓
Cloudflare Worker
  ↓
CodeCoachAgent
  ↓
Tool checkClosureAnswer
  ↓
Réponse pédagogique
```

C'est important.

Un modèle de langage est excellent pour expliquer, reformuler, analogiser. Il est moins fiable pour être l'arbitre ultime du vrai/faux. Donc on ne lui donne pas le sifflet. On lui donne le rôle de coach.

<div class="codecoach-meme">RÉPARTITION DES RÔLES

LLM       : expliquer, questionner, guider
Tool      : vérifier, calculer, exécuter
Humain    : prédire, se tromper, comprendre

Si tu inverses les rôles, bienvenue dans le chaos.</div>

</section>

<section class="codecoach-step" markdown="1" data-next-label="Ok, et WebAssembly ? →">

## Pourquoi WebAssembly est intéressant ici

Dans cette première version, la validation est encore en TypeScript. C'est volontaire : je voulais d'abord vérifier le produit pédagogique minimal.

Mais la prochaine étape logique, c'est WebAssembly.

Pourquoi ? Parce qu'un environnement d'apprentissage a souvent besoin d'une brique qui **exécute** ou **valide** du code de manière contrôlée.

WebAssembly est intéressant parce qu'il donne une cible portable, rapide, relativement isolée, et compatible avec le navigateur comme avec certains runtimes serveur.

Attention, phrase obligatoire pour éviter les takes LinkedIn éclatés :

<div class="codecoach-callout">
<strong>WASM n'est pas une sandbox magique.</strong> C'est une brique utile dans une stratégie d'isolation. Pas un bouclier divin contre toutes les mauvaises idées.
</div>

Dans ce prototype, WASM servira d'abord à remplacer le validateur TypeScript par un module minimal :

```text
answer: "2"
  ↓
WASM validator
  ↓
{ correct: true, hint: "..." }
```

Puis, plus tard, on pourra imaginer :

- des tests unitaires embarqués ;
- des mini-interpréteurs ;
- des exercices Rust → WASM ;
- des corrections déterministes ;
- des environnements plus riches.

</section>

<section class="codecoach-step" markdown="1" data-next-label="Et le RAG alors ? →">

## Où intervient le RAG ?

Un coach générique, c'est utile. Un coach qui connaît **tes articles**, **tes analogies**, **tes exercices**, c'est beaucoup plus intéressant.

Le mini-RAG actuel est volontairement simple : un petit corpus local avec quelques passages sur les closures, WASM et le rôle du coach.

Mais la cible est claire : brancher l'agent à mes contenus existants.

```text
Mes articles WASM
Mes explications
Mes exemples
Mes exercices
Mes prompts pédagogiques
      ↓
RAG
      ↓
Coach IA qui répond avec mon contexte
```

C'est là que le blog devient autre chose qu'une archive. Il devient une base de connaissances interactive.

<figure class="codecoach-figure">
  <img src="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=1400&q=80" alt="Personne travaillant sur un ordinateur portable">
  <figcaption>Le rêve : chaque article devient un petit atelier, pas juste un PDF avec du CSS.</figcaption>
</figure>

</section>

<section class="codecoach-step" markdown="1" data-next-label="Donne-moi les règles du jeu →">

## Les règles pédagogiques que je veux imposer à l'agent

Un agent de code utilisé n'importe comment devient vite un distributeur automatique de solutions.

Donc le prompt système du coach impose quelques règles :

```text
- Ne donne pas la solution trop tôt.
- Demande l'intuition de l'utilisateur.
- Appelle un outil quand une réponse doit être vérifiée.
- Si la réponse est fausse, donne un indice, pas une humiliation.
- Si la réponse est correcte, renforce le modèle mental.
- Ne prétends jamais avoir vérifié sans avoir appelé le tool.
```

Cette dernière règle est centrale.

Un agent ne doit pas dire :

> "J'ai vérifié."

s'il a juste halluciné une vérification dans son salon intérieur.

Il doit appeler un outil. Obtenir un résultat. Puis expliquer.

C'est la différence entre :

```text
confiance esthétique : "ça a l'air vrai"
```

et :

```text
confiance instrumentée : "j'ai un résultat observable"
```

</section>

<section class="codecoach-step" markdown="1" data-next-label="Et la sécurité ? →">

## La partie pas sexy mais obligatoire : éviter de se faire spammer

Dès qu'un article appelle un modèle en ligne, il y a un risque simple : quelqu'un peut essayer d'utiliser ton endpoint comme proxy gratuit.

Donc le Worker est protégé par deux couches :

1. **CORS whitelist** : seuls les navigateurs venant de mon blog peuvent lire la réponse.
2. **Cloudflare Turnstile** : le navigateur doit fournir un token anti-bot.

Concrètement, le blog récupère un token Turnstile invisible, puis l'envoie au Worker :

```json
{
  "message": "Je pense que la réponse est 2...",
  "turnstileToken": "..."
}
```

Le Worker vérifie ce token côté serveur avant d'appeler l'agent.

CORS seul ne suffit pas. CORS bloque surtout le navigateur. Un serveur ou un `curl` motivé s'en fiche. Turnstile sert donc de vraie barrière anti-abus pour les routes qui consomment Workers AI.

</section>

<section class="codecoach-step" markdown="1" data-next-label="Conclusion →">

## Ce que ça change pour apprendre à coder

Apprendre à coder avec l'IA ne devrait pas devenir :

> "je demande, je copie, j'oublie."

Ça devrait devenir :

> "je prédis, je teste, je discute, je corrige mon modèle mental."

C'est une nuance énorme.

Le code n'est pas seulement une production. C'est une conversation avec une machine très stricte. Si ton modèle mental est faux, le programme te le dit. Si ton coach est bien conçu, il transforme ce signal en apprentissage.

L'article interactif de 2026, dans ma tête, ressemble à ça :

```text
Lire
  ↓
Prédire
  ↓
Tester
  ↓
Se tromper
  ↓
Comprendre
  ↓
Recommencer
```

Et oui, l'IA est là. Mais elle n'est pas là pour penser à ta place.

Elle est là pour te forcer gentiment à penser mieux.

</section>

<section class="codecoach-step" markdown="1">

## Si tu veux creuser

Le prototype utilisé dans cet article repose sur :

- Cloudflare Workers ;
- Cloudflare Think ;
- Workers AI ;
- Durable Objects SQLite ;
- un mini-RAG local ;
- Cloudflare Turnstile ;
- bientôt un validateur WebAssembly.

La prochaine étape sera de remplacer le validateur TypeScript par un vrai module WASM minimal, puis de brancher le RAG existant sur mes anciens articles.

Si tu as joué l'exercice jusqu'au bout, tu as déjà vu le point essentiel :

<div class="codecoach-callout">
<strong>La valeur n'est pas dans la réponse.</strong><br>
La valeur est dans la boucle qui t'oblige à formuler une hypothèse, à la tester, puis à ajuster ton intuition.
</div>

Bienvenue dans l'apprentissage du code version 2026.

</section>

</div>
