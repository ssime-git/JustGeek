---
layout: post-interactive
title: "Qu'est-ce qu'un Agent IA ? La Boucle que le Tool Calling Cache"
date: 2026-03-29
author: "Sébastien Sime"
categories: [IA, LLM, Agents]
tags: [llm, agent, tool-calling, boucle-autonome, ia]
---

## Introduction

Tout le monde confond. On dit "Claude Code est un LLM" ou "GPT-4 avec tools est un agent". **Faux.**

Un **LLM** (Large Language Model) génère du texte. Rien d'autre.  
Un **Agent** est un **système** qui utilise un LLM comme cerveau, mais ajoute des **mains** (outils) et une **boucle autonome**.

Dans cet article, je vais te montrer exactement ce qui change, avec du code que tu peux exécuter dans ton navigateur.


## Partie 1 : Le LLM Seul

### Ce qu'il fait vraiment

Un LLM ne comprend rien. Il prédit la suite de caractères la plus probable.

**Entrée :** `"Quelle heure est-il ?"`  
**Sortie :** `"Je suis un modèle de langage, je n'ai pas accès à l'heure..."`

Le LLM n'a pas accès à vos fichiers, à internet, ni à l'heure. Il s'arrête après avoir généré sa réponse.


## Demo 1 : Voyez ce que génère vraiment un LLM

Le simulateur ci-dessous reçoit une question et génère une réponse. Observe bien : il ne fait qu'écrire du texte.

<div class="mini-demo" id="demo-llm-only">
  <div class="demo-header">
    <span class="demo-badge">Demo 1</span>
    <span class="demo-title">LLM seul : juste du texte</span>
  </div>
  
  <div class="demo-flow">
    <div class="flow-box user">Toi<br>"Quelle heure ?"</div>
    <div class="flow-arrow">→</div>
    <div class="flow-box llm">LLM<br>(prédiction)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-box output">"Je ne sais pas..."</div>
  </div>
  
  <button class="demo-run-btn" data-demo="llm-only">Simuler la réponse</button>
  <div class="demo-output" id="output-llm-only"></div>
</div>

**Le constat :** Le LLM s'arrête net. Il ne peut pas chercher l'heure, ni faire quoi que ce soit d'autre.


## Partie 2 : Le Tool Calling

Les LLM modernes peuvent générer des **appels structurés** sous forme de JSON. Mais attention au piège.

### Ce que génère le LLM

Quand on lui dit "Tu peux appeler des outils", il répond ainsi :

```json
{
  "tool_call": {
    "name": "get_current_time",
    "arguments": {}
  }
}
```

**Mais ce n'est pas l'agent !** Le LLM a juste généré du texte formaté. Quelqu'un doit **lire ce JSON et exécuter** la fonction.


## Demo 2 : Le Tool Call en Action

Voyons ce que le LLM génère concrètement. Ce JSON ne s'exécute pas tout seul.

<div class="mini-demo" id="demo-tool-call">
  <div class="demo-header">
    <span class="demo-badge">Demo 2</span>
    <span class="demo-title">Tool Calling : le LLM génère un JSON</span>
  </div>
  
  <div class="demo-flow">
    <div class="flow-box user">Toi<br>"Quelle heure ?"</div>
    <div class="flow-arrow">→</div>
    <div class="flow-box llm">LLM<br>(décide)</div>
    <div class="flow-arrow">→</div>
    <div class="flow-box json">JSON généré<br><code>{"tool":"get_time"}</code></div>
  </div>
  
  <div class="demo-halt">ET LÀ, LE LLM S'ARRÊTE</div>
  
  <button class="demo-run-btn" data-demo="tool-call">Voir le JSON généré</button>
  <div class="demo-output" id="output-tool-call"></div>
</div>

**La question clé :** Qui exécute ce JSON ? Certainement pas le LLM.


## Partie 3 : Le Programme Intermédiaire

Entre le LLM et l'outil, il faut un **programme** qui :
1. Parse le JSON
2. Appelle la vraie fonction
3. Renvoie le résultat au LLM

```python
# C'EST TON CODE, PAS LE LLM
llm_response = {"tool": "get_time", "args": {}}

if llm_response["tool"] == "get_time":
    result = get_current_time()  # ← Exécution réelle ici
    # ← Ensuite on renvoie au LLM
```

Sans ce programme, le JSON reste inerte. C'est toi qui dois faire la boucle manuellement.


## Demo 3 : Ce qui Manque (la boucle)

Voici le code minimal qu'il faut ajouter pour que ça fonctionne. Sans ça, strictement rien ne se passe.

<div class="mini-demo" id="demo-missing-loop">
  <div class="demo-header">
    <span class="demo-badge">Demo 3</span>
    <span class="demo-title">La boucle manquante (ton code)</span>
  </div>
  
  <div class="demo-flow">
    <div class="flow-box json">JSON<br><code>{"tool":"get_time"}</code></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-box code">Ton code<br>parse + exécute</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-box result">Résultat<br>"14h30"</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-box llm">LLM<br>reçoit le résultat</div>
  </div>
  
  <button class="demo-run-btn" data-demo="missing-loop">Exécuter la boucle</button>
  <div class="demo-output" id="output-missing-loop"></div>
</div>

**Cette boucle, c'est ce qui manque.** Le LLM seul ne l'a pas. C'est ce qui transforme un simple générateur de texte en système capable d'agir.


## Partie 4 : L'Agent Autonome

L'agent ajoute une **boucle autonome** qui tourne sans vous. Voici l'architecture complète :

```
Vous (objectif) → LLM (décide) → JSON (tool call)
                                         ↓
                              Programme (exécute)
                                         ↓
                              Résultat → LLM (analyse)
                                         ↓
                        Nouveau tool call ? → Oui → Boucle
                                        ↓ Non
                                     Terminé
```

**La différence cruciale :** L'agent gère tout seul les itérations. Tu donnes un objectif, il s'occupe du reste.


## Demo 4 : L'Agent Complet en Python

Voici un agent minimal que tu peux exécuter toi-même. Il lit un fichier, analyse le résultat, et répond, tout seul.

<div class="mini-demo agent-full" id="demo-agent-full">
  <div class="demo-header">
    <span class="demo-badge demo-badge-final">Demo 4</span>
    <span class="demo-title">L'Agent : boucle autonome complète</span>
  </div>
  
  <div class="agent-status-mini" id="agent-status-mini">Prêt</div>
  
  <div class="agent-visualizer-mini">
    <div class="mini-step" data-step="1">Objectif</div>
    <div class="mini-arrow">→</div>
    <div class="mini-step" data-step="2">Décide</div>
    <div class="mini-arrow">→</div>
    <div class="mini-step" data-step="3">Exécute</div>
    <div class="mini-arrow">→</div>
    <div class="mini-step" data-step="4">Résultat</div>
    <div class="mini-arrow">→</div>
    <div class="mini-step" data-step="5">Répond</div>
  </div>
  
  <div class="agent-code-mini">
    <pre><code>def run_agent(goal):
    messages = [{"role": "user", "content": goal}]
    
    for i in range(5):  # Boucle autonome
        # 1. LLM décide
        response = llm.generate(messages, tools)
        
        if response.type == "final":
            return response.content  # Terminé
        
        # 2. Exécute l'outil
        result = execute_tool(response.tool_call)
        
        # 3. Continue la boucle
        messages.append({"role": "tool", "content": result})</code></pre>
  </div>
  
  <button class="demo-run-btn btn-primary" data-demo="agent-full">Exécuter l'Agent Autonome</button>
  <div class="demo-output agent-console" id="output-agent-full"></div>
</div>


## Résumé Visuel

| | LLM seul | Tool Calling | Agent |
|---|---|---|---|
| Génère du texte | Oui | Oui | Oui |
| Décide d'agir | Non | Oui | Oui |
| Exécute l'action | Non | Non | Oui |
| Boucle autonome | Non | Non | Oui |


## L'Analogie Finale

- **LLM** : Un médecin qui t'explique l'opération par téléphone. Tu dois tout faire toi-même.
- **Tool Calling** : Le médecin te prescrit un soin. Tu dois aller chercher le médicament toi-même.
- **Agent** : Le médecin est dans la salle d'opération avec ses scalpels. Il opère, vérifie, recouse, sans rien te demander entre deux.

L'agent n'est pas plus intelligent. Il est autonome. C'est toute la différence.


<!-- Turnstile Script -->
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>

<div class="question-block" data-worker-url="https://rag-blog-worker.seb-sime.workers.dev/api/ask">
  <h3>Une question sur les Agents ?</h3>
  <p>Pose ta question sur la différence LLM vs Agent, le function calling, ou la boucle autonome.</p>

  <div id="rag-status">Initialisation du système RAG...</div>

  <div class="question-input-wrapper">
    <input
      type="text"
      id="user-question"
      placeholder="Ex: Pourquoi le LLM ne peut pas exécuter directement ?"
      disabled
    />
    <button id="ask-button" disabled>Chargement...</button>
  </div>

  <div id="answer-container"></div>
</div>

<!-- Hidden container for Turnstile -->
<div id="turnstile-container" style="display:none;"></div>


## Références

- [Function Calling - OpenAI](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [ReAct Paper - Reasoning + Acting](https://arxiv.org/abs/2210.03629)
