---
layout: post-interactive
title: "Qu'est-ce qu'un Agent IA ? Construis-en un en 50 lignes"
date: 2026-03-29
author: "Sébastien Sime"
categories: [IA, LLM, Agents]
tags: [llm, agent, tool-calling, boucle-autonome, ia, python]
---

Tu utilises ChatGPT tous les jours. Tu crois parler à une IA intelligente. En réalité, tu parles à un **perroquet statistique enfermé dans une boîte**. Il prédit le mot suivant, point final.

L'agent, lui, a les clés de la boîte. Il peut sortir, agir, et revenir avec des résultats.

Dans cet article, tu vas **construire un agent de zéro**, étape par étape, avec du vrai code Python qui s'exécute dans ton navigateur. Pas de mock, pas de simulation. Du code réel.


## Pour t'expliquer avec une analogie

Imagine un médecin :

- **LLM seul** : Il t'explique l'opération par téléphone. "Coupe ici, recouds là." Tu dois tout faire toi-même.
- **LLM + Tool Calling** : Il te prescrit un médicament. Tu dois aller le chercher à la pharmacie.
- **Agent** : Il est dans la salle d'opération. Il coupe, vérifie, recoud, sans rien te demander.

L'agent n'est pas plus intelligent. Il est **autonome**. C'est toute la différence.

On va construire ce médecin-chirurgien, étape par étape.


## Étape 1 : Le LLM seul (le perroquet)

Un LLM ne "comprend" rien. Il prédit la suite de caractères la plus probable. Demande-lui l'heure, il va t'expliquer qu'il ne peut pas la connaître.

<div class="ascii-art"><code>┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│     Toi     │ ──── │     LLM     │ ──── │   Texte     │
│  "Quelle    │      │  (prédit)   │      │  "Je ne     │
│   heure?"   │      │             │      │   sais pas" │
└─────────────┘      └─────────────┘      └─────────────┘
                            │
                            ▼
                         [STOP]
                     Pas d'action possible</code></div>

**Exécute ce code pour voir (n'hésite pas à scroller vers le bas pour voir tout le code) :**

<div class="pyodide-cell" id="demo-step1">
  <div class="demo-header">
    <span class="demo-badge">Étape 1</span>
    <span class="demo-title">Le LLM seul : juste du texte</span>
  </div>
  <textarea class="pyodide-code" rows="12" readonly>
# Simulation d'un LLM basique
# En vrai, c'est un appel API à OpenAI/Anthropic

def llm_generate(prompt):
    """
    Le LLM ne fait que générer du texte.
    Il n'a accès à RIEN d'autre.
    """
    print(f"[INPUT]  Prompt reçu: '{prompt}'")
    print(f"[LLM]    Génération en cours...")
    
    # Le LLM génère une réponse (simulée mais réaliste)
    response = "Je suis un modèle de langage. Je n'ai pas accès à l'heure actuelle, ni à vos fichiers, ni à internet."
    
    print(f"[OUTPUT] '{response}'")
    print(f"\n>>> Le LLM s'arrête ici. Il ne peut rien faire d'autre.")
    return response

# Test
llm_generate("Quelle heure est-il ?")
</textarea>
  <div class="pyodide-controls">
    <button data-pyodide-action="run">Exécuter</button>
    <button data-pyodide-action="clear">Effacer</button>
    <span class="pyodide-status"></span>
  </div>
  <div class="pyodide-output"></div>
</div>

**Constat :** Le LLM génère du texte, puis s'arrête. Il ne peut pas aller chercher l'heure. Il est enfermé dans sa boîte.


## Étape 2 : Le Tool Calling (la prescription)

Les LLM modernes peuvent générer des **appels structurés** en JSON. Mais comment le LLM sait-il quels outils existent ?

### Le prompt système : la clé de tout

Quand tu appelles l'API d'OpenAI ou Anthropic avec des outils, tu envoies quelque chose comme ça :

```json
{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Quelle heure est-il ?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_time",
        "description": "Retourne l'heure actuelle du serveur",
        "parameters": {"type": "object", "properties": {}}
      }
    },
    {
      "type": "function",
      "function": {
        "name": "calculate",
        "description": "Évalue une expression mathématique",
        "parameters": {
          "type": "object",
          "properties": {
            "expression": {"type": "string", "description": "L'expression à calculer"}
          }
        }
      }
    }
  ]
}
```

Le LLM **voit** cette liste d'outils dans son contexte. Il peut alors faire du **pattern matching** : par exemple "L'utilisateur demande l'heure → j'ai un outil `get_time` → je génère un JSON pour l'appeler."

C'est du texte qui génère du texte. Pas de magie.

### Ce que le LLM génère

Quand il juge qu'un outil est pertinent, il répond :

<div class="ascii-art"><code>┌─────────────┐      ┌─────────────┐      ┌─────────────────────┐
│     Toi     │ ──── │     LLM     │ ──── │   JSON (texte)      │
│  "Quelle    │      │  (prédit)   │      │  {"tool":"get_time"}│
│   heure?"   │      │             │      │                     │
└─────────────┘      └─────────────┘      └─────────────────────┘
                                                    │
                                                    ▼
                                                   ???
                                          Qui exécute ce JSON ?</code></div>

Mais attention : **ce n'est que du texte**. Le LLM a juste écrit du JSON bien formaté. Personne ne l'a exécuté.

**Exécute ce code pour voir le JSON brut :**

<div class="pyodide-cell" id="demo-step2">
  <div class="demo-header">
    <span class="demo-badge">Étape 2</span>
    <span class="demo-title">Tool Calling : le LLM génère du JSON</span>
  </div>
  <textarea class="pyodide-code" rows="20" readonly>
import json

def llm_with_tools(prompt, available_tools):
    """
    Le LLM sait qu'il a des outils disponibles.
    Il génère un JSON structuré pour les appeler.
    """
    print(f"[INPUT]  Prompt: '{prompt}'")
    print(f"[INPUT]  Outils disponibles: {available_tools}")
    print(f"[LLM]    Analyse de la demande...")
    
    # Le LLM décide d'appeler un outil (simulé)
    tool_call = {
        "type": "tool_call",
        "tool": "get_time",
        "arguments": {}
    }
    
    print(f"[LLM]    Décision: appeler un outil")
    print(f"[OUTPUT] JSON généré:")
    print(json.dumps(tool_call, indent=2))
    
    print(f"\n>>> Le LLM a généré du JSON. Mais QUI va l'exécuter ?")
    print(f">>> Ce JSON est inerte. C'est juste du texte.")
    
    return tool_call

# Test
tools = ["get_time", "read_file", "calculate"]
llm_with_tools("Quelle heure est-il ?", tools)
</textarea>
  <div class="pyodide-controls">
    <button data-pyodide-action="run">Exécuter</button>
    <button data-pyodide-action="clear">Effacer</button>
    <span class="pyodide-status"></span>
  </div>
  <div class="pyodide-output"></div>
</div>

**Question clé :** Le LLM a généré `{"tool": "get_time"}`. Mais qui va réellement appeler la fonction `get_time()` ? Pas le LLM. C'est **ton code**.


## Étape 3 : Le Dispatcher (ton code)

Entre le LLM et l'outil, il faut un **programme intermédiaire** qui :
1. Parse le JSON du LLM
2. Trouve la bonne fonction
3. L'exécute vraiment
4. Récupère le résultat

<div class="ascii-art"><code>┌─────────┐    ┌─────────┐    ┌──────────────┐    ┌─────────┐    ┌──────────┐
│   Toi   │───▶│   LLM   │───▶│    JSON      │───▶│ TON CODE│───▶│  OUTIL   │
│         │    │         │    │ {"tool":...} │    │ (parse) │    │ get_time │
└─────────┘    └─────────┘    └──────────────┘    └─────────┘    └──────────┘
                                                       │              │
                                                       │   Exécute    │
                                                       │◀─────────────┘
                                                       │
                                                       ▼
                                                  "14:32:05"</code></div>

C'est ce code que personne ne te montre. Le voici :

<div class="pyodide-cell" id="demo-step3">
  <div class="demo-header">
    <span class="demo-badge">Étape 3</span>
    <span class="demo-title">Le Dispatcher : parser et exécuter</span>
  </div>
  <textarea class="pyodide-code" rows="35" readonly>
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# 1. DÉFINIR LES OUTILS (les vraies fonctions)
# ═══════════════════════════════════════════════════════════
def get_time():
    """Retourne l'heure actuelle"""
    return datetime.now().strftime("%H:%M:%S")

def calculate(expression):
    """Évalue une expression mathématique"""
    return str(eval(expression))

# Registre des outils
TOOLS = {
    "get_time": get_time,
    "calculate": calculate
}

print("=== OUTILS DISPONIBLES ===")
print(f"Fonctions: {list(TOOLS.keys())}")

# ═══════════════════════════════════════════════════════════
# 2. LE JSON DU LLM (simulé)
# ═══════════════════════════════════════════════════════════
llm_output = '{"type": "tool_call", "tool": "get_time", "arguments": {}}'

print(f"\n=== JSON REÇU DU LLM ===")
print(llm_output)

# ═══════════════════════════════════════════════════════════
# 3. TON CODE : PARSER ET EXÉCUTER
# ═══════════════════════════════════════════════════════════
print(f"\n=== PARSING ===")
parsed = json.loads(llm_output)
tool_name = parsed["tool"]
tool_args = parsed.get("arguments", {})

print(f"Outil demandé: {tool_name}")
print(f"Arguments: {tool_args}")

print(f"\n=== EXÉCUTION ===")
if tool_name in TOOLS:
    result = TOOLS[tool_name](**tool_args)
    print(f">>> {tool_name}() exécuté")
    print(f">>> Résultat: {result}")
else:
    print(f">>> Erreur: outil '{tool_name}' inconnu")

print(f"\n>>> Ce résultat doit maintenant être renvoyé au LLM.")
</textarea>
  <div class="pyodide-controls">
    <button data-pyodide-action="run">Exécuter</button>
    <button data-pyodide-action="clear">Effacer</button>
    <span class="pyodide-status"></span>
  </div>
  <div class="pyodide-output"></div>
</div>

**C'est ce code qui manque au LLM seul.** Sans lui, le JSON reste du texte inerte. Avec lui, le JSON devient une action réelle.

Mais il manque encore quelque chose : **la boucle**.


## Étape 4 : La Boucle Autonome (l'agent)

Un agent, c'est quand tu mets tout ça dans une **boucle while**. Le LLM reprend la main après chaque outil et décide : encore un outil, ou réponse finale ?

<div class="ascii-art"><code>                              ┌──────────────────────────────────────┐
                              │                                      │
                              ▼                                      │
┌─────────┐    ┌─────────────────────────┐    ┌──────────────┐      │
│   Toi   │───▶│          LLM            │───▶│  tool_call?  │      │
│ "Quelle │    │   (analyse contexte)    │    │              │      │
│ heure?" │    └─────────────────────────┘    └──────────────┘      │
└─────────┘                                          │              │
                                          ┌──────────┴──────────┐   │
                                          ▼                     ▼   │
                                    ┌──────────┐          ┌─────────┐
                                    │   OUI    │          │   NON   │
                                    │ Exécuter │          │ Réponse │
                                    │  outil   │          │ finale  │
                                    └──────────┘          └─────────┘
                                          │                     │
                                          │                     ▼
                                          │               ┌──────────┐
                                          │               │  "Il est │
                                          └───────────────│  14:32"  │
                                            Résultat      └──────────┘
                                            renvoyé
                                            au LLM</code></div>

**Sans intervention humaine.** Tu donnes un objectif, l'agent fait le reste.

Voici l'agent complet :

<div class="pyodide-cell" id="demo-step4">
  <div class="demo-header">
    <span class="demo-badge demo-badge-final">Étape 4</span>
    <span class="demo-title">L'Agent Complet : la boucle autonome</span>
  </div>
  <textarea class="pyodide-code" rows="75" readonly>
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# ÉTAPE 1 : DÉFINIR LES OUTILS
# ═══════════════════════════════════════════════════════════
def get_time():
    return datetime.now().strftime("%H:%M:%S")

def calculate(expression):
    return str(eval(expression))

def read_file(path):
    FILES = {
        "config.json": '{"version": "2.1.0", "debug": true}',
        "package.json": '{"name": "mon-app", "dependencies": {"react": "18.2.0"}}'
    }
    return FILES.get(path, f"Fichier '{path}' non trouvé")

TOOLS = {
    "get_time": get_time,
    "calculate": calculate,
    "read_file": read_file
}

# ═══════════════════════════════════════════════════════════
# ÉTAPE 2 : SIMULER LE LLM
# ═══════════════════════════════════════════════════════════
def llm_decide(messages):
    """
    Simule la décision du LLM.
    En production, c'est un appel API à OpenAI/Anthropic.
    """
    last_msg = messages[-1]
    context = " ".join(m.get("content", "") for m in messages).lower()
    
    # Vérifier ce qu'on a déjà collecté
    has_time = any(m["role"] == "tool" and ":" in m.get("content", "") for m in messages)
    has_calc = any(m["role"] == "tool" and m.get("tool") == "calculate" for m in messages)
    
    # Logique de décision
    if "heure" in context and not has_time:
        return {"type": "tool_call", "tool": "get_time", "arguments": {}}
    
    if ("+" in context or "-" in context or "*" in context) and not has_calc:
        # Extraire l'expression (simplifié)
        for expr in ["2+2", "10*5", "100-42"]:
            if expr.replace("*", "x") in context or expr in context:
                return {"type": "tool_call", "tool": "calculate", "arguments": {"expression": expr}}
        return {"type": "tool_call", "tool": "calculate", "arguments": {"expression": "2+2"}}
    
    # Construire la réponse finale
    parts = []
    for m in messages:
        if m["role"] == "tool":
            if ":" in m["content"] and m.get("tool") == "get_time":
                parts.append(f"Il est {m['content']}")
            elif m.get("tool") == "calculate":
                parts.append(f"Le résultat est {m['content']}")
    
    answer = ". ".join(parts) if parts else "Je n'ai pas pu répondre."
    return {"type": "answer", "content": answer}

# ═══════════════════════════════════════════════════════════
# ÉTAPE 3 : LA BOUCLE AGENT
# ═══════════════════════════════════════════════════════════
def run_agent(goal, max_iterations=5):
    print("=" * 60)
    print(f"AGENT DÉMARRÉ")
    print(f"Objectif: {goal}")
    print("=" * 60)
    
    messages = [{"role": "user", "content": goal}]
    
    for i in range(max_iterations):
        print(f"\n--- Itération {i + 1} ---")
        
        # 1. Le LLM analyse et décide
        print(f"[LLM] Analyse du contexte ({len(messages)} messages)...")
        decision = llm_decide(messages)
        
        # 2. Si réponse finale, on arrête
        if decision["type"] == "answer":
            print(f"[LLM] Type: RÉPONSE FINALE")
            print(f"\n{'=' * 60}")
            print(f"RÉPONSE: {decision['content']}")
            print(f"Agent terminé en {i + 1} itération(s)")
            print("=" * 60)
            return decision["content"]
        
        # 3. Sinon, exécuter l'outil
        tool_name = decision["tool"]
        tool_args = decision.get("arguments", {})
        
        print(f"[LLM] Type: TOOL_CALL")
        print(f"[LLM] Outil: {tool_name}({tool_args})")
        
        # Exécution réelle
        print(f"[EXEC] Appel de {tool_name}()...")
        result = TOOLS[tool_name](**tool_args)
        print(f"[EXEC] Résultat: {result}")
        
        # 4. Ajouter aux messages et continuer
        messages.append({
            "role": "tool",
            "tool": tool_name,
            "content": result
        })
        print(f"[LOOP] Résultat ajouté au contexte. Retour au LLM...")
    
    return "Nombre maximum d'itérations atteint"

# ═══════════════════════════════════════════════════════════
# EXÉCUTER L'AGENT
# ═══════════════════════════════════════════════════════════
run_agent("Quelle heure est-il et combien fait 2+2 ?")
</textarea>
  <div class="pyodide-controls">
    <button data-pyodide-action="run">Exécuter</button>
    <button data-pyodide-action="clear">Effacer</button>
    <span class="pyodide-status"></span>
  </div>
  <div class="pyodide-output"></div>
</div>

**Observe bien la console :**
1. Le LLM analyse → décide d'appeler `get_time()`
2. L'outil s'exécute → résultat ajouté au contexte
3. Le LLM reprend → décide d'appeler `calculate()`
4. L'outil s'exécute → résultat ajouté
5. Le LLM reprend → génère la réponse finale

**C'est ça, un agent.** Une boucle qui tourne jusqu'à ce que le LLM décide qu'il a fini.


## Étape 5 : À toi de jouer

L'agent ci-dessus a deux outils : `get_time()` et `calculate()`. Mais il manque quelque chose : **la lecture de fichiers**.

### L'exercice

Complète le code ci-dessous pour ajouter l'outil `read_file` :

1. Crée une fonction `read_file(path)` qui simule la lecture d'un fichier
2. Ajoute-la au dictionnaire `TOOLS`
3. Ajoute une condition dans le dispatcher pour détecter quand l'utilisateur demande de lire un fichier
4. Teste avec l'objectif : `"Lis le fichier config.json"`

<div class="pyodide-cell" id="demo-step5">
  <div class="demo-header">
    <span class="demo-badge" style="background: #8b5cf6;">Exercice</span>
    <span class="demo-title">Ajoute l'outil read_file</span>
  </div>
  <textarea class="pyodide-code" rows="35">
import json
from datetime import datetime

# Outils existants
def get_time():
    return datetime.now().strftime("%H:%M:%S")

def calculate(expression):
    return str(eval(expression))

# TODO: Ajoute ici la fonction read_file(path)
# Elle doit retourner le contenu d'un fichier simulé
# Utilise un dictionnaire FILES pour simuler les fichiers disponibles


# TODO: Ajoute read_file au dictionnaire TOOLS
TOOLS = {"get_time": get_time, "calculate": calculate}

# Change l'objectif pour tester ton outil !
GOAL = "Lis le fichier config.json"

print(f"Objectif: {GOAL}\n")

# Dispatcher simplifié
if "heure" in GOAL.lower():
    print("[LLM] Je dois appeler get_time()")
    result = get_time()
    print(f"[EXEC] Résultat: {result}")
    print(f"\n[RÉPONSE] Il est {result}")
elif "+" in GOAL or "calcul" in GOAL.lower():
    expr = "2+2"
    print(f"[LLM] Je dois appeler calculate('{expr}')")
    result = calculate(expr)
    print(f"[EXEC] Résultat: {result}")
    print(f"\n[RÉPONSE] {expr} = {result}")
# TODO: Ajoute une condition pour détecter "lis" ou "fichier" dans GOAL
# et appeler read_file avec le bon chemin
else:
    print("[LLM] Je ne sais pas quel outil utiliser.")
</textarea>
  <div class="pyodide-controls">
    <button data-pyodide-action="run">Exécuter</button>
    <button data-pyodide-action="clear">Effacer</button>
    <span class="pyodide-status"></span>
  </div>
  <div class="pyodide-output"></div>
</div>

<details>
<summary><strong>Voir la solution</strong></summary>

<div class="solution-code">
<pre><code>import json
from datetime import datetime

# Outils existants
def get_time():
    return datetime.now().strftime("%H:%M:%S")

def calculate(expression):
    return str(eval(expression))

# SOLUTION: La fonction read_file
def read_file(path):
    FILES = {
        "config.json": '{"version": "2.1.0", "debug": true}',
        "package.json": '{"name": "mon-app", "react": "18.2.0"}'
    }
    return FILES.get(path, f"Fichier '{path}' non trouvé")

# SOLUTION: Ajout au dictionnaire TOOLS
TOOLS = {"get_time": get_time, "calculate": calculate, "read_file": read_file}

GOAL = "Lis le fichier config.json"

print(f"Objectif: {GOAL}\n")

# Dispatcher avec read_file
if "heure" in GOAL.lower():
    print("[LLM] Je dois appeler get_time()")
    result = get_time()
    print(f"[EXEC] Résultat: {result}")
    print(f"\n[RÉPONSE] Il est {result}")
elif "+" in GOAL or "calcul" in GOAL.lower():
    expr = "2+2"
    print(f"[LLM] Je dois appeler calculate('{expr}')")
    result = calculate(expr)
    print(f"[EXEC] Résultat: {result}")
    print(f"\n[RÉPONSE] {expr} = {result}")
# SOLUTION: Condition pour read_file
elif "lis" in GOAL.lower() or "fichier" in GOAL.lower():
    # Extraire le nom du fichier (simplifié)
    path = "config.json" if "config" in GOAL else "package.json"
    print(f"[LLM] Je dois appeler read_file('{path}')")
    result = read_file(path)
    print(f"[EXEC] Résultat: {result}")
    print(f"\n[RÉPONSE] Contenu de {path}: {result}")
else:
    print("[LLM] Je ne sais pas quel outil utiliser.")</code></pre>
</div>

</details>


## Résumé : LLM vs Tool Calling vs Agent

| | LLM seul | Tool Calling | Agent |
|---|---|---|---|
| Génère du texte | Oui | Oui | Oui |
| Peut demander une action | Non | Oui | Oui |
| Exécute l'action | Non | Non | Oui |
| Boucle autonome | Non | Non | Oui |
| Gère plusieurs étapes | Non | Non | Oui |


## Ce que tu as appris

1. **Un LLM** génère du texte, rien d'autre
2. **Le Tool Calling** permet au LLM de générer des JSON structurés
3. **Le Dispatcher** (ton code) parse le JSON et exécute les fonctions
4. **L'Agent** met tout ça dans une boucle autonome

Un agent, c'est **50 lignes de Python**. Pas de magie, pas de framework complexe. Juste une boucle while, un dispatcher, et des outils.

La prochaine fois que tu utilises Claude ou Codex avec des outils, tu sauras exactement ce qui se passe sous le capot.


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
      placeholder="Ex: Comment ajouter un nouvel outil à l'agent ?"
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
