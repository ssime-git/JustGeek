---
layout: post-interactive
title: "Les Transformers Expliqués : De l'Attention à ChatGPT"
date: 2025-11-23
author: "Sébastien Sime"
categories: [Machine Learning, NLP]
tags: [transformers, attention, deep-learning, ia]
---

## Introduction

Les **Transformers** ont révolutionné le traitement du langage naturel (NLP) et sont à la base de tous les grands modèles de langage modernes comme GPT, BERT, et bien d'autres. Introduits en 2017 par Vaswani et al. dans le célèbre article "Attention is All You Need", les Transformers ont remplacé les architectures récurrentes (RNN, LSTM) pour devenir la référence en NLP.

Dans cet article, nous allons explorer en détail le fonctionnement des Transformers, en mettant l'accent sur le mécanisme d'**attention** qui en est le cœur.

## Le Problème des Architectures Récurrentes

Avant les Transformers, les modèles de séquence utilisaient principalement des **réseaux de neurones récurrents** (RNN) et des **LSTM**. Ces architectures avaient plusieurs limitations :

### Traitement Séquentiel

Les RNN traitent les mots un par un, dans l'ordre. Cela pose deux problèmes majeurs :

1. **Parallélisation impossible** : On ne peut pas traiter plusieurs mots en même temps, ce qui ralentit l'entraînement
2. **Dépendances longue distance** : Les informations du début de la phrase peuvent se perdre quand on arrive à la fin

### Gradient Vanishing

Même avec les LSTM, il est difficile de capturer des dépendances sur de très longues séquences. Le gradient a tendance à disparaître lors de la rétropropagation à travers de nombreux pas de temps.

## L'Idée Révolutionnaire : L'Attention

Les Transformers proposent une solution élégante : **et si chaque mot pouvait directement regarder tous les autres mots de la phrase ?**

C'est exactement ce que fait le mécanisme d'**auto-attention** (self-attention). Au lieu de traiter les mots séquentiellement, chaque mot calcule une "attention" sur tous les autres mots pour déterminer lesquels sont importants pour le comprendre.

### Exemple Concret

Prenons la phrase : **"Le chat mange la souris"**

Quand le modèle traite le mot "mange", il va calculer des scores d'attention pour tous les autres mots :

- **"Le"** : faible attention (article peu informatif)
- **"chat"** : forte attention (sujet de l'action)
- **"la"** : faible attention
- **"souris"** : forte attention (objet de l'action)

Le mot "mange" va donc se concentrer sur "chat" et "souris" pour mieux se comprendre dans le contexte.

## Le Mécanisme d'Auto-Attention (Self-Attention)

Le mécanisme d'attention utilise trois concepts clés : les **Query (Q)**, **Key (K)**, et **Value (V)**.

### Les Trois Matrices : Q, K, V

Pour chaque mot de la phrase, on calcule trois vecteurs :

1. **Query (Q)** : "Que cherche ce mot ?"
2. **Key (K)** : "Qu'est-ce que ce mot offre ?"
3. **Value (V)** : "Quelle information ce mot contient ?"

Ces trois vecteurs sont obtenus en multipliant l'embedding du mot par trois matrices de poids apprises pendant l'entraînement.

### Calcul de l'Attention

Le score d'attention entre deux mots est calculé en trois étapes :

1. **Score** : Produit scalaire entre la Query d'un mot et la Key d'un autre
   ```
   score(mot_i, mot_j) = Q_i · K_j
   ```

2. **Normalisation** : Division par la racine carrée de la dimension (pour stabiliser les gradients)
   ```
   score_normalized = score / sqrt(d_k)
   ```

3. **Softmax** : Conversion en probabilités
   ```
   attention_weights = softmax(score_normalized)
   ```

4. **Weighted Sum** : Somme pondérée des Values
   ```
   output = sum(attention_weights * V)
   ```

### Formule Mathématique

La formule complète de l'attention est :

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Où :
- `Q`, `K`, `V` sont les matrices Query, Key, Value
- `d_k` est la dimension des vecteurs Key
- `QK^T` est le produit matriciel donnant les scores d'attention

## Multi-Head Attention

Une des innovations clés des Transformers est l'utilisation de **plusieurs têtes d'attention** en parallèle.

### Pourquoi Plusieurs Têtes ?

Chaque tête d'attention peut se spécialiser dans la capture d'un type de relation différent :

- **Tête 1** : Relations syntaxiques (sujet-verbe)
- **Tête 2** : Relations sémantiques (synonymes, antonymes)
- **Tête 3** : Références anaphoriques (pronoms → noms)

### Fonctionnement

Au lieu d'avoir une seule attention, on en calcule 8 ou 12 en parallèle (selon le modèle), chacune avec ses propres matrices Q, K, V. Les résultats sont ensuite concaténés et projetés.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

où head_i = Attention(Q*W_Qi, K*W_Ki, V*W_Vi)
```

## Positional Encoding

Il y a un problème : contrairement aux RNN, les Transformers ne traitent pas les mots dans l'ordre. La phrase "Le chat mange la souris" et "La souris mange le chat" produiraient le même résultat !

Pour résoudre ce problème, on ajoute un **encodage positionnel** (positional encoding) aux embeddings des mots.

### Encodage Sinusoïdal

L'article original utilise des fonctions sinusoïdales pour encoder la position :

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Où :
- `pos` est la position du mot dans la phrase
- `i` est la dimension
- `d_model` est la dimension du modèle

Cet encodage a l'avantage de permettre au modèle de généraliser à des séquences plus longues que celles vues pendant l'entraînement.

## Architecture Complète d'un Transformer

Un Transformer complet se compose de deux parties :

### Encoder

1. **Embeddings** + Positional Encoding
2. **Multi-Head Attention** (auto-attention)
3. **Add & Norm** (connexion résiduelle + normalisation)
4. **Feed-Forward Network** (deux couches linéaires avec ReLU)
5. **Add & Norm**

Ces blocs sont répétés N fois (typiquement 6 ou 12 fois).

### Decoder

Similaire à l'encoder, mais avec une couche supplémentaire :

1. **Masked Multi-Head Attention** (pour éviter de regarder le futur)
2. **Add & Norm**
3. **Multi-Head Attention** (attention croisée avec l'encoder)
4. **Add & Norm**
5. **Feed-Forward Network**
6. **Add & Norm**

## Applications des Transformers

Les Transformers sont utilisés dans de nombreuses tâches :

### Traitement du Langage

- **BERT** : Pré-entraînement bidirectionnel pour la compréhension
- **GPT** : Génération de texte autoregressive
- **T5** : Approche text-to-text unifiée

### Au-delà du Texte

- **Vision Transformers (ViT)** : Classification d'images
- **DALL-E** : Génération d'images à partir de texte
- **Whisper** : Reconnaissance vocale
- **AlphaFold** : Prédiction de structure de protéines

## Avantages des Transformers

1. **Parallélisation** : Tous les mots sont traités en même temps
2. **Dépendances longue distance** : Chaque mot peut directement interagir avec tous les autres
3. **Interprétabilité** : On peut visualiser les matrices d'attention
4. **Flexibilité** : S'adapte à différentes modalités (texte, images, audio)

## Limitations

1. **Complexité quadratique** : O(n²) en mémoire et calcul pour une séquence de longueur n
2. **Besoin de données** : Nécessite beaucoup de données pour l'entraînement
3. **Coût computationnel** : Très gourmand en ressources GPU

## Évolutions Récentes

Pour résoudre la limitation de complexité, plusieurs variantes ont été proposées :

- **Longformer** : Attention sparse pour les documents longs
- **Reformer** : Utilisation de LSH (Locality-Sensitive Hashing)
- **Linear Transformers** : Attention linéaire en O(n)

## Conclusion

Les Transformers ont révolutionné le machine learning et continuent d'évoluer. Le mécanisme d'attention, simple mais puissant, permet de capturer des relations complexes dans les données de manière parallélisable et efficace.

Avec l'avènement des grands modèles de langage (LLM) comme GPT-4 ou Claude, les Transformers sont devenus le socle de l'intelligence artificielle moderne.

---

<div class="question-block" data-worker-url="https://rag-blog-worker.seb-sime.workers.dev/api/ask">
  <h3>💬 Une question sur l'article ?</h3>
  <p>Posez votre question et obtenez une réponse basée sur le contenu de cet article grâce au système RAG local.</p>

  <div id="rag-status">⏳ Initialisation du système RAG local...</div>

  <div class="question-input-wrapper">
    <input
      type="text"
      id="user-question"
      placeholder="Ex: Quelle est la différence entre Q, K et V dans l'attention ?"
      disabled
    />
    <button id="ask-button" disabled>⏳ Chargement...</button>
  </div>

  <div id="answer-container"></div>
</div>

---

## Références

- Vaswani, A., et al. (2017). "Attention is All You Need"
- Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners" (GPT-3)
- Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition"
