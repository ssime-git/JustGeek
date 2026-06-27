const CODECOACH_WORKER_URL = 'https://blog-codecoach-2026.seb-sime.workers.dev';
const CODECOACH_TURNSTILE_SITE_KEY = '0x4AAAAAACEY6vkwXOKRVPHk';

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatCoachText(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

function initJourneyReveal() {
  const journey = document.querySelector('.codecoach-journey');
  if (!journey) return;

  const steps = Array.from(journey.querySelectorAll('.codecoach-step'));
  if (steps.length === 0) return;

  const progress = document.createElement('div');
  progress.className = 'codecoach-progress';
  progress.innerHTML = `
    <div class="codecoach-progress-track"><div class="codecoach-progress-bar"></div></div>
    <div class="codecoach-progress-label" aria-live="polite"></div>
  `;
  journey.prepend(progress);

  const bar = progress.querySelector('.codecoach-progress-bar');
  const label = progress.querySelector('.codecoach-progress-label');

  let currentIndex = 0;

  function updateProgress() {
    const percent = Math.round(((currentIndex + 1) / steps.length) * 100);
    bar.style.width = `${percent}%`;
    label.textContent = `Étape ${currentIndex + 1}/${steps.length} — ${percent}% du voyage débloqué`;
  }

  function reveal(index) {
    steps.forEach((step, i) => {
      step.classList.toggle('is-hidden', i > index);
      step.classList.toggle('is-visible', i <= index);
    });
    currentIndex = index;
    updateProgress();
  }

  steps.forEach((step, index) => {
    if (index < steps.length - 1 && !step.querySelector('.codecoach-next')) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'codecoach-next';
      button.textContent = step.dataset.nextLabel || 'Débloquer la suite →';
      button.addEventListener('click', () => {
        reveal(index + 1);
        steps[index + 1].scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
      step.appendChild(button);
    }
  });

  reveal(0);
}

async function getTurnstileToken(container) {
  if (typeof turnstile === 'undefined') {
    throw new Error('Turnstile n’est pas encore chargé. Recharge la page ou réessaie dans une seconde.');
  }

  return new Promise((resolve, reject) => {
    container.innerHTML = '';
    turnstile.render(container, {
      sitekey: CODECOACH_TURNSTILE_SITE_KEY,
      size: 'invisible',
      callback: (token) => resolve(token),
      'error-callback': () => reject(new Error('Échec Turnstile. Réessaie.')),
      'expired-callback': () => reject(new Error('Token Turnstile expiré. Réessaie.')),
    });
  });
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(data?.error || `Erreur HTTP ${response.status}`);
  }

  return data;
}

function renderResult(target, html, kind = 'ok') {
  target.className = `codecoach-result ${kind === 'error' ? 'error' : kind === 'loading' ? 'loading' : ''}`.trim();
  target.innerHTML = html;
}

function initCodeCoachExercises() {
  document.querySelectorAll('.codecoach-exercise').forEach((block) => {
    const input = block.querySelector('input[name="answer"]');
    const result = block.querySelector('.codecoach-result');
    const tokenContainer = block.querySelector('.codecoach-turnstile');
    const checkButton = block.querySelector('[data-action="check"]');
    const agentButton = block.querySelector('[data-action="ask-agent"]');

    if (!input || !result) return;

    async function checkOnly() {
      const answer = input.value.trim();
      if (!answer) {
        renderResult(result, 'Écris d’abord ton intuition. Même si elle est fausse. Surtout si elle est fausse.', 'error');
        return;
      }

      renderResult(result, 'Validation déterministe en cours…', 'loading');
      const data = await postJson(`${CODECOACH_WORKER_URL}/api/check/closure`, { answer });
      const payload = data.result;
      renderResult(
        result,
        `<strong>${payload.correct ? '✅ Correct.' : '🧠 Pas encore.'}</strong><br>${escapeHtml(payload.hint)}<br><small>Réponse normalisée : <code>${escapeHtml(payload.normalizedAnswer)}</code></small>`
      );
    }

    async function askAgent() {
      const answer = input.value.trim();
      if (!answer) {
        renderResult(result, 'Donne une réponse avant d’appeler le coach. Le coach n’apprend rien si tu ne risques rien.', 'error');
        return;
      }

      renderResult(result, 'Le coach vérifie ton raisonnement…', 'loading');

      let turnstileToken = null;
      if (tokenContainer) {
        turnstileToken = await getTurnstileToken(tokenContainer);
      }

      const message = `Je pense que la réponse de l'exercice closure-counter est ${answer}. Vérifie ma réponse avec le tool, puis explique-moi sans me noyer.`;
      const data = await postJson(`${CODECOACH_WORKER_URL}/api/coach/closure`, {
        name: `blog-reader-${Date.now()}`,
        message,
        turnstileToken,
      });

      const payload = data.result;
      const tools = payload.toolCalls?.map((tool) => `<li><code>${escapeHtml(tool.toolName)}</code></li>`).join('') || '';
      renderResult(
        result,
        `<strong>Coach IA</strong><p>${formatCoachText(payload.text || 'Pas de réponse textuelle.')}</p>${tools ? `<details><summary>Outil appelé</summary><ul>${tools}</ul></details>` : ''}`
      );

      if (typeof turnstile !== 'undefined') {
        turnstile.reset();
      }
    }

    checkButton?.addEventListener('click', async () => {
      try {
        checkButton.disabled = true;
        await checkOnly();
      } catch (error) {
        renderResult(result, escapeHtml(error.message), 'error');
      } finally {
        checkButton.disabled = false;
      }
    });

    agentButton?.addEventListener('click', async () => {
      try {
        agentButton.disabled = true;
        await askAgent();
      } catch (error) {
        renderResult(result, escapeHtml(error.message), 'error');
      } finally {
        agentButton.disabled = false;
      }
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initJourneyReveal();
  initCodeCoachExercises();
});
