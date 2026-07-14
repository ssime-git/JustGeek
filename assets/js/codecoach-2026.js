const CODECOACH_WORKER_URL = 'https://blog-codecoach-2026.seb-sime.workers.dev';
const CODECOACH_TURNSTILE_SITE_KEY = '0x4AAAAAACEY6vkwXOKRVPHk';

const EXERCISES = {
  'closure-counter': {
    checkPath: '/api/check/closure',
    coachPath: '/api/coach/closure',
    emptyMessage: "Écris d'abord ton intuition. Même si elle est fausse. Surtout si elle est fausse.",
    buildMessage: (answer) => `Je pense que la réponse de l'exercice closure-counter est ${answer}. Vérifie ma réponse avec le tool, puis explique-moi sans me noyer.`,
  },
  'double-function': {
    checkPath: '/api/check/double',
    coachPath: '/api/coach/double',
    emptyMessage: "Écris une expression pour remplacer le trou. Pas besoin d'être brillant : commence par une hypothèse.",
    buildMessage: (answer) => `Pour compléter function double(n), je propose ${answer}. Vérifie avec le tool puis donne-moi un feedback de coach, court et utile.`,
  },
  'typescript-type': {
    checkPath: '/api/check/typescript-type',
    coachPath: '/api/coach/typescript-type',
    emptyMessage: "Propose un type. L'idée n'est pas de briller, c'est de rendre ton hypothèse visible.",
    buildMessage: (answer) => `Pour compléter let score: ___ = 42 en TypeScript, je propose ${answer}. Vérifie avec le tool puis explique le contrat de type en langage simple.`,
  },
  'async-bug': {
    checkPath: '/api/check/async-bug',
    coachPath: '/api/coach/async-bug',
    emptyMessage: "Décris ce que tu penses être le bug avant d'appeler le coach.",
    buildMessage: (answer) => `Dans la fonction fetchUser, voici mon diagnostic du bug : ${answer}. Vérifie avec le tool checkAsyncBug puis donne-moi un feedback de coach.`,
  },
};

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
    label.textContent = `Étape ${currentIndex + 1}/${steps.length} — ${percent}% du parcours débloqué`;
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
    throw new Error("Turnstile n'est pas encore chargé. Recharge la page ou réessaie dans une seconde.");
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

function renderCheckPayload(payload) {
  const tests = payload.tests
    ? `<details><summary>Tests mentaux utilisés</summary><ul>${payload.tests.map((test) => `<li><code>${test.input}</code> → <code>${test.expected}</code></li>`).join('')}</ul></details>`
    : '';

  return `<strong>${payload.correct ? '✅ Correct.' : '🧠 Pas encore.'}</strong><br>
    ${escapeHtml(payload.hint)}<br>
    <small>Réponse normalisée : <code>${escapeHtml(payload.normalizedAnswer)}</code></small>
    ${tests}`;
}

function initCodeCoachExercises() {
  document.querySelectorAll('.codecoach-exercise').forEach((block) => {
    const exerciseId = block.dataset.exercise || 'closure-counter';
    const config = EXERCISES[exerciseId];
    const input = block.querySelector('input[name="answer"]');
    const result = block.querySelector('.codecoach-result');
    const tokenContainer = block.querySelector('.codecoach-turnstile');
    const checkButton = block.querySelector('[data-action="check"]');
    const agentButton = block.querySelector('[data-action="ask-agent"]');

    if (!config || !input || !result) return;

    async function checkOnly() {
      const answer = input.value.trim();
      if (!answer) {
        renderResult(result, config.emptyMessage, 'error');
        return;
      }

      renderResult(result, 'Validation déterministe en cours…', 'loading');
      const data = await postJson(`${CODECOACH_WORKER_URL}${config.checkPath}`, { answer });
      renderResult(result, renderCheckPayload(data.result));
    }

    async function askAgent() {
      const answer = input.value.trim();
      if (!answer) {
        renderResult(result, "Donne une réponse avant d'appeler le coach. Le coach n'apprend rien si tu ne risques rien.", 'error');
        return;
      }

      // Show initial thinking state with streaming UI
      const thinkingId = `thinking-${Date.now()}`;
      let reasoningText = '';
      let responseText = '';
      const toolsUsed = [];

      renderResult(
        result,
        `<div class="codecoach-thinking-stream" id="${thinkingId}">
          <div class="thinking-dots">
            <span class="thinking-label">Le coach réfléchit</span>
            <span class="dot-pulse">...</span>
          </div>
          <div class="thinking-reasoning" style="display:none"></div>
          <div class="thinking-bar"><div class="thinking-bar-fill"></div></div>
        </div>`,
        'loading'
      );

      const thinkingEl = () => document.getElementById(thinkingId);
      const reasoningEl = () => thinkingEl()?.querySelector('.thinking-reasoning');
      const labelEl = () => thinkingEl()?.querySelector('.thinking-label');

      let turnstileToken = null;
      if (tokenContainer) {
        try {
          turnstileToken = await getTurnstileToken(tokenContainer);
        } catch (e) {
          renderResult(result, escapeHtml(e.message), 'error');
          return;
        }
      }

      // Update label
      const label = labelEl();
      if (label) label.textContent = 'Le coach analyse ton hypothèse';

      try {
        const response = await fetch(`${CODECOACH_WORKER_URL}${config.coachPath}`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            name: `blog-reader-${exerciseId}-${Date.now()}`,
            message: config.buildMessage(answer),
            turnstileToken,
          }),
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
          throw new Error(err.error || `HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let reasoningStarted = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          let eventType = '';
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ') && eventType) {
              const data = line.slice(6);
              try {
                const parsed = JSON.parse(data);

                if (eventType === 'reasoning' && parsed.text) {
                  if (!reasoningStarted) {
                    reasoningStarted = true;
                    const rel = reasoningEl();
                    if (rel) {
                      rel.style.display = 'block';
                      const lbl = labelEl();
                      if (lbl) lbl.textContent = 'Pensée du coach';
                    }
                  }
                  reasoningText += parsed.text;
                  const rel = reasoningEl();
                  if (rel) rel.textContent = reasoningText;
                } else if (eventType === 'text' && parsed.text) {
                  // First text chunk: collapse the thinking panel
                  if (reasoningStarted) {
                    const tel = thinkingEl();
                    if (tel) tel.classList.add('thinking-done');
                  }
                  responseText += parsed.text;
                } else if (eventType === 'tool' && parsed.toolName) {
                  toolsUsed.push(parsed);
                }
              } catch {
                // skip malformed chunks
              }
              eventType = '';
            }
          }
        }

        // Stream complete — show final answer after collapse animation
        setTimeout(() => {
          const tools = toolsUsed.length
            ? `<details><summary>Outils appelés (${toolsUsed.length})</summary><ul>${toolsUsed.map(t => `<li><code>${escapeHtml(t.toolName)}</code></li>`).join('')}</ul></details>`
            : '';
          renderResult(
            result,
            `<strong>Coach</strong>
            <p>${formatCoachText(responseText || 'Pas de réponse.')}</p>
            ${tools}`
          );
        }, 400);

      } catch (error) {
        renderResult(result, escapeHtml(error.message), 'error');
      } finally {
        if (typeof turnstile !== 'undefined') {
          turnstile.reset();
        }
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
  // Warmup: pre-instantiate the coach DO to avoid cold start on first click
  fetch(`${CODECOACH_WORKER_URL}/api/warmup`).catch(() => {});

  initJourneyReveal();
  initCodeCoachExercises();
  initJsCells();
});

function initJsCells() {
  document.querySelectorAll('.js-cell').forEach((cell) => {
    const btn = cell.querySelector('[data-action="run-js"]');
    const output = cell.querySelector('.js-cell-output');
    const pre = cell.querySelector('.js-cell-code');
    if (!btn || !output || !pre) return;

    btn.addEventListener('click', () => {
      const code = pre.textContent || '';
      output.style.display = 'block';
      output.className = 'js-cell-output running';
      output.textContent = '▶ Exécution…';

      // Simulate fetch returning a real Response so the Promise chain is intact
      // but WITHOUT await — the bug manifests exactly as in prod
      const logs = [];
      const fakeConsole = { log: (...args) => logs.push(args.map(String).join(' ')) };

      const fakeFetch = (_url) =>
        Promise.resolve(new Response(JSON.stringify({ name: 'Alice', id: 1 })));

      // Wrap in AsyncFunction so top-level await works
      const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
      const wrapped = new AsyncFunction('fetch', 'console', code);

      wrapped(fakeFetch, fakeConsole)
        .then((returnVal) => {
          // The function returns data.name which will be undefined because of the bug
          const lines = [...logs];
          output.className = 'js-cell-output done';
          output.innerHTML =
            lines.map((l) => `<span class="out-line">${escapeHtml(l)}</span>`).join('') +
            `<span class="out-line muted">→ valeur retournée : <strong>${escapeHtml(String(returnVal))}</strong></span>`;

          // Reveal the question after execution
          const question = cell.closest('section')?.querySelector('.js-cell-question');
          if (question) {
            question.style.display = 'block';
            question.classList.add('js-cell-question-reveal');
          }
        })
        .catch((err) => {
          output.className = 'js-cell-output error';
          output.textContent = `Erreur : ${err.message}`;
          const question = cell.closest('section')?.querySelector('.js-cell-question');
          if (question) {
            question.style.display = 'block';
            question.classList.add('js-cell-question-reveal');
          }
        });
    });
  });
}
