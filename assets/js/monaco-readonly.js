(() => {
  'use strict';

  const MONACO_VERSION = '0.52.0';
  const MONACO_BASE_URL = `https://cdn.jsdelivr.net/npm/monaco-editor@${MONACO_VERSION}/min`;
  const languageAliases = {
    bash: 'shell',
    csharp: 'csharp',
    css: 'css',
    go: 'go',
    html: 'html',
    java: 'java',
    javascript: 'javascript',
    js: 'javascript',
    json: 'json',
    markdown: 'markdown',
    php: 'php',
    plaintext: 'plaintext',
    python: 'python',
    py: 'python',
    ruby: 'ruby',
    rust: 'rust',
    scss: 'scss',
    shell: 'shell',
    sql: 'sql',
    typescript: 'typescript',
    ts: 'typescript',
    xml: 'xml',
    yaml: 'yaml',
    yml: 'yaml',
  };

  function color(name, fallback) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  }

  function detectLanguage(block) {
    const classes = [
      block.className,
      block.parentElement?.className,
      block.querySelector('code')?.className,
      block.querySelector('pre')?.className,
    ].filter(Boolean).join(' ');
    const match = classes.match(/(?:language-|lang-)([a-z0-9+#-]+)/i);
    return languageAliases[match?.[1]?.toLowerCase()] || 'plaintext';
  }

  function loadMonaco() {
    if (window.monaco) return Promise.resolve(window.monaco);

    return new Promise((resolve, reject) => {
      const loader = document.createElement('script');
      loader.src = `${MONACO_BASE_URL}/vs/loader.js`;
      loader.async = true;
      loader.onload = () => {
        window.require.config({ paths: { vs: `${MONACO_BASE_URL}/vs` } });
        window.require(['vs/editor/editor.main'], () => resolve(window.monaco), reject);
      };
      loader.onerror = reject;
      document.head.append(loader);
    });
  }

  function defineTheme(monaco) {
    const dark = document.body.classList.contains('dark-mode');
    const background = color('--code-bg', dark ? '#141519' : '#f1efe7');
    const foreground = color('--ink', dark ? '#e7e4da' : '#202124');

    monaco.editor.defineTheme('justgeek-readonly', {
      base: dark ? 'vs-dark' : 'vs',
      inherit: true,
      rules: [],
      colors: {
        'editor.background': background,
        'editor.foreground': foreground,
        'editor.lineHighlightBackground': background,
        'editorLineNumber.foreground': color('--faint', dark ? '#767773' : '#8e8c83'),
        'editorLineNumber.activeForeground': color('--soft', foreground),
        'editorGutter.background': background,
        'editorCursor.foreground': color('--accent', '#e45f35'),
        'editor.selectionBackground': color('--accent-bg', '#e45f3530'),
        'editorIndentGuide.background1': color('--code-line', '#d8d5cc'),
      },
    });
  }

  function createEditor(monaco, fallback) {
    const source = fallback.querySelector('.rouge-code pre') || fallback.querySelector('pre');
    if (!source || fallback.dataset.monacoEnhanced) return;

    const host = document.createElement('div');
    const lineCount = Math.max(source.textContent.split('\n').length, 1);
    const lineHeight = 22;
    host.className = 'monaco-code-editor';
    host.style.height = `${Math.min(Math.max(lineCount * lineHeight + 24, 82), 520)}px`;
    fallback.before(host);

    const editor = monaco.editor.create(host, {
      value: source.textContent.replace(/^\n/, '').replace(/\n$/, ''),
      language: detectLanguage(fallback),
      theme: 'justgeek-readonly',
      readOnly: true,
      domReadOnly: true,
      minimap: { enabled: false },
      lineNumbers: 'on',
      lineNumbersMinChars: 3,
      glyphMargin: false,
      folding: false,
      scrollBeyondLastLine: false,
      renderLineHighlight: 'none',
      roundedSelection: false,
      overviewRulerLanes: 0,
      hideCursorInOverviewRuler: true,
      lineHeight,
      fontFamily: 'IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace',
      fontSize: 14,
      wordWrap: 'off',
      automaticLayout: true,
      padding: { top: 12, bottom: 12 },
      scrollbar: { verticalScrollbarSize: 10, horizontalScrollbarSize: 10 },
    });

    fallback.hidden = true;
    fallback.dataset.monacoEnhanced = 'true';
    host.addEventListener('monaco-dispose', () => editor.dispose(), { once: true });
  }

  async function enhanceCodeBlocks() {
    if (!window.matchMedia('(min-width: 768px)').matches) return;
    const fallbacks = [...document.querySelectorAll('.highlight')].filter((block) => block.querySelector('pre'));
    if (!fallbacks.length) return;

    try {
      const monaco = await loadMonaco();
      defineTheme(monaco);
      fallbacks.forEach((fallback) => createEditor(monaco, fallback));
    } catch {
      // Le HTML Rouge visible reste le repli en cas de CDN indisponible.
    }
  }

  enhanceCodeBlocks();
})();
