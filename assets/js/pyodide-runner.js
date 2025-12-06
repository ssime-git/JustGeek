(function () {
  const cells = document.querySelectorAll('.pyodide-cell');
  if (!cells.length) {
    return;
  }

  const PYODIDE_VERSION = '0.24.1';
  const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
  let pyodidePromise = null;
  let packagesLoaded = false;

  function ensurePyodideLoaded() {
    if (!pyodidePromise) {
      pyodidePromise = new Promise((resolve, reject) => {
        const start = async () => {
          try {
            const pyodide = await globalThis.loadPyodide({ indexURL: INDEX_URL });
            // Load NumPy and matplotlib automatically
            await pyodide.loadPackage(['numpy', 'matplotlib']);
            packagesLoaded = true;
            resolve(pyodide);
          } catch (error) {
            reject(error);
          }
        };

        if (globalThis.loadPyodide) {
          start();
          return;
        }

        const script = document.createElement('script');
        script.src = `${INDEX_URL}pyodide.js`;
        script.async = true;
        script.onload = start;
        script.onerror = () => reject(new Error('Failed to load Pyodide runtime.'));
        document.head.appendChild(script);
      });
    }

    return pyodidePromise;
  }

  function setStatus(statusEl, message, type = 'info') {
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.dataset.statusType = type;
  }

  function clearOutput(outputEl) {
    if (outputEl) {
      outputEl.textContent = '';
    }
  }

  function appendOutput(outputEl, text, isError = false) {
    if (!outputEl || !text) {
      return;
    }
    const span = document.createElement('span');
    span.className = isError ? 'pyodide-output-line error' : 'pyodide-output-line';
    span.textContent = text;
    outputEl.appendChild(span);
  }

  function appendImage(outputEl, base64Data) {
    if (!outputEl || !base64Data) {
      return;
    }
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${base64Data}`;
    img.style.maxWidth = '100%';
    img.style.height = 'auto';
    img.style.display = 'block';
    img.style.margin = '10px 0';
    outputEl.appendChild(img);
  }

  cells.forEach((cell) => {
    const textarea = cell.querySelector('textarea.pyodide-code');
    const runButton = cell.querySelector('[data-pyodide-action="run"]');
    const clearButton = cell.querySelector('[data-pyodide-action="clear"]');
    const outputEl = cell.querySelector('.pyodide-output');
    const statusEl = cell.querySelector('.pyodide-status');

    if (!textarea || !runButton || !outputEl) {
      return;
    }

    runButton.addEventListener('click', async () => {
      setStatus(statusEl, 'Loading Pyodide + NumPy + Matplotlib…', 'info');
      runButton.disabled = true;
      if (clearButton) clearButton.disabled = true;
      try {
        const pyodide = await ensurePyodideLoaded();
        setStatus(statusEl, 'Running…', 'info');

        clearOutput(outputEl);
        const code = textarea.value;
        pyodide.globals.set('code_to_run', code);

        const result = await pyodide.runPythonAsync(`
import sys, io, traceback
import numpy as np
import matplotlib
matplotlib.use('agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import builtins
import base64

_stdout, _stderr = sys.stdout, sys.stderr
stdout = io.StringIO()
stderr = io.StringIO()
sys.stdout = stdout
sys.stderr = stderr
success = True
plot_images = []

# Create execution namespace with builtins and loaded modules
exec_globals = {
    '__builtins__': builtins,
    'np': np,
    'numpy': np,
    'plt': plt,
    'matplotlib': matplotlib
}

try:
    exec(code_to_run, exec_globals)

    # Capture any matplotlib figures
    figs = [plt.figure(num) for num in plt.get_fignums()]
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plot_images.append(img_str)
        buf.close()
    plt.close('all')  # Close all figures

except Exception:
    success = False
    traceback.print_exc(file=stderr)
finally:
    sys.stdout = _stdout
    sys.stderr = _stderr
stdout.getvalue(), stderr.getvalue(), success, plot_images
        `);

        const [stdoutText, stderrText, success, plotImages] = result.toJs({ create_proxies: false });

        if (stdoutText) {
          appendOutput(outputEl, stdoutText, false);
        }
        if (stderrText) {
          appendOutput(outputEl, stderrText, true);
        }

        // Display matplotlib plots
        if (plotImages && plotImages.length > 0) {
          plotImages.forEach(imgData => {
            appendImage(outputEl, imgData);
          });
        }

        if (!stdoutText && !stderrText && (!plotImages || plotImages.length === 0)) {
          appendOutput(outputEl, '(no output)', false);
        }
        setStatus(statusEl, success ? 'Finished' : 'Finished with errors', success ? 'success' : 'error');
      } catch (error) {
        console.error('Pyodide execution failed', error); // eslint-disable-line no-console
        clearOutput(outputEl);
        appendOutput(outputEl, String(error), true);
        setStatus(statusEl, 'Execution failed', 'error');
      } finally {
        runButton.disabled = false;
        if (clearButton) clearButton.disabled = false;
      }
    });

    if (clearButton) {
      clearButton.addEventListener('click', () => {
        clearOutput(outputEl);
        setStatus(statusEl, 'Output cleared', 'info');
      });
    }
  });
})();
