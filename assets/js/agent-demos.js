(function () {
  // Demo 1: LLM Only - Just generates text
  const demoLlmOnly = document.getElementById('demo-llm-only');
  if (demoLlmOnly) {
    const btn = demoLlmOnly.querySelector('[data-demo="llm-only"]');
    const output = document.getElementById('output-llm-only');
    
    btn?.addEventListener('click', () => {
      output.innerHTML = `
        <div class="demo-log">
          <div class="log-line"><span class="timestamp">[T+0ms]</span> LLM reçoit : "Quelle heure est-il ?"</div>
          <div class="log-line"><span class="timestamp">[T+50ms]</span> LLM prédit la réponse...</div>
          <div class="log-line"><span class="timestamp">[T+200ms]</span> Sortie : "Je suis un modèle de langage et je n'ai pas accès à l'heure actuelle..."</div>
          <div class="log-line"><span class="timestamp">[T+201ms]</span> <strong>FIN. Le LLM s'arrête ici.</strong></div>
        </div>
      `;
      
      // Animate flow
      demoLlmOnly.querySelectorAll('.flow-box').forEach((box, i) => {
        setTimeout(() => box.classList.add('active'), i * 500);
      });
    });
  }

  // Demo 2: Tool Call - Parse JSON with Python
  const demoToolCall = document.getElementById('demo-tool-call');
  if (demoToolCall) {
    const btn = demoToolCall.querySelector('[data-demo="tool-call"]');
    const output = document.getElementById('output-tool-call');
    
    const toolCallCode = `
import json

llm_output = '{"tool": "get_time", "args": {}}'
print("[PARSE]Parsing JSON...")
parsed = json.loads(llm_output)
print(f"[TOOL]{parsed['tool']}")
print(f"[ARGS]{parsed['args']}")
`;
    
    const ensurePyodide = (() => {
      let pyodidePromise = null;
      return async () => {
        if (pyodidePromise) return pyodidePromise;
        const INDEX_URL = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/';
        pyodidePromise = new Promise((resolve, reject) => {
          const start = async () => {
            try {
              const pyodide = await globalThis.loadPyodide({ indexURL: INDEX_URL });
              resolve(pyodide);
            } catch (e) {
              reject(e);
            }
          };
          start();
        });
        return pyodidePromise;
      };
    })();
    
    btn?.addEventListener('click', async () => {
      btn.disabled = true;
      output.innerHTML = '';
      
      try {
        const pyodide = await ensurePyodide();
        
        pyodide.setStdout({ batched: (text) => {
          const lines = text.split('\n').filter(l => l.trim());
          lines.forEach(line => {
            if (line.includes('[PARSE]')) {
              const msg = line.replace('[PARSE]', '');
              const el = document.createElement('div');
              el.className = 'step-output step-llm';
              el.innerHTML = `<strong>Parser:</strong> ${msg}`;
              output.appendChild(el);
            } else if (line.includes('[TOOL]')) {
              const tool = line.replace('[TOOL]', '');
              const el = document.createElement('div');
              el.className = 'step-output step-tool';
              el.innerHTML = `<strong>Outil détecté:</strong> ${tool}`;
              output.appendChild(el);
            } else if (line.includes('[ARGS]')) {
              const args = line.replace('[ARGS]', '');
              const el = document.createElement('div');
              el.className = 'step-output step-result';
              el.innerHTML = `<strong>Arguments:</strong> ${args}`;
              output.appendChild(el);
            }
          });
          output.scrollTop = output.scrollHeight;
        }});
        
        await pyodide.runPythonAsync(toolCallCode);
        
      } catch (e) {
        const el = document.createElement('div');
        el.className = 'step-output step-error';
        el.innerHTML = `<strong>Erreur:</strong> ${e}`;
        output.appendChild(el);
      } finally {
        btn.disabled = false;
      }
    });
    
    // Preload Pyodide
    ensurePyodide();
  }

  // Demo 3: Missing Loop - Execute tool with Python
  const demoMissingLoop = document.getElementById('demo-missing-loop');
  if (demoMissingLoop) {
    const btn = demoMissingLoop.querySelector('[data-demo="missing-loop"]');
    const output = document.getElementById('output-missing-loop');
    
    const toolExecutionCode = `
import json

llm_response = '{"tool": "get_time", "args": {}}'
print("[PARSE]Parsing JSON...")
parsed = json.loads(llm_response)

def get_current_time():
    return "14h32"

print("[EXEC]Exécution de get_time()...")
if parsed["tool"] == "get_time":
    result = get_current_time()
    print(f"[RESULT]{result}")
    print("[SEND]Envoi du résultat au LLM...")
`;
    
    const ensurePyodide = (() => {
      let pyodidePromise = null;
      return async () => {
        if (pyodidePromise) return pyodidePromise;
        const INDEX_URL = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/';
        pyodidePromise = new Promise((resolve, reject) => {
          const start = async () => {
            try {
              const pyodide = await globalThis.loadPyodide({ indexURL: INDEX_URL });
              resolve(pyodide);
            } catch (e) {
              reject(e);
            }
          };
          start();
        });
        return pyodidePromise;
      };
    })();
    
    btn?.addEventListener('click', async () => {
      btn.disabled = true;
      output.innerHTML = '';
      
      try {
        const pyodide = await ensurePyodide();
        
        pyodide.setStdout({ batched: (text) => {
          const lines = text.split('\n').filter(l => l.trim());
          lines.forEach(line => {
            if (line.includes('[PARSE]')) {
              const msg = line.replace('[PARSE]', '');
              const el = document.createElement('div');
              el.className = 'step-output step-llm';
              el.innerHTML = `<strong>Parser:</strong> ${msg}`;
              output.appendChild(el);
            } else if (line.includes('[EXEC]')) {
              const msg = line.replace('[EXEC]', '');
              const el = document.createElement('div');
              el.className = 'step-output step-tool';
              el.innerHTML = `<strong>Exécution:</strong> ${msg}`;
              output.appendChild(el);
            } else if (line.includes('[RESULT]')) {
              const result = line.replace('[RESULT]', '');
              const el = document.createElement('div');
              el.className = 'step-output step-result';
              el.innerHTML = `<strong>Résultat:</strong> ${result}`;
              output.appendChild(el);
            } else if (line.includes('[SEND]')) {
              const msg = line.replace('[SEND]', '');
              const el = document.createElement('div');
              el.className = 'step-output step-final';
              el.innerHTML = `<strong>Retour:</strong> ${msg}`;
              output.appendChild(el);
            }
          });
          output.scrollTop = output.scrollHeight;
        }});
        
        await pyodide.runPythonAsync(toolExecutionCode);
        
      } catch (e) {
        const el = document.createElement('div');
        el.className = 'step-output step-error';
        el.innerHTML = `<strong>Erreur:</strong> ${e}`;
        output.appendChild(el);
      } finally {
        btn.disabled = false;
      }
    });
    
    // Preload Pyodide
    ensurePyodide();
  }

  // Demo 4: Full Agent - Complete autonomous loop with real code execution
  const demoAgent = document.getElementById('demo-agent-full');
  if (demoAgent) {
    const btn = demoAgent.querySelector('[data-demo="agent-full"]');
    const output = document.getElementById('output-agent-full');
    
    const ensurePyodide = (() => {
      let pyodidePromise = null;
      return async () => {
        if (pyodidePromise) return pyodidePromise;
        const INDEX_URL = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/';
        const statusEl = demoAgent.querySelector('.agent-status-mini');
        statusEl.textContent = 'Chargement Pyodide...';
        pyodidePromise = new Promise((resolve, reject) => {
          const start = async () => {
            try {
              const pyodide = await globalThis.loadPyodide({ indexURL: INDEX_URL });
              statusEl.textContent = 'Prêt';
              resolve(pyodide);
            } catch (e) {
              statusEl.textContent = 'Erreur';
              reject(e);
            }
          };
          start();
        });
        return pyodidePromise;
      };
    })();
    
    const addStepOutput = (content, type = 'info') => {
      const line = document.createElement('div');
      line.className = `step-output step-${type}`;
      line.innerHTML = content;
      output.appendChild(line);
      output.scrollTop = output.scrollHeight;
    };
    
    const agentCode = `
import json

print("[START]===== AGENT AUTONOME =====")
print("[START]Objectif: Quelle version de React ?")
print()

# Étape 1: Initialiser les messages
messages = [{"role": "user", "content": "Quelle version de React ?"}]
print("[INIT]Messages initialisés:")
print(f"[INIT]  {messages}")
print()

# Étape 2: Définir les outils
def read_file(path):
    print(f"[TOOL_EXEC]  -> Exécution réelle: lecture de {path}")
    files = {"package.json": '{"dependencies":{"react":"^18.2.0"}}'}
    result = files.get(path, "Non trouvé")
    print(f"[TOOL_EXEC]  -> Résultat: {result}")
    return result

TOOLS = {"read_file": read_file}

# Étape 3: Simuler le LLM
def simulate_llm(messages):
    last_msg = messages[-1].get("content", "") if messages else ""
    print(f"[LLM_THINK]Le LLM analyse le contexte...")
    print(f"[LLM_THINK]  Dernier message: {last_msg}")
    
    if "dependencies" in last_msg or "react" in last_msg.lower():
        print(f"[LLM_FINAL]Le LLM décide: C'est la réponse finale!")
        return {"type": "final", "content": "React 18.2.0 est installé"}
    else:
        print(f"[LLM_DECIDE]Le LLM décide: Je dois appeler read_file")
        return {"type": "tool_call", "tool": "read_file", "args": {"path": "package.json"}}

# Étape 4: La boucle autonome
print("[LOOP]===== DÉBUT DE LA BOUCLE =====")
for iteration in range(3):
    print()
    print(f"[LOOP]--- Itération {iteration + 1} ---")
    
    # Le LLM décide
    print(f"[LOOP]1. LLM décide...")
    response = simulate_llm(messages)
    
    # Vérifier si c'est fini
    if response["type"] == "final":
        print(f"[LOOP]2. Réponse finale détectée!")
        print(f"[LOOP]3. Réponse: {response['content']}")
        print()
        print("[END]===== AGENT TERMINÉ =====")
        break
    
    # Exécuter l'outil
    print(f"[LOOP]2. Exécution de l'outil: {response['tool']}")
    result = TOOLS[response["tool"]](**response["args"])
    
    # Ajouter le résultat aux messages
    print(f"[LOOP]3. Ajout du résultat aux messages")
    messages.append({"role": "tool", "content": result})
    print(f"[LOOP]4. Messages mis à jour: {len(messages)} messages")
    print(f"[LOOP]5. Retour à l'étape 1 (LLM reprend la main)")
`;
    
    btn?.addEventListener('click', async () => {
      btn.disabled = true;
      output.innerHTML = '';
      
      try {
        const pyodide = await ensurePyodide();
        
        pyodide.setStdout({ batched: (text) => {
          const lines = text.split('\n');
          
          lines.forEach(line => {
            if (!line.trim()) return;
            
            if (line.includes('[START]')) {
              const msg = line.replace('[START]', '');
              addStepOutput(`<strong>Démarrage:</strong> ${msg}`, 'llm');
            } else if (line.includes('[INIT]')) {
              const msg = line.replace('[INIT]', '');
              addStepOutput(`<strong>Initialisation:</strong> ${msg}`, 'llm');
            } else if (line.includes('[LOOP]')) {
              const msg = line.replace('[LOOP]', '');
              addStepOutput(`<strong>Boucle:</strong> ${msg}`, 'tool');
            } else if (line.includes('[LLM_THINK]')) {
              const msg = line.replace('[LLM_THINK]', '');
              addStepOutput(`<strong>LLM pense:</strong> ${msg}`, 'llm');
            } else if (line.includes('[LLM_DECIDE]')) {
              const msg = line.replace('[LLM_DECIDE]', '');
              addStepOutput(`<strong>LLM décide:</strong> ${msg}`, 'tool');
            } else if (line.includes('[LLM_FINAL]')) {
              const msg = line.replace('[LLM_FINAL]', '');
              addStepOutput(`<strong>LLM final:</strong> ${msg}`, 'final');
            } else if (line.includes('[TOOL_EXEC]')) {
              const msg = line.replace('[TOOL_EXEC]', '');
              addStepOutput(`<strong>Exécution outil:</strong> ${msg}`, 'result');
            } else if (line.includes('[END]')) {
              const msg = line.replace('[END]', '');
              addStepOutput(`<strong>Fin:</strong> ${msg}`, 'final');
            } else {
              addStepOutput(line, 'info');
            }
          });
        }});
        
        await pyodide.runPythonAsync(agentCode);
        
      } catch (e) {
        addStepOutput(`<strong>Erreur:</strong> ${e.toString()}`, 'error');
      } finally {
        btn.disabled = false;
      }
    });
    
    // Preload Pyodide
    ensurePyodide();
  }
})();
