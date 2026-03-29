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

  // Demo 2: Tool Call - Shows JSON generation
  const demoToolCall = document.getElementById('demo-tool-call');
  if (demoToolCall) {
    const btn = demoToolCall.querySelector('[data-demo="tool-call"]');
    const output = document.getElementById('output-tool-call');
    
    btn?.addEventListener('click', () => {
      output.innerHTML = `
        <div class="demo-log">
          <div class="log-line"><span class="timestamp">[T+0ms]</span> LLM analyse : "Quelle heure ?"</div>
          <div class="log-line"><span class="timestamp">[T+100ms]</span> LLM décide : "Je dois appeler get_time()"</div>
          <div class="log-line"><span class="timestamp">[T+150ms]</span> JSON généré :</div>
          <pre class="json-output">{
  "tool_call": {
    "name": "get_current_time",
    "arguments": {}
  }
}</pre>
          <div class="log-line"><span class="timestamp">[T+151ms]</span> <strong>LE LLM S'ARRÊTE. Le JSON attend.</strong></div>
        </div>
      `;
      
      demoToolCall.querySelectorAll('.flow-box').forEach((box, i) => {
        setTimeout(() => box.classList.add('active'), i * 400);
      });
      
      setTimeout(() => {
        demoToolCall.querySelector('.demo-halt')?.classList.add('blink');
      }, 1600);
    });
  }

  // Demo 3: Missing Loop - Shows what code needs to be added
  const demoMissingLoop = document.getElementById('demo-missing-loop');
  if (demoMissingLoop) {
    const btn = demoMissingLoop.querySelector('[data-demo="missing-loop"]');
    const output = document.getElementById('output-missing-loop');
    
    btn?.addEventListener('click', () => {
      output.innerHTML = `
        <div class="demo-log">
          <div class="log-line"><span class="timestamp">[T+0ms]</span> Reçoit JSON : {"tool":"get_time"}</div>
          <div class="log-line"><span class="timestamp">[T+10ms]</span> Parse le JSON...</div>
          <div class="log-line"><span class="timestamp">[T+20ms]</span> Appelle get_current_time()</div>
          <div class="log-line code-call"><span class="timestamp">[T+25ms]</span> -> Exécution réelle : time.now()</div>
          <div class="log-line"><span class="timestamp">[T+30ms]</span> Résultat : "14:32:15"</div>
          <div class="log-line"><span class="timestamp">[T+40ms]</span> Envoie au LLM pour analyse...</div>
          <div class="log-line"><span class="timestamp">[T+200ms]</span> LLM répond : "Il est 14h32"</div>
        </div>
      `;
      
      demoMissingLoop.querySelectorAll('.flow-box').forEach((box, i) => {
        setTimeout(() => box.classList.add('active'), i * 300);
      });
    });
  }

  // Demo 4: Full Agent - Complete autonomous loop
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
    
    const updateStep = (step) => {
      const steps = demoAgent.querySelectorAll('.mini-step');
      steps.forEach((s, i) => {
        if (i < step) s.classList.add('active');
        else s.classList.remove('active');
      });
    };
    
    const addStepOutput = (stepName, content, type = 'info') => {
      const line = document.createElement('div');
      line.className = `step-output step-${type}`;
      line.innerHTML = `<strong>${stepName}:</strong> ${content}`;
      output.appendChild(line);
      output.scrollTop = output.scrollHeight;
    };
    
    const agentCode = `
import json

def simulate_llm(messages, tools):
    last = messages[-1].get("content", "") if messages else ""
    if "dependencies" in last:
        return {"type": "final", "content": "React 18.2.0 est installé"}
    return {"type": "tool_call", "tool": "read_file", "args": {"path": "package.json"}}

def read_file(path):
    files = {"package.json": '{"dependencies":{"react":"^18.2.0"}}'}
    return files.get(path, "Non trouvé")

TOOLS = {"read_file": read_file}

def run_agent(goal):
    messages = [{"role": "user", "content": goal}]
    
    for i in range(3):
        print(f"[ITERATION_{i+1}]")
        resp = simulate_llm(messages, TOOLS)
        
        if resp["type"] == "final":
            print(f"[FINAL]{resp['content']}")
            return resp["content"]
        
        print(f"[TOOL_CALL]{resp['tool']}|{resp['args']}")
        result = TOOLS[resp["tool"]](**resp["args"])
        print(f"[RESULT]{result}")
        messages.append({"role": "tool", "content": result})
    
    return "Trop d'iterations"

run_agent("Quelle version de React ?")
`;
    
    btn?.addEventListener('click', async () => {
      btn.disabled = true;
      output.innerHTML = '';
      updateStep(1);
      
      try {
        const pyodide = await ensurePyodide();
        let iterationCount = 0;
        
        pyodide.setStdout({ batched: (text) => {
          const lines = text.split('\n').filter(l => l.trim());
          
          lines.forEach(line => {
            if (line.includes('[ITERATION_')) {
              iterationCount++;
              addStepOutput(`Itération ${iterationCount}`, 'Le LLM reçoit le contexte et décide', 'llm');
              updateStep(2);
            } else if (line.includes('[TOOL_CALL]')) {
              const parts = line.replace('[TOOL_CALL]', '').split('|');
              addStepOutput('Appel d\'outil', `${parts[0]}(${parts[1]})`, 'tool');
              updateStep(3);
            } else if (line.includes('[RESULT]')) {
              const result = line.replace('[RESULT]', '');
              addStepOutput('Résultat', result, 'result');
              updateStep(4);
            } else if (line.includes('[FINAL]')) {
              const answer = line.replace('[FINAL]', '');
              addStepOutput('Réponse finale', answer, 'final');
              updateStep(5);
            }
          });
        }});
        
        await pyodide.runPythonAsync(agentCode);
        
      } catch (e) {
        addStepOutput('Erreur', e.toString(), 'error');
      } finally {
        btn.disabled = false;
      }
    });
    
    // Preload Pyodide
    ensurePyodide();
  }
})();
