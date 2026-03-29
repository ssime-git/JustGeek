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

  // Demo 4: Full Agent - Pyodide execution
  const demoAgentFull = document.getElementById('demo-agent-full');
  if (demoAgentFull) {
    const PYODIDE_VERSION = '0.24.1';
    const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
    let pyodidePromise = null;
    
    const btn = demoAgentFull.querySelector('[data-demo="agent-full"]');
    const output = document.getElementById('output-agent-full');
    const statusEl = document.getElementById('agent-status-mini');
    
    function updateStep(stepNum) {
      demoAgentFull.querySelectorAll('.mini-step').forEach(step => {
        step.classList.remove('active');
        if (step.dataset.step === String(stepNum)) {
          step.classList.add('active');
        }
      });
    }
    
    function ensurePyodide() {
      if (!pyodidePromise) {
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
          
          if (globalThis.loadPyodide) {
            start();
          } else {
            const script = document.createElement('script');
            script.src = `${INDEX_URL}pyodide.js`;
            script.onload = start;
            document.head.appendChild(script);
          }
        });
      }
      return pyodidePromise;
    }
    
    const agentCode = `
import json

def simulate_llm(messages, tools):
    """Simule le LLM qui décide ou répond"""
    last = messages[-1].get("content", "") if messages else ""
    
    if "dependencies" in last:
        return {"type": "final", "content": "React 18.2.0 est installé"}
    
    return {"type": "tool_call", "tool": "read_file", "args": {"path": "package.json"}}

def read_file(path):
    """Outil qui lit un fichier"""
    files = {"package.json": '{"dependencies":{"react":"^18.2.0"}}'}
    return files.get(path, "Non trouvé")

TOOLS = {"read_file": read_file}

def run_agent(goal):
    messages = [{"role": "user", "content": goal}]
    
    for i in range(3):
        print(f"Iteration {i+1}")
        print("-" * 35)
        
        # LLM décide
        resp = simulate_llm(messages, TOOLS)
        
        if resp["type"] == "final":
            print(f"Reponse finale: {resp['content']}")
            return resp["content"]
        
        # Exécute l'outil
        print(f"Appel: {resp['tool']}({resp['args']})")
        result = TOOLS[resp["tool"]](**resp["args"])
        print(f"Resultat: {result}")
        
        messages.append({"role": "tool", "content": result})
    
    return "Trop d'iterations"

print("=" * 40)
print("AGENT AUTONOME - La boucle en action")
print("=" * 40)
run_agent("Quelle version de React ?")
print("=" * 40)
`;
    
    btn?.addEventListener('click', async () => {
      btn.disabled = true;
      output.innerHTML = '';
      
      try {
        const pyodide = await ensurePyodide();
        
        pyodide.setStdout({ batched: (text) => {
          output.innerHTML += text.replace(/\n/g, '<br>');
          output.scrollTop = output.scrollHeight;
          
          // Update visualizer based on output
          if (text.includes('Iteration 1')) updateStep(2);
          else if (text.includes('Appel:')) updateStep(3);
          else if (text.includes('Resultat:')) updateStep(4);
          else if (text.includes('Reponse finale')) updateStep(5);
        }});
        
        updateStep(1);
        await pyodide.runPythonAsync(agentCode);
        updateStep(5);
        
      } catch (e) {
        output.innerHTML += `<div class="error">Erreur: ${e}</div>`;
      } finally {
        btn.disabled = false;
      }
    });
    
    // Preload Pyodide
    ensurePyodide();
  }
})();
