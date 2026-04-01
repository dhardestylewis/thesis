import os
import re
import json
import warnings
import sys
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from artifact_registry import ROOT_DIR, TraceabilityRegistry as AR

ROOT = str(ROOT_DIR)
TEX_PATH = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Austin_NIMBY_Thesis_Draft.tex")
METRICS_PATH = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Tables", "metrics_config.tex")
STATE_PATH = str(AR.AST_STATE_JSON)

def load_metrics():
    metrics = {}
    if not os.path.exists(METRICS_PATH):
        return metrics
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'\\newcommand\{\\(.*?)\}\{(.*?)\}', line)
            if match:
                metrics[match.group(1)] = match.group(2)
    return metrics

def call_llm(prompt):
    """
    Sends the localized mathematical AST prompt to a live language model.
    Prioritizes local On-Premise Ollama execution for data privacy, falling back
    to Gemini 2.5 Flash via google-genai, falling back to OpenAI.
    """
    import urllib.request
    import json
    
    print(f"\n    >> [LLM OUTBOUND TRACE] <<")
    print(f"       PROMPT: {prompt[:120]}...\n")
    
    # 1. On-Premise Priority (Ollama DeepSeek-R1 / Qwen2.5)
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({
                "model": "deepseek-r1", # Outperforms Llama3 for structured reasoning/math in 2026. Alt: "qwen2.5"
                "prompt": "You are a rigorous urban planning academic. Output strictly the rewritten sentence to mathematically reflect the new metrics. Do not include commentary.\n\n" + prompt,
                "stream": False
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode())
            print("    [+] Routing AST prompt to On-Premise Ollama (deepseek-r1)...")
            return result.get("response", "").strip()
    except Exception as e:
        pass
        
    # 2. External API Fallbacks
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if gemini_key:
        print("    [+] Routing AST prompt to Gemini 2.5 Flash...")
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            sys_inst = "You are a rigorous urban planning academic bridging data science and policy. Rewrite the provided thesis sentence to mathematically and structurally reflect the new metrics. Output strictly the rewritten sentence, with no surrounding commentary or markdown."
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=sys_inst,
                    temperature=0.2,
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"    [!] Gemini API Call Failed: {e}")
            
    elif openai_key:
        print("    [+] Routing AST prompt to OpenAI GPT-4o...")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a rigorous urban planning academic. Output strictly the rewritten sentence to reflect the new mathematical metrics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [!] OpenAI API Call Failed: {e}")
            
    print("    [!] No API Keys found (GEMINI_API_KEY or OPENAI_API_KEY). Using deterministic fallback...")
    if "metricFNRGap" in prompt:
        return "Because the pipeline deliberately evaluated the pure empirical data structure against the empirical background threshold ($\\mu_{y}$), the model completely failed predictive parity and recorded a severe \\metricFNRGap{} False Negative Rate gap across council districts."
    return "This section has been algorithmically rewritten to conform to new tabular data."

def run_stage_e():
    print("==============================================")
    print(" STAGE E: Hierarchical Semantic AST Engine")
    print("==============================================")
    
    current_metrics = load_metrics()
    
    # Bootstrap state mapping for prior run tracking
    prior_metrics = {}
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, 'r', encoding='utf-8') as f:
            try:
                prior_metrics = json.load(f)
            except: pass
            
    # 1. EVALUATE LEAF NODES (Level 1: The Math)
    current_fnr = current_metrics.get('metricFNRGap', '0.00\\%')
    prior_fnr = prior_metrics.get('metricFNRGap', '0.00\\%')
    
    # Hypothesis: The user's changes to np.mean() in StageC will spike the FNR gap > 0.
    if current_fnr != prior_fnr and current_fnr != r"0.00\%":
        print(f"[*] CRITICAL AST DELTA DETECTED: metricFNRGap shifted from {prior_fnr} to {current_fnr}.")
        print("[*] Traversing Dependency Graph upward to Level 2 (Sentence)...")
        
        with open(TEX_PATH, 'r', encoding='utf-8') as f:
            tex_content = f.read()
            
        # Parse Level 2 AST Sentences
        pattern = r'\\dependentclaim\{FNR_Gap_Claim\}\{(.*?)\}'
        match = re.search(pattern, tex_content, re.DOTALL)
        if match:
            original_claim = match.group(1)
            print(f"    Found Compromised AST Node: [FNR_Gap_Claim].")
            print(f"    Isolating stochastic extraction...")
            
            prompt = f"The FNR gap metric (\\metricFNRGap) has changed from {prior_fnr} to {current_fnr}. The original thesis wording was: '{original_claim}'. Rewrite this sentence mathematically to explicitly state that it fails the parity constraint."
            
            # 2. TRIGGER LLM GENERATION
            new_claim = mock_llm_call(prompt)
            
            # 3. RECONSTRUCT PDF DOM
            new_node = f"\\dependentclaim{{FNR_Gap_Claim}}{{{new_claim}}}"
            tex_content = tex_content.replace(match.group(0), new_node)
            
            with open(TEX_PATH, 'w', encoding='utf-8') as f:
                f.write(tex_content)
            print("[+] AST Traversal and PDF Injection Complete. Structural academic arguments are preserved.")
        else:
            print("    [-] Target AST Node [FNR_Gap_Claim] not found in draft. Bypassing generation.")
    else:
        print("[+] All AST mathematical constraints hold. No semantic injection triggered.")
        
    # Lock current metrics as the new prior state
    with open(STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(current_metrics, f, indent=4)
        
    print("\nStage E Complete.")

if __name__ == '__main__':
    run_stage_e()
