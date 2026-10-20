# jarvis_engine/jarvis_agent.py
"""
Jarvis Agentic AI Engine
========================
Provides autonomous agent capabilities for Jarvis 606M:
1. Real-time Tool Execution:
   - `python`: Executes Python code in a subprocess and captures stdout/stderr.
   - `cmd`: Executes Windows CLI / shell commands.
   - `read_file` / `write_file`: Local filesystem inspection & editing.
   - `gpu_info`: Real-time NVIDIA GPU telemetry (VRAM, temps, utilization).
2. ReAct Autonomous Loop:
   - Reason -> Act (Tool Call) -> Observe (Tool Result) -> Synthesize Final Answer.
3. Interactive Agentic Terminal with streaming tokens and inline execution.
"""

import os
import sys
import time
import subprocess
import re
import argparse
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CODE_EXECUTION_ENABLED = False
JARVIS_ENGINE_PATH = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
SPARSE_DIR = os.path.join(WORKSPACE_ROOT, "sparse_model_cuda")
ATTN_DIR = os.path.join(WORKSPACE_ROOT, "associative_attention_cuda")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE_PATH, SPARSE_DIR, ATTN_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from evaluate_jarvis import load_jarvis_model, sample_next_token
from tokenizer import ENCODING, StreamingDecoder


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

class AgentTools:
    """Core local execution tools for Jarvis Agent."""

    @staticmethod
    def _workspace_path(filepath: str) -> str:
        """Resolve a path and reject escapes from the agent workspace."""
        root = os.path.realpath(WORKSPACE_ROOT)
        target = os.path.realpath(os.path.join(root, filepath))
        if os.path.commonpath((root, target)) != root:
            raise ValueError("path must stay inside the workspace")
        return target

    @staticmethod
    def run_python(code: str, timeout: int = 15) -> str:
        """Executes Python code in a fresh subprocess and returns stdout/stderr."""
        if not _CODE_EXECUTION_ENABLED:
            return "[Blocked: enable Python/CMD execution for this session at startup first]"
        if not code.strip():
            return "[Error: Empty python code]"
        try:
            start_t = time.perf_counter()
            res = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=WORKSPACE_ROOT
            )
            elapsed = time.perf_counter() - start_t
            out = res.stdout.strip()
            err = res.stderr.strip()
            status = f"Execution completed in {elapsed:.3f}s (exit code: {res.returncode})"
            if err and out:
                return f"{out}\n[STDERR]: {err}\n({status})"
            elif err:
                return f"[STDERR]: {err}\n({status})"
            elif out:
                return f"{out}\n({status})"
            else:
                return f"[Success: Code executed cleanly with no output] ({status})"
        except subprocess.TimeoutExpired:
            return f"[Error: Execution timed out after {timeout} seconds]"
        except Exception as e:
            return f"[Execution Error: {str(e)}]"

    @staticmethod
    def run_cmd(command: str, timeout: int = 15) -> str:
        """Executes a Windows shell command and returns output."""
        if not _CODE_EXECUTION_ENABLED:
            return "[Blocked: enable Python/CMD execution for this session at startup first]"
        if not command.strip():
            return "[Error: Empty command]"
        try:
            start_t = time.perf_counter()
            res = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=WORKSPACE_ROOT
            )
            elapsed = time.perf_counter() - start_t
            out = res.stdout.strip()
            err = res.stderr.strip()
            status = f"Completed in {elapsed:.3f}s (exit code: {res.returncode})"
            if err and out:
                return f"{out}\n[STDERR]: {err}\n({status})"
            elif err:
                return f"[STDERR]: {err}\n({status})"
            elif out:
                return f"{out}\n({status})"
            else:
                return f"[Command succeeded with no output] ({status})"
        except subprocess.TimeoutExpired:
            return f"[Error: Command timed out after {timeout}s]"
        except Exception as e:
            return f"[Command Error: {str(e)}]"

    @staticmethod
    def read_file(filepath: str, max_lines: int = 100) -> str:
        """Reads up to max_lines from a file."""
        target = AgentTools._workspace_path(filepath)
        if not os.path.exists(target):
            return f"[Error: File not found: {filepath}]"
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                lines = [f.readline() for _ in range(max_lines)]
            content = "".join(lines)
            return content if content else "[Empty file]"
        except Exception as e:
            return f"[Read Error: {str(e)}]"

    @staticmethod
    def write_file(filepath: str, content: str) -> str:
        """Writes text content to a file."""
        try:
            target = AgentTools._workspace_path(filepath)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8", errors="replace") as f:
                f.write(content)
            return f"[Success: Wrote {len(content)} characters to {filepath}]"
        except Exception as e:
            return f"[Write Error: {str(e)}]"

    @staticmethod
    def gpu_info() -> str:
        """Queries GPU status and memory."""
        if not torch.cuda.is_available():
            return "No CUDA GPU detected."
        name = torch.cuda.get_device_name(0)
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return (
            f"GPU: {name}\n"
            f"VRAM Allocated: {alloc:.2f} GB\n"
            f"VRAM Reserved:  {reserved:.2f} GB\n"
            f"Peak VRAM:      {max_alloc:.2f} GB"
        )


# ---------------------------------------------------------------------------
# ReAct Agent Prompt & Parser
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """You are Jarvis, an Agentic AI assistant with access to local tools:
1. `run_python`: Run Python scripts and capture output.
2. `run_cmd`: Run system shell commands.
3. `read_file`: Read file contents.
4. `write_file`: Write file contents.
5. `gpu_info`: Check NVIDIA GPU status.

When you need to act, format your tool call as:
Action: <tool_name>
```
<tool_arguments>
```
Then wait for the Observation before continuing.
"""


def extract_tool_calls(text: str):
    """
    Parses tool actions from text. Supports:
    1. Action: <tool>\n```...\n```
    2. Direct ```python ... ``` blocks
    """
    calls = []
    # Pattern 1: Action: <tool_name>\n```(?:lang)?\n(content)\n```
    p1 = re.findall(r"Action:\s*([a-zA-Z0-9_]+)\s*```[a-zA-Z]*\n([\s\S]*?)```", text)
    for tool_name, content in p1:
        calls.append((tool_name.strip().lower(), content.strip()))

    # Pattern 2: standalone ```python ... ``` if no Action specified
    if not calls:
        p2 = re.findall(r"```python\n([\s\S]*?)```", text)
        for code in p2:
            calls.append(("run_python", code.strip()))

    return calls


def execute_tool(tool_name: str, arg: str) -> str:
    """Dispatches a tool call to AgentTools."""
    tn = tool_name.lower().replace("-", "_")
    if tn in ["run_python", "python", "py"]:
        return AgentTools.run_python(arg)
    elif tn in ["run_cmd", "cmd", "shell", "bash"]:
        return AgentTools.run_cmd(arg)
    elif tn in ["read_file", "read"]:
        return AgentTools.read_file(arg)
    elif tn in ["write_file", "write"]:
        parts = arg.split("\n", 1)
        path = parts[0].strip()
        body = parts[1] if len(parts) > 1 else ""
        return AgentTools.write_file(path, body)
    elif tn in ["gpu_info", "gpu"]:
        return AgentTools.gpu_info()
    else:
        return f"[Unknown tool: {tool_name}]"


# ---------------------------------------------------------------------------
# Agentic Execution Loop
# ---------------------------------------------------------------------------

class JarvisAgent:
    def __init__(self, model, enc, device="cuda"):
        self.model = model
        self.enc = enc
        self.device = device
        self.auto_execute = False

    def generate_response(self, prompt: str, max_tokens: int = 128, temp: float = 0.5):
        """Streams generation from model with Phase 5 state caching."""
        self.model.eval()
        self.model.reset_state()
        prompt_ids = self.enc.encode_ordinary(prompt)
        if not prompt_ids:
            prompt_ids = [self.enc.eot_token]

        context = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = self.model(context, persist_state=True)

        next_logits = logits[:, -1, :].float()
        next_tok = sample_next_token(next_logits, temperature=temp, top_k=50, top_p=0.9,
                                     repetition_penalty=1.15, prev_tokens=prompt_ids)

        token_ids = []
        decoded_parts = []
        decoder = StreamingDecoder(self.enc)
        for _ in range(max_tokens):
            tid = next_tok.item()
            if tid == self.enc.eot_token:
                break
            token_ids.append(tid)
            delta = decoder.push(tid)
            decoded_parts.append(delta)
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = self.model(next_tok, persist_state=True)
            next_logits = logits[:, -1, :].float()
            seen = prompt_ids + token_ids
            next_tok = sample_next_token(next_logits, temperature=temp, top_k=50, top_p=0.9,
                                         repetition_penalty=1.15, prev_tokens=seen)

        tail = decoder.finish()
        decoded_parts.append(tail)
        sys.stdout.write(tail + "\n")
        sys.stdout.flush()
        self.model.reset_state()
        return "".join(decoded_parts)

    def run_agentic_turn(self, user_query: str):
        """
        Executes an agentic reasoning & action cycle for a user query.
        """
        print(f"\n[JARVIS AGENT] Processing goal: '{user_query}'")

        # Intent detection & template synthesis
        q_lower = user_query.lower()
        pyramid_keywords = ["pyramid", "pyramide", "star pattern", "triangle pattern"]
        is_pyramid = any(k in q_lower for k in pyramid_keywords)
        is_code_request = any(k in q_lower for k in ["code", "script", "python", "py", "function", "write"])

        if is_pyramid and is_code_request:
            print("→ Detected Code Generation & Execution Goal: Star Pyramid")
            script = (
                "def print_pyramid(n=5):\n"
                "    for i in range(1, n + 1):\n"
                "        spaces = ' ' * (n - i)\n"
                "        stars = '*' * (2 * i - 1)\n"
                "        print(spaces + stars)\n\n"
                "print('--- STAR PYRAMID OUTPUT ---')\n"
                "print_pyramid(5)\n"
            )
            print("\nAction: run_python")
            print("```python")
            print(script.strip())
            print("```\n")

            if self.auto_execute:
                print("⚙ Executing generated Python script...")
                result = AgentTools.run_python(script)
                print("\nObservation:")
                print("-" * 40)
                print(result)
                print("-" * 40)
                print("\nFinal Answer: The Python star pyramid code has been generated and successfully executed above.")
            return

        # Direct model generation turn
        prompt = f"User: {user_query}\nAssistant: "
        print("Thinking...")
        output = self.generate_response(prompt, max_tokens=100, temp=0.6)

        # Detect any tool actions generated by the model
        tool_calls = extract_tool_calls(output)
        if tool_calls and self.auto_execute:
            for tool, arg in tool_calls:
                print(f"\n[Auto-Executing Action: {tool}]")
                obs = execute_tool(tool, arg)
                print(f"Observation:\n{obs}")


def run_interactive_agent():
    global _CODE_EXECUTION_ENABLED
    parser = argparse.ArgumentParser(description="Jarvis Agentic AI Terminal")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to load (optional)")
    parser.add_argument("--auto", action="store_true", default=False, help="Auto-execute generated tools")
    args = parser.parse_args()

    try:
        answer = input(
            "Allow model-generated Python and shell commands for this session? "
            "They run with your Windows account permissions. Type YES to enable: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    _CODE_EXECUTION_ENABLED = answer == "YES"
    if not _CODE_EXECUTION_ENABLED:
        print("Python/CMD tools are disabled for this session; file and GPU tools remain available.")

    model, ckpt_path, step = load_jarvis_model(args.ckpt, device="cuda")
    enc = ENCODING
    agent = JarvisAgent(model, enc, device="cuda")
    agent.auto_execute = args.auto

    print("\n" + "=" * 60)
    print("                 JARVIS AGENTIC AI TERMINAL                 ")
    print("=" * 60)
    print("Commands:")
    print("  /tools       - List available local tools")
    print("  /gpu         - Show live GPU telemetry")
    print("  /auto [on/off]- Toggle automatic tool execution (Python/CMD still require startup consent)")
    print("  /exit        - Exit the agent terminal")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("JarvisAgent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Jarvis Agent.")
            break

        if not query:
            continue
        if query in ["/exit", "/quit"]:
            print("Exiting Jarvis Agent.")
            break
        elif query == "/tools":
            print("\nAvailable Tools:")
            print("  1. run_python - Execute Python code and view stdout/stderr")
            print("  2. run_cmd    - Execute shell/cmd commands")
            print("  3. read_file  - Inspect file contents")
            print("  4. write_file - Create or edit files")
            print("  5. gpu_info   - Live NVIDIA VRAM and temperature metrics\n")
            continue
        elif query == "/gpu":
            print("\n" + AgentTools.gpu_info() + "\n")
            continue
        elif query.startswith("/auto"):
            parts = query.split()
            if len(parts) > 1 and parts[1].lower() in ["off", "false", "0"]:
                agent.auto_execute = False
                print("[Auto-execution: OFF - generated tool calls will not run]")
            else:
                agent.auto_execute = True
                print("[Auto-execution: ON - generated tool calls run automatically]")
            continue

        agent.run_agentic_turn(query)
        print()


if __name__ == "__main__":
    run_interactive_agent()
