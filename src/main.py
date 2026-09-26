"""CLI entry point."""
import argparse
from pathlib import Path
import sys
from src.agent import Agent
from src.model import ModelError, TextModel
from src.providers import configured_providers

def main() -> int:
    parser=argparse.ArgumentParser(description="Text-only AI coding harness")
    parser.add_argument("task",nargs="?",help="Engineering task; if omitted, prompt interactively")
    parser.add_argument("--repo",default=".",help="Target repository (default: current directory)")
    parser.add_argument("--verify",default="python -m pytest -q",help="Verification command")
    parser.add_argument("--list-providers",action="store_true",help="List providers with configured API keys")
    parser.add_argument("--tui",action="store_true",help="Launch the interactive terminal interface")
    args=parser.parse_args()
    if args.list_providers:
        print(", ".join(configured_providers()) or "No providers configured")
        return 0
    if args.tui or args.task is None:
        from src.tui import run_tui
        run_tui(args.repo,args.verify)
        return 0
    task=args.task.strip()
    if not task: parser.error("a task is required")
    try:
        model=TextModel()
        agent=Agent(Path(args.repo),model,verify_command=args.verify)
        result=agent.run(task)
        print(result.get("summary") or "Harness finished")
        status=result.get("status",result.get("state","unknown"))
        print(f"Success: {bool(result.get('success',False))}; status: {status}; verification passed: {result.get('verification',{}).get('passed',False)}")
        if result.get("error"): print(f"Error: {result['error']}",file=sys.stderr)
        if result.get("verification",{}).get("stdout"): print(result["verification"]["stdout"])
        if result.get("verification",{}).get("stderr"): print(result["verification"]["stderr"],file=sys.stderr)
        if model.active_provider: print(f"Provider used: {model.active_provider}")
        return 0 if result.get("success",False) else 1
    except (ModelError, OSError, ValueError) as exc:
        print(f"Error: {exc}",file=sys.stderr); return 2

if __name__=="__main__": raise SystemExit(main())
