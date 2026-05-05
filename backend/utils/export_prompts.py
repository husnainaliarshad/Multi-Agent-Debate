import os
from core.config import DEFAULT_PROPOSER_PROMPT, DEFAULT_CRITIC_PROMPT, DEFAULT_JUDGE_PROMPT

def export_prompts(output_file="prompts.txt"):
    """Export all system prompts to a text file for course submission."""
    prompts = {
        "Proposer Agent Prompt": DEFAULT_PROPOSER_PROMPT,
        "Critic Agent Prompt": DEFAULT_CRITIC_PROMPT,
        "Judge Agent Prompt (IRAC)": DEFAULT_JUDGE_PROMPT,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== GEN AI PROJECT PROMPT SUBMISSION ===\n\n")
        for name, content in prompts.items():
            f.write(f"--- {name} ---\n")
            f.write(content)
            f.write("\n\n" + "="*50 + "\n\n")
            
    print(f"Prompts exported to {output_file}")
    return os.path.abspath(output_file)

if __name__ == "__main__":
    export_prompts()
