from __future__ import annotations

import os

import gradio as gr

from inference import run_episode


def evaluate(task_level: str, agent_type: str, model_name: str, seed: int, case_id: str):
    normalized_case_id = case_id.strip() or None
    result = run_episode(
        task_level=task_level,
        agent_type=agent_type,
        model_name=model_name,
        seed=seed,
        case_id=normalized_case_id,
        emit_stdout=False,
    )
    log_text = "\n".join(result["logs"])
    return log_text, result["total_reward"], result["task_score"]


with gr.Blocks(title="ClinTrialEnv OpenEnv Runner") as demo:
    gr.Markdown("# ClinTrialEnv OpenEnv Runner")
    gr.Markdown("Run easy, medium, or hard episodes with deterministic baseline or OpenAI agent.")

    with gr.Row():
        task_level = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Task")
        agent_type = gr.Dropdown(choices=["baseline", "openai"], value="baseline", label="Agent")

    model_name = gr.Textbox(value="gpt-4.1-mini", label="OpenAI Model")

    with gr.Row():
        seed = gr.Number(value=7, precision=0, label="Seed")
        case_id = gr.Textbox(value="", label="Optional Case ID")

    run_btn = gr.Button("Run Episode")

    logs = gr.Textbox(label="Logs", lines=22)
    total_reward = gr.Number(label="Total Reward")
    final_score = gr.Number(label="Final Task Score")

    run_btn.click(
        fn=evaluate,
        inputs=[task_level, agent_type, model_name, seed, case_id],
        outputs=[logs, total_reward, final_score],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
