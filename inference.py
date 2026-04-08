import json
import time
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal

# --- Pydantic Schemas for Robustness ---
class Report(BaseModel):
    patient_id: str
    clause_violated: str
    severity: Literal["minor", "major", "critical"]
    regulation_ref: Optional[str] = None

class ActionSchema(BaseModel):
    action_type: Literal["submit_reports", "finish"]
    reports: Optional[List[Report]] = Field(default_factory=list)

# --- Mock Environment ---
class ClinTrialEnv:
    def __init__(self, task_level="medium"):
        self.task_level = task_level
        self.max_steps = 10 if task_level == "easy" else (25 if task_level == "medium" else 50)
        self.current_step = 0
        self.history = set() # For duplicate tracking

    def reset(self):
        # Returns instantly (No blocking)
        self.current_step = 0
        self.history = set()
        state = {
            "episode_id": "EP_8492", 
            "task_level": self.task_level, 
            "current_step": self.current_step, 
            "max_steps": self.max_steps
        }
        return state, {}

    def step(self, action_dict):
        # Max steps enforcement
        self.current_step += 1
        done = self.current_step >= self.max_steps
        reward = 0.0

        # JSON Parsing Robustness (Catch errors, do not crash)
        try:
            action = ActionSchema(**action_dict)
        except ValidationError as e:
            return {"error": "schema_validation_failed"}, 0.0, done, {"error": str(e)}

        if action.action_type == "finish":
            done = True
            return {"status": "finished"}, 0.0, done, {}

        if action.action_type == "submit_reports":
            if not action.reports:
                # No-op penalty
                return {"status": "empty_submission"}, 0.0, done, {}
            
            step_score = 0.0
            for report in action.reports:
                sig = f"{report.patient_id}_{report.clause_violated}"
                if sig in self.history:
                    continue # Duplicate, 0 reward
                self.history.add(sig)
                
                # Mock scoring logic (e.g., patient + clause match)
                step_score += 0.6 
            
            # Strict clamping
            reward = max(0.0, min(1.0, step_score))

        return {"current_step": self.current_step}, reward, done, {}

# --- Mock Agent (No API Dependency) ---
class MockAgent:
    def __init__(self):
        self.step_count = 0
        
    def act(self, state):
        self.step_count += 1
        if self.step_count == 1:
            # Valid submission
            return {
                "action_type": "submit_reports", 
                "reports": [{"patient_id": "P004", "clause_violated": "Section 3.1", "severity": "major", "regulation_ref": "ICH E6 R2 4.5.2"}]
            }
        elif self.step_count == 2:
            # Duplicate to test penalty
            return {
                "action_type": "submit_reports", 
                "reports": [{"patient_id": "P004", "clause_violated": "Section 3.1", "severity": "major", "regulation_ref": "ICH E6 R2 4.5.2"}]
            }
        elif self.step_count == 3:
            # Invalid schema (wrong enum) to test robustness
            return {
                "action_type": "submit_reports", 
                "reports": [{"patient_id": "P005", "clause_violated": "Section 4.1", "severity": "invalid_enum"}]
            }
        else:
            # Finish
            return {"action_type": "finish"}

def run_inference():
    """
    Simulates the OpenEnv inference loop for ClinTrialEnv.
    Produces strict, parser-friendly logs required for the hackathon benchmark.
    """
    env = ClinTrialEnv(task_level="medium")
    agent = MockAgent()
    
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    
    print(f"[START] Episode EP_8492 | Task: medium")
    
    while not done:
        # Safe loop termination
        if state.get("current_step", 0) >= env.max_steps:
            break
            
        print(f"[STEP] {env.current_step + 1}/{env.max_steps}")
        
        # Agent acts
        action_dict = agent.act(state)
        
        # Log action cleanly with strict, compact JSON formatting (no spaces)
        action_summary = {"action_type": action_dict.get("action_type")}
        if "reports" in action_dict:
            action_summary["reports"] = len(action_dict["reports"]) if isinstance(action_dict["reports"], list) else 0
        print(f"[ACTION] {json.dumps(action_summary, separators=(',', ':'))}")
        
        # Env steps
        next_state, reward, done, info = env.step(action_dict)
        
        print(f"[REWARD] {reward:.4f}")
        total_reward += reward
        state = next_state
        time.sleep(0.1) # Slight delay for realistic log pacing
        
    print(f"[END] Episode finished. Total Reward: {total_reward:.4f}")

if __name__ == "__main__":
    run_inference()
