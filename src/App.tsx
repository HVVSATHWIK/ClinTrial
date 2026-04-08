import React, { useState, useRef, useEffect } from 'react';
import { 
  Activity, 
  Clock, 
  Database, 
  FileText, 
  Send, 
  Terminal, 
  CheckCircle,
  Code,
  ShieldAlert,
  Calculator,
  ListChecks,
  Microscope,
  RotateCcw
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<'patients' | 'ground_truth'>('patients');

  // --- INTERACTIVE STATE ---
  const [task, setTask] = useState<'easy' | 'medium' | 'hard'>('medium');
  const [step, setStep] = useState(1);
  const [maxSteps, setMaxSteps] = useState(25);
  const [tp, setTp] = useState(0);
  const [fp, setFp] = useState(0);
  const [fn, setFn] = useState(1); // 1 ground truth deviation for this mock
  const [totalReward, setTotalReward] = useState(0);
  const [isDone, setIsDone] = useState(false);
  const [logs, setLogs] = useState<string[]>([`[START] Episode EP_8492 | Task: medium`]);
  const [actionInput, setActionInput] = useState(`{\n  "action_type": "submit_reports",\n  "reports": [\n    {\n      "patient_id": "P004",\n      "clause_violated": "Section 3.1",\n      "severity": "major",\n      "regulation_ref": "ICH E6 R2 4.5.2"\n    }\n  ]\n}`);
  
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs to bottom
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // --- HANDLERS ---
  const handleTaskSelect = (newTask: 'easy' | 'medium' | 'hard') => {
    setTask(newTask);
    const mSteps = newTask === 'easy' ? 10 : newTask === 'medium' ? 25 : 50;
    setMaxSteps(mSteps);
    setStep(1);
    setTp(0);
    setFp(0);
    setFn(1);
    setTotalReward(0);
    setIsDone(false);
    setLogs([`[START] Episode EP_8492 | Task: ${newTask}`]);
    setActionInput(`{\n  "action_type": "submit_reports",\n  "reports": []\n}`);
  };

  const handleStepSubmit = () => {
    if (isDone) return;

    let parsedAction;
    try {
      parsedAction = JSON.parse(actionInput);
    } catch (e) {
      const newLogs = [
        ...logs,
        `[STEP] ${step}/${maxSteps}`,
        `[ERROR] Invalid JSON schema`,
        `[REWARD] 0.0000`
      ];
      advanceStep(newLogs, totalReward);
      return;
    }

    // Format action log strictly
    const actionSummary: any = { action_type: parsedAction.action_type };
    if (parsedAction.reports) {
      actionSummary.reports = Array.isArray(parsedAction.reports) ? parsedAction.reports.length : 0;
    }
    const actionLog = `[ACTION] ${JSON.stringify(actionSummary)}`;

    if (parsedAction.action_type === 'finish') {
      const newLogs = [...logs, `[STEP] ${step}/${maxSteps}`, actionLog];
      handleFinish(newLogs, totalReward);
      return;
    }

    if (parsedAction.action_type === 'submit_reports') {
      let stepReward = 0;
      let newTp = tp;
      let newFp = fp;
      let newFn = fn;

      if (!parsedAction.reports || parsedAction.reports.length === 0) {
        // No-op penalty
        stepReward = 0.0;
      } else {
        // Mock Evaluation Logic
        parsedAction.reports.forEach((r: any) => {
          if (r.patient_id === 'P004' && (r.clause_violated === 'Section 3.1' || r.clause_violated === 'Section 3.2')) {
            if (newTp === 0) {
              stepReward += 0.6; // Mock score for finding it
              newTp = 1;
              newFn = 0;
            } else {
              // Duplicate submission -> 0 reward
              stepReward += 0.0;
            }
          } else {
            // False positive
            newFp += 1;
          }
        });
      }

      const clampedReward = Math.max(0.0, Math.min(1.0, stepReward));
      
      const newLogs = [
        ...logs,
        `[STEP] ${step}/${maxSteps}`,
        actionLog,
        `[REWARD] ${clampedReward.toFixed(4)}`
      ];
      
      setTp(newTp);
      setFp(newFp);
      setFn(newFn);
      setTotalReward(prev => prev + clampedReward);
      
      advanceStep(newLogs, totalReward + clampedReward);
    } else {
      // Invalid action type
      const newLogs = [
        ...logs,
        `[STEP] ${step}/${maxSteps}`,
        `[ERROR] Invalid action_type`,
        `[REWARD] 0.0000`
      ];
      advanceStep(newLogs, totalReward);
    }
  };

  const advanceStep = (currentLogs: string[], currentTotalReward: number) => {
    if (step >= maxSteps) {
      handleFinish(currentLogs, currentTotalReward);
    } else {
      setLogs(currentLogs);
      setStep(s => s + 1);
    }
  };

  const handleFinish = (currentLogs: string[], finalReward: number) => {
    setIsDone(true);
    setLogs([...currentLogs, `[END] Episode finished. Total Reward: ${finalReward.toFixed(4)}`]);
  };

  // Helper to render logs with colors
  const renderLogLine = (line: string, idx: number) => {
    if (line.startsWith('[START]') || line.startsWith('[END]')) return <div key={idx} className="text-blue-400">{line}</div>;
    if (line.startsWith('[STEP]')) return <div key={idx} className="text-gray-500 mt-2">{line}</div>;
    if (line.startsWith('[ACTION]')) return <div key={idx} className="text-gray-300">{line}</div>;
    if (line.startsWith('[REWARD]')) {
      const val = parseFloat(line.split(' ')[1]);
      return <div key={idx} className={val > 0 ? "text-green-400 font-bold" : "text-red-500 font-bold"}>{line}</div>;
    }
    if (line.startsWith('[ERROR]')) return <div key={idx} className="text-red-400">{line}</div>;
    return <div key={idx} className="text-gray-400">{line}</div>;
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-gray-300 font-sans flex flex-col selection:bg-blue-500/30">
      {/* HEADER */}
      <header className="bg-[#111111] border-b border-gray-800/60 px-6 py-3 flex items-center justify-between sticky top-0 z-20">
        <div>
          <h1 className="text-lg font-semibold text-gray-100 flex items-center gap-2 tracking-tight">
            <Microscope className="w-6 h-6 text-blue-500" />
            ClinTrialEnv OpenEnv Dashboard
          </h1>
          <p className="text-xs text-gray-400 mt-0.5">Strict RL Environment for Protocol Deviation Detection</p>
        </div>
        <nav className="flex gap-6 text-sm font-medium text-gray-400">
          <a href="#" className="hover:text-gray-100 transition-colors text-blue-400">Environment State</a>
          <a href="#" className="hover:text-gray-100 transition-colors">Reward Specs</a>
          <a href="#" className="hover:text-gray-100 transition-colors">Inference Logs</a>
        </nav>
      </header>

      {/* MAIN LAYOUT */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* LEFT PANEL (TASK SELECTOR & GRADERS) */}
        <aside className="w-80 border-r border-gray-800/60 bg-[#0f0f0f] flex flex-col overflow-y-auto shrink-0">
          <div className="p-5">
            <h2 className="text-[10px] font-bold tracking-widest text-gray-500 uppercase mb-4 flex items-center gap-2">
              <ListChecks className="w-3.5 h-3.5" /> Task Graders & Weights
            </h2>
            <div className="space-y-4">
              {/* Task 1 */}
              <div 
                onClick={() => handleTaskSelect('easy')}
                className={`border rounded-md p-3 cursor-pointer transition-colors ${task === 'easy' ? 'bg-[#1a1a1a] border-blue-500/50 shadow-sm' : 'bg-[#141414] border-gray-800/60 hover:bg-[#1a1a1a]'}`}
              >
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-sm font-medium text-gray-200 leading-tight">Structured Detection</h3>
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-green-500/10 text-green-400 border border-green-500/20 font-medium uppercase tracking-wide">Easy</span>
                </div>
                <div className="bg-[#0a0a0a] rounded p-2 mt-2 border border-gray-800/40">
                  <div className="text-[10px] text-gray-400 font-mono space-y-1">
                    <div className="flex justify-between"><span>patient_match</span><span className="text-blue-400">0.5</span></div>
                    <div className="flex justify-between"><span>clause_match</span><span className="text-blue-400">0.5</span></div>
                  </div>
                </div>
              </div>
              {/* Task 2 */}
              <div 
                onClick={() => handleTaskSelect('medium')}
                className={`border rounded-md p-3 cursor-pointer transition-colors ${task === 'medium' ? 'bg-[#1a1a1a] border-blue-500/50 shadow-sm' : 'bg-[#141414] border-gray-800/60 hover:bg-[#1a1a1a]'}`}
              >
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-sm font-medium text-gray-200 leading-tight">Severity Classification</h3>
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 font-medium uppercase tracking-wide">Medium</span>
                </div>
                <div className="bg-[#0a0a0a] rounded p-2 mt-2 border border-gray-800/40">
                  <div className="text-[10px] text-gray-400 font-mono space-y-1">
                    <div className="flex justify-between"><span>patient_match</span><span className="text-blue-400">0.3</span></div>
                    <div className="flex justify-between"><span>clause_match</span><span className="text-blue-400">0.3</span></div>
                    <div className="flex justify-between"><span>severity_match</span><span className="text-amber-400">0.4</span></div>
                  </div>
                </div>
              </div>
              {/* Task 3 */}
              <div 
                onClick={() => handleTaskSelect('hard')}
                className={`border rounded-md p-3 cursor-pointer transition-colors ${task === 'hard' ? 'bg-[#1a1a1a] border-blue-500/50 shadow-sm' : 'bg-[#141414] border-gray-800/60 hover:bg-[#1a1a1a]'}`}
              >
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-sm font-medium text-gray-200 leading-tight">Multi-Protocol Contradiction</h3>
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20 font-medium uppercase tracking-wide">Hard</span>
                </div>
                <div className="bg-[#0a0a0a] rounded p-2 mt-2 border border-gray-800/40">
                  <div className="text-[10px] text-gray-400 font-mono space-y-1">
                    <div className="flex justify-between"><span>patient_match</span><span className="text-blue-400">0.2</span></div>
                    <div className="flex justify-between"><span>clause_match</span><span className="text-blue-400">0.2</span></div>
                    <div className="flex justify-between"><span>severity_match</span><span className="text-amber-400">0.2</span></div>
                    <div className="flex justify-between"><span>regulation_ref</span><span className="text-red-400">0.4</span></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* MAIN PANEL */}
        <main className="flex-1 overflow-y-auto p-6 space-y-6 bg-[#0a0a0a]">
          
          {/* SECTION 1: EPISODE STATUS & REWARD MATH */}
          <section className="grid grid-cols-5 gap-4">
            <div className="bg-[#111] border border-gray-800/60 rounded-md p-4 flex flex-col shadow-sm">
              <span className="text-[11px] text-gray-500 mb-1.5 uppercase tracking-wide font-medium">Episode ID</span>
              <span className="text-sm font-mono text-gray-200">EP_8492</span>
            </div>
            <div className="bg-[#111] border border-gray-800/60 rounded-md p-4 flex flex-col shadow-sm">
              <span className="text-[11px] text-gray-500 mb-1.5 uppercase tracking-wide font-medium">Current Step</span>
              <div className="flex items-baseline gap-2">
                <span className="text-xl font-semibold text-gray-100">{isDone ? maxSteps : step}</span>
                <span className="text-xs text-gray-500 font-mono">/ {maxSteps} max</span>
              </div>
            </div>
            <div className="bg-[#111] border border-gray-800/60 rounded-md p-4 flex flex-col shadow-sm">
              <span className="text-[11px] text-gray-500 mb-1.5 uppercase tracking-wide font-medium">Deviations (TP/FP/FN)</span>
              <div className="flex items-baseline gap-4">
                <span className="text-xl font-semibold text-green-400" title="True Positives">{tp}</span>
                <span className="text-xl font-semibold text-red-400" title="False Positives">{fp}</span>
                <span className="text-xl font-semibold text-amber-400" title="False Negatives">{fn}</span>
              </div>
            </div>
            <div className="col-span-2 bg-[#111] border border-blue-500/30 rounded-md p-4 flex flex-col shadow-sm relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full blur-2xl -mr-10 -mt-10 pointer-events-none"></div>
              <span className="text-[11px] text-blue-400 mb-1.5 uppercase tracking-wide font-medium flex items-center gap-1.5">
                <Calculator className="w-3.5 h-3.5" /> Step Reward Formula
              </span>
              <div className="text-[10px] font-mono text-gray-400 space-y-1">
                <div><span className="text-gray-500">For each valid report:</span></div>
                <div className="text-blue-300">score = Σ (component_match * weight)</div>
                <div className="mt-1"><span className="text-gray-500">Step Reward:</span></div>
                <div className="text-green-400">reward = max(0.0, min(1.0, score))</div>
              </div>
            </div>
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* SECTION 2: PROTOCOL VIEWER (STATE) */}
            <section className="bg-[#111] border border-gray-800/60 rounded-md overflow-hidden flex flex-col h-80 shadow-sm">
              <div className="bg-[#161616] border-b border-gray-800/60 px-4 py-2.5 flex justify-between items-center">
                <h3 className="text-xs font-medium text-gray-200 flex items-center gap-2 uppercase tracking-wide">
                  <FileText className="w-3.5 h-3.5 text-blue-400" /> state.context.protocol
                </h3>
              </div>
              <div className="p-4 overflow-y-auto font-mono text-[11px] text-gray-400 leading-relaxed whitespace-pre-wrap">
{`3.0 STUDY PROCEDURES
3.1 Screening Visit (Day -14 to Day 0)
Prior to any study-related procedures, written informed consent must be obtained.
Bloodwork (CBC, CMP) must be collected within 7 days prior to Randomization.

3.2 Visit Schedule
Visit 1 (Randomization): Day 1
Visit 2: Day 14 (± 2 days)
Visit 3: Day 28 (± 3 days)

4.0 INVESTIGATIONAL PRODUCT
4.1 Storage
Study drug must be stored at 2°C to 8°C (36°F to 46°F).`}
              </div>
            </section>

            {/* SECTION 3: ACTION SUBMISSION PANEL (STRICT SCHEMA) */}
            <section className="bg-[#111] border border-gray-800/60 rounded-md overflow-hidden flex flex-col h-80 shadow-sm">
              <div className="bg-[#161616] border-b border-gray-800/60 px-4 py-2.5 flex justify-between items-center">
                <h3 className="text-xs font-medium text-gray-200 flex items-center gap-2 uppercase tracking-wide">
                  <Code className="w-3.5 h-3.5 text-blue-400" /> Action Schema
                </h3>
                <span className="text-[9px] text-red-400 border border-red-400/30 bg-red-400/10 px-2 py-0.5 rounded flex items-center gap-1">
                  <ShieldAlert className="w-3 h-3" /> Strict Validation
                </span>
              </div>
              <div className="p-4 flex flex-col flex-1">
                <textarea 
                  value={actionInput}
                  onChange={(e) => setActionInput(e.target.value)}
                  disabled={isDone}
                  className="w-full flex-1 bg-[#0a0a0a] border border-gray-800/80 rounded text-[11px] font-mono text-gray-300 p-3 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all resize-none disabled:opacity-50"
                />
                <div className="mt-4 flex justify-between items-center">
                  <div className="text-[9px] text-gray-500 font-mono leading-tight">
                    <span className="text-red-400 font-semibold">Strict Pydantic-style validation:</span><br/>
                    Missing fields, wrong types, or invalid enums = 0.0 reward
                  </div>
                  <button 
                    onClick={handleStepSubmit}
                    disabled={isDone}
                    className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:text-gray-500 text-white text-xs font-medium px-4 py-2 rounded flex items-center gap-2 transition-colors shadow-sm shrink-0"
                  >
                    {isDone ? <RotateCcw className="w-3.5 h-3.5" /> : <Send className="w-3.5 h-3.5" />}
                    {isDone ? 'Episode Finished' : 'env.step(action)'}
                  </button>
                </div>
              </div>
            </section>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* SECTION 4: DATA & GROUND TRUTH */}
            <section className="bg-[#111] border border-gray-800/60 rounded-md overflow-hidden flex flex-col h-80 shadow-sm">
              <div className="bg-[#161616] border-b border-gray-800/60 flex">
                <button 
                  onClick={() => setActiveTab('patients')}
                  className={`px-4 py-2.5 text-xs font-medium uppercase tracking-wide flex items-center gap-2 border-b-2 transition-colors ${activeTab === 'patients' ? 'border-blue-500 text-gray-200' : 'border-transparent text-gray-500 hover:text-gray-300'}`}
                >
                  <Database className="w-3.5 h-3.5" /> state.context.patients
                </button>
                <button 
                  onClick={() => setActiveTab('ground_truth')}
                  className={`px-4 py-2.5 text-xs font-medium uppercase tracking-wide flex items-center gap-2 border-b-2 transition-colors ${activeTab === 'ground_truth' ? 'border-amber-500 text-gray-200' : 'border-transparent text-gray-500 hover:text-gray-300'}`}
                >
                  <CheckCircle className="w-3.5 h-3.5" /> Ground Truth
                </button>
              </div>
              
              {activeTab === 'patients' ? (
                <div className="overflow-x-auto p-0">
                  <table className="w-full text-left text-[11px] whitespace-nowrap">
                    <thead className="bg-[#111] text-gray-500 border-b border-gray-800/60 uppercase tracking-wider">
                      <tr>
                        <th className="px-4 py-3 font-medium">Patient ID</th>
                        <th className="px-4 py-3 font-medium">Visit No.</th>
                        <th className="px-4 py-3 font-medium">Bloodwork Date</th>
                        <th className="px-4 py-3 font-medium">Consent Date</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800/40 text-gray-300">
                      <tr className="hover:bg-[#161616] transition-colors">
                        <td className="px-4 py-2.5 font-mono text-blue-400">P001</td>
                        <td className="px-4 py-2.5">1</td>
                        <td className="px-4 py-2.5 font-mono text-gray-400">2023-09-28</td>
                        <td className="px-4 py-2.5 font-mono text-gray-400">2023-09-25</td>
                      </tr>
                      <tr className="hover:bg-[#161616] transition-colors">
                        <td className="px-4 py-2.5 font-mono text-blue-400">P004</td>
                        <td className="px-4 py-2.5">1</td>
                        <td className="px-4 py-2.5 font-mono text-red-400">2023-09-15</td>
                        <td className="px-4 py-2.5 font-mono text-gray-400">2023-09-25</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="p-4 overflow-y-auto font-mono text-[11px] text-amber-500/90 leading-relaxed whitespace-pre-wrap bg-[#1a140a] relative">
                  <div className="absolute top-3 right-3 bg-red-500/10 text-red-400 border border-red-500/30 px-2 py-1 rounded text-[9px] uppercase tracking-widest font-bold">
                    Debug UI Only - NEVER exposed in state()
                  </div>
{`{
  "ground_truth": [
    {
      "patient_id": "P004",
      "clause_violated": "Section 3.1",
      "severity": "major",
      "regulation_ref": "ICH E6 R2 4.5.2",
      "reason": "Bloodwork was 10 days prior to randomization, exceeding the 7-day limit."
    }
  ]
}`}
                </div>
              )}
            </section>

            {/* SECTION 5: STRICT INFERENCE LOGS */}
            <section className="bg-[#0a0a0a] border border-gray-800/60 rounded-md overflow-hidden flex flex-col h-80 shadow-sm relative">
              <div className="bg-[#161616] border-b border-gray-800/60 px-4 py-2.5 flex justify-between items-center">
                <h3 className="text-xs font-medium text-gray-200 flex items-center gap-2 uppercase tracking-wide">
                  <Terminal className="w-3.5 h-3.5 text-gray-400" /> Inference Logs
                </h3>
              </div>
              <div className="p-4 overflow-y-auto font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
                {logs.map((log, i) => renderLogLine(log, i))}
                <div ref={logsEndRef} />
              </div>
            </section>
          </div>

        </main>
      </div>
    </div>
  );
}
