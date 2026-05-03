
**Instructions for the User:** 
Copy the entire text below (from "--- BEGIN PROMPT ---" downwards) and paste it into Gemini along with your Research Paper (PDF/Text).

--- BEGIN PROMPT ---

**System Role:** You are an expert Computer Science Post-Doctoral Researcher, IEEE Transactions Reviewer, and Lead Software Architect. 

**Mission:**
I am sharing a research paper (attached) and the exact context of my current codebase (below). My current codebase implements parts of this paper (or foundational concepts related to it). 
I need you to:
1. Analyze how much of the paper is currently implemented based on my codebase context.
2. Identify a specific, highly novel extension to just *one part* of this paper. I do not want to implement the whole paper. The novelty MUST be grounded in robust methodologies typically found in "IEEE Transactions on Evolutionary Computation" or similar top-tier journals. Look into the theoretical concepts often found in the paper's references to form this novelty.
3. Formulate the structure for a **new research paper** based on this novelty.
4. Devise a strict, comprehensive **Codebase Implementation Plan** that I can hand directly to Claude (another AI agent) so it can build this novelty perfectly on top of my existing code.

---

### Step 1: Deep Research & Gap Analysis
Review the attached paper and my codebase context. Tell me exactly what I have successfully implemented so far compared to the paper, and what is left out. 

### Step 2: Novelty Generation
Propose 2-3 specific, mathematically robust novelties that extend *one part* of my existing implementation (e.g., modifying the pheromone update rule, introducing a dynamic heuristic, hybridizing the multi-entry cluster logic with another algorithm, etc.).
- Choose the best one and explain *why* it is IEEE-Transactions-worthy.
- Cite specific methodologies (e.g., adaptive evaporation, quantum-behaved ants, reinforcement learning-driven alpha/beta tuning).

### Step 3: The "New Paper" Concept
Draft an outline for my NEW research paper based on the chosen novelty:
- **Proposed Title**
- **Abstract snippet**
- **Formulation / Mathematical Contribution**
- **Expected Results / Metrics to track**

### Step 4: The Instructions for Claude (Codebase Plan)
Create a strict step-by-step architectural plan for Claude. The plan must include:
- Which files to modify or create.
- The exact algorithms/math to code.
- How to integrate it with the existing 3D Plotly visualization and Multi-Entry logic.
- Keep the scope restricted *only* to the single part of the paper we are extending. 

---

### CODEBASE CONTEXT (What I currently have):
Since you cannot see my files, here is the complete summary of my existing Python codebase.

**Overview:**
The project is an Ant Colony Optimization (ACO) suite primarily designed for the Traveling Salesman Problem (TSP), and a specialized "Multi-Entry Region" routing problem, paired with real-time 3D visualizations.

**1. `aco_tsp.py` (Standard ACO)**
- Object-oriented `ACO_TSP` class.
- standard variables: `n_ants`, `n_iter`, `alpha`, `beta`, `evaporation`.
- Construct path using probabilistic transition rules (Pheromone^alpha * Heuristic^beta).
- Pheromones are dynamically updated (only top 5 paths update the pheromone matrix to boost convergence).
- Evaporation applies globally per iteration.

**2. `aco_multi_entry.py` (Clustered/Region-Based ACO)**
- Designed to route through "Regions" (circles with center and radius) instead of just single nodes.
- Each region has `num_entry` (e.g., 8) nodes on its circumference.
- Ants start at a random node, and the path must visit *every region exactly once*.
- Cost function penalizes visiting the same region twice (adds 1000 to cost).
- Pheromone updates are based on the global best path found in the iteration limit.

**3. `plotly_aco_3d.py` & `main.py` (Real-Time 3D Visualization)**
- Converts 2D points to 3D.
- Renders an interactive Plotly 3D scatter/line simulation.
- Animates iterations frame-by-frame:
  - Ant positions are smoothly interpolated between cities.
  - Pheromone intensities are mapped to edge colors and thickness.
  - Global best path highlights instantly.
  - Includes a UI slider and play/pause/speed configurations.

**4. `analysis.py` (Metrics & Plotting)**
- Runs algorithm sweeps (e.g., tracking performance over variations of `n_ants`).
- Calculates stability across multiple runs and plots convergence history via `matplotlib`.

---

**END OF PROMPT.** Please begin your analysis based on the attached paper and the codebase context provided above.