# fjsp-slga-mod
A modification to Self Learning Genetic Algorithm (SLGA) for Flexible Jop Shop Scheduling Problem (FJSP)

## Table of Contents

- [Overview](#overview)
- [Reference Paper](#reference-paper)
- [Repository Structure](#repository-structure)
- [Building](#building)
- [Usage](#usage)
- [Documented Assumptions](#documented-assumptions)
- [Known Limitations](#known-limitations)
- [Proposed Modification](#proposed-modification)


---



## Overview

The Flexible Job-Shop Scheduling Problem (FJSP) is an extension of the classical Job-Shop Scheduling Problem in which each operation can be processed on one of several alternative machines. The objective is typically to minimize the makespan $C_{max}$, making FJSP a challenging NP-hard combinatorial optimization problem.

The Self-Learning Genetic Algorithm (SLGA), proposed by Chen et al. (2020), integrates reinforcement learning into a genetic algorithm to dynamically adjust crossover and mutation probabilities during the evolutionary process. The method combines SARSA and Q-Learning to balance exploration and exploitation throughout the search.

This project provides a C implementation of the original SLGA for FJSP, together with a modified version that introduces a adaptive mutation strategy. Both algorithms are evaluated on benchmark instances from the Kacem and Brandimarte datasets, and their performance is compared in terms of makespan and solution quality.

## Reference Paper

https://www.sciencedirect.com/science/article/pii/S0360835220304885

## Repository Structure

| File | Description |
|---|---|
| `rl.h` / `rl.c`     | Reinforcement learning module |
| `fjsp.h` / `fjsp.c` | FJSP instance parser, chromosome encoding, decoders, fitness |
| `ga.h` / `ga.c`     | Genetic operators: crossover, mutation, selection, repair |
| `main.c`            | Main SLGA loop, elite retention, benchmark runner, CLI |

### `rl.h` / `rl.c` — Reinforcement Learning Module

Implements the RL component that dynamically adjusts the crossover and mutation probabilities ($P_c$, $P_m$) once per generation, replacing the fixed values used in a traditional GA. Responsibilities:

- **State computation** (Eq. 6–9): computes $f^*$ (average population fitness ratio), $d^*$ (diversity ratio), $m^*$ (best individual fitness ratio) — all normalized against the generation-1 population — combines them into a continuous value $S^*$, and discretizes it into 20 states of equal width (0.05 intervals).
- **Action set**: 10 predefined actions, each jointly defining a sub-range for $P_c$ and $P_m$ (e.g. action `a1` → $P_c \in [0.40, 0.45)$, $P_m \in [0.01, 0.03)$). The actual $P_c$/$P_m$ values are sampled uniformly at random within the chosen action's sub-range.
- **Reward**: computes $r_c$ (improvement of the best individual, evaluates $P_c$'s effectiveness) and $r_m$ (improvement of the population average, evaluates $P_m$'s effectiveness), combined as $r = r_c + r_m$.
- **Q-table**: a $20 \times 10$ table, initialized to zero. Updated using either SARSA (Eq. 3) or Q-learning (Eq. 4), depending on a conversion condition based on the number of elapsed generations relative to $(N_{states} \times N_{actions})/2 = 100$.
- **$\varepsilon$-greedy policy**: selects the next action, exploiting the best known action with probability $\varepsilon=0.85$ and exploring randomly otherwise.
- **Orchestrator** (`rl_init`, `rl_step`): ties the above together and is called once per generation from `main.c`, returning the $P_c$/$P_m$ values to use.

### `fjsp.h` / `fjsp.c` — Instance Parsing, Encoding, and Decoding

- **Parser**: reads the `.txt` instance format used by the Brandimarte and Kacem benchmark sets (jobs, machines, operations, and their valid machine/processing-time options).
- **Chromosome**: dual-string encoding (`os[]` + `ms[]`), where `os[]` encodes the operation execution order and `ms[]` encodes, position by position, the machine assigned to each operation. Includes random chromosome generation for population initialization.
- **Decoders**: translate a chromosome into an actual schedule and compute $C_{max}$.
  - `decode_makespan` — semi-active decoder (baseline, appends each operation to the end of its machine's queue).
  - `decode_makespan_active` — active decoder (first-fit: inserts each operation into the first idle gap where it fits). Used throughout the project.
  - `decode_makespan_bestfit` — experimental active decoder (best-fit: evaluates all idle gaps and picks the one minimizing the operation's completion time). Empirically produced identical final results to first-fit in full GA runs; kept for reference, not used by default.
- **Fitness**: $fitness = 1/C_{max}$.

### `ga.h` / `ga.c` — Genetic Operators

- **Crossover** (`crossover`): implements POX (Precedence Preserving Order-based Crossover) on the OS string. The MS string is inherited **coupled** to the same POX mechanism — each operation carries the machine assignment from whichever parent it was copied from — instead of an independent two-point crossover. This fix eliminated a bug where the original (uncoupled) approach caused `repair_chromosome` to destroy up to ~41% of inherited MS genes on partial-flexibility instances (Brandimarte); after the fix, the destruction rate measured 0% across all tested instances.
- **Mutation** (`mutate`): swap mutation on the OS string, plus a validity-preserving swap mutation on the MS string (only swaps two machine assignments if the swap is valid in both directions), following Lei (2012).
- **Repair** (`repair_chromosome`): safety-net function that reassigns a random valid machine to any operation left with an invalid machine assignment. After the crossover fix, this is only exercised as a consequence of OS mutation (never by crossover).
- **Selection** (`tournament_select_k`): binary tournament used to select parents for crossover. Not specified in the original paper (which only confirms the replacement/survivor strategy); tournament selection was chosen as a common default from the wider GA literature.

### `main.c` — Main Loop, Replacement, and Benchmark Runner

- **Population management**: initializes and frees the population (chromosomes + fitness array).
- **Main SLGA loop** (`run_slga`): for each generation, calls the RL module to get $P_c$/$P_m$, optionally applies the proposed adaptive-mutation modification (see [Proposed Modification](#proposed-modification)), generates offspring via selection + crossover + mutation, and replaces the population.
- **Replacement**: elite retention (Sec. 3.1.4 of the paper) — the current population and its offspring are pooled, and the best individuals are kept. This implementation *softens* strict elitism: a `DIVERSITY_RESERVE_FRACTION` (default 0.15) of survivor slots are filled with random individuals from the non-elite pool instead of the strict best, in order to preserve population diversity; the single best individual overall is always guaranteed to survive.
- **Benchmark runner** (`run_benchmark`): runs both the baseline and modified SLGA across `N_SEEDS` independent seeds on a given instance, reporting best (BSL) and average (ASL) $C_{max}$, plus per-variant execution time.
- **CLI**: accepts instance names (resolved automatically against `instances/brandimarte/` and `instances/kacem/`), full file paths, or a directory (runs every `.txt` file inside it).
## Building

```bash
make
```

## Usage

```bash
./slga_mod mk01
./slga_mod mk01 mk02 mk03
./slga_mod instances/brandimarte
```
## Documented Assumptions

### Decoder:
Active (first-fit). Paper does not specify; tested best-fit as well, no measurable difference in practice.
### Parent selection:
binary tournament. Paper only confirms replacement strategy (elite retention), not parent selection.
### Mutation probability application:
per-gene (tested per-chromosome, performed worse empirically).
### Population size (N) and max generations (Max_t):
not published in the paper; chosen independently (state your values).


## Known Limitations

### State saturation: 
agent spends 56-83% of generations in the maximum state (18-19), due to cumulative ratios (Eq 6-9) combined with elitist selection that never regresses.

### SARSA/Q-learning convergence:
with epsilon=0.85, ~81% of action choices already match argmax, so the two update rules coincide most of the time in practice.

### Gap vs. paper's reported results: 
partially attributable to unknown search budget (N, Max_t not published).


## Proposed Modification

### Summary

A single additional mechanism is proposed on top of the base SLGA: **adaptive mutation based on population diversity**. It scales up the effective mutation probability $P_m$ (obtained from the RL module) when population diversity is low, in order to counteract premature convergence — a known limitation of genetic algorithms using elitist replacement strategies.

Note: an earlier version of this project also included a second mechanism (reintroduction of random individuals on stagnation detection). It was found, during validation, that a configuration bug made this second mechanism inactive (0 individuals reintroduced) across every combination of instance and population size actually tested, meaning all reported results already reflect the adaptive-mutation-only version. The mechanism was therefore set innactive.

### Idea and Design

While SLGA already adjusts $P_c$/$P_m$ dynamically via its RL module, that adjustment operates over a predefined, discrete action space, and is not specifically designed to respond to a loss of population diversity. The adaptive mutation mechanism complements the RL-provided $P_m$ by scaling it according to the population's current diversity $d^*$ (already computed as part of the state calculation, Eq. 7): the lower the diversity, the higher the multiplier applied, encouraging more exploration exactly when it is most needed. When diversity is high ($d^* \geq 1.0$), the multiplier is effectively 1.0 (no change from the RL-provided value).

This mechanism is non-invasive with respect to the rest of SLGA — it only adjusts the effective $P_m$ value used for mutation, without altering the genetic operators, the RL module, or the replacement strategy — and always preserves the best individual found so far, via the elite retention strategy.

### Parameters

| Parameter | Value | Description |
|---|---|---|
| `ADAPTIVE_MUT_MAX_MULT` | 1.2 (fixed) | Maximum multiplier applied to $P_m$ when diversity is at its minimum |

### Algorithmic Description

```
function adaptive_mutation(Pm, d_star, mult_max):
    d_clamped = clamp(d_star, 0, 1)
    mult = 1 + (mult_max - 1) * (1 - d_clamped)
    Pm_effective = min(Pm * mult, 1.0)
    return Pm_effective
```

### Results Summary

Evaluated with `POP_SIZE=30`, `MAX_GENERATIONS=3000`, and 10 seeds per instance across the 10 Brandimarte instances: the modified version (SLGA-M) improved the average solution (ASL) in 8 of 10 instances relative to the unmodified implementation (SLGA-P), with an average improvement of ~0.6% and a maximum of ~2.67% (MK06). This improvement generally comes at the cost of a modest increase in execution time (2%–6% in most instances), attributable to the higher effective mutation rate increasing the number of mutation attempts per generation.

