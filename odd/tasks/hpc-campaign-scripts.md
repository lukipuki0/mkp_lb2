# HPC campaign scripts for GWO-DE and WOA-ABC

Slurm scripts for the PUCV Océano cluster. One script per hybrid, same layout: one 9-CPU job that runs the size groups in order, nine instances in parallel per group (L8).
Branch: feat/hpc-campaign-scripts. Cluster conventions come from submit_adaptive_mkp_arrays.sh and run_adaptive_mkp_array.sh.

## Specs

- S1. GWO-DE script runs `gwo_de_dtw` competitive campaign. User answer: "gwo_de_dtw (competitivo)". Four policies, seeds 1..31 (campaign defaults).
- S2. WOA-ABC script runs policy mode. User answers: "Cuatro políticas", "31 semillas (1..31)".
- S3. `woa_abc.campaign` accepts `--files` and `--indices`. User answer: "Agregar --files e --indices". Omitting both keeps today's behaviour: all nine files, indices 0 1 2.
- S4. "la idea es correr las 9 intancias en paralelo ideal tipo son los 3 primeros indices". Each size group is 3 files x indices 0 1 2 = 9 instances, run as 9 parallel processes inside the job (one per CPU). One process = one instance, all 31 seeds x 4 policies.
- S5. "quiero que sea primeo las de 100 items despues 250 despues las de 500". Inside the job the groups run in order: 250 starts after the nine 100 processes end, 500 after the nine 250 processes end. A failed instance does not stop the others or the next group; the job exits 1 at the end if any failed.
- S6. "basate en los demas sh pen la config ... debe ser asi la config": one `sbatch` job with the header of run_parallel_hpc.sh: `--partition=CPU`, `--nodes=1`, `--ntasks=1`, `--cpus-per-task=9`, `--threads-per-core=1`, `--mem=32G`, `--output=./.hpc/logs/<name>_%j.log`, `--error=./.hpc/errors/<name>_%j.error`, `--qos=long`. Submitted from the repo root (`sbatch run_<name>_hpc.sh`); conda env `mkp_env` from `~/miniconda3` (override `MKP_CONDA_SH`), activated without `set -u`; one thread per process; `bash run_<name>_hpc.sh --dry-run` prints the 27 instances and commands without running or writing.

- S7. "cambio el como se guardan los resultados que se guarde resultados/_nombre_exoerimento_ejecucion/intancia/ modo". User answers: name "Algoritmo + job de Slurm"; WOA-ABC seeds "Igual que GWO–DE"; instance "Dos niveles: mknapcb1/inst_00". Layout of one job: `resultados/<algoritmo>_<SLURM_JOB_ID>/mknapcb<F>/inst_<II>/` with `runs.csv` (31 seeds x 4 modes), `statistics.csv`, `convergence_comparison.png/.pdf`, `run.log`, and `<modo>/best_run/` (winning seed: highest best fitness, tie lower seed). Same for both hybrids.

## Tasks

- T1 [S3] woa_abc.campaign --files/--indices, tests, README. Route: worker. Commit: 9178402. Done: RED observed (unsupported keyword/unrecognized options), GREEN 123 woa_abc tests OK.
- T2 [S1 S4 S5 S6] run_gwo_de_hpc.sh plus tests. Route: inline. Commit: fa831ce (0efad99 was the superseded array design). RED seen, 6 tests GREEN.
- T3 [S2 S4 S5 S6] run_woa_abc_hpc.sh plus tests. Route: inline. Commit: fa831ce (0efad99 was the superseded array design). RED seen, 6 tests GREEN.
- T4 [S6] Time and QOS. Superseded by L8: header `--qos=long`, no `--time` (same as run_parallel_hpc.sh). Commit: with T2/T3.
- T6 [S7] `python -m woa_abc.seed_campaign`: one instance, seeds x four policies, GWO-DE layout and CSV columns, best run per policy. Route: worker failed (agent error, not retried), done inline. Commit: 888cff2. RED (module missing), GREEN 7 tests; WOA suite 129 OK.
- T7 [S7] Scripts write `resultados/<algoritmo>_<job>/mknapcb<F>/inst_<II>/`: each process runs in a staging folder and moves its instance folder in on success. Route: inline. Commit: 414712f. RED 8 failures, GREEN 6 tests.
- T8 [S7] Re-verify. Route: gentle-ai-verify unavailable (agent error, not retried); parent self-checks instead. Commit: none. Done: L14 probe plus 33 tests (hpc scripts, seed_campaign, woa_abc.campaign, gwo_de_dtw.campaign) OK, no leftovers under resultados/.
- T5 [S1-S6] Independent verification (risk item 5, cluster environment). Route: gentle-ai-verify. Commit: none. Done: second verify on fa831ce, S1-S6 pass, no blocking findings; real Slurm/Conda on Océano not executed. Native assess (committed range from 9d17b07): unassessable, untracked user files in the tree; no review started.

## Log

- L1 User: "ves los .sh quiere crear uno para gwo de y otro para woa abc ve como hacerlo son para un cluster hpc oceano de la pucv, si necesitas preguntat hacelo"
- L2 User answers: GWO-DE "gwo_de_dtw (competitivo)"; WOA-ABC "Cuatro políticas"; seeds "31 semillas (1..31)"; partition "Agregar --files e --indices".
- L3 User: "la idea es correr las 9 intancias en paralelo ideal tipo son los 3 primeros indices quiero que sea primeo las de 100 items despues 250 despues las de 500"
- L4 Evidence: gwo_de_dtw.campaign already has --files/--indices/--seeds and keeps the best run per policy across the seeds of one invocation, so seeds of one instance must stay in one task. woa_abc.campaign today takes one --seed and always all 27 instances. Old note: 31 WOA-ABC runs on mknapcb1 summed ~22 h on the local PC (older code).
- L5 Decision: afterany, not afterok, so one failed task does not cancel the next size group. Failures stay visible in errors/.
- L6 User, after the local n=500 measurement showed >16 min for the first GWO-DE run: "no si es mucho mas rapido en el hoc" / "hpc". Calibration stopped and its partial output removed; time limits are not derived from the local PC.
- L7 Evidence: the fake-worker test found that the WOA-ABC worker fell through into the submitter after the seed loop and exited 2. Fixed with `exit 0` after `worker` in both scripts.
- L8 User, while T5 ran: "basate en los demas sh pen la config" followed by the header of run_parallel_hpc.sh (`--job-name=ddtw_2k`, `--partition=CPU`, `--nodes=1`, `--ntasks=1`, `--cpus-per-task=9`, `--threads-per-core=1`, `--mem=32G`, `--output=./.hpc/logs/mkp_par9_%j.log`, `--error=./.hpc/errors/mkp_par9_%j.error`, `--qos=long`) and "debe ser asi la config". S4-S6 rewritten: one job with 9 CPUs instead of three arrays of 1-CPU tasks.
- L9 Evidence from T5 (gentle-ai-verify, array design): S1-S5 pass, S6 unverifiable on this PC. Advisory: `set -u` while sourcing conda.sh can kill the job before Python (also true for run_adaptive_mkp_array.sh); run_parallel_hpc.sh does not use `set -u`. Under sbatch the script runs from a spool copy, so the repo is taken from SLURM_SUBMIT_DIR, not from the script path.
- L10 Evidence (single-job design): RED 10 failures against the array scripts; GREEN 6/6 after fixing two test-helper bugs (posix() on a file; fake python3 exiting 1 on its last `[[ ]]`, which had made the GWO failure test pass by coincidence). The fake conda.sh reads unset variables, so the job test also covers the `set +u` around conda.
- L11 ALGORITMO_CENTRAL.md and "ESWA_Journal_Article_Template (1).zip" were found missing from the working tree (still in HEAD; not removed by any commit of this branch; cause not established). User choice: "Dejarlos borrados". Left unstaged and outside this feature's commits.
- L12 User: "cambio el como se guardan los resultados que se guarde resultados/_nombre_exoerimento_ejecucion/intancia/ modo se entiende? botta del git lo de algoritmo central y eswa". Removed both files from Git: commit 45a895d (still in history). S7 added; answers in S7.
- L15 Scope note: S7 was verified by the parent, not by an independent verifier (the subagent failed twice with an agent error and was not retried). The real Oceano Slurm/Conda run is still unexecuted, as in L9.
- L14 Evidence (real one-instance probe, `gwo_de_dtw.campaign --files mknapcb1.txt --indices 0 --seeds 1 2 --pop 8 --iterations 20 --window 10`): `--out` holds campaign_config.json, resumen.txt, summary.csv, statistics.csv and mknapcb1/inst_00/ (runs.csv, statistics.csv, convergence_comparison.png/.pdf, four policy folders). Root summary.csv is byte-identical to the instance runs.csv and root statistics.csv to the instance statistics.csv (`cmp` IGUAL twice), so dropping the root duplicates loses nothing and L13 holds.
- L13 Decision: GWO-DE needs no Python change (its campaign already writes `<stem>/inst_<NN>/<policy>/best_run`), but nine processes cannot share one `--out` (must be empty; root files campaign_config.json, resumen.txt, summary.csv, statistics.csv). Each process uses `<exp>/.staging/<stem>_inst_<NN>/` and moves `<stem>/inst_<NN>` into `<exp>/` on success; campaign_config.json and resumen.txt move into the instance folder; the root summary.csv/statistics.csv repeat the instance files for a one-instance campaign. A failed process leaves its staging folder. WOA-ABC gets a new command instead of changing woa_abc.campaign.
