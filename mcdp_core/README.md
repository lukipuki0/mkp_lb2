# MCDP core

This package contains the model extracted from `MCDP-DL-DRL`, without PyTorch,
training code, or metaheuristics.  It is the common evaluator that can later
be used by the standalone methods and by the DTW/DDTW framework.

## Representation

An instance is a binary machine--part incidence matrix
`A in {0,1}^{M x P}`.  `A[i,j]=1` means that machine `i` processes part `j`.
A solution is the vector `x=(x_1,...,x_M)`, where `x_i` is the cell assigned
to machine `i` and `x_i in {0,...,K-1}`.

The part-cell assignment is not an independent decision variable.  For each
part `j`, it is derived by majority vote among the cells containing machines
that process that part.  Ties are assigned to the lowest cell index, matching
the source implementation.

## Constraints

The core enforces:

* every machine is assigned to exactly one available cell;
* each cell contains at most `C` machines;
* cell labels are in `0,...,K-1`.

The capacity configuration is allowed to be infeasible (`M > K*C`) so that
experiments can report this condition explicitly instead of failing while
constructing an instance.

## Objective

For part `j`, let `n_j` be the number of machines that process it and let
`v_jk` be the number of those machines assigned to cell `k`.  The source model
minimizes the number of inter-cell exceptional elements:

`Z(x) = sum_j [ n_j - max_k v_jk ]`.

The implementation counts processing ones that lie outside the majority cell;
zeros inside a machine-cell block are not penalized.  This definition should be
kept consistent when the benchmark protocol is designed.

## Workflow

`MCDP_State` represents a partial or complete assignment.  `MCDP_Environment`
generates constructive assignments and feasible move, swap, and 3-swap
neighbors.  `MCDP_Instance.evaluate()` returns cost, feasibility, derived part
assignments, and the machine assignment.  `MCDPResult` and the save helpers
persist complete per-run records in CSV or JSON.

The input loader accepts blocks headed by `matriz 1`, `matriz 2`, etc., with
comma- or whitespace-separated binary rows.  The ten source matrices are
available in `mcdp_core/instances/instancias.txt`; the original copy remains
untouched in `MCDP-DL-DRL/mcdp/Instancias/instancias.txt`.
