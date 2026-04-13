#!/bin/bash
# sweep.sh — First experiment sweep.
# Edit NITERS/NEVAL for faster/slower runs.
# Usage: bash sweep.sh

NITERS=20000
NEVAL=2000

echo "=== 1. Sanity check: D=2, x-pred v-loss 1-step ==="
python run.py --dataset 8gaussians --obs_dim 2 --pred_space x --loss_space v --steps 1 --n_iters $NITERS --n_eval $NEVAL

echo ""
echo "=== 2. Tier 1 (v-loss): all pred_spaces, D=16, 8gaussians ==="
for PS in x eps v u; do
    python run.py --dataset 8gaussians --obs_dim 16 --pred_space $PS --loss_space v --steps 1 --n_iters $NITERS --n_eval $NEVAL
done

echo ""
echo "=== 3. Tier 2 (x-loss): all pred_spaces, D=16, 8gaussians ==="
for PS in x eps v u; do
    python run.py --dataset 8gaussians --obs_dim 16 --pred_space $PS --loss_space x --steps 1 --n_iters $NITERS --n_eval $NEVAL
done

echo ""
echo "=== 4. Dimension sweep: pred=x loss=v, D in {2,8,16,64,256} ==="
for D in 2 8 16 64 256; do
    python run.py --dataset 8gaussians --obs_dim $D --pred_space x --loss_space v --steps 1 --n_iters $NITERS --n_eval $NEVAL
done

echo ""
echo "=== 5. 1-step vs 4-step: pred=x, moons D=64 ==="
python run.py --dataset moons --obs_dim 64 --pred_space x --loss_space v --steps 1 --n_iters $NITERS --n_eval $NEVAL
python run.py --dataset moons --obs_dim 64 --pred_space x --loss_space v --steps 4 --n_iters $NITERS --n_eval $NEVAL
python run.py --dataset moons --obs_dim 64 --pred_space u --loss_space v --steps 1 --n_iters $NITERS --n_eval $NEVAL
python run.py --dataset moons --obs_dim 64 --pred_space u --loss_space v --steps 4 --n_iters $NITERS --n_eval $NEVAL

echo ""
echo "=== 6. Target smoothing: pred=x loss=v, moons D=64 ==="
for TAU in 0.0 0.1 0.2 0.3; do
    python run.py --dataset moons --obs_dim 64 --pred_space x --loss_space v --steps 1 --tau $TAU --n_iters $NITERS --n_eval $NEVAL
done

echo ""
echo "=== Done. Results in exp/ ==="
