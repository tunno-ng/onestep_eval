#!/bin/bash
# sweep.sh — Run the first set of minimal experiments.
# Edit NITERS and NEVAL for faster/slower runs.
# Usage: bash sweep.sh

NITERS=20000
NEVAL=2000

echo "=== 1. Sanity check: D=2, x/x, 1-step ==="
python run.py --dataset 8gaussians --obs_dim 2 --pred_space x --loss_space x --steps 1 --n_iters $NITERS --n_eval $NEVAL

echo ""
echo "=== 2. x vs u, D=16, 1-step ==="
python run.py --dataset 8gaussians --obs_dim 16 --pred_space x --loss_space x --steps 1 --n_iters $NITERS --n_eval $NEVAL
python run.py --dataset 8gaussians --obs_dim 16 --pred_space u --loss_space u --steps 1 --n_iters $NITERS --n_eval $NEVAL

echo ""
echo "=== 3. Cross-space loss: pred_space != loss_space, D=16 ==="
python run.py --dataset 8gaussians --obs_dim 16 --pred_space x --loss_space u --steps 1 --n_iters $NITERS --n_eval $NEVAL
python run.py --dataset 8gaussians --obs_dim 16 --pred_space u --loss_space x --steps 1 --n_iters $NITERS --n_eval $NEVAL

echo ""
echo "=== 4. Dimension sweep: D in {2, 8, 16, 64, 256}, pred=x loss=x ==="
for D in 2 8 16 64 256; do
    python run.py --dataset 8gaussians --obs_dim $D --pred_space x --loss_space x --steps 1 --n_iters $NITERS --n_eval $NEVAL
done

echo ""
echo "=== 5. 1-step vs 4-step, moons D=64 ==="
python run.py --dataset moons --obs_dim 64 --pred_space x --loss_space x --steps 1 --n_iters $NITERS --n_eval $NEVAL
python run.py --dataset moons --obs_dim 64 --pred_space x --loss_space x --steps 4 --n_iters $NITERS --n_eval $NEVAL
python run.py --dataset moons --obs_dim 64 --pred_space u --loss_space u --steps 1 --n_iters $NITERS --n_eval $NEVAL
python run.py --dataset moons --obs_dim 64 --pred_space u --loss_space u --steps 4 --n_iters $NITERS --n_eval $NEVAL

echo ""
echo "=== 6. Target smoothing, moons D=64, pred=x loss=x ==="
for TAU in 0.0 0.1 0.2 0.3; do
    python run.py --dataset moons --obs_dim 64 --pred_space x --loss_space x --steps 1 --tau $TAU --n_iters $NITERS --n_eval $NEVAL
done

echo ""
echo "=== 7. v and eps baselines, moons D=64, 1-step ==="
python run.py --dataset moons --obs_dim 64 --pred_space v --loss_space v --steps 1 --n_iters $NITERS --n_eval $NEVAL
python run.py --dataset moons --obs_dim 64 --pred_space eps --loss_space eps --steps 1 --n_iters $NITERS --n_eval $NEVAL

echo ""
echo "=== Done. Check exp/ for results. ==="
