"""
tests.py — Sanity checks for transform consistency and u-pred JVP.

Run with: python tests.py

All checks print pass/fail with max absolute error.
"""
import torch
import numpy as np

from transforms import (
    x_to_v, x_to_eps, eps_to_x, eps_to_v, v_to_x, v_to_eps,
    compute_V_theta, verify_transforms,
)
from paths import interpolate, sample_t, sample_r_t, T_EPS
from models import build_model

TOL = 1e-4
BATCH = 64
D = 16
device = torch.device("cpu")


def check(name: str, err: float, tol: float = TOL):
    status = "PASS" if err < tol else "FAIL"
    print(f"  [{status}] {name:50s}  err={err:.2e}  (tol={tol:.2e})")
    return err < tol


def test_algebraic_transforms():
    print("\n=== Algebraic transform consistency ===")
    rng = torch.Generator()
    rng.manual_seed(0)

    x   = torch.randn(BATCH, D)
    eps = torch.randn(BATCH, D)
    t   = sample_t(BATCH, device)      # (B, 1), in (T_EPS, 1-T_EPS)
    z_t = interpolate(x, eps, t)

    results = verify_transforms(x, eps, z_t, t)
    all_pass = True
    for name, err in results.items():
        ok = check(name, err)
        all_pass = all_pass and ok

    # Extra: v is constant along the path
    v_true = eps - x
    v_from_x   = x_to_v(x, z_t, t)
    v_from_eps = eps_to_v(eps, z_t, t)
    check("v true == x_to_v(x)", (v_true - v_from_x).abs().max().item())
    check("v true == eps_to_v(eps)", (v_true - v_from_eps).abs().max().item())

    # x-space round-trip via eps
    x_rt = eps_to_x(x_to_eps(x, z_t, t), z_t, t)
    check("x -> eps -> x", (x - x_rt).abs().max().item())

    # eps-space round-trip via x
    eps_rt = x_to_eps(eps_to_x(eps, z_t, t), z_t, t)
    check("eps -> x -> eps", (eps - eps_rt).abs().max().item())

    # z_t reconstruction from v
    z_t_from_v = v_to_x(v_true, z_t, t) + t * v_true   # x + t*v
    check("z_t = x + t*v", (z_t - z_t_from_v).abs().max().item())

    return all_pass


def test_u_pred_jvp():
    print("\n=== u-pred JVP and V_theta ===")
    model = build_model("u", D, hidden_dim=64, n_layers=3, time_emb_dim=16)
    model.eval()

    r, t = sample_r_t(BATCH, device)
    x    = torch.randn(BATCH, D)
    eps  = torch.randn(BATCH, D)
    z_t  = interpolate(x, eps, t)

    # Test that compute_V_theta runs without error
    try:
        u_theta, V_theta = compute_V_theta(model, z_t, r, t)
        check("V_theta shape correct", 0.0 if V_theta.shape == (BATCH, D) else 1.0)
        check("u_theta shape correct", 0.0 if u_theta.shape == (BATCH, D) else 1.0)
        print("  [PASS] compute_V_theta runs without error")
    except Exception as e:
        print(f"  [FAIL] compute_V_theta raised: {e}")
        return False

    # Test that loss can backpropagate
    try:
        model.train()
        u_theta, V_theta = compute_V_theta(model, z_t, r, t)
        v_true = eps - x
        loss = ((V_theta - v_true) ** 2).mean()
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters()
                        if p.grad is not None)
        check("u-pred loss backprop, grad_norm > 0", 0.0 if grad_norm > 0 else 1.0)
        print(f"  [INFO] grad_norm = {grad_norm:.4f}")
        model.zero_grad()
    except Exception as e:
        print(f"  [FAIL] u-pred backprop raised: {e}")
        return False

    # Boundary condition: at r=t, V_theta should equal u_theta
    # (because (t-r)*jvp_val = 0 when t=r)
    r_eq_t = t.clone()   # r = t
    # (t - r) = 0 => V_theta = u_theta
    try:
        u_theta_eq, V_theta_eq = compute_V_theta(model, z_t, r_eq_t, t)
        err = (V_theta_eq - u_theta_eq).abs().max().item()
        check("V_theta == u_theta when r=t", err, tol=1e-4)
    except Exception as e:
        print(f"  [FAIL] boundary condition check raised: {e}")

    return True


def test_time_sampling():
    print("\n=== Time sampling ===")
    t = sample_t(10000, device)
    check("t >= T_EPS",   (t < T_EPS).float().max().item(), tol=0.5)
    check("t <= 1-T_EPS", (t > 1.0 - T_EPS).float().max().item(), tol=0.5)
    print(f"  [INFO] t range: [{t.min():.4f}, {t.max():.4f}]")

    r, t_pair = sample_r_t(10000, device)
    gap = t_pair - r
    check("r < t always (gap > 0)", (gap <= 0).float().max().item(), tol=0.5)
    check("gap >= T_EPS", (gap < T_EPS * 0.5).float().max().item(), tol=0.5)
    print(f"  [INFO] gap range: [{gap.min():.4f}, {gap.max():.4f}]")
    print(f"  [INFO] r range:   [{r.min():.4f}, {r.max():.4f}]")
    print(f"  [INFO] t range:   [{t_pair.min():.4f}, {t_pair.max():.4f}]")


def test_standard_models():
    print("\n=== Standard model shapes ===")
    for ps in ["x", "eps", "v"]:
        m = build_model(ps, D, 64, 3, 16)
        z_t = torch.randn(BATCH, D)
        t   = sample_t(BATCH, device)
        out = m(z_t, t)
        ok  = out.shape == (BATCH, D)
        check(f"StandardMLP pred_space={ps} output shape", 0.0 if ok else 1.0)


def test_generation_sanity():
    print("\n=== Generation sanity (output shape) ===")
    from sample import generate

    for ps in ["x", "eps", "v"]:
        m = build_model(ps, D, 64, 3, 16)
        gen, traj = generate(m, 50, D, ps, 1, device)
        ok_gen  = gen.shape == (50, D)
        ok_traj = len(traj) == 2       # initial + 1 step
        check(f"1-step gen shape [{ps}]", 0.0 if ok_gen else 1.0)
        check(f"1-step traj len [{ps}]",  0.0 if ok_traj else 1.0)

    for ps in ["x", "v"]:
        m = build_model(ps, D, 64, 3, 16)
        gen, traj = generate(m, 50, D, ps, 4, device)
        ok_traj = len(traj) == 5       # initial + 4 steps
        check(f"4-step traj len [{ps}]", 0.0 if ok_traj else 1.0)

    # u-pred generation
    m_u = build_model("u", D, 64, 3, 16)
    gen, traj = generate(m_u, 50, D, "u", 1, device)
    check("1-step u gen shape", 0.0 if gen.shape == (50, D) else 1.0)
    gen, traj = generate(m_u, 50, D, "u", 4, device)
    check("4-step u traj len", 0.0 if len(traj) == 5 else 1.0)


if __name__ == "__main__":
    print("Running sanity checks...")
    test_time_sampling()
    test_algebraic_transforms()
    test_standard_models()
    test_u_pred_jvp()
    test_generation_sanity()
    print("\nDone.")
