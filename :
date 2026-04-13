
# 基本設定と変換

## パス定義
$$
z_t = (1 - t)x + t\epsilon
$$

$$
v = \epsilon - x
$$

---

## 空間間の線形変換

### x → ε, v
$$
\epsilon = \frac{z_t - (1 - t)x}{t}
$$

$$
v = \frac{z_t - x}{t}
$$

---

### ε → x, v
$$
x = \frac{z_t - t\epsilon}{1 - t}
$$

$$
v = \frac{\epsilon - z_t}{1 - t}
$$

---

### v → x, ε
$$
x = z_t - t v
$$

$$
\epsilon = z_t + (1 - t)v
$$

---

## 平均速度 (MeanFlow)
$$
u(z_t, r, t) = \frac{1}{t-r} \int_r^t v(z_\tau, \tau) d\tau
$$

---

## MeanFlow identity (iMF)
$$
v(z_t,t) = u(z_t,r,t) + (t-r) \frac{d}{dt}u(z_t,r,t)
$$

---

# Tier 1: v-loss

## x-pred + v-loss
$$
\mathcal L_{x \to v}
=
\mathbb E \left[
\left\|
\frac{z_t - x_\theta}{t} - (\epsilon - x)
\right\|^2
\right]
$$

---

## ε-pred + v-loss
$$
\mathcal L_{\epsilon \to v}
=
\mathbb E \left[
\left\|
\frac{\epsilon_\theta - z_t}{1 - t} - (\epsilon - x)
\right\|^2
\right]
$$

---

## v-pred + v-loss
$$
\mathcal L_{v \to v}
=
\mathbb E \left[
\|v_\theta - (\epsilon - x)\|^2
\right]
$$

---

## u-pred + v-loss
$$
V_\theta = u_\theta + (t-r) \frac{d}{dt}u_\theta
$$

$$
\mathcal L_{u \to v}
=
\mathbb E \left[
\|V_\theta - (\epsilon - x)\|^2
\right]
$$

---

# Tier 2: x-loss

## x-pred + x-loss
$$
\mathcal L_{x \to x}
=
\mathbb E \left[
\|x_\theta - x\|^2
\right]
$$

---

## ε-pred + x-loss
$$
\mathcal L_{\epsilon \to x}
=
\mathbb E \left[
\left\|
\frac{z_t - t\epsilon_\theta}{1 - t} - x
\right\|^2
\right]
$$

---

## v-pred + x-loss
$$
\mathcal L_{v \to x}
=
\mathbb E \left[
\|z_t - t v_\theta - x\|^2
\right]
$$

---

## u-pred + x-loss
$$
x_\theta^{(u)} = z_t - t V_\theta
$$

$$
\mathcal L_{u \to x}
=
\mathbb E \left[
\|z_t - t V_\theta - x\|^2
\right]
$$

---

# ε-loss

## x-pred + ε-loss
$$
\mathcal L_{x \to \epsilon}
=
\mathbb E \left[
\left\|
\frac{z_t - (1 - t)x_\theta}{t} - \epsilon
\right\|^2
\right]
$$

---

## ε-pred + ε-loss
$$
\mathcal L_{\epsilon \to \epsilon}
=
\mathbb E \left[
\|\epsilon_\theta - \epsilon\|^2
\right]
$$

---

## v-pred + ε-loss
$$
\mathcal L_{v \to \epsilon}
=
\mathbb E \left[
\|z_t + (1 - t)v_\theta - \epsilon\|^2
\right]
$$

---

## u-pred + ε-loss
$$
\epsilon_\theta^{(u)} = z_t + (1 - t)V_\theta
$$

$$
\mathcal L_{u \to \epsilon}
=
\mathbb E \left[
\|z_t + (1 - t)V_\theta - \epsilon\|^2
\right]
$$
