import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Physical Constants ===
e = 1e-19
m = 1e-27

# === Field from Line Segment ===
def field_from_segment(x0, x1, y0, q, X, Y, num=100):
    xs = np.linspace(x0, x1, num)
    ys = np.full_like(xs, y0)
    Ex, Ey = np.zeros_like(X), np.zeros_like(Y)
    k = 2e8
    dq = q / num
    for xi, yi in zip(xs, ys):
        dx = X - xi
        dy = Y - yi
        r2 = dx**2 + dy**2 + 1e-10
        r = np.sqrt(r2)
        E = k * dq / r2
        Ex += E * dx / r
        Ey += E * dy / r
    return Ex, Ey

# === Geometry ===
plate_len = 0.01
gap_y = 0.007
plate_spacing = 0.013
x1, x2, x3 = -plate_spacing, 0, plate_spacing

# === Grid ===
xlim = (-0.04, 0.04)
ylim = (-0.015, 0.015)
X, Y = np.meshgrid(np.linspace(*xlim, 160), np.linspace(*ylim, 80))

# === Electric Field Calculation ===
Ex_top, Ey_top = field_from_segment(x2 - plate_len/2, x2 + plate_len/2, gap_y, 1e-9, X, Y)
Ex_bot, Ey_bot = field_from_segment(x2 - plate_len/2, x2 + plate_len/2, -gap_y, 1e-9, X, Y)
Ex = -(Ex_top + Ex_bot) * 1e5
Ey = -(Ey_top + Ey_bot) * 1e5
field_mag = np.sqrt(Ex**2 + Ey**2)

# === Masking Field Region ===
mask = (Y < gap_y) & (Y > -gap_y) & (X > x1 - 0.00575) & (X < x3 + 0.00575)
Ex_masked = np.where(mask, Ex, 0)
Ey_masked = np.where(mask, Ey, 0)
field_mag_masked = np.where(mask, field_mag, 0)

# === Electron Beam Setup ===
n_particles = 10
dt = 1e-11
steps = 500
initial_speed = 4e7

max_vertical_offset = gap_y * 0.95
positions = np.zeros((n_particles, 2))
positions[:, 0] = xlim[0]
positions[:, 1] = np.linspace(-max_vertical_offset / 4, max_vertical_offset / 4, n_particles)

velocities = np.zeros((n_particles, 2))
velocities[:, 0] = initial_speed
velocities[:, 1] = positions[:, 1] * 1  # Stronger outward push

trajectories = [positions.copy()]

# === Interpolator ===
def interpolate_field(x, y):
    i = np.clip(((x - xlim[0]) / (xlim[1] - xlim[0]) * X.shape[1]).astype(int), 0, X.shape[1]-1)
    j = np.clip(((y - ylim[0]) / (ylim[1] - ylim[0]) * Y.shape[0]).astype(int), 0, Y.shape[0]-1)
    return np.array([Ex_masked[j, i], Ey_masked[j, i]])

# === Time Evolution ===
lens_entry = x1 - 0.005
lens_exit = x3 + 0.005

for _ in range(steps):
    for i in range(n_particles):
        x, y = positions[i]
        if lens_entry < x < lens_exit:
            Ex_local, Ey_local = interpolate_field(x, y)
            ax = -e * Ex_local / m
            if(x>x2):
                ay = - e * Ey_local  / m
            else:
                ay = e * Ey_local / m 
            velocities[i, 0] += ax * dt
            velocities[i, 1] += ay * dt
        positions[i] += velocities[i] * dt
    trajectories.append(positions.copy())

# === Plot & Animate ===
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect('equal')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.axhline(0, color='gray', linestyle='--', lw=0.8)

def draw_plate(x, label=None):
    ax.plot([x - plate_len/2, x + plate_len/2], [gap_y, gap_y], color='black', lw=2)
    ax.plot([x - plate_len/2, x + plate_len/2], [-gap_y, -gap_y], color='black', lw=2)
    if label:
        ax.text(x, gap_y + 0.002, label, ha='center', fontsize=8)

draw_plate(x1, label='+0 V')
draw_plate(x2, label='Positively Charged')
draw_plate(x3, label='+0 V')

ax.contourf(X, Y, field_mag_masked, levels=50, cmap='YlOrBr', alpha=0.6)
ax.streamplot(X, Y, Ex_masked, Ey_masked, color='blue', density=0.6, arrowsize=0, linewidth=0.8)

# === Initialize Animated Lines ===
lines = [ax.plot([], [], color='red', linestyle='--', linewidth=0.8)[0] for _ in range(n_particles)]

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    for i, line in enumerate(lines):
        path = np.array([p[i] for p in trajectories[:frame+1]])
        line.set_data(path[:, 0], path[:, 1])
    return lines

ani = FuncAnimation(fig, update, frames=len(trajectories),
                    init_func=init, blit=True, interval=25)

plt.tight_layout()
plt.show()
