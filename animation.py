import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict

data = pd.read_csv("output.csv", names=["step", "x", "y", "mass"])
steps = data["step"].unique()

grouped = data.groupby("step")

fig, ax = plt.subplots()

paths = defaultdict(lambda: {"x": [], "y": []})

N = len(grouped.get_group(steps[0]))

# create line objects once
lines = [ax.plot([], [], linewidth=1)[0] for _ in range(N)]
scat = ax.scatter([], [], cmap='plasma')



def update(frame):
    step_data = grouped.get_group(frame)

    x = step_data["x"].values
    y = step_data["y"].values
    m = step_data["mass"].values

    # center
    max_idx = m.argmax()
    cx, cy = x[max_idx], y[max_idx]

    x = x - cx
    y = y - cy

    # update paths
    for i in range(N):
        paths[i]["x"].append(x[i])
        paths[i]["y"].append(y[i])

        lines[i].set_data(paths[i]["x"], paths[i]["y"])

    # uniform scatter
    scat.set_offsets(list(zip(x, y)))
    scat.set_sizes([30] * N)
    scat.set_color("black")
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_aspect('equal')
    ax.set_title(f"Step: {frame}")

    return lines + [scat]

ani = FuncAnimation(fig, update, frames=steps[::5], interval=10, blit=True)

plt.show()