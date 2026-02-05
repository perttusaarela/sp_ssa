import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spatial import points_in_polygon


class PolygonDrawer:
    def __init__(self, ax, points, filename="polygons.json"):
        self.ax = ax
        self.fig = ax.figure

        self.points = np.asarray(points)

        self.current = []
        self.polygons = []

        self.line, = ax.plot([], [], "ro-")
        self.preview_line, = ax.plot([], [], "r-",  linewidth=1)
        self.closing_line, = ax.plot([], [], "r--", linewidth=1)
        self.finished_lines = []

        self.filename = filename
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.color_idx = 0
        self.polygon_colors = []

        self.cid_click = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_click
        )
        self.cid_key = self.fig.canvas.mpl_connect(
            "key_press_event", self.on_key
        )

        self.cid_scroll = self.fig.canvas.mpl_connect(
            "scroll_event", self.on_scroll
        )

        self.cid_motion = self.fig.canvas.mpl_connect(
            "motion_notify_event", self.on_move
        )

        self.count_text = self.ax.text(
            0.02, 0.98, "",
            transform=self.ax.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Left click → add vertex
        if event.button == 1:
            self.current.append((event.xdata, event.ydata))
            self._update_current()

        # Right click → close polygon
        elif event.button == 3 and len(self.current) >= 3:
            self._close_polygon()

    def on_key(self, event):
        if event.key == "n":
            if len(self.current) >= 3:
                self._close_polygon()
            self.current = []
            self._update_current()

        elif event.key == "w":
            self.save(self.filename)
            print("Saved polygons.json")

        elif event.key == "q":
            plt.close(self.fig)

        # Delete latest vertex
        elif event.key == "backspace":
            self._delete_last_vertex()

        # Delete last polygon
        elif event.key == "d":
            self._delete_last_polygon()

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return

        base_scale = 1.2

        if event.button == "up":
            scale_factor = 1 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            return

        xdata = event.xdata
        ydata = event.ydata

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])

        self.ax.set_xlim(
            [xdata - new_width * (1 - relx),
             xdata + new_width * relx]
        )
        self.ax.set_ylim(
            [ydata - new_height * (1 - rely),
             ydata + new_height * rely]
        )

        self.fig.canvas.draw_idle()

    def on_move(self, event):
        if event.inaxes != self.ax or not self.current:
            self.preview_line.set_data([], [])
            self.closing_line.set_data([], [])
            self.count_text.set_text("")
            self.fig.canvas.draw_idle()
            return

        x_last, y_last = self.current[-1]

        self.preview_line.set_data(
            [x_last, event.xdata],
            [y_last, event.ydata]
        )

        # Temporary polygon = current + mouse
        temp_poly = np.array(self.current + [(event.xdata, event.ydata)])

        # Closing edge: mouse → first vertex (only if polygon can close)
        if len(self.current) >= 2:
            x_first, y_first = self.current[0]
            self.closing_line.set_data(
                [event.xdata, x_first],
                [event.ydata, y_first]
            )
            inside = points_in_polygon(self.points, temp_poly)
            count = inside.sum()
            self.count_text.set_text(f"Points inside: {count}")
        else:
            self.closing_line.set_data([], [])
            self.count_text.set_text("")

        self.fig.canvas.draw_idle()

    def _close_polygon(self):
        poly = np.array(self.current + [self.current[0]])
        self.polygons.append(poly)

        color = self.colors[self.color_idx % len(self.colors)]
        self.color_idx += 1

        line, = self.ax.plot(
            poly[:, 0], poly[:, 1],
            color=color, linewidth=2
        )
        self.finished_lines.append(line)
        self.polygon_colors.append(color)
        self.current = []
        self.preview_line.set_data([], [])
        self.closing_line.set_data([], [])
        self.count_text.set_text("")
        self._update_current()

    def _update_current(self):
        if self.current:
            xs, ys = zip(*self.current)
        else:
            xs, ys = [], []

        self.line.set_data(xs, ys)
        if not self.current:
            self.preview_line.set_data([], [])
            self.closing_line.set_data([], [])
            self.count_text.set_text("")
        self.fig.canvas.draw_idle()

    def _delete_last_vertex(self):
        if self.current:
            self.current.pop()
            self._update_current()

    def _delete_last_polygon(self):
        if self.polygons:
            self.polygons.pop()

            line = self.finished_lines.pop()
            line.remove()

            self.fig.canvas.draw_idle()

    def save(self, filename):
        data = [
            {
                "vertices": poly.tolist(),
                "color": color
            }
            for poly, color in zip(self.polygons, self.polygon_colors)
        ]

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def get_polygons(self):
        return self.polygons


def create_map(ax):
    borders = pd.read_csv("kola_borders.csv")
    boundary = pd.read_csv("kola_boundary.csv")
    coast = pd.read_csv("kola_coast.csv")
    lakes = pd.read_csv("kola_lakes.csv")

    ax.plot(coast.V1, coast.V2, color="#4F4F4F", zorder=1)
    ax.plot(lakes.V1, lakes.V2, color="#B3E5FC", zorder=1)
    ax.plot(borders.V1, borders.V2, color="#81C784", zorder=1)
    ax.plot(boundary.V1, boundary.V2, color="#4DB6AC", zorder=2)

    # Compute bounds (reuse your logic, just cleaner)
    x_min = min(
        coast.V1.min(), lakes.V1.min(),
        borders.V1.min(), boundary.V1.min()
    )
    x_max = max(
        coast.V1.max(), lakes.V1.max(),
        borders.V1.max(), boundary.V1.max()
    )
    y_min = min(
        coast.V2.min(), lakes.V2.min(),
        borders.V2.min(), boundary.V2.min()
    )
    y_max = max(
        coast.V2.max(), lakes.V2.max(),
        borders.V2.max(), boundary.V2.max()
    )

    # Moss data
    df = pd.read_csv("moss_data.csv")
    coords = df.values[:, 2:4]
    ax.scatter(
        coords[:, 0], coords[:, 1],
        marker="x", color="#D3D3D3", zorder=3
    )

    ax.set_xlim(x_min - 30000, x_max + 30000)
    ax.set_ylim(y_min - 30000, y_max + 30000)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    return coords


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8), frameon=False)

    points = create_map(ax)
    #ax.figure.savefig("plots/kola.pdf", bbox_inches="tight")
    drawer = PolygonDrawer(ax, points)

    ax.set_title(
        "Left click: add vertex | Right click: close polygon\n"
        "n: new polygon | w: save | q: quit\n"
        "backspace: delete last vertex | d: delete last polygon\n",
        fontsize=10
    )

    plt.show()

    polygons = drawer.get_polygons()
    print(f"{len(polygons)} polygons drawn")