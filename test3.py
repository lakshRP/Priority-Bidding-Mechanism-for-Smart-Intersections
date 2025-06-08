import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from collections import deque
import argparse
import time

# ---------------------------------------------
# Simulation constants
# ---------------------------------------------
DT = 0.05              # Time step for smooth animation (s)
CYCLE_TIME = 10.0      # Signal cycle duration (s)
HEADWAY = 2.0          # Minimum headway between departures (s)
LINK_LENGTH = 1.0      # Normalized link length
QUEUE_SPACING = 0.1    # Spacing between queued vehicles
NUM_LANES = 2          # Lanes per directed link
LANE_WIDTH = 0.2       # Lateral spacing between lanes

# ---------------------------------------------
# Vehicle class
# ---------------------------------------------
class Vehicle:
    def __init__(self, vid, path_edges, path_nodes, current_link, lane, bid, urgency=False):
        self.id = vid
        self.path_edges = path_edges
        self.path_nodes = path_nodes
        self.current_link = current_link
        self.lane = lane
        self.pos = 0.0
        self.bid = bid
        self.urgency = urgency
        self.state = 'traveling'

# ---------------------------------------------
# Link class
# ---------------------------------------------
class Link:
    def __init__(self, origin, dest):
        self.origin = np.array(origin, dtype=float)
        self.dest = np.array(dest, dtype=float)
        self.length = LINK_LENGTH
        self.queue = deque()
        self.num_lanes = NUM_LANES
        delta = self.dest - self.origin
        self.unit = delta / np.linalg.norm(delta)
        self.perp = np.array([-self.unit[1], self.unit[0]])

# ---------------------------------------------
# Intersection class with Priority Bidding Mechanism
# ---------------------------------------------
class Intersection:
    def __init__(self, node_id, incoming_links):
        self.id = node_id
        self.incoming_links = incoming_links
        self.schedule = []
        self.next_departure = {}
        # Algorithm parameters
        self.weights = np.array([1.0, 1.0, 1.0, 1.0])  # queue, bid, urgency, spillover
        self.beta = 1.0
        self.epsilon = 0.1
        self.alpha = 0.5
        self.C = CYCLE_TIME
        self.headway = HEADWAY

    def prepare(self):
        self.next_departure = {link: 0.0 for link in self.incoming_links}

    def compute_schedule(self, t):
        entries = [(link, veh) for link in self.incoming_links for veh in link.queue]
        if not entries:
            self.schedule = []
            return
        # Local queue lengths
        local_q = {link: len(link.queue) for link in self.incoming_links}
        # Spillover: average upstream queues
        upstream = {}
        for link in self.incoming_links:
            up_links = [l for l in self.incoming_links if tuple(l.dest) == tuple(link.origin)]
            if up_links:
                avg_up = sum(len(l.queue) for l in up_links) / len(up_links)
            else:
                avg_up = 0.0
            upstream[link] = self.alpha * avg_up
        # Compute scores and softmax
        scores, probs = [], []
        for link, veh in entries:
            f = np.array([local_q[link], veh.bid, float(veh.urgency), upstream[link]])
            s = float(np.dot(self.weights, f))
            scores.append(s)
        exp_s = np.exp(self.beta * np.array(scores))
        probs = exp_s / exp_s.sum()
        # Enforce minimum for emergencies
        for i, (link, veh) in enumerate(entries):
            if veh.urgency and probs[i] < self.epsilon:
                probs[i] = self.epsilon
        probs /= probs.sum()
        # Print auction details
        alloc = {}
        for (link, veh), s, p in zip(entries, scores, probs):
            print(f"[Intersection {self.id}] t={t:.1f}s | Veh {veh.id} | score={s:.2f} | prob={p:.2f}")
            alloc[link] = alloc.get(link, 0) + p
        # Build schedule intervals
        self.schedule = []
        start = 0.0
        for link in self.incoming_links:
            dur = alloc.get(link, 0) * self.C
            self.schedule.append((link, start, start + dur))
            start += dur

    def is_green(self, link, t):
        tmod = t % self.C
        return any(lnk == link and st <= tmod < ed for lnk, st, ed in self.schedule)

# ---------------------------------------------
# RoadNetwork class
# ---------------------------------------------
class RoadNetwork:
    def __init__(self, size):
        self.size = size
        G = nx.grid_2d_graph(size, size)
        self.G = G.to_directed()
        self.links = {(u, v): Link(u, v) for u, v in self.G.edges()}
        self.intersections = {
            node: Intersection(node, [self.links[(u, node)] for u in self.G.predecessors(node)])
            for node in self.G.nodes()
        }
        for inter in self.intersections.values():
            inter.prepare()

    def shortest_path_edges(self, src, dest):
        path = nx.shortest_path(self.G, src, dest)
        return list(zip(path[:-1], path[1:])), path

# ---------------------------------------------
# Simulation class
# ---------------------------------------------
class Simulation:
    def __init__(self, size, n_vehicles, duration, dt=DT, cycle_time=CYCLE_TIME, headway=HEADWAY):
        self.net = RoadNetwork(size)
        self.dt = dt
        self.C = cycle_time
        self.headway = headway
        self.current_time = 0.0
        self.steps = int(duration / dt)
        self.cost_history = []
        self.time_history = []
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)
        self.ax.set_facecolor('#f0f0f0')
        self.vehicles = []
        self.spawn_vehicles(n_vehicles)

    def spawn_vehicles(self, n):
        boundary = [node for node in self.net.G.nodes() if 0 in node or self.net.size - 1 in node]
        for vid in range(n):
            src = random.choice(boundary)
            dest = random.choice(boundary)
            while dest == src:
                dest = random.choice(boundary)
            edges, nodes = self.net.shortest_path_edges(src, dest)
            first = self.net.links[edges[0]]
            lane = random.randrange(first.num_lanes)
            veh = Vehicle(vid, edges, nodes, first, lane, random.random(), random.random() < 0.1)
            self.vehicles.append(veh)
            first.queue.append(veh)

    def init_anim(self):
        self.ax.clear()
        self.draw_static()
        return []

    def draw_static(self):
        for link in self.net.links.values():
            for i in range(link.num_lanes + 1):
                off = (i - link.num_lanes / 2) * LANE_WIDTH
                p0 = link.origin + link.perp * off
                p1 = link.dest + link.perp * off
                self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='gray', lw=1)
        self.ax.axis('off')
        self.ax.set_xlim(-1, self.net.size)
        self.ax.set_ylim(-1, self.net.size)

    def step(self, frame):
        # Start-of-cycle scheduling
        if frame % int(self.C / self.dt) == 0:
            total_q = sum(len(link.queue) for link in self.net.links.values())
            self.cost_history.append(total_q)
            self.time_history.append(self.current_time)
            for inter in self.net.intersections.values():
                inter.compute_schedule(self.current_time)
        # Serve green-phase departures
        for inter in self.net.intersections.values():
            for link in inter.incoming_links:
                if inter.is_green(link, self.current_time) and inter.next_departure[link] <= self.current_time and link.queue:
                    veh = link.queue.popleft()
                    veh.state = 'traveling'
                    idx = veh.path_edges.index((tuple(link.origin), tuple(link.dest)))
                    if idx + 1 < len(veh.path_edges):
                        veh.current_link = self.net.links[veh.path_edges[idx + 1]]
                        veh.pos = 0.0
                    else:
                        veh.current_link = None
                        veh.state = 'arrived'
                    inter.next_departure[link] = self.current_time + self.headway
        # Vehicle movement and queuing
        for veh in list(self.vehicles):
            if veh.state == 'traveling' and veh.current_link:
                link = veh.current_link
                speed = self.headway / self.C
                next_pos = veh.pos + self.dt * speed
                if next_pos >= link.length:
                    veh.pos = link.length
                    veh.state = 'queued'
                    link.queue.append(veh)
                else:
                    veh.pos = next_pos
            elif veh.current_link is None or veh.state == 'arrived':
                self.vehicles.remove(veh)
        # Redraw
        self.ax.clear()
        self.draw_static()
        # Draw queued vehicles
        for link in self.net.links.values():
            for i, veh in enumerate(link.queue):
                off = (veh.lane - (link.num_lanes - 1) / 2) * LANE_WIDTH
                pos = link.dest - link.unit * QUEUE_SPACING * (i + 1) + link.perp * off
                self.ax.scatter(pos[0], pos[1], s=30, color='red', zorder=2)
        # Draw traveling vehicles
        for veh in self.vehicles:
            if veh.state == 'traveling' and veh.current_link:
                link = veh.current_link
                off = (veh.lane - (link.num_lanes - 1) / 2) * LANE_WIDTH
                pos = link.origin + link.unit * veh.pos + link.perp * off
                self.ax.scatter(pos[0], pos[1], s=30, color='blue', zorder=3)
        # Draw signal phases & dynamic route lines
        for inter in self.net.intersections.values():
            x, y = inter.id
            is_green = any(inter.is_green(link, self.current_time) for link in inter.incoming_links)
            color = 'green' if is_green else 'red'
            self.ax.scatter(x, y, s=200, alpha=0.2, color=color, zorder=1)
        for veh in self.vehicles:
            if veh.current_link:
                if veh.state == 'traveling':
                    link = veh.current_link
                    off = (veh.lane - (link.num_lanes - 1) / 2) * LANE_WIDTH
                    curr = link.origin + link.unit * veh.pos + link.perp * off
                else:
                    link = veh.current_link
                    off = (veh.lane - (link.num_lanes - 1) / 2) * LANE_WIDTH
                    curr = link.dest - link.unit * QUEUE_SPACING + link.perp * off
                dest = np.array(veh.path_nodes[-1], dtype=float)
                self.ax.plot([curr[0], dest[0]], [curr[1], dest[1]], color='gray', lw=0.5, alpha=0.3, zorder=0)
        q_last = self.cost_history[-1] if self.cost_history else 0
        self.ax.set_title(f"Time {self.current_time:.2f}s  Queue={q_last}")
        self.current_time += self.dt
        return []

    def run(self):
        self.anim = FuncAnimation(
            self.fig, self.step, init_func=self.init_anim,
            frames=self.steps, interval=DT * 1000, blit=False, repeat=False
        )
        plt.show()

# ---------------------------------------------
# Runtime-scaling experiment
# ---------------------------------------------
def run_runtime_scaling(grid_size):
    vehicle_counts = [10, 20, 50, 100, 200]
    avg_times = []
    for n in vehicle_counts:
        sim = Simulation(grid_size, n, duration=50.0)
        sim.step(0)  # warm up
        t0 = time.perf_counter()
        for frame in range(1, 101): sim.step(frame)
        t1 = time.perf_counter()
        avg = (t1 - t0) / 100
        avg_times.append(avg)
        print(f"N={n:3d} -> avg step time {avg*1000:.3f} ms")
    plt.figure(figsize=(8,5))
    plt.plot(vehicle_counts, avg_times, marker='o', linewidth=2)
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Avg Step Time (s)')
    plt.title('Per‑Cycle Runtime vs Vehicle Count')
    plt.grid(True)
    plt.show()

# ---------------------------------------------
# Main entrypoint
# ---------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=4)
    parser.add_argument('--nveh', type=int, default=20)
    parser.add_argument('--duration', type=float, default=200.0)
    parser.add_argument('--runtime-scaling', action='store_true', help='Run runtime-scaling experiment')
    args = parser.parse_args()
    if args.runtime_scaling:
        run_runtime_scaling(args.size)
    else:
        sim = Simulation(args.size, args.nveh, args.duration)
        sim.run()