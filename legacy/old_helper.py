import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from collections import deque
import argparse

# ---------------------------------------------
# Simulation constants
# ---------------------------------------------
DT = 0.05              # Time step for smooth animation (s)
CYCLE_TIME = 10.0      # Signal cycle duration (s)
HEADWAY = 2.0          # Minimum headway between departures (s)
LINK_LENGTH = 1.0      # Normalized link length for movement scaling
QUEUE_SPACING = 0.1    # Distance between queued vehicles
NUM_LANES = 2          # Number of lanes per directed link
LANE_WIDTH = 0.2       # Lateral spacing between lanes

# ---------------------------------------------
# Vehicle class: tracks routing, bids, and state
# ---------------------------------------------
class Vehicle:
    def __init__(self, vid, path_edges, path_nodes, current_link, lane, bid, urgency=False):
        self.id = vid
        self.path_edges = path_edges      # Sequence of edges (u,v) for routing
        self.path_nodes = path_nodes      # Sequence of node coordinates
        self.current_link = current_link  # Link object where the vehicle currently is
        self.lane = lane                  # Lane index on the link
        self.pos = 0.0                    # Position along current link [0, LINK_LENGTH]
        self.bid = bid                    # Bid value [0,1] representing driver priority
        self.urgency = urgency            # Emergency flag
        self.state = 'traveling'          # 'traveling' or 'queued'

# ---------------------------------------------
# Link class: holds geometry and queue of vehicles
# ---------------------------------------------
class Link:
    def __init__(self, origin, dest):
        self.origin = np.array(origin, dtype=float)
        self.dest = np.array(dest, dtype=float)
        self.length = LINK_LENGTH
        self.queue = deque()              # Vehicles waiting at end of link
        self.num_lanes = NUM_LANES
        # Unit direction vector and its perpendicular for lane offsets
        delta = self.dest - self.origin
        self.unit = delta / np.linalg.norm(delta)
        self.perp = np.array([-self.unit[1], self.unit[0]])

# ---------------------------------------------
# Intersection class: runs priority bidding per cycle
# ---------------------------------------------
class Intersection:
    def __init__(self, node_id, incoming_links):
        self.id = node_id
        self.incoming_links = incoming_links  # List of Link objects feeding into this intersection
        self.schedule = []                    # List of tuples (link, start, end) for green intervals
        self.next_departure = {}              # Next allowable departure time per link
        # Algorithm parameters (will be set in prepare):
        self.weights = None     # Weight vector for feature scoring
        self.beta = None        # Softmax temperature
        self.epsilon = None     # Min probability for urgent vehicles
        self.alpha = None       # Diffusion weight
        self.C = None           # Cycle time
        self.headway = None     # Headway

    def prepare(self, weights, beta, epsilon, alpha, C, headway):
        """
        Initialize algorithm parameters for bidding.
        weights: np.array of length 4 (queue, bid, urgency, spillover)
        beta: softmax inverse temperature
        epsilon: minimum prob for urgencies
        alpha: spillover diffusion weight
        C: cycle time
        headway: service headway
        """
        self.weights = weights
        self.beta = beta
        self.epsilon = epsilon
        self.alpha = alpha
        self.C = C
        self.headway = headway
        self.next_departure = {link: 0.0 for link in self.incoming_links}

    def compute_schedule(self, t):
        """
        1) Gather queued vehicles and compute augmented features with spillover.
        2) Compute scores, apply softmax, enforce urgency minimum.
        3) Compute VCG-style payments and print auction details.
        4) Build green-time schedule intervals.
        """
        # Step 1: Collect entries
        entries = [(link, veh) for link in self.incoming_links for veh in link.queue]
        if not entries:
            self.schedule = []
            return
        # Compute local and spillover features
        local_queues = {link: len(link.queue) for link in self.incoming_links}
        # Spillover: average upstream queue lengths
        upstream_pressure = {}
        for link in self.incoming_links:
            # Upstream nodes are those feeding into link.origin
            up_links = [l for l in self.incoming_links if tuple(l.dest) == tuple(link.origin)]
            if not up_links:
                spill = 0.0
            else:
                spill = sum(len(l.queue) for l in up_links) / len(up_links)
            upstream_pressure[link] = self.alpha * spill
        # Build feature vectors and scores
        features = []
        scores = []
        for link, veh in entries:
            f = np.array([
                local_queues[link],             # local queue length
                veh.bid,                        # bid value
                float(veh.urgency),             # urgency flag
                upstream_pressure[link]         # spillover term
            ])
            s = float(np.dot(self.weights, f))
            features.append((link, veh, f))
            scores.append(s)
        # Step 2: Softmax allocation
        exp_scores = np.exp(self.beta * np.array(scores))
        probs = exp_scores / exp_scores.sum()
        # enforce minimum for emergencies
        for i, (_, veh) in enumerate([(lnk,v) for lnk,v in entries]):
            if veh.urgency and probs[i] < self.epsilon:
                probs[i] = self.epsilon
        probs /= probs.sum()
        # Step 3: Print auction and compute payments
        alloc = {}
        for (link, veh, f), p, s in zip(features, probs, scores):
            print(f"[Intersection {self.id}] t={t:.1f}s | Veh {veh.id} | features={f.tolist()} | score={s:.2f} | prob={p:.2f}")
            alloc[link] = alloc.get(link, 0) + p
        # VCG-style payment for winners (approximate externality)
        for (link, veh), p in zip(entries, probs):
            pay = sum(p_k * v_k.bid for (_,v_k), p_k in zip(entries, probs) if v_k!=veh) - (1-p)*veh.bid
            pay = max(pay, 0.0)
            print(f"  --> Veh {veh.id} pays {pay:.2f}")
        # Step 4: Build green schedule intervals
        self.schedule = []
        start = 0.0
        for link in self.incoming_links:
            dur = alloc.get(link, 0) * self.C
            self.schedule.append((link, start, start+dur))
            start += dur

    def is_green(self, link, t):
        """Return True if link is in green phase at time t."""
        tmod = t % self.C
        for lnk, st, ed in self.schedule:
            if lnk == link and st <= tmod < ed:
                return True
        return False

# ---------------------------------------------
# RoadNetwork: builds grid and links
# ---------------------------------------------
class RoadNetwork:
    def __init__(self, size):
        self.size = size
        G = nx.grid_2d_graph(size, size)
        self.G = G.to_directed()
        self.links = {(u,v): Link(u,v) for u,v in self.G.edges()}
        self.intersections = {
            node: Intersection(node, [self.links[(u,node)] for u in self.G.predecessors(node)])
            for node in self.G.nodes()
        }

    def shortest_path_edges(self, src, dest):
        path = nx.shortest_path(self.G, src, dest)
        return list(zip(path[:-1], path[1:])), path

# ---------------------------------------------
# Simulation: orchestrates traffic flow and visualization
# ---------------------------------------------
class Simulation:
    def __init__(self, size, n_vehicles, duration, dt=DT, cycle_time=CYCLE_TIME, headway=HEADWAY):
        # Initialize network and parameters
        self.net = RoadNetwork(size)
        self.dt = dt
        self.C = cycle_time
        self.headway = headway
        self.current_time = 0.0
        self.steps = int(duration/dt)
        self.cost_history, self.time_history = [], []
        # Figure setup
        self.fig, self.ax = plt.subplots(figsize=(6,6), dpi=100)
        self.ax.set_facecolor('#f0f0f0')
        # Prepare intersections with algorithm params
        weights = np.array([1.0,1.0,1.0,1.0])  # adjust length to 4
        beta, epsilon, alpha = 1.0, 0.1, 0.5
        for inter in self.net.intersections.values():
            inter.prepare(weights, beta, epsilon, alpha, cycle_time, headway)
        # Spawn vehicles
        self.vehicles = []
        self.spawn_vehicles(n_vehicles)

    def spawn_vehicles(self, n):
        """Randomly place n vehicles at boundary nodes with random destinations."""
        boundary = [node for node in self.net.G.nodes() if 0 in node or self.net.size-1 in node]
        for vid in range(n):
            src = random.choice(boundary)
            dest = random.choice(boundary)
            while dest==src:
                dest = random.choice(boundary)
            edges, nodes = self.net.shortest_path_edges(src,dest)
            first = self.net.links[edges[0]]
            lane = random.randrange(first.num_lanes)
            bid = random.random()
            urgency = (random.random()<0.1)
            veh = Vehicle(vid, edges, nodes, first, lane, bid, urgency)
            self.vehicles.append(veh)
            first.queue.append(veh)

    def init_anim(self):
        self.ax.clear()
        self.draw_static()
        return []

    def draw_static(self):
        """Draw static road geometry."""
        for link in self.net.links.values():
            for i in range(link.num_lanes+1):
                off = (i-link.num_lanes/2)*LANE_WIDTH
                p0 = link.origin + link.perp*off
                p1 = link.dest   + link.perp*off
                self.ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color='gray',lw=1)
        self.ax.axis('off')
        self.ax.set_xlim(-1,self.net.size)
        self.ax.set_ylim(-1,self.net.size)

    def step(self, frame):
        # 1) Start of cycle: compute schedules
        if frame % int(self.C/self.dt)==0:
            total_q = sum(len(link.queue) for link in self.net.links.values())
            self.cost_history.append(total_q)
            self.time_history.append(self.current_time)
            for inter in self.net.intersections.values():
                inter.compute_schedule(self.current_time)
        # 2) Serve green & update vehicle state
        for inter in self.net.intersections.values():
            for link in inter.incoming_links:
                if inter.is_green(link, self.current_time) and inter.next_departure[link] <= self.current_time and link.queue:
                    veh = link.queue.popleft()
                    veh.state = 'traveling'
                    idx = veh.path_edges.index((tuple(link.origin), tuple(link.dest)))
                    if idx + 1 < len(veh.path_edges):
                        # proceed to next link
                        veh.current_link = self.net.links[veh.path_edges[idx + 1]]
                        veh.pos = 0.0
                    else:
                        # reached final destination
                        veh.current_link = None
                        veh.state = 'arrived'
                    inter.next_departure[link] = self.current_time + self.headway
        # 3) Move vehicles along links (smooth queuing)
        for veh in list(self.vehicles):
            if veh.state=='traveling' and veh.current_link:
                link=veh.current_link
                speed=self.headway/self.C
                nxt=veh.pos+self.dt*speed
                if nxt>=link.length:
                    # treat as arriving at intersection, will be served next cycle
                    veh.pos=link.length
                    veh.state='queued'
                    link.queue.append(veh)
                else:
                    veh.pos=nxt
        # 4) Redraw all
        self.ax.clear(); self.draw_static()
        # queued (red)
        for link in self.net.links.values():
            for i,veh in enumerate(link.queue):
                off=(veh.lane-(link.num_lanes-1)/2)*LANE_WIDTH
                pos=link.dest-link.unit*QUEUE_SPACING*(i+1)+link.perp*off
                self.ax.scatter(pos[0],pos[1],s=30,color='red',zorder=2)
        # traveling (blue)
        for veh in self.vehicles:
            if veh.state=='traveling' and veh.current_link:
                link=veh.current_link
                off=(veh.lane-(link.num_lanes-1)/2)*LANE_WIDTH
                pos=link.origin+link.unit*veh.pos+link.perp*off
                self.ax.scatter(pos[0],pos[1],s=30,color='blue',zorder=3)
        # signals
        for inter in self.net.intersections.values():
            x,y=inter.id
            green=any(inter.is_green(link,self.current_time) for link in inter.incoming_links)
            color='green' if green else 'red'
            self.ax.scatter(x,y,s=200,alpha=0.2,color=color,zorder=1)
        self.ax.set_title(f"Time {self.current_time:.2f}s  Queue={self.cost_history[-1]}")
        self.current_time+=self.dt
        return []

    def run(self):
        self.anim=FuncAnimation(self.fig,self.step,init_func=self.init_anim,frames=self.steps,interval=DT*1000,blit=False,repeat=False)
        plt.show()
        # efficiency plot
        plt.figure(); plt.plot(self.time_history,self.cost_history,lw=2)
        plt.xlabel('Time (s)'); plt.ylabel('Total Queue Length')
        plt.title('Queue Length Over Time'); plt.grid(True)
        plt.show()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--size',type=int,default=4)
    p.add_argument('--nveh',type=int,default=20)
    p.add_argument('--duration',type=float,default=200.0)
    args=p.parse_args()
    Simulation(args.size,args.nveh,args.duration).run()
