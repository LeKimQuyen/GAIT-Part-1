'''
Task 1: Implement Basic Q-Learning for Level 0
'''

import json, os
import pygame, random
from dataclasses import dataclass
from typing import Tuple, List, Dict


# ==========================================
# STEP 1: LOAD DATA FROM JSON
# ==========================================

def load_config():
    path = os.path.join(os.path.dirname(__file__), "default-config.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print("Loaded default-config.json")
    return cfg
CFG = load_config()

def load_layout(level: int):
    path = os.path.join(os.path.dirname(__file__), "layouts.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            layout = json.load(f)
        print("Loaded layouts.json")
    return layout[str(level)]


# ==========================================
# STEP 2: SETUP WINDOW AND BASIC CONSTANTS
# ==========================================

# Value from CFG
EPISODES = int(CFG["episodes"])
ALPHA = float(CFG["alpha"])
GAMMA = float(CFG["gamma"])
EPS_START = float(CFG["epsilonStart"])
EPS_END = float(CFG["epsilonEnd"])
EPS_DECAY_EP = int(CFG["epsilonDecayEpisodes"])
MAX_STEPS = int(CFG["maxStepsPerEpisode"])
FPS_VISUAL = int(CFG["fpsVisual"])
FPS_FAST = int(CFG["fpsFast"])
TILE_SIZE = int(CFG["tileSize"])
random.seed(int(CFG["seed"]))

# Pygame
WIDTH_TILES, HEIGHT_TILES = 12, 12
WIDTH, HEIGHT = WIDTH_TILES * TILE_SIZE, HEIGHT_TILES * TILE_SIZE
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GridWorld - Part 1 Q-learning, Level 0")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)

# Colors
COL_BG = (25, 28, 34)
COL_GRID = (45, 50, 58)
COL_AGENT = (74, 222, 128)
# COL_APPLE = (252, 92, 101)
COL_TEXT = (240, 240, 240)

# Actions: 0 up, 1 right, 2 down, 3 left
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
A_UP, A_RIGHT, A_DOWN, A_LEFT = 0, 1, 2, 3
ALL_ACTIONS = [A_UP, A_RIGHT, A_DOWN, A_LEFT]

# Animation
DURATION = 800

# Level 0 layout (only agent and apples on the right side)
LEVEL0 = load_layout(0)
LEVEL0 = [row.ljust(WIDTH_TILES)[:WIDTH_TILES] for row in LEVEL0]


# ==========================================
# STEP 3: ENVIRONMENT
# ==========================================

@dataclass
class StepResult:
    next_state: Tuple[int, int, int]   # (agent_x, agent_y, apple_mask)
    reward: float
    done: bool
    info: dict

class GridWorld:
    def __init__(self, layout: List[str]):
        self.layout = layout
        self.w, self.h = len(layout[0]), len(layout)
        self.rocks, self.fires, self.keys, self.chests = set(), set(), set(), set()
        self.apples, self.apple_index = [], {}
        self.monsters = []
        self.start = (0, 0)
        # Animation
        self.apple_imgs = load_images()
        self.apple_state = 0
        self.fire_state = self.fire_dir = 1

        # Parse the map once
        for y, row in enumerate(layout):
            for x, ch in enumerate(row):
                p = (x, y)
                if ch == "A":
                    self.apple_index[p] = len(self.apples)
                    self.apples.append(p)
                elif ch == "S": self.start = p
                # support other symbols for later levels if present
                elif ch == "R": self.rocks.add(p)
                elif ch == "F": self.fires.add(p)
                elif ch == "K": self.keys.add(p)
                elif ch == "C": self.chests.add(p)
                elif ch == "M": self.monsters.append(p)

        self.reset()

    def reset(self) -> Tuple[int, int, int]:
        self.agent = self.start
        self.collected_keys = 0
        self.opened_chests = set()
        self.alive = True
        # Build an apple mask with one bit per apple, all set to 1 at start
        self.apple_mask = 0
        for i in range(len(self.apples)):
            self.apple_mask |= (1 << i)
        self.step_count = 0
        return self.encode_state()

    def encode_state(self) -> Tuple[int, int, int]:
        return (self.agent[0], self.agent[1], self.apple_mask)

    # Movement helpers
    def in_bounds(self, p): return 0 <= p[0] < self.w and 0 <= p[1] < self.h
    
    def blocked(self, p): return p in self.rocks
    
    def cell_contains_monster(self, p): return p in self.monsters

    def try_move(self, p, a):
        dx, dy = ACTIONS[a]
        np = (p[0] + dx, p[1] + dy)
        if not self.in_bounds(np): return p
        if self.blocked(np): return p
        return np

    def step(self, action: int) -> StepResult:
        self.step_count += 1
        reward, done = -0.01, False     #Todo: explain why -0.01 for each step

        # 1) Move agent
        self.agent = self.try_move(self.agent, action)

        # 2) Deadly checks for later levels
        if self.agent in self.fires or self.cell_contains_monster(self.agent):
            self.alive = False
            return StepResult(self.encode_state(), reward, True, {"event": "death"})

        # 3) Apple collection logic for Level 0
        if self.agent in self.apple_index:
            idx = self.apple_index[self.agent]
            if (self.apple_mask >> idx) & 1:
                self.apple_mask &= ~(1 << idx)
                reward += 1.0

        # 4) Episode end condition when all apples gone
        if self.apple_mask == 0:
            done = True

        # 5) Monster update for later levels, skipped here
        # self.update_monsters()

        return StepResult(self.encode_state(), reward, done, {})


# ==========================================
# STEP 5: Q-TABLE
# ==========================================

class QTable:
    def __init__(self):
        self.q: Dict[Tuple[Tuple[int,int,int], int], float] = {}

    def get(self, s, a): return self.q.get((s, a), 0.0)
    
    def set(self, s, a, v): self.q[(s, a)] = v
    
    def best_value(self, s): return max(self.get(s, a) for a in ALL_ACTIONS)

    def best_actions(self, s):
        vals = [self.get(s, a) for a in ALL_ACTIONS]
        m = max(vals)
        return [a for a, v in zip(ALL_ACTIONS, vals) if v == m]

# Linear epsilon decay
def linear_epsilon(ep, start, end, decay_ep):
    if decay_ep <= 0: return end
    t = min(ep / decay_ep, 1.0)
    return start + t * (end - start)

# Epsilon-greedy policy
def epsilon_greedy(qtab: QTable, s, eps):
    '''
    If eps = 0.7
    => 30% choose random action. 70% choose action with best value
    '''
    if random.random() < eps:
        return random.choice(ALL_ACTIONS)   # choose random action
    best = qtab.best_actions(s) # choose actions with best (max) value
    return random.choice(best)  # random tie-breaking for multiple same value actions

# Q-learning update
def q_learning_update(qtab: QTable, s, a, r, sp, alpha, gamma):
    current = qtab.get(s, a)
    target = r + gamma * qtab.best_value(sp)
    qtab.set(s, a, current + alpha * (target - current))
    
    
# ==========================================
# STEP 6: DRAW GRIDWORLD
# ==========================================    
    
def draw_grid(env: GridWorld, episode, step, epsilon, total_reward):
    screen.fill(COL_BG)

    # grid lines
    for x in range(env.w):
        for y in range(env.h):
            rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, COL_GRID, rect, 1)

    # apples from bitmask
    for p, idx in env.apple_index.items():
        if (env.apple_mask >> idx) & 1:
            cx, cy = p[0]*TILE_SIZE + TILE_SIZE//2, p[1]*TILE_SIZE + TILE_SIZE//2
            # pygame.draw.circle(screen, COL_APPLE, (cx, cy), TILE_SIZE//3)
            apple_rect = env.apple_imgs[env.apple_state].get_rect(center=(cx, cy))
            screen.blit(env.apple_imgs[env.apple_state], apple_rect)


    # agent
    ax, ay = env.agent
    # Todo: replace agent interface
    pygame.draw.rect(screen, COL_AGENT, (ax*TILE_SIZE+8, ay*TILE_SIZE+8, TILE_SIZE-16, TILE_SIZE-16), border_radius=6)

    # HUD
    hud = [
        f"Ep {episode+1}/{EPISODES}  step {step}  eps {epsilon:.3f}",
        f"Apples left {bin(env.apple_mask).count('1')}",
        f"Return {total_reward:.2f}  Part 1 Q-learning Level 0",
        "V toggles fast mode. R resets."
    ]
    for i, t in enumerate(hud):
        screen.blit(font.render(t, True, COL_TEXT), (10, 8 + i*20))

    pygame.display.flip()
    
def load_images():
    path = os.path.join(os.path.dirname(__file__), "assets/")

    # Apples
    apple0 = pygame.transform.scale(pygame.image.load(path + 'apple-0.png').convert_alpha(), (TILE_SIZE, TILE_SIZE))
    apple1 = pygame.transform.scale(pygame.image.load(path + 'apple-1.png').convert_alpha(), (TILE_SIZE, TILE_SIZE))
    
    return [apple0, apple1]
    
# ==========================================
# STEP 7: MAIN TRAINING LOOP AND CONTROLS
# ==========================================        
    
def run_training():
    env = GridWorld(LEVEL0)
    qtab = QTable()
    visualize, running = True, True
    
    # For animation
    apple_timer = pygame.USEREVENT
    pygame.time.set_timer(apple_timer, DURATION)
    
    for ep in range(EPISODES):
        s = env.reset()
        ep_reward, steps = 0.0, 0
        eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)

        while running:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v: visualize = not visualize
                    if event.key == pygame.K_r:
                        qtab = QTable()
                        s = env.reset()
                        ep_reward, steps = 0.0, 0
                        eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
                if event.type == apple_timer:
                    env.apple_state += 1
                    if env.apple_state == 2:
                        env.apple_state = 0
            if not running: break

            a = epsilon_greedy(qtab, s, eps)
            res = env.step(a)
            q_learning_update(qtab, s, a, res.reward, res.next_state, ALPHA, GAMMA)
            s = res.next_state
            ep_reward += res.reward
            steps += 1

            if visualize:
                draw_grid(env, ep, steps, eps, ep_reward)
                clock.tick(FPS_VISUAL)
            else:
                if steps % 5 == 0: draw_grid(env, ep, steps, eps, ep_reward)
                clock.tick(FPS_FAST)

            if res.done or steps >= MAX_STEPS:
                draw_grid(env, ep, steps, eps, ep_reward)
                break

    pygame.quit()

if __name__ == "__main__":
    run_training()