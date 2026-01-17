'''
Part 1 full tasks within one single file
'''

import json, os, pygame, random
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

def load_layouts():
    path = os.path.join(os.path.dirname(__file__), "layouts.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            layout = json.load(f)
        print("Loaded layouts.json")
    return layout
LEVELS = load_layouts()

def load_images():
    path = os.path.join(os.path.dirname(__file__), "assets/")

    # Apples
    apple = pygame.transform.scale(pygame.image.load(path + 'apple.png').convert_alpha(), (TILE_SIZE // 1.5, TILE_SIZE // 1.5))
    
    # Fires
    fire_0 = pygame.transform.scale(pygame.image.load(path + 'fire-0.png').convert_alpha(), (TILE_SIZE // 1.5, TILE_SIZE // 1.5))
    fire_1 = pygame.transform.scale(pygame.image.load(path + 'fire-1.png').convert_alpha(), (TILE_SIZE // 1.5, TILE_SIZE // 1.5))
    fire_2 = pygame.transform.scale(pygame.image.load(path + 'fire-2.png').convert_alpha(), (TILE_SIZE // 1.5, TILE_SIZE // 1.5))

    # Key
    key = pygame.transform.scale(pygame.image.load(path + 'key.png').convert_alpha(), (TILE_SIZE // 2, TILE_SIZE // 2))

    # Chest
    chest_closed = pygame.transform.scale(pygame.image.load(path + 'chest-closed.png').convert_alpha(), (TILE_SIZE, TILE_SIZE))
    chest_opened = pygame.transform.scale(pygame.image.load(path + 'chest-opened.png').convert_alpha(), (TILE_SIZE, TILE_SIZE))
    
    return apple, [fire_0, fire_1, fire_2], key, [chest_closed, chest_opened]

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
pygame.init()
WIDTH_TILES, HEIGHT_TILES = 12, 12
WIDTH, HEIGHT = WIDTH_TILES * TILE_SIZE, HEIGHT_TILES * TILE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Part 1 - GridWorld")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 14)
font_med = pygame.font.SysFont("consolas", 16)
font_big = pygame.font.SysFont("consolas", 24, bold=True)
font_title = pygame.font.SysFont("consolas", 32, bold=True)

# Colors
COL_BG, COL_GRID = (25, 28, 34), (45, 50, 58)
COL_AGENT = (74, 222, 128)
COL_ROCK = (100, 100, 100)
COL_MONSTER, COL_TEXT = (148, 0, 211), (240, 240, 240)
BUTTON_COLOR = (60, 70, 90)
BUTTON_HOVER = (80, 100, 130)
BUTTON_TEXT = (255, 255, 255)

# Actions: 0 up, 1 right, 2 down, 3 left
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
A_UP, A_RIGHT, A_DOWN, A_LEFT = 0, 1, 2, 3 
ALL_ACTIONS = [A_UP, A_RIGHT, A_DOWN, A_LEFT]

# Animation
DURATION = 800
FIRE_TIMER = pygame.USEREVENT


# ==========================================
# STEP 3: ENVIRONMENT
# ==========================================

@dataclass
class StepResult:
    next_state: Tuple[int, int, int, int, int]    # (agent_x, agent_y, apple_mask, key_mask, chest_mask)
    reward: float
    done: bool
    info: dict

class GridWorld:
    def __init__(self, layout: List[str], monster_prob=0.4):
        self.layout = [r.ljust(WIDTH_TILES)[:WIDTH_TILES] for r in layout]
        self.w, self.h = WIDTH_TILES, HEIGHT_TILES
        self.monster_prob = monster_prob
        self.rocks, self.fires = set(), set()
        self.apples, self.apple_idx = [], {}
        self.keys, self.key_idx = [], {}
        self.collected_keys = 0
        self.chests, self.chest_idx = [], {}
        self.monster_starts = []
        self.start = (0,0)
        # Animation
        self.apple_img, self.fire_imgs, self.key_img, self.chest_imgs = load_images()
        self.fire_state = self.fire_dir = 1
        
        # Parse the map once
        for y, row in enumerate(self.layout):
            for x, ch in enumerate(row):
                p = (x, y)
                if ch == "A":
                    self.apple_idx[p] = len(self.apples)
                    self.apples.append(p)
                elif ch == "S": self.start = p
                elif ch == "R": self.rocks.add(p)
                elif ch == "F": self.fires.add(p)
                elif ch == "K":
                    self.key_idx[p] = len(self.keys)
                    self.keys.append(p)
                elif ch == "C":
                    self.chest_idx[p] = len(self.chests)
                    self.chests.append(p)
                elif ch == "M": self.monster_starts.append(p)
        self.reset()
    
    def reset(self) -> Tuple[int, int, int, int, int]:
        self.agent = self.start
        self.collected_keys = 0
        self.opened_chests = set()
        self.alive, self.step_count = True, 0
        # Reset masks
        self.apple_mask = (1 << len(self.apples)) - 1
        self.key_mask = (1 << len(self.keys)) - 1
        self.chest_mask = (1 << len(self.chests)) - 1
        self.monsters = list(self.monster_starts)
        return self.encode_state()
    
    def encode_state(self) -> Tuple[int, int, int, int, int]:
        return (self.agent[0], self.agent[1], self.apple_mask, 
                self.key_mask, self.chest_mask)
    
    def in_bounds(self, p):
        return 0 <= p[0] < self.w and 0 <= p[1] < self.h

    def try_move(self, p, a):
        dx, dy = ACTIONS[a]
        np = (p[0] + dx, p[1] + dy)
        if not self.in_bounds(np) or np in self.rocks:
            return p
        return np
    
    def step(self, action: int) -> StepResult:
        self.step_count += 1
        reward = 0.0
        
        # 1) Move agent
        self.agent = self.try_move(self.agent, action)
        
        # 2) Deadly checks for fires and monsters
        if self.agent in self.fires or self.agent in self.monsters:
            self.alive = False
            return StepResult(self.encode_state(), reward, True, {"death": True})
        
        # 3) Apple collection logic for Level 0
        if self.agent in self.apple_idx:
            idx = self.apple_idx[self.agent]
            if (self.apple_mask >> idx) & 1:
                self.apple_mask &= ~(1 << idx)
                reward += 1.0
        
        # 4) Key collection logic for Level 2-3 (No reward)
        if self.agent in self.key_idx:
            idx = self.key_idx[self.agent]
            if (self.key_mask >> idx) & 1:
                self.key_mask &= ~(1 << idx)
                self.collected_keys += 1
        
        # 5) Chest collection logic for Level 2-3 (requires key)
        if self.agent in self.chest_idx and self.collected_keys > 0:
            idx = self.chest_idx[self.agent]
            if (self.chest_mask >> idx) & 1:
                self.chest_mask &= ~(1 << idx)
                self.collected_keys -= 1
                reward += 2.0
        
        # 6) Update monsters logic for Level 4-5
        self.update_monsters()
        
        # 7) Deadly checks for fires and monsters after update monsters
        if self.agent in self.monsters:
            self.alive = False
            return StepResult(self.encode_state(), reward, True, {"death": True})
        
        # 8) Episode end condition when all apples and chests gone
        done = (self.apple_mask == 0 and self.chest_mask == 0)
        return StepResult(self.encode_state(), reward, done, {})
    
    def update_monsters(self):
        new = []
        for m in self.monsters:
            if random.random() < self.monster_prob:     # chance to move
                # Find legal movements
                moves = [self.try_move(m, a) for a in ALL_ACTIONS]
                moves = [p for p in moves if p != m]    # If new positions different to m
                if moves:
                    new.append(random.choice(moves))
                else:
                    new.append(m)
            else:
                new.append(m)   # no movement
        self.monsters = new

class Button:
    def __init__(self, x, y, w, h, text, action=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action = action
        self.hovered = False
    
    def draw(self, surf):
        color = BUTTON_HOVER if self.hovered else BUTTON_COLOR
        pygame.draw.rect(surf, color, self.rect, border_radius=8)
        pygame.draw.rect(surf, COL_TEXT, self.rect, 2, border_radius=8)
        text_surf = font_med.render(self.text, True, BUTTON_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surf.blit(text_surf, text_rect)
    
    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)
    
    def check_click(self, pos):
        return self.rect.collidepoint(pos)


# ==========================================
# STEP 4: Q-TABLE
# ==========================================

class QTable:
    def __init__(self):
        # {(((agent_x, agent_y, apple_mask, key_mask, chest_mask), action), value)}
        self.q: Dict[Tuple[Tuple[int, int, int, int, int], int], float] = {}
    
    def get(self, s, a):
        return self.q.get((s,a), 0.0)
    
    def set(self, s, a, v):
        self.q[(s,a)] = v
        
    def best_value(self, s):
        return max(self.get(s,a) for a in ALL_ACTIONS)
    
    def best_actions(self, s):
        vals = [self.get(s,a) for a in ALL_ACTIONS]
        m = max(vals)
        return [a for a,v in zip(ALL_ACTIONS, vals) if v == m]


# ==========================================
# STEP 5: Main algorithm functions
# ==========================================

def linear_epsilon(ep, start, end, decay_ep):
    if decay_ep <= 0: return end
    t = min(ep / decay_ep, 1.0)
    return start + t * (end - start)

def epsilon_greedy(qtab: QTable, s, eps):
   if random.random() < eps:
       return random.choice(ALL_ACTIONS)   # choose random action
   best = qtab.best_actions(s) # choose actions with best (max) value
   return random.choice(best)  # random tie-breaking for multiple same value actions

def q_learning_update(qtab: QTable, s, a, r, sp, alpha, gamma):
    current = qtab.get(s, a)
    target = r + gamma * qtab.best_value(sp)    # Choose the best (max) value of the state
    qtab.set(s, a, current + alpha * (target - current))

def sarsa_update(qtab: QTable, s, a, r, sp, ap, alpha, gamma):
    current = qtab.get(s, a)
    target = r + gamma * qtab.get(sp, ap)       # Get value of the next state
    qtab.set(s, a, current + alpha * (target - current))


# ==========================================
# STEP 6: Intrinsic Reward for Level 6 (Task 5)
# ==========================================

class IntrinsicRewardTracker:
    def __init__(self, strength=0.5):
        self.strength = strength
        self.visit_counts = {}
        
    def reset(self):
        self.visit_counts = {}
        
    def get(self, s):
        # Return 0 for the first visit
        n = self.visit_counts.get(s, 0)
        r = self.strength / ((n + 1) ** 0.5)
        self.visit_counts[s] = n + 1
        return r


# ==========================================
# STEP 7: DRAW GRIDWORLD
# ==========================================    

def draw(env: GridWorld, ep, step, eps, ret, alg, lvl, is_intrinsic=False):
    screen.fill(COL_BG)
    
    # Grid lines
    for x in range(env.w):
        for y in range(env.h):
            rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, COL_GRID, rect, 1)
    
    # Rocks -> a shade of dark gray rectangle
    for p in env.rocks:
        pygame.draw.rect(screen, COL_ROCK, (p[0]*TILE_SIZE+5, p[1]*TILE_SIZE+5, TILE_SIZE-10, TILE_SIZE-10))
    
    # Fires -> animation with 3 states
    for p in env.fires:
        cx, cy = p[0]*TILE_SIZE + TILE_SIZE//2, p[1]*TILE_SIZE + TILE_SIZE//2
        fire_rect = env.fire_imgs[env.fire_state].get_rect(center=(cx, cy))
        screen.blit(env.fire_imgs[env.fire_state], fire_rect)
    
    # Apples -> render an image if that apple is not collected
    for p, idx in env.apple_idx.items():
        if (env.apple_mask >> idx) & 1:
            cx, cy = p[0]*TILE_SIZE + TILE_SIZE//2, p[1]*TILE_SIZE + TILE_SIZE//2
            apple_rect = env.apple_img.get_rect(center=(cx, cy))
            screen.blit(env.apple_img, apple_rect)
    
    # Keys -> render a key image if that key is not collected
    for p, idx in env.key_idx.items():
        if (env.key_mask >> idx) & 1:
            cx, cy = p[0]*TILE_SIZE + TILE_SIZE//2, p[1]*TILE_SIZE + TILE_SIZE//2
            key_rect = env.key_img.get_rect(center=(cx, cy))
            screen.blit(env.key_img, key_rect)

    # Chests -> two states of chest corresponding with collected and not collected
    for p, idx in env.chest_idx.items():
        cx, cy = p[0]*TILE_SIZE + TILE_SIZE//2, p[1]*TILE_SIZE + TILE_SIZE//2
        if (env.chest_mask >> idx) & 1:
            state = 0       # Closed chest image
        else: state = 1     # Opened chest image
        chest_rect = env.chest_imgs[state].get_rect(center=(cx, cy))
        screen.blit(env.chest_imgs[state], chest_rect)
    
    # Monsters -> a DarkViolet circle
    for p in env.monsters:
        pygame.draw.circle(screen, COL_MONSTER, (p[0]*TILE_SIZE+TILE_SIZE//2, p[1]*TILE_SIZE+TILE_SIZE//2), TILE_SIZE//3)
    
    # Agent -> a Bright Green rectangle
    ax, ay = env.agent
    pygame.draw.rect(screen, COL_AGENT, (ax*TILE_SIZE+8, ay*TILE_SIZE+8, TILE_SIZE-16, TILE_SIZE-16), border_radius=6)
    
    itext = " +Intrinsic" if is_intrinsic else ""
    hud = [
        f"{alg}{itext} | Level {lvl}",
        f"Ep {ep+1} | Step {step} | eps {eps:.3f}",
        f"Collected Keys {env.collected_keys} | Chests left {bin(env.chest_mask).count('1')} | Apples left {bin(env.apple_mask).count('1')}",
        f"Return {ret:.2f}",
        "V fast mode | S skip to end | Q quit"
    ]
    for i, t in enumerate(hud):
        screen.blit(font_med.render(t, True, COL_TEXT), (10, 8+i*22))
    pygame.display.flip()

def draw_comparison_chart(returns_dict, title):
    """Draw training curves comparing different algorithms"""
    screen.fill(COL_BG)
    
    # Title
    title_surf = font_big.render(title, True, (100, 255, 100))
    screen.blit(title_surf, (WIDTH//2 - title_surf.get_width()//2, 20))
    
    # Chart area - adjusted for better proportions
    chart_x, chart_y = 100, 100
    chart_w, chart_h = WIDTH - 320, HEIGHT - 180
    
    # Draw axes
    pygame.draw.line(screen, COL_TEXT, (chart_x, chart_y + chart_h), 
                    (chart_x + chart_w, chart_y + chart_h), 2)
    pygame.draw.line(screen, COL_TEXT, (chart_x, chart_y), 
                    (chart_x, chart_y + chart_h), 2)
    
    # X-axis label
    ep_label = font.render("Episodes", True, COL_TEXT)
    screen.blit(ep_label, (chart_x + chart_w//2 - ep_label.get_width()//2, chart_y + chart_h + 25))
    
    # Y-axis label (vertical)
    ret_label = "Return"
    y_start = chart_y + chart_h//2 - (len(ret_label) * 15)//2
    for i, ch in enumerate(ret_label):
        ch_surf = font.render(ch, True, COL_TEXT)
        screen.blit(ch_surf, (20, y_start + i * 15))
    
    # Find max values for scaling
    max_episodes = max(len(returns) for returns in returns_dict.values())
    all_returns = []
    for returns in returns_dict.values():
        all_returns.extend(returns)
    max_return = max(all_returns) if all_returns else 1
    min_return = min(all_returns) if all_returns else 0
    
    # Add padding to y-axis
    y_padding = (max_return - min_return) * 0.1
    max_return += y_padding
    min_return = max(0, min_return - y_padding)
    
    # Colors for different curves
    colors = [
        (74, 222, 128),   # Green
        (252, 92, 101),   # Red
        (255, 215, 0),    # Yellow
        (148, 0, 211)     # Purple
    ]
    
    # Draw curves
    for idx, (name, returns) in enumerate(returns_dict.items()):
        color = colors[idx % len(colors)]
        
        # Smooth data with moving average
        window = 50
        smoothed = []
        for i in range(len(returns)):
            start = max(0, i - window//2)
            end = min(len(returns), i + window//2)
            smoothed.append(sum(returns[start:end]) / (end - start))
        
        # Plot points
        points = []
        for i, ret in enumerate(smoothed):
            x = chart_x + int((i / max(max_episodes - 1, 1)) * chart_w)
            y_range = max_return - min_return if max_return != min_return else 1
            y_val = chart_y + chart_h - int(((ret - min_return) / y_range) * chart_h)
            # Clamp y values to chart bounds
            y_val = max(chart_y, min(chart_y + chart_h, y_val))
            points.append((x, y_val))
        
        # Draw line
        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, 2)
        
        # Legend - positioned to the right of chart
        legend_x = chart_x + chart_w + 30
        legend_y = chart_y + 30 + idx * 30
        pygame.draw.line(screen, color, (legend_x, legend_y), 
                        (legend_x + 40, legend_y), 3)
        name_surf = font.render(name, True, COL_TEXT)
        screen.blit(name_surf, (legend_x + 50, legend_y - 8))
    
    # Grid lines and Y-axis labels
    for i in range(5):
        y = chart_y + int((i / 4) * chart_h)
        pygame.draw.line(screen, (60, 60, 60), (chart_x, y), (chart_x + chart_w, y), 1)
        val = max_return - (i / 4) * (max_return - min_return)
        val_surf = font.render(f"{val:.1f}", True, COL_TEXT)
        screen.blit(val_surf, (chart_x - 60, y - 8))
    
    # X-axis labels (episodes)
    for i in range(5):
        x = chart_x + int((i / 4) * chart_w)
        ep_val = int((i / 4) * max_episodes)
        ep_surf = font.render(str(ep_val), True, COL_TEXT)
        screen.blit(ep_surf, (x - ep_surf.get_width()//2, chart_y + chart_h + 10))
    
    # Instructions
    inst_surf = font_med.render("Press space to continue...", True, COL_TEXT)
    screen.blit(inst_surf, (WIDTH//2 - inst_surf.get_width()//2, HEIGHT - 40))
    
    pygame.display.flip()


# ==========================================
# STEP 8: Support functions
# ==========================================    

def show_main_menu():
    """Main menu: single level or comparison"""
    btn_w = 300
    btn_h = 60
    dis_y = 80
    mid_x = WIDTH//2 - btn_w//2
    mid_y = HEIGHT//2 - btn_h
    
    buttons = [
        Button(mid_x, mid_y - dis_y, btn_w, btn_h, "Train Single Level", "single"),
        Button(mid_x, mid_y, btn_w, btn_h, "Compare Algorithms", "compare"),
        Button(mid_x, mid_y + dis_y + 20, btn_w, btn_h, "Quit", "quit")
    ]
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.MOUSEMOTION:
                for btn in buttons:
                    btn.check_hover(event.pos)
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn in buttons:
                    if btn.check_click(event.pos):
                        return btn.action
        
        screen.fill(COL_BG)
        title = font_title.render("GridWorld RL Training", True, (100, 255, 100))
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 80))
        
        for btn in buttons:
            btn.draw(screen)
        
        pygame.display.flip()
        clock.tick(30)

def show_level_menu():
    """Select level and algorithm for single training"""
    btn_w = 450
    btn_h = 45
    dis_y = 60
    mid_x = WIDTH//2 - btn_w//2
    mid_y = HEIGHT//2 - btn_h//2
    
    buttons = [
        Button(mid_x, mid_y + dis_y * -3, btn_w, btn_h, "Level 0: Q-Learning (Task 1)", (0, "q")),
        Button(mid_x, mid_y + dis_y * -2, btn_w, btn_h, "Level 1: SARSA (Task 2)", (1, "s")),
        Button(mid_x, mid_y + dis_y * -1, btn_w, btn_h, "Level 2: Q-Learning (Task 3)", (2, "q")),
        Button(mid_x, mid_y, btn_w, btn_h, "Level 3: SARSA (Task 3)", (3, "s")),
        Button(mid_x, mid_y + dis_y * 1, btn_w, btn_h, "Level 4: Q-Learning (Task 4)", (4, "q")),
        Button(mid_x, mid_y + dis_y * 2, btn_w, btn_h, "Level 5: SARSA (Task 4)", (5, "s")),
        Button(mid_x, mid_y + dis_y * 3, btn_w, btn_h, "Level 6: Intrinsic with Q-Learning (Task 5)", (6, "qi")),
        Button(WIDTH//2-90, mid_y + dis_y * 4 + 10, 180, 35, "Back", "back")
    ]
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.MOUSEMOTION:
                for btn in buttons:
                    btn.check_hover(event.pos)
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn in buttons:
                    if btn.check_click(event.pos):
                        return btn.action
        
        screen.fill(COL_BG)
        title = font_big.render("Select Level to Train", True, (100, 255, 100))
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 40))
        
        for btn in buttons:
            btn.draw(screen)
        
        pygame.display.flip()
        clock.tick(30)

def show_comparison_menu():
    """Select what to compare"""
    btn_w = 400
    btn_h = 55
    dis_y = 80
    mid_x = WIDTH//2 - btn_w//2
    mid_y = HEIGHT//2 - btn_h//2
    
    buttons = [
        Button(mid_x, mid_y - dis_y, btn_w, btn_h, "Task 2: Q-Learning vs SARSA (Fire)", "task2"),
        Button(mid_x, mid_y, btn_w, btn_h, "Task 4: Monsters (Both Algorithms)", "task4"),
        Button(mid_x, mid_y + dis_y, btn_w, btn_h, "Task 5: With/Without Intrinsic", "task5"),
        Button(WIDTH//2 - 180//2, mid_y + dis_y * 2 + 20, 180, 40, "Back", "back")
    ]
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.MOUSEMOTION:
                for btn in buttons:
                    btn.check_hover(event.pos)
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn in buttons:
                    if btn.check_click(event.pos):
                        return btn.action
        
        screen.fill(COL_BG)
        title = font_big.render("Select Comparison", True, (100, 255, 100))
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 60))
        
        for btn in buttons:
            btn.draw(screen)
        
        pygame.display.flip()
        clock.tick(30)

def run_single_level(level, alg, intrinsic=False):
    """Train single level"""
    random.seed(CFG["seed"])
    # Get layout and setup environment
    env = GridWorld(LEVELS[str(level)])
    
    # Start training
    if alg == "q" or alg == "qi":
        use_intr = (alg == "qi" or intrinsic)
        returns = train_q_learning(env, level, use_intr)
        alg_name = "Q-Learning"
    else:
        returns = train_sarsa(env, level)
        alg_name = "SARSA"
    
    if returns is None:
        return "quit"
    
    return show_results_menu(alg_name, level, returns, alg == "qi" or intrinsic)

def show_results_menu(alg, lvl, returns, is_intrinsic=False):
    """Show results with options"""
    avg_100 = sum(returns[-100:])/100 if len(returns) >= 100 else sum(returns)/len(returns)
    
    btn_w = 180
    btn_h = 50
    
    buttons = [
        Button(WIDTH//2-200, 350, btn_w, btn_h, "Retry", "retry"),
        Button(WIDTH//2+20, 350, btn_w, btn_h, "New Level", "menu"),
        Button(WIDTH//2-90, 420, btn_w, btn_h, "Main Menu", "main")
    ]
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.MOUSEMOTION:
                for btn in buttons:
                    btn.check_hover(event.pos)
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn in buttons:
                    if btn.check_click(event.pos):
                        return btn.action
        
        screen.fill(COL_BG)
        
        itext = " +Intrinsic" if is_intrinsic else ""
        title = f"{alg}{itext} - Level {lvl} Complete!"
        title = font_big.render(title, True, (100, 255, 100))
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 50))
        
        results = [
            f"Total Episodes: {len(returns)}",
            f"Average Return (last 100): {avg_100:.2f}",
            f"Average Return (all): {sum(returns)/len(returns):.2f}",
            f"Best Return: {max(returns):.2f}",
            f"Worst Return: {min(returns):.2f}"
        ]
        
        for i, text in enumerate(results):
            y = 150 + i * 35
            screen.blit(font_med.render(text, True, COL_TEXT), (WIDTH//2 - 180, y))
        
        for btn in buttons:
            btn.draw(screen)
        
        pygame.display.flip()
        clock.tick(30)

def run_comparison_task2():
    """Compare Q-Learning vs SARSA on Level 1 (fire)"""    
    # Show loading screen
    screen.fill(COL_BG)
    screen.blit(font_big.render("Training Q-Learning on Level 1...", True, COL_TEXT), (WIDTH//2 - 250, HEIGHT//2))
    screen.blit(font_med.render("Press S to skip visualization", True, COL_TEXT), (WIDTH//2 - 150, HEIGHT//2 + 50))
    pygame.display.flip()
    
    random.seed(CFG["seed"])
    env = GridWorld(LEVELS["1"])
    returns_q = train_q_learning(env, 1, False)
    if returns_q is None:
        return "quit"
    
    # Show loading screen for second training
    screen.fill(COL_BG)
    screen.blit(font_big.render("Training SARSA on Level 1...", True, COL_TEXT), (WIDTH//2 - 200, HEIGHT//2))
    screen.blit(font_med.render("Press S to skip visualization", True, COL_TEXT), (WIDTH//2 - 150, HEIGHT//2 + 50))
    pygame.display.flip()
    
    random.seed(CFG["seed"])
    env = GridWorld(LEVELS["1"])
    returns_s = train_sarsa(env, 1)
    if returns_s is None:
        return "quit"
    
    # Show comparison chart
    draw_comparison_chart({
        "Q-Learning": returns_q,
        "SARSA": returns_s
    }, "Task 2: Q-Learning vs SARSA (Level 1)")
    
    waiting = True
    while waiting:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return "quit"
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    waiting = False
    
    return "menu"

def run_comparison_task4():
    """Compare algorithms on monster levels"""
    results = {}
    
    for lvl in [4, 5]:        
        # Show loading screen
        screen.fill(COL_BG)
        screen.blit(font_big.render(f"Training Q-Learning on Level {lvl}...", True, COL_TEXT), (WIDTH//2 - 250, HEIGHT//2))
        screen.blit(font_med.render("Press S to skip visualization", True, COL_TEXT), (WIDTH//2 - 150, HEIGHT//2 + 50))
        pygame.display.flip()
        
        random.seed(CFG["seed"])
        env = GridWorld(LEVELS[str(lvl)])
        returns_q = train_q_learning(env, lvl, False)
        if returns_q is None:
            return "quit"
        results[f"L{lvl} Q-Learn"] = returns_q
        
        # Show loading screen
        screen.fill(COL_BG)
        screen.blit(font_big.render(f"Training SARSA on Level {lvl}...", True, COL_TEXT), (WIDTH//2 - 200, HEIGHT//2))
        screen.blit(font_med.render("Press S to skip visualization", True, COL_TEXT), (WIDTH//2 - 150, HEIGHT//2 + 50))
        pygame.display.flip()
        
        random.seed(CFG["seed"])
        env = GridWorld(LEVELS[str(lvl)])
        returns_s = train_sarsa(env, lvl)
        if returns_s is None:
            return "quit"
        results[f"L{lvl} SARSA"] = returns_s
    
    draw_comparison_chart(results, "Task 4: Monster Levels (4 & 5)")
    
    waiting = True
    while waiting:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return "quit"
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    waiting = False
    
    return "menu"

def run_comparison_task5():
    """Compare with/without intrinsic reward"""
    # Show loading screen
    screen.fill(COL_BG)
    screen.blit(font_big.render("Training WITHOUT intrinsic reward...", True, COL_TEXT), (WIDTH//2 - 250, HEIGHT//2))
    screen.blit(font_med.render("Press S to skip visualization", True, COL_TEXT), (WIDTH//2 - 150, HEIGHT//2 + 50))
    pygame.display.flip()
    
    random.seed(CFG["seed"])
    env = GridWorld(LEVELS["6"])
    returns_without = train_q_learning(env, 6, False)
    if returns_without is None:
        return "quit"
    
    # Show loading screen
    screen.fill(COL_BG)
    screen.blit(font_big.render("Training WITH intrinsic reward...", True, COL_TEXT), (WIDTH//2 - 230, HEIGHT//2))
    screen.blit(font_med.render("Press S to skip visualization", True, COL_TEXT), (WIDTH//2 - 150, HEIGHT//2 + 50))
    pygame.display.flip()
    
    random.seed(CFG["seed"])
    env = GridWorld(LEVELS["6"])
    returns_with = train_q_learning(env, 6, True)
    if returns_with is None:
        return "quit"
    
    draw_comparison_chart({
        "Without Intrinsic": returns_without,
        "With Intrinsic": returns_with
    }, "Task 5: Intrinsic Reward Comparison")
    
    waiting = True
    while waiting:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return "quit"
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    waiting = False
    
    return "menu"


# ==========================================
# STEP 9: Training functions
# ==========================================    

def train_q_learning(env: GridWorld, lvl, is_intrinsic=False):
    qtab = QTable()
    intrinsic = IntrinsicRewardTracker(0.5) if is_intrinsic else None
    visualize, running, skip = True, True, False
    returns = []
    
    for ep in range(EPISODES):
        if not running or skip: break
        
        s = env.reset()
        if intrinsic: intrinsic.reset()
        ep_reward, steps = 0.0, 0
        eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
        
        while running and not skip:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return None
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        return None
                    if e.key == pygame.K_v:
                        visualize = not visualize
                    if e.key == pygame.K_s:
                        skip = True
                        break
                if e.type == FIRE_TIMER:
                    env.fire_state += env.fire_dir
                    if env.fire_state == 2:
                        env.fire_dir = -1
                    elif env.fire_state == 0:
                        env.fire_dir = 1
            if not running or skip: break
            # Q-Learning implementation where the action is only chosen after finished update
            a = epsilon_greedy(qtab, s, eps)    # choose action
            res = env.step(a)
            # Calculate intrinsic reward for Level 6
            total_r = res.reward
            if intrinsic:
                total_r += intrinsic.get(res.next_state)
            q_learning_update(qtab, s, a, total_r, res.next_state, ALPHA, GAMMA)
            s = res.next_state      # assign only state
            ep_reward += res.reward
            steps += 1
            
            if visualize:
                draw(env, ep, steps, eps, ep_reward, "Q-Learning", lvl, is_intrinsic)
                clock.tick(FPS_VISUAL)
            else:
                if steps % 10 == 0:
                    draw(env, ep, steps, eps, ep_reward, "Q-Learning", lvl, is_intrinsic)
                clock.tick(FPS_FAST)
            
            if res.done or steps >= MAX_STEPS:
                break
        
        returns.append(ep_reward)
    
    # Skip to the end of the training
    if skip:
        print(f"Skipping visualization, training remaining episodes in background...")
        for ep in range(ep + 1, EPISODES):
            s = env.reset()
            if intrinsic: intrinsic.reset()
            ep_reward, steps = 0.0, 0
            eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
            
            while steps < MAX_STEPS:
                a = epsilon_greedy(qtab, s, eps)
                res = env.step(a)
                total_r = res.reward
                if intrinsic:
                    total_r += intrinsic.get(res.next_state)
                q_learning_update(qtab, s, a, total_r, res.next_state, ALPHA, GAMMA)
                s = res.next_state
                ep_reward += res.reward
                steps += 1
                
                if res.done:
                    break
            
            returns.append(ep_reward)
            
            # Show progress every 50 episodes
            if (ep + 1) % 50 == 0:
                print(f"Episode {ep + 1}/{EPISODES} complete (background training)")
    
    return returns

def train_sarsa(env: GridWorld, lvl):
    qtab = QTable()
    visualize, running, skip = True, True, False
    returns = []
    
    for ep in range(EPISODES):
        if not running or skip: break
        
        s = env.reset()
        ep_reward, steps = 0.0, 0
        eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
        a = epsilon_greedy(qtab, s, eps)    # Choose action before update
        
        while running and not skip:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return None
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        return None
                    if e.key == pygame.K_v:
                        visualize = not visualize
                    if e.key == pygame.K_s:
                        skip = True
                        break
                if e.type == FIRE_TIMER:
                    env.fire_state += env.fire_dir
                    if env.fire_state == 2:
                        env.fire_dir = -1
                    elif env.fire_state == 0:
                        env.fire_dir = 1
            if not running or skip: break
            
            res = env.step(a)
            # SARSA implementation where the next state and action were chosen before update
            ap = epsilon_greedy(qtab, res.next_state, eps)  # choose next action
            sarsa_update(qtab, s, a, res.reward, res.next_state, ap, ALPHA, GAMMA)
            s, a = res.next_state, ap   # assign action for the next update (before update)
            ep_reward += res.reward
            steps += 1
            
            if visualize:
                draw(env, ep, steps, eps, ep_reward, "SARSA", lvl)
                clock.tick(FPS_VISUAL)
            else:
                if steps % 10 == 0:
                    draw(env, ep, steps, eps, ep_reward, "SARSA", lvl)
                clock.tick(FPS_FAST)
            
            if res.done or steps >= MAX_STEPS:
                break
        
        returns.append(ep_reward)
    
    # Skip to the end of the training
    if skip:
        print(f"Skipping visualization, training remaining episodes in background...")
        for ep in range(ep + 1, EPISODES):
            s = env.reset()
            ep_reward, steps = 0.0, 0
            eps = linear_epsilon(ep, EPS_START, EPS_END, EPS_DECAY_EP)
            a = epsilon_greedy(qtab, s, eps)
            
            while steps < MAX_STEPS:
                res = env.step(a)
                ap = epsilon_greedy(qtab, res.next_state, eps)
                sarsa_update(qtab, s, a, res.reward, res.next_state, ap, ALPHA, GAMMA)
                s, a = res.next_state, ap
                ep_reward += res.reward
                steps += 1
                
                if res.done:
                    break
            
            returns.append(ep_reward)
            
            # Show progress every 50 episodes
            if (ep + 1) % 50 == 0:
                print(f"Episode {ep + 1}/{EPISODES} complete (background training)")
    
    return returns


# ==========================================
# STEP 10: Run main()
# ==========================================    

def main():
    while True:
        # Return "single" or "compare" or "quit"
        mode = show_main_menu()
        
        if mode == "quit":
            break   
        else:
            # Set animation for fires
            pygame.time.set_timer(FIRE_TIMER, int(DURATION / 5))
            
            if mode == "single":
                while True:
                    # Return "quit" or "back" or tuple()
                    choice = show_level_menu()
                    
                    if choice == "quit":
                        pygame.quit()
                        return
                    elif choice == "back":
                        break
                    elif isinstance(choice, tuple):
                        level, alg = choice
                        result = run_single_level(level, alg)
                        
                        while result == "retry":
                            result = run_single_level(level, alg)
                        
                        if result == "quit":
                            pygame.quit()
                            return
                        elif result == "main":
                            break    
            elif mode == "compare":
                while True:
                    choice = show_comparison_menu()
                    
                    if choice == "quit":
                        pygame.quit()
                        return
                    elif choice == "back":
                        break
                    elif choice == "task2":
                        result = run_comparison_task2()
                        if result == "quit":
                            pygame.quit()
                            return
                    elif choice == "task4":
                        result = run_comparison_task4()
                        if result == "quit":
                            pygame.quit()
                            return
                    elif choice == "task5":
                        result = run_comparison_task5()
                        if result == "quit":
                            pygame.quit()
                            return
    
    pygame.quit()

if __name__ == "__main__":
    main()