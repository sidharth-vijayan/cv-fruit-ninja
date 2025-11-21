# game_engine.py
import math
import random
import time
from typing import List, Tuple, Optional, Any
import numpy as np
import pygame
import cv2

from utils import line_intersects_circle, polyline_intersects_circle, save_high_score, load_high_scores

# ---------- Simple game objects ----------
class Fruit:
    def __init__(self, image: pygame.Surface, pos: Tuple[float,float], vel: Tuple[float,float], radius: float, kind: str = "fruit"):
        self.image = image
        self.pos = list(pos)
        self.vel = list(vel)
        self.radius = radius
        self.kind = kind
        self.sliced = False
        self.spawn_time = time.time()

    def update(self, dt: float):
        g = 600.0
        self.vel[1] += g * dt
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt

    def draw(self, surf: pygame.Surface):
        if self.image:
            rect = self.image.get_rect(center=(int(self.pos[0]), int(self.pos[1])))
            surf.blit(self.image, rect)

class FruitHalf:
    def __init__(self, surface: pygame.Surface, pos: Tuple[float,float], vel: Tuple[float,float], ang: float, ang_vel: float, ttl: float = 3.5):
        self.surface = surface
        self.pos = list(pos)
        self.vel = list(vel)
        self.angle = ang
        self.ang_vel = ang_vel
        self.age = 0.0
        self.ttl = ttl

    def update(self, dt: float):
        g = 600.0
        self.age += dt
        self.vel[1] += g * dt
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        self.angle += self.ang_vel * dt

    def alive(self) -> bool:
        return self.age < self.ttl and -2000 < self.pos[0] < 2000 and -2000 < self.pos[1] < 2000

    def draw(self, surf: pygame.Surface):
        if self.surface is None:
            return
        rotated = pygame.transform.rotate(self.surface, math.degrees(self.angle))
        rect = rotated.get_rect(center=(int(self.pos[0]), int(self.pos[1])))
        surf.blit(rotated, rect)

class Particle:
    def __init__(self, pos: Tuple[float,float], vel: Tuple[float,float], color: Tuple[int,int,int]=(255,200,50), ttl: float = 0.6):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.ttl = ttl
        self.age = 0.0

    def update(self, dt: float):
        self.age += dt
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt

    def alive(self) -> bool:
        return self.age < self.ttl

class ScorePopup:
    def __init__(self, text: str, pos: Tuple[float,float], color=(255,255,255), ttl=0.9):
        self.text = text
        self.pos = list(pos)
        self.start_pos = pos
        self.color = color
        self.age = 0.0
        self.ttl = ttl
        self.alpha = 255

    def update(self, dt: float):
        self.age += dt
        # float up and fade
        self.pos[1] -= 40 * dt
        self.alpha = int(255 * max(0.0, 1.0 - (self.age / self.ttl)))

    def alive(self) -> bool:
        return self.age < self.ttl

# ---------- mask slicing helper (same approach as before) ----------
def slice_image_along_line(surface: pygame.Surface, p1: Tuple[float,float], p2: Tuple[float,float]):
    # ensure proper format
    if surface.get_bitsize() not in (32,):
        surface = surface.convert_alpha()

    arr_rgb = pygame.surfarray.array3d(surface).copy()  # (w,h,3)
    arr_rgb = np.transpose(arr_rgb, (1,0,2))  # (h,w,3)
    arr_alpha = pygame.surfarray.array_alpha(surface).copy()
    arr_alpha = np.transpose(arr_alpha, (1,0))  # (h,w)
    h,w = arr_alpha.shape
    ys, xs = np.mgrid[0:h, 0:w]

    x1,y1 = float(p1[0]), float(p1[1])
    x2,y2 = float(p2[0]), float(p2[1])
    vx = x2 - x1
    vy = y2 - y1
    rel_x = xs - x1
    rel_y = ys - y1
    cross = vx * rel_y - vy * rel_x
    nonzero_mask = arr_alpha > 0
    left_mask = (cross > 0) & nonzero_mask
    right_mask = (cross <= 0) & nonzero_mask

    def make_surface_from_mask(mask):
        if not mask.any():
            return None
        out_rgba = np.zeros((h,w,4), dtype=np.uint8)
        out_rgba[..., :3] = arr_rgb
        out_rgba[..., 3] = (arr_alpha * mask).astype(np.uint8)
        out_rgb_t = np.transpose(out_rgba[..., :3], (1,0,2)).copy()
        out_alpha_t = np.transpose(out_rgba[..., 3], (1,0)).copy()
        surf = pygame.Surface((w,h), pygame.SRCALPHA, 32)
        pygame.surfarray.blit_array(surf, out_rgb_t)
        pygame.surfarray.pixels_alpha(surf)[:] = out_alpha_t
        ys_idx, xs_idx = np.where(mask)
        miny, maxy = ys_idx.min(), ys_idx.max()
        minx, maxx = xs_idx.min(), xs_idx.max()
        rect_w = maxx - minx + 1
        rect_h = maxy - miny + 1
        cropped = pygame.Surface((rect_w, rect_h), pygame.SRCALPHA, 32)
        cropped.blit(surf, (0,0), (minx, miny, rect_w, rect_h))
        return cropped, (int(minx), int(miny))

    left_res = make_surface_from_mask(left_mask)
    right_res = make_surface_from_mask(right_mask)
    return left_res, right_res

# ---------- Game engine ----------
class GameEngine:
    def __init__(self, screen: pygame.Surface, assets: dict, fps_display: bool = True, calib: Optional[dict]=None):
        self.screen = screen
        self.w, self.h = screen.get_size()
        self.assets = assets
        self.fruits: List[Fruit] = []
        self.halves: List[FruitHalf] = []
        self.particles: List[Particle] = []
        self.popups: List[ScorePopup] = []
        self.score = 0
        self.lives = 3
        self.combo = 0
        self.last_spawn = 0.0
        self.spawn_interval = 1.0
        self.fps_display = fps_display
        self.font = pygame.font.SysFont("Segoe UI", 20)
        self.font_big = pygame.font.SysFont("Segoe UI", 48)
        self.level = 1
        self.level_timer = 60.0
        self.start_time = time.time()
        self.high_scores = load_high_scores()
        self.level_overlay_timer = 0.0
        self.calib = calib or {"min_speed":800.0,"thickness":14.0,"smooth_alpha":0.35}
        self.record_writer = None  # cv2.VideoWriter if recording
        self.record_path = None
        

    # spawn
    def spawn_fruit(self):
        kind = "fruit" if random.random() >= 0.3 else "bomb"
        if kind == "bomb":
            img = self.assets.get("bomb")
            radius = max(30, img.get_width()//2) if img is not None else 32
        else:
            fruits = self.assets.get("fruits", [])
            if fruits:
                img = random.choice(fruits)
            else:
                surf = pygame.Surface((96,96), pygame.SRCALPHA)
                color = random.choice([(255,0,0),(0,200,0),(255,150,0),(255,0,255)])
                pygame.draw.circle(surf,color,(48,48),44)
                img = surf
            radius = max(20, img.get_width()//2)
        x = random.randint(80, self.w-80)
        y = self.h + 60
        vx = random.uniform(-120,120)
        vy = random.uniform(-900, -700)
        f = Fruit(img, (x,y), (vx,vy), radius, kind)
        self.fruits.append(f)

    # sound helper: dynamic pitch change via resampling numpy array (small shifts)
    def play_pitch_pan(self, base_sound: pygame.mixer.Sound, pan_x: float, pitch_scale: float):
        # base_sound is a pygame Sound; we'll try to get its samples, resample and play
        try:
            arr = pygame.sndarray.array(base_sound)  # (N,channels)
            if arr is None or len(arr) < 2:
                base_sound.play()
                return
            # resample by simple linear interpolation
            length = arr.shape[0]
            new_length = max(16, int(length / pitch_scale))
            idx = np.linspace(0, length-1, new_length)
            idx0 = np.floor(idx).astype(int)
            idx1 = np.clip(idx0+1, 0, length-1)
            frac = idx - idx0
            # if stereo, arr shape (N,2)
            if arr.ndim == 2 and arr.shape[1] == 2:
                left = (1-frac)[:,None]*arr[idx0,0:1] + frac[:,None]*arr[idx1,0:1]
                right = (1-frac)[:,None]*arr[idx0,1:2] + frac[:,None]*arr[idx1,1:2]
                res = np.concatenate([left, right], axis=1).astype(arr.dtype)
            else:
                mono = arr if arr.ndim ==1 else arr[:,0]
                mono = mono.astype(np.float32)
                new = (1-frac)*mono[idx0] + frac*mono[idx1]
                res = np.column_stack([new, new]).astype(arr.dtype)
            snd = pygame.sndarray.make_sound(res.copy())
            # pan_x in 0..w -> left/right
            pan = max(0.0, min(1.0, pan_x / float(self.w)))
            left_v = 1.0 - pan
            right_v = pan
            ch = pygame.mixer.find_channel(force=True)
            ch.set_volume(left_v, right_v)
            ch.play(snd)
        except Exception:
            try:
                # fallback simple pan
                ch = pygame.mixer.find_channel(force=True)
                pan = max(0.0, min(1.0, pan_x / float(self.w)))
                ch.set_volume(1.0-pan, pan)
                ch.play(base_sound)
            except Exception:
                base_sound.play()

    # create halves (masked)
    def _create_halves_from_image(self, image: pygame.Surface, center_pos: Tuple[float,float], seg: Tuple[Tuple[float,float],Tuple[float,float]]) -> List[FruitHalf]:
        w,h = image.get_size()
        cx,cy = center_pos
        top_left_x = cx - w/2.0
        top_left_y = cy - h/2.0
        (x1_s,y1_s),(x2_s,y2_s) = seg
        p1_img = (x1_s - top_left_x, y1_s - top_left_y)
        p2_img = (x2_s - top_left_x, y2_s - top_left_y)
        # clamp
        def clamp_pt(p):
            x,y = p
            x = max(-w, min(w*2, x))
            y = max(-h, min(h*2, y))
            return (x,y)
        p1_img = clamp_pt(p1_img)
        p2_img = clamp_pt(p2_img)
        # ----------------------------
        # FAST PATH: use prewarmed vertical halves if available
        # ----------------------------
        try:
            # check if this fruit image is one of the preloaded fruits
            fruit_list = self.assets.get("fruits", [])
            pre = self.assets.get("prewarmed_halves", [])
            idx = fruit_list.index(image)
            left_surf, right_surf = pre[idx]

            # compute approximate world positions so halves appear centered correctly
            w = image.get_width()
            h = image.get_height()

            # image top-left in world coords
            cx, cy = center_pos
            top_left_x = cx - w/2
            top_left_y = cy - h/2

            # left half center in world coords
            lw = left_surf.get_width()
            lh = left_surf.get_height()
            l_cx = top_left_x + lw/2
            l_cy = top_left_y + lh/2

            # right half center in world coords
            rw = right_surf.get_width()
            rh = right_surf.get_height()
            r_cx = top_left_x + lw + rw/2
            r_cy = top_left_y + rh/2

            # give them random velocities similar to mask version
            dx = p2_img[0] - p1_img[0]
            dy = p2_img[1] - p1_img[1]
            norm = math.hypot(dx, dy) + 1e-6
            nx, ny = dx/norm, dy/norm
            perp_x, perp_y = -ny, nx

            speed_base = 200 + random.uniform(-80, 80)

            halves = []

            lv = (-nx*speed_base + perp_x*random.uniform(80,260),
                -ny*speed_base + perp_y*random.uniform(40,160))
            rv = (-nx*speed_base - perp_x*random.uniform(80,260),
                -ny*speed_base - perp_y*random.uniform(40,160))

            # assign angles
            la = random.uniform(-0.5, 0.5)
            ra = random.uniform(-0.5, 0.5)
            lav = random.uniform(-6, -2)
            rav = random.uniform(2, 6)

            halves.append(FruitHalf(left_surf,  (l_cx, l_cy), lv, la, lav))
            halves.append(FruitHalf(right_surf, (r_cx, r_cy), rv, ra, rav))

            return halves
        except Exception:
            pass
        left_res, right_res = slice_image_along_line(image, p1_img, p2_img)
        halves: List[FruitHalf] = []
        dx = x2_s - x1_s
        dy = y2_s - y1_s
        norm = math.hypot(dx,dy) + 1e-6
        nx, ny = dx/norm, dy/norm
        perp_x, perp_y = -ny, nx
        speed_base = 200 + random.uniform(-80,80)
        if left_res is not None:
            left_surf, (offx,offy) = left_res
            half_cx = top_left_x + offx + left_surf.get_width()/2.0
            half_cy = top_left_y + offy + left_surf.get_height()/2.0
            left_vel = (-nx*speed_base + perp_x*random.uniform(80,260), -ny*speed_base + perp_y*random.uniform(40,160))
            left_ang = random.uniform(-0.5,0.5)
            left_ang_vel = random.uniform(-6,-2)
            halves.append(FruitHalf(left_surf, (half_cx,half_cy), left_vel, left_ang, left_ang_vel, ttl=3.5))
        if right_res is not None:
            right_surf, (offx,offy) = right_res
            half_cx = top_left_x + offx + right_surf.get_width()/2.0
            half_cy = top_left_y + offy + right_surf.get_height()/2.0
            right_vel = (-nx*speed_base - perp_x*random.uniform(80,260), -ny*speed_base - perp_y*random.uniform(40,160))
            right_ang = random.uniform(-0.5,0.5)
            right_ang_vel = random.uniform(2,6)
            halves.append(FruitHalf(right_surf, (half_cx,half_cy), right_vel, right_ang, right_ang_vel, ttl=3.5))
            return halves

    def slice_fruit(self, fruit: Fruit, seg: Tuple[Tuple[float,float],Tuple[float,float]], swipe_speed: Optional[float]=None):
        if fruit.sliced:
            return
        fruit.sliced = True
        # audio: pan + pitch dependent on swipe_speed
        try:
            sounds = self.assets.get("sounds", {})
            base = sounds.get("slice") if fruit.kind!="bomb" else sounds.get("bomb")
            if base:
                pitch = 1.0
                if swipe_speed is not None:
                    # map speed to pitch factor gently
                    pitch = max(0.7, min(1.6, 1.0 + (swipe_speed - 800.0)/2000.0))
                # play modified
                self.play_pitch_pan(base, fruit.pos[0], pitch)
        except Exception:
            pass

        if fruit.kind == "bomb":
            self.lives -= 1
            self.combo = 0
            # fewer bomb particles
        for _ in range(14):  # instead of 26
            angle = random.uniform(0, math.pi*2)
            speed = random.uniform(120,500)
            vx = math.cos(angle)*speed
            vy = math.sin(angle)*speed
            self.particles.append(Particle(tuple(fruit.pos),(vx,vy),ttl=1.0))
        else:
            self.combo += 1
            pts = 10 * min(5, self.combo)
            self.score += pts
            self.popups.append(ScorePopup(f"+{pts}", (fruit.pos[0], fruit.pos[1]), color=(200,255,180), ttl=0.9))
            halves = self._create_halves_from_image(fruit.image, (fruit.pos[0], fruit.pos[1]), seg)
            for h in halves:
                self.halves.append(h)
            # fewer fruit particles
        for _ in range(6):  # instead of 10
            angle = random.uniform(0, math.pi*2)
            speed = random.uniform(80,260)
            vx = math.cos(angle)*speed
            vy = math.sin(angle)*speed
            self.particles.append(Particle(tuple(fruit.pos),(vx,vy),ttl=0.6))
            # combo flame popup maybe
            if self.combo >= 3:
                self.popups.append(ScorePopup(f"x{self.combo}!", (fruit.pos[0], fruit.pos[1]-20), color=(255,180,60), ttl=0.9))
        try:
            self.fruits.remove(fruit)
        except ValueError:
            pass

    # recording helpers (video only via OpenCV VideoWriter)
    def start_recording(self, path: str = None, fps: int = 30):
        if self.record_writer is not None:
            return
        if path is None:
            path = f"demo_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.record_path = path
        self.record_writer = cv2.VideoWriter(path, fourcc, fps, (self.w, self.h))
        print("Recording started:", path)

    def stop_recording(self):
        if self.record_writer is not None:
            self.record_writer.release()
            print("Recording saved:", self.record_path)
            self.record_writer = None
            self.record_path = None

    def write_frame(self, surf: pygame.Surface):
        if self.record_writer is None:
            return
        # grab pixels from surface
        arr = pygame.surfarray.array3d(surf)  # (w,h,3)
        arr = np.transpose(arr, (1,0,2))  # (h,w,3)
        # convert RGB to BGR for OpenCV
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        self.record_writer.write(arr)

    # main update loop
    def update(self, dt: float, swipe_segments: Optional[List[Tuple[Tuple[float,float],Tuple[float,float]]]]=None, swipe_polyline: Optional[List[Tuple[float,float]]]=None, swipe_speed: Optional[float]=None):
        if swipe_segments is None:
            swipe_segments=[]
        if swipe_polyline is None:
            swipe_polyline=[]
        now = time.time()
        if now - self.last_spawn > max(0.35, self.spawn_interval - (self.level-1)*0.08):
            self.spawn_fruit()
            self.last_spawn = now

        for f in list(self.fruits):
            f.update(dt)
            if f.pos[1] - f.radius > self.h + 80 and not f.sliced:
                if f.kind != "bomb":
                    self.lives -= 1
                    self.combo = 0
                try:
                    self.fruits.remove(f)
                except ValueError:
                    pass
                continue

            # robust polyline check
            if swipe_polyline and polyline_intersects_circle(swipe_polyline, (f.pos[0], f.pos[1]), f.radius, thickness=self.calib.get("thickness",14.0)):
                # pass endpoints as slice line, and provide swipe_speed for doppler
                try:
                    self.slice_fruit(f, (swipe_polyline[0], swipe_polyline[-1]), swipe_speed)
                except Exception:
                    self.slice_fruit(f, ((f.pos[0]-20,f.pos[1]), (f.pos[0]+20,f.pos[1])), swipe_speed)
                continue

            for seg in swipe_segments:
                p1,p2 = seg
                if not f.sliced and line_intersects_circle(p1,p2,(f.pos[0],f.pos[1]), f.radius + self.calib.get("thickness",14.0)):
                    self.slice_fruit(f, seg, swipe_speed)
                    break

        for h in list(self.halves):
            h.update(dt)
            if not h.alive():
                try:
                    self.halves.remove(h)
                except ValueError:
                    pass

        for p in list(self.particles):
            p.update(dt)
            if not p.alive():
                try:
                    self.particles.remove(p)
                except ValueError:
                    pass

        for pop in list(self.popups):
            pop.update(dt)
            if not pop.alive():
                try:
                    self.popups.remove(pop)
                except ValueError:
                    pass

        elapsed = now - self.start_time
        if elapsed > self.level_timer:
            self.level += 1
            self.start_time = now
            self.spawn_interval = max(0.6, self.spawn_interval * 0.95)
            self.level_overlay_timer = 2.0

    # drawing & UI
    def draw(self, surf: pygame.Surface, fps: Optional[float]=None, detection_confidence: float=0.0, recording: bool=False):
        surf.fill((30,30,40))

        # draw fruits
        for f in self.fruits:
            f.draw(surf)

        # draw halves
        for h in self.halves:
            h.draw(surf)

        # draw particles
        for p in self.particles:
            alpha = int(255 * max(0, 1 - p.age / p.ttl))
            pygame.draw.circle(surf, p.color, (int(p.pos[0]), int(p.pos[1])), 3)

        # draw score popups
        for pop in self.popups:
            txt = self.font.render(pop.text, True, pop.color)
            img = txt.copy()
            img.set_alpha(pop.alpha)
            surf.blit(img, (int(pop.pos[0]) - txt.get_width()//2,
                            int(pop.pos[1]) - txt.get_height()//2))

        # -------- HUD SECTION (score, level bar, combo, overlays) --------

        # score (shadow + text)
        txt = self.font.render(f"Score: {self.score}", True, (255,255,255))
        shadow = self.font.render(f"Score: {self.score}", True, (0,0,0))
        surf.blit(shadow, (12,12))
        surf.blit(txt, (10,10))
        
        # lives
        life_txt = self.font.render(f"Lives: {self.lives}", True, (255,200,200))
        surf.blit(life_txt, (10, 40))

        # combos (small HUD indicator, not the big popup)
        combo_txt = self.font.render(f"Combo: {self.combo}", True, (200,200,255))
        surf.blit(combo_txt, (10, 70))

        # FPS (optional)
        if fps is not None:
            fps_txt = self.font.render(f"{int(fps)} FPS", True, (180,180,180))
            surf.blit(fps_txt, (self.w - 90, 10))

        # Detection confidence (top right)
        conf_txt = self.font.render(f"Conf: {detection_confidence:.2f}", True, (180,255,180))
        surf.blit(conf_txt, (self.w - 130, 40))

        # level progress bar
        bar_w = 300
        frac = min(1.0, (time.time() - self.start_time) / self.level_timer)
        pygame.draw.rect(surf, (60,60,60),
                        (self.w//2 - bar_w//2, 8, bar_w, 10))
        pygame.draw.rect(surf, (80,200,120),
                        (self.w//2 - bar_w//2, 8, int(bar_w * frac), 10))

        # combo popup
        if self.combo >= 3:
            combo_txt = self.font_big.render(f"COMBO x{self.combo}", True, (255,180,60))
            surf.blit(combo_txt,
                    (self.w//2 - combo_txt.get_width()//2, 60))

        # combo flame icon (optional, uses fixed HUD Y)
        hud_y = 10
        if self.combo >= 3:
            flame_pos = (320, hud_y + 18)
            pygame.draw.circle(surf, (255,140,0), flame_pos, 12)
            surf.blit(self.font.render(f"x{self.combo}", True, (20,20,20)),
                    (flame_pos[0]-10, flame_pos[1]-10))

        # level transition overlay
        if self.level_overlay_timer > 0:
            alpha = int(255 * (self.level_overlay_timer / 2.0))
            overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            overlay.fill((10,10,10, max(0, min(200, alpha))))
            surf.blit(overlay, (0,0))

            txt = self.font_big.render(f"LEVEL {self.level}", True, (255,255,255))
            surf.blit(txt, (self.w//2 - txt.get_width()//2,
                            self.h//2 - txt.get_height()//2))

            self.level_overlay_timer = max(0.0, self.level_overlay_timer - 1.0/60.0)

        # recording indicator & caption overlay
        if recording:
            cap = self.font.render("REC ● Manual live demo — no automation",
                                True, (255,80,80))
            surf.blit(cap, (10, self.h - 30))


    # endgame
    def game_over(self) -> bool:
        return self.lives <= 0

    def finalize(self, player_name: str = "Player"):
        save_high_score(player_name, self.score)
