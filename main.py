# main.py
import os
import sys
import time
import math
import json
import cv2
import numpy as np
import pygame

from tracking import HandTracker
from game_engine import GameEngine
from utils import load_high_scores

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

ASSETS_DIR = "assets"
CALIB_FILE = "calib.json"


# ---------- helpers ----------
def synth_sound(freq=880, duration_s=0.08, volume=0.5, sample_rate=44100):
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    envelope = np.exp(-12 * t)
    samples = (wave * envelope * volume * (2**15 - 1)).astype(np.int16)
    stereo = np.column_stack([samples, samples])
    try:
        snd = pygame.sndarray.make_sound(stereo)
        return snd
    except Exception:
        return None


def load_assets():
    assets = {"fruits": [], "bomb": None, "sounds": {}}
    fruits_dir = os.path.join(ASSETS_DIR, "fruits")

    # fruits
    if os.path.isdir(fruits_dir):
        for fname in sorted(os.listdir(fruits_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(fruits_dir, fname)
                try:
                    img = pygame.image.load(path).convert_alpha()
                    assets["fruits"].append(
                        pygame.transform.smoothscale(img, (64, 64))
                    )
                except Exception:
                    pass

    # bomb
    bomb_path = os.path.join(ASSETS_DIR, "bomb.png")
    if os.path.exists(bomb_path):
        try:
            img = pygame.image.load(bomb_path).convert_alpha()
            assets["bomb"] = pygame.transform.smoothscale(img, (64, 64))
        except Exception:
            assets["bomb"] = None

    # sounds
    sounds_dir = os.path.join(ASSETS_DIR, "sounds")
    if os.path.isdir(sounds_dir):
        for fname in os.listdir(sounds_dir):
            if fname.lower().endswith((".wav", ".ogg", ".mp3")):
                key = os.path.splitext(fname)[0]
                try:
                    assets["sounds"][key] = pygame.mixer.Sound(
                        os.path.join(sounds_dir, fname)
                    )
                except Exception:
                    pass

    # fallback sounds
    if "slice" not in assets["sounds"]:
        s = synth_sound(freq=1200, duration_s=0.05, volume=0.6)
        if s:
            assets["sounds"]["slice"] = s

    if "bomb" not in assets["sounds"]:
        s = synth_sound(freq=200, duration_s=0.3, volume=0.8)
        if s:
            assets["sounds"]["bomb"] = s

    return assets


def prewarm_vertical_halves(assets):
    halves = []
    fruits = assets.get("fruits", [])
    for img in fruits:
        w, h = img.get_size()
        left = pygame.Surface((w//2, h), pygame.SRCALPHA, 32)
        right = pygame.Surface((w - w//2, h), pygame.SRCALPHA, 32)
        left.blit(img, (0, 0), (0, 0, w//2, h))
        right.blit(img, (0, 0), (w//2, 0, w - w//2, h))
        halves.append((left.copy(), right.copy()))
    assets["prewarmed_halves"] = halves


assets = load_assets()
prewarm_vertical_halves(assets)


def load_calib():
    default = {"min_speed": 900.0, "thickness": 14.0, "smooth_alpha": 0.35}
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k in default:
                if k not in cfg:
                    cfg[k] = default[k]
            return cfg
        except Exception:
            return default.copy()
    return default.copy()


def save_calib(cfg):
    try:
        with open(CALIB_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def load_knife_sprite():
    path = os.path.join(ASSETS_DIR, "knife.png")
    if os.path.exists(path):
        try:
            surf = pygame.image.load(path).convert_alpha()
            return pygame.transform.smoothscale(surf, (160, 72))
        except Exception:
            pass

    # fallback
    surf = pygame.Surface((160, 72), pygame.SRCALPHA)
    pygame.draw.polygon(surf, (220, 220, 220),
                        [(12, 36), (112, 8), (146, 36), (112, 64)])
    pygame.draw.rect(surf, (50, 50, 50), (0, 32, 24, 8))
    pygame.draw.circle(surf, (80, 80, 80), (14, 36), 8)
    return surf


# ---------- Slider Widget ----------
class Slider:
    def __init__(self, x, y, w, label, min_v, max_v, step, value):
        self.rect = pygame.Rect(x, y, w, 24)
        self.label = label
        self.min = min_v
        self.max = max_v
        self.step = step
        self.value = value
        self.drag = False

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self.drag = True
                self.set_from_pos(ev.pos[0])

        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.drag = False

        elif ev.type == pygame.MOUSEMOTION and self.drag:
            self.set_from_pos(ev.pos[0])

    def set_from_pos(self, px):
        x = px - self.rect.x
        frac = max(0.0, min(1.0, x / float(self.rect.w)))
        val = self.min + frac * (self.max - self.min)
        if self.step:
            val = round(val / self.step) * self.step
        self.value = val

    def draw(self, surf, font):
        pygame.draw.rect(surf, (60, 60, 60), self.rect)
        fill_w = int(
            (self.value - self.min) / (self.max - self.min) * self.rect.w
        )
        pygame.draw.rect(
            surf, (140, 200, 255),
            (self.rect.x, self.rect.y, fill_w, self.rect.h)
        )
        txt = font.render(f"{self.label}: {self.value:.2f}", True,
                          (220, 220, 220))
        surf.blit(txt, (self.rect.x, self.rect.y - 22))


# ---------- MAIN ----------
def main():
    pygame.init()
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    except Exception:
        pygame.mixer.init()

    win_w, win_h = 1280, 720
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        "CV Fruit Ninja - Manual Demo (with Calibration + Recording)"
    )
    clock = pygame.time.Clock()

    assets = load_assets()
    calib = load_calib()
    engine = GameEngine(screen, assets, calib=calib)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open webcam. Exiting.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    tracker = HandTracker(
        max_len=calib.get("dlen", 48),
        detection_conf=0.6,
        tracking_conf=0.5,
        smooth_alpha=calib.get("smooth_alpha", 0.35)
    )

    knife = load_knife_sprite()

    show_cam_panel = True
    draw_hand_overlay = True
    recording = False

    font_small = pygame.font.SysFont("Arial", 16)
    font_med = pygame.font.SysFont("Arial", 20)
    font_big = pygame.font.SysFont("Arial", 36)

    # states
    state = "start"
    slider_speed = Slider(80, 220, 420, "Min swipe speed",
                          200.0, 2200.0, 10.0, calib.get("min_speed", 900.0))
    slider_thickness = Slider(
        80, 300, 420, "Collision thickness",
        2.0, 40.0, 1.0, calib.get("thickness", 14.0)
    )
    slider_alpha = Slider(80, 380, 420, "Smoothing alpha",
                          0.05, 0.9, 0.01, calib.get("smooth_alpha", 0.35))
    sliders = [slider_speed, slider_thickness, slider_alpha]

    btn_play = pygame.Rect(80, 460, 160, 44)
    btn_test = pygame.Rect(260, 460, 160, 44)
    btn_save = pygame.Rect(80, 520, 160, 36)
    btn_back = pygame.Rect(260, 520, 160, 36)

    player_name = "Player"

    # MAIN LOOP ----------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]

        events = pygame.event.get()
        for ev in events:
            if ev.type == pygame.QUIT:
                engine.finalize(player_name)
                if recording:
                    engine.stop_recording()
                cap.release()
                tracker.close()
                pygame.quit()
                return

        # ----------------------- START SCREEN -----------------------
        if state == "start":
            for ev in events:
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_RETURN:
                    state = "calib"

                if ev.type == pygame.MOUSEBUTTONDOWN:
                    if btn_play.collidepoint(ev.pos):
                        state = "play"
                    if btn_test.collidepoint(ev.pos):
                        state = "calib"

            screen.fill((18, 18, 22))
            title = font_big.render(
                "CV Fruit Ninja — Manual Demo", True, (255, 255, 255)
            )
            screen.blit(title, (80, 80))

            instr = font_med.render(
                "Press Enter or click 'Calibrate' to configure slicing sensitivity",
                True, (200, 200, 200))
            screen.blit(instr, (80, 150))

            pygame.draw.rect(screen, (60, 140, 60), btn_play)
            screen.blit(font_med.render(
                "Play", True, (255, 255, 255)), (btn_play.x+40, btn_play.y+10))

            pygame.draw.rect(screen, (80, 80, 200), btn_test)
            screen.blit(font_med.render(
                "Calibrate", True, (255, 255, 255)), (btn_test.x+20, btn_test.y+10))

            pygame.display.flip()
            clock.tick(30)
            continue

        # ----------------------- CALIBRATION SCREEN -----------------------
        if state == "calib":
            for ev in events:
                for s in sliders:
                    s.handle_event(ev)

                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    mx, my = ev.pos

                    if btn_play.collidepoint((mx, my)):
                        calib["min_speed"] = slider_speed.value
                        calib["thickness"] = slider_thickness.value
                        calib["smooth_alpha"] = slider_alpha.value
                        save_calib(calib)
                        tracker.smooth_alpha = calib["smooth_alpha"]
                        engine.calib = calib.copy()
                        state = "play"

                    if btn_test.collidepoint((mx, my)):
                        engine.spawn_fruit()

                    if btn_save.collidepoint((mx, my)):
                        calib["min_speed"] = slider_speed.value
                        calib["thickness"] = slider_thickness.value
                        calib["smooth_alpha"] = slider_alpha.value
                        save_calib(calib)
                        engine.calib = calib.copy()

                    if btn_back.collidepoint((mx, my)):
                        state = "start"

                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    state = "start"

            screen.fill((25, 25, 30))
            title = font_big.render(
                "Calibration & Test", True, (255, 255, 255))
            screen.blit(title, (60, 60))

            help_text = font_med.render(
                "Adjust values. Click 'Play' when ready.",
                True, (200, 200, 200))
            screen.blit(help_text, (60, 120))

            for s in sliders:
                s.draw(screen, font_small)

            pygame.draw.rect(screen, (60, 200, 120), btn_play)
            screen.blit(font_med.render("PLAY", True, (0, 0, 0)),
                        (btn_play.x+40, btn_play.y+8))

            pygame.draw.rect(screen, (180, 180, 80), btn_test)
            screen.blit(font_med.render("SPAWN TEST FRUIT", True, (0, 0, 0)),
                        (btn_test.x+2, btn_test.y+8))

            pygame.draw.rect(screen, (120, 160, 220), btn_save)
            screen.blit(font_small.render("Save", True, (0, 0, 0)),
                        (btn_save.x+10, btn_save.y+6))

            pygame.draw.rect(screen, (200, 120, 120), btn_back)
            screen.blit(font_small.render("Back", True, (0, 0, 0)),
                        (btn_back.x+10, btn_back.y+6))

            diag = font_small.render(
                f"min_speed={slider_speed.value:.1f}  thickness={slider_thickness.value:.1f}  alpha={slider_alpha.value:.2f}",
                True, (220, 220, 220)
            )
            screen.blit(diag, (60, 440))

            pygame.display.flip()
            clock.tick(30)
            continue

        # ----------------------- PLAY MODE -----------------------

        # defaults (avoid undefined variable bugs)
        swipe_segments = []
        scaled_segments = []
        scaled_poly = []
        swipe_speed = 0.0
        
        # process tracking
        track = tracker.process(frame)

        # HAND OVERLAY --------------------------------------------------
        if draw_hand_overlay and track.get("hand_present", False) and track.get("landmarks") is not None:

            rgb_draw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                mp_drawing.draw_landmarks(
                    rgb_draw,
                    track["landmarks"],
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
            except Exception:
                pass

            # fingertip trail
            now_ts = time.time()
            pts = list(tracker.recent)
            for (x, y, ts) in pts:
                age = now_ts - ts
                if age > 0.6:
                    continue
                frac = 1 - age / 0.6
                r = int(7 * frac)
                color = (int(200 * frac), int(200 * frac), int(255 * frac))
                cv2.circle(rgb_draw, (int(x), int(y)), r, color, -1)

            # bounding box
            lm = track["landmarks"].landmark
            xs = [int(p.x * w_frame) for p in lm]
            ys = [int(p.y * h_frame) for p in lm]
            minx, maxx = max(0, min(xs)), min(w_frame - 1, max(xs))
            miny, maxy = max(0, min(ys)), min(h_frame - 1, max(ys))

            overlay = rgb_draw.copy()
            cv2.rectangle(overlay, (minx, miny), (maxx, maxy),
                          (10, 120, 220), -1)
            cv2.addWeighted(overlay, 0.25, rgb_draw, 0.75, 0, rgb_draw)
            cv2.rectangle(rgb_draw, (minx, miny),
                          (maxx, maxy), (255, 255, 255), 1)

            conf = track.get("detection_confidence", 0.0)
            cv2.putText(rgb_draw, f"conf:{conf:.2f}", (minx, maxy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

            # dynamic skeleton
            if len(pts) >= 2:
                x1, y1, t1 = pts[-2]
                x2, y2, t2 = pts[-1]
                dt = max(1e-6, t2 - t1)
                swipe_speed = math.hypot(x2 - x1, y2 - y1) / dt

            thick = 1 + int(min(6, swipe_speed / 300))

            if conf < 0.5:
                t = conf / 0.5
                col = (int(255 * t), int(255 * t), int(255 * (1 - t)))
            else:
                t = (conf - 0.5) / 0.5
                col = (255, int(255 * (1 - t)), 0)

            for conn in mp.solutions.hands.HAND_CONNECTIONS:
                a = conn[0]; b = conn[1]
                xA = int(lm[a].x * w_frame); yA = int(lm[a].y * h_frame)
                xB = int(lm[b].x * w_frame); yB = int(lm[b].y * h_frame)

                # compute segment length safely
                seg_len = max(1.0, math.hypot(xB - xA, yB - yA))

                # safe thickness
                value = thick * (seg_len / 40.0)
                seg_thick = int(value)
                if seg_thick < 1:
                    seg_thick = 1
                elif seg_thick > 12:
                    seg_thick = 12

                # draw line
                cv2.line(rgb_draw, (xA, yA), (xB, yB), col, seg_thick, lineType=cv2.LINE_AA)


            for p in lm:
                x, y = int(p.x * w_frame), int(p.y * h_frame)
                r = 3 + int(conf * 3)
                cv2.circle(rgb_draw, (x, y), r, col, -1)

            frame = cv2.cvtColor(rgb_draw, cv2.COLOR_RGB2BGR)

            # get slices
            swipe_segments = tracker.get_swipe_segments(
                min_speed_px_s=calib.get("min_speed", 900.0),
                min_length_px=20.0
            )
            recent_poly = tracker.get_recent_polyline()

            scale_x = win_w / float(w_frame)
            scale_y = win_h / float(h_frame)

            scaled_segments = [
                ((p1[0] * scale_x, p1[1] * scale_y),
                 (p2[0] * scale_x, p2[1] * scale_y))
                for (p1, p2) in swipe_segments
            ]
            scaled_poly = [(x * scale_x, y * scale_y)
                           for (x, y) in recent_poly]

        # ENGINE UPDATE ---------------------------------------------------------
        dt = clock.tick(60) / 1000.0
        engine.update(dt, swipe_segments=scaled_segments,
                      swipe_polyline=scaled_poly, swipe_speed=swipe_speed)

        # camera panel frame
        cam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_rgb = np.rot90(cam_rgb)
        cam_surf = pygame.surfarray.make_surface(cam_rgb)
        cam_surf = pygame.transform.scale(
            cam_surf,
            (int(w_frame * (win_w / w_frame)),
             int(h_frame * (win_h / h_frame)))
        )

        # draw engine scene
        engine.draw(screen, fps=clock.get_fps(),
                    detection_confidence=track.get("detection_confidence", 0.0),
                    recording=recording)

        # draw swipe trail
        now_ts = time.time()
        trail_points = [(x * (win_w / w_frame),
                         y * (win_h / h_frame), ts)
                        for (x, y, ts) in tracker.recent
                        if now_ts - ts <= 0.25]

        if len(trail_points) >= 2:
            for i in range(1, len(trail_points)):
                x1, y1, t1 = trail_points[i-1]
                x2, y2, t2 = trail_points[i]
                age = now_ts - t2
                alpha = int(255 * max(0, 1 - age / 0.25))

                seg_surf = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
                pygame.draw.line(seg_surf, (255, 255, 255, alpha),
                                 (x1, y1), (x2, y2), 6)
                screen.blit(seg_surf, (0, 0))

        # knife sprite
        if track.get("hand_present") and track.get("index_tip") is not None:
            ix, iy = track["index_tip"]
            ix_s = ix * (win_w / w_frame)
            iy_s = iy * (win_h / h_frame)

            pts = list(tracker.recent)
            angle_deg = 0
            if len(pts) >= 2:
                x1, y1, _ = pts[-2]
                x2, y2, _ = pts[-1]
                dx = (x2 - x1)
                dy = (y2 - y1)
                if abs(dx) + abs(dy) > 1e-6:
                    angle_deg = -math.degrees(math.atan2(dy, dx))

            rot = pygame.transform.rotate(knife, angle_deg)
            rect = rot.get_rect(center=(int(ix_s + 24), int(iy_s)))
            screen.blit(rot, rect.topleft)

        # camera side panel
        if show_cam_panel:
            panel_w, panel_h = 360, 270
            panel_x = win_w - panel_w - 16
            panel_y = 12

            cam_panel = pygame.transform.smoothscale(
                cam_surf, (panel_w, panel_h))
            bg = pygame.Surface((panel_w + 12, panel_h + 48), pygame.SRCALPHA)
            bg.fill((8, 8, 8, 220))

            screen.blit(bg, (panel_x - 6, panel_y - 6))
            screen.blit(cam_panel, (panel_x, panel_y))

            pygame.draw.rect(
                screen, (200, 200, 200),
                (panel_x, panel_y, panel_w, panel_h), 2
            )
            lbl = font_med.render(
                "LIVE CAM — Manual Demo",
                True, (255, 80, 80)
            )
            screen.blit(lbl, (panel_x + 8, panel_y + panel_h + 8))
            pygame.draw.circle(
                screen, (255, 0, 0),
                (panel_x + panel_w - 16, panel_y + 16), 7
            )

        hint = font_small.render(
            "Press R to restart | N start/stop recording | C toggle cam | M toggle overlay | Esc quit",
            True, (200, 200, 200)
        )
        screen.blit(hint, (10, win_h - 22))

        pygame.display.flip()

        # RECORDING -----------------------------------------------------
        if recording and engine.record_writer is not None:
            engine.write_frame(screen)

        # KEY HANDLING --------------------------------------------------
        for ev in events:
            if ev.type == pygame.KEYDOWN:

                if ev.key == pygame.K_ESCAPE:
                    engine.finalize(player_name)
                    cap.release()
                    tracker.close()
                    pygame.quit()
                    return

                if ev.key == pygame.K_c:
                    show_cam_panel = not show_cam_panel

                if ev.key == pygame.K_m:
                    draw_hand_overlay = not draw_hand_overlay

                if ev.key == pygame.K_n:   # recording toggle
                    if not recording:
                        engine.start_recording(
                            path=f"demo_{int(time.time())}.mp4", fps=30)
                        recording = True
                    else:
                        engine.stop_recording()
                        recording = False

                if ev.key == pygame.K_r:   # restart
                    engine = GameEngine(screen, assets, calib=calib)
                    if recording:
                        engine.stop_recording()
                        recording = False

        # GAME OVER ----------------------------------------------------
        if engine.game_over():
            engine.draw(screen)
            over_text = font_big.render(
                "GAME OVER", True, (255, 255, 255))
            screen.blit(over_text, (
                win_w // 2 - over_text.get_width() // 2,
                win_h // 2 - over_text.get_height() // 2
            ))
            pygame.display.flip()

            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        engine.finalize(player_name)
                        if recording:
                            engine.stop_recording()
                        cap.release()
                        tracker.close()
                        pygame.quit()
                        return

                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_RETURN:
                            engine.finalize(player_name)
                            waiting = False

                        elif ev.key == pygame.K_r:
                            engine = GameEngine(screen, assets, calib=calib)
                            waiting = False

                time.sleep(0.05)

    # end loop
    cap.release()
    tracker.close()
    pygame.quit()


if __name__ == "__main__":
    main()
