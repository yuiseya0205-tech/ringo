from ultralytics import YOLO
import cv2
import numpy as np
import random
import math
import time

import json
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

bgm_process = None
diet = False
pose = YOLO("yolov8n-pose.pt")

person_colors = {}
person_last_pos = {}
pose_hold = {}
team_score = 0
best_scores = []
diet_scores = []
play_minutes = 1
team_name = "Player"
input_mode = False
last_apple_time = 0
diet_mode = False
seated_mode = False
cutting_mode = False
sword_len = 140

FONT_CANDIDATES = [
    "C:/Windows/Fonts/meiryo.ttc",
]

_font_cache = {}

def get_font(size):
    if size in _font_cache:
        return _font_cache[size]
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                f = ImageFont.truetype(path, size)
                _font_cache[size] = f
                return f
            except Exception:
                continue
    f = ImageFont.load_default()
    _font_cache[size] = f
    return f

lang = "ja"
lang_dropdown_open = False
mouse_click = None
lang_options = [("ja", "日本語"), ("en", "English")]

LANG = {
    "team_label":   {"ja": "プレイヤー名:", "en": "PlayerName:"},
    "name_label":   {"ja": "プレイヤー名:", "en": "PlayerName:"},
    "diet_mode":    {"ja": "ダイエットモード:", "en": "DIET MODE:"},
    "seated_mode":  {"ja": "着席モード:", "en": "SEATED MODE:"},
    "cutting_mode": {"ja": "斬撃モード:", "en": "CUTTING MODE:"},
    "on":           {"ja": "オン", "en": "ON"},
    "off":          {"ja": "オフ", "en": "OFF"},
    "toggle_diet":    {"ja": "Dキーでダイエットモード切替", "en": "Press D to toggle diet"},
    "toggle_seated":  {"ja": "Kキーで着席モード切替", "en": "Press K to toggle seated"},
    "toggle_cutting": {"ja": "Jキーで斬撃モード切替", "en": "Press J to toggle cutting"},
    "enter_name":     {"ja": "Tキーで名前入力", "en": "Press T to enter name"},
    "play_time":  {"ja": "プレイ時間:", "en": "PLAY TIME:"},
    "min":        {"ja": "分", "en": "min"},
    "set_minutes":{"ja": "1〜9キーで分数設定", "en": "Press 1-9 to set minutes"},
    "start":      {"ja": "Sキーでスタート", "en": "Press S to START"},
    "ranking":    {"ja": "ランキング", "en": "RANKING"},
    "result":     {"ja": "結果", "en": "RESULT"},
    "best":       {"ja": "ベスト:", "en": "BEST:"},
    "score":      {"ja": "スコア:", "en": "SCORE:"},
    "home":       {"ja": "Hキーでホームへ", "en": "Press H to HOME"},
    "time":       {"ja": "残り時間:", "en": "TIME:"},
    "diet_short":   {"ja": "ダイエット:", "en": "DIET:"},
    "seated_short": {"ja": "着席:", "en": "SEATED:"},
    "cutting_short":{"ja": "斬撃:", "en": "CUTTING:"},
    "language":   {"ja": "言語", "en": "Language"},
}

def L(key):
    return LANG[key][lang]

queued_texts = []

def qtext(out, s, x, y, size=20, color=(255, 255, 255)):
    if all(ord(c) < 128 for c in s):
        scale = size / 24.0
        thickness = max(1, int(size / 14))
        cv2.putText(out, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    else:
        queued_texts.append((s, x, y - size, size, color))

def flush_texts(out):
    global queued_texts
    if not queued_texts:
        return
    pil_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for s, x, y, size, color in queued_texts:
        font = get_font(size)
        draw.text((x, y), s, font=font, fill=(color[2], color[1], color[0]))
    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    out[:, :, :] = arr
    queued_texts = []

def measure_text_width(s, size):
    if all(ord(c) < 128 for c in s):
        scale = size / 24.0
        thickness = max(1, int(size / 14))
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        return tw
    else:
        font = get_font(size)
        bbox = font.getbbox(s)
        return bbox[2] - bbox[0]

def draw_lang_dropdown(out, w):
    dd_w, dd_h = 130, 32
    dd_x = w - dd_w - 20
    dd_y = 20

    label = L("language")
    label_size = 20
    label_w = measure_text_width(label, label_size)
    label_gap = 16
    label_x = dd_x - label_w - label_gap
    qtext(out, label, label_x, dd_y + 22, label_size, (255, 255, 255))

    cv2.rectangle(out, (dd_x, dd_y), (dd_x + dd_w, dd_y + dd_h), (70, 70, 70), -1)
    cv2.rectangle(out, (dd_x, dd_y), (dd_x + dd_w, dd_y + dd_h), (255, 255, 255), 1)
    cur_label = dict(lang_options)[lang]
    qtext(out, cur_label + "  v", dd_x + 10, dd_y + 22, 18, (255, 255, 255))

    option_rects = []
    if lang_dropdown_open:
        for i, (code, label_txt) in enumerate(lang_options):
            oy = dd_y + dd_h + i * dd_h
            cv2.rectangle(out, (dd_x, oy), (dd_x + dd_w, oy + dd_h), (50, 50, 50), -1)
            cv2.rectangle(out, (dd_x, oy), (dd_x + dd_w, oy + dd_h), (255, 255, 255), 1)
            qtext(out, label_txt, dd_x + 10, oy + 22, 18, (255, 255, 255))
            option_rects.append((code, dd_x, oy, dd_x + dd_w, oy + dd_h))

    return (dd_x, dd_y, dd_x + dd_w, dd_y + dd_h), option_rects

def handle_lang_click(main_rect, option_rects):
    global lang, lang_dropdown_open, mouse_click
    if mouse_click is None:
        return
    mx, my = mouse_click
    x0, y0, x1, y1 = main_rect
    if x0 <= mx <= x1 and y0 <= my <= y1:
        lang_dropdown_open = not lang_dropdown_open
        mouse_click = None
        return
    if lang_dropdown_open:
        for code, ox0, oy0, ox1, oy1 in option_rects:
            if ox0 <= mx <= ox1 and oy0 <= my <= oy1:
                lang = code
                lang_dropdown_open = False
                mouse_click = None
                return
        lang_dropdown_open = False
    mouse_click = None

def on_mouse(event, x, y, flags, param):
    global mouse_click
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click = (x, y)

def point_seg_dist(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    l2 = dx*dx + dy*dy
    if l2 == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1)*dx + (py - y1)*dy) / l2
    t = max(0, min(1, t))
    cx2 = x1 + t*dx
    cy2 = y1 + t*dy
    return math.hypot(px - cx2, py - cy2)

if os.path.exists("rank.json"):
    try:
        with open("rank.json","r") as f:
            best_scores = json.load(f)
    except:
        best_scores = []
else:
    best_scores = []

if os.path.exists("diet_rank.json"):
    try:
        with open("diet_rank.json","r") as f:
            diet_scores = json.load(f)
    except:
        diet_scores = []
else:
    diet_scores = []

def save_rank():
    with open("rank.json","w") as f:
        json.dump(best_scores,f)

def save_diet_rank():
    with open("diet_rank.json","w") as f:
        json.dump(diet_scores,f)

def find_person_id(x, y):
    for pid, (px, py) in person_last_pos.items():
        if math.hypot(px - x, py - y) < 150:
            return pid
    return None

def is_star_pose(kps):
    lwx, lwy = kps[9]
    rwx, rwy = kps[10]
    lsx, lsy = kps[5]
    rsx, rsy = kps[6]
    lfx, lfy = kps[15]
    rfx, rfy = kps[16]
    lhx, lhy = kps[11]
    rhx, rhy = kps[12]
    return abs(lwx - lsx) > 80 and abs(rwx - rsx) > 80 and abs(lfx - lhx) > 60 and abs(rfx - rhx) > 60

pairs = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]

apple_img = cv2.imread("apple.png", cv2.IMREAD_UNCHANGED)
bomb_img = cv2.imread("bomb.png", cv2.IMREAD_UNCHANGED)
green_img = cv2.imread("green.png", cv2.IMREAD_UNCHANGED)
explosion_img = cv2.imread("explosion.gif", cv2.IMREAD_UNCHANGED)

explosions = []
slices = []

def make_half(png, side):
    p = png.copy()
    hh, ww = p.shape[:2]
    if p.shape[2] != 4:
        return p
    if side == "left":
        p[:, ww//2:, 3] = 0
    else:
        p[:, :ww//2, 3] = 0
    return p

apple_left = make_half(apple_img, "left") if apple_img is not None else None
apple_right = make_half(apple_img, "right") if apple_img is not None else None
green_left = make_half(green_img, "left") if green_img is not None else None
green_right = make_half(green_img, "right") if green_img is not None else None

def add_slice(img_left, img_right, x, y, scale):
    slices.append({"img": img_left, "x": x, "y": y, "vx": -4, "vy": -5, "angle": random.randint(0,360), "va": -10, "scale": scale, "frame": 0})
    slices.append({"img": img_right, "x": x, "y": y, "vx": 4, "vy": -5, "angle": random.randint(0,360), "va": 10, "scale": scale, "frame": 0})

def draw_png_alpha(img, png, x, y, scale=1.0, angle=0, alpha=1.0):
    if png is None:
        return
    png = rotate_png(png, angle)
    h, w = png.shape[:2]
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    if x < 0 or y < 0 or x + nw > img.shape[1] or y + nh > img.shape[0]:
        return
    r = cv2.resize(png, (nw, nh))
    if r.shape[2] != 4:
        return
    b, g, rr, a = cv2.split(r)
    mask = (a.astype(np.float32) / 255.0) * alpha
    roi = img[y:y+nh, x:x+nw]
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1.0 - mask) + r[:, :, c] * mask
    img[y:y+nh, x:x+nw] = roi

def add_explosion(x, y):
    explosions.append({
        "x": x,
        "y": y,
        "frame": 0,
        "scale": 0.15,
        "angle": random.randint(0, 360)
    })
def rotate_png(png, angle):
    h, w = png.shape[:2]
    m = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(png, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def draw_png(img, png, x, y, scale=1.0, angle=0):
    if png is None:
        return
    png = rotate_png(png, angle)
    h, w = png.shape[:2]
    nw = int(w * scale)
    nh = int(h * scale)
    if x < 0 or y < 0 or x+nw > img.shape[1] or y+nh > img.shape[0]:
        return
    r = cv2.resize(png, (nw, nh))
    if r.shape[2] == 4:
        b, g, r2, a = cv2.split(r)
        mask = a / 255.0
        for c in range(3):
            img[y:y+nh, x:x+nw, c] = img[y:y+nh, x:x+nw, c] * (1 - mask) + r[:, :, c] * mask

cap = cv2.VideoCapture(0)
cv2.namedWindow("virtual_multi_stickman", cv2.WINDOW_NORMAL)
cv2.resizeWindow("virtual_multi_stickman", 1280, 720)
cv2.setMouseCallback("virtual_multi_stickman", on_mouse)

mode = "menu"
apples = []
bombs = []
greens = []
angles = []
start = 0
final_score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    seated_range = min(300, w)
    seated_x0 = w//2 - seated_range//2
    seated_x1 = w//2 + seated_range//2

    if mode == "menu":
        name_or_team = (L("name_label") if diet_mode else L("team_label")) + team_name
        qtext(out, name_or_team, 40, 45, 24, (255,255,255))
        qtext(out, L("diet_mode") + (L("on") if diet_mode else L("off")), 40, 75, 20, (0,255,255))
        qtext(out, L("seated_mode") + (L("on") if seated_mode else L("off")), 40, 100, 20, (0,255,0))
        qtext(out, L("cutting_mode") + (L("on") if cutting_mode else L("off")), 40, 125, 20, (0,180,255))
        qtext(out, L("toggle_diet"), 40, 150, 18, (200,200,200))
        qtext(out, L("toggle_seated"), 40, 175, 18, (200,200,200))
        qtext(out, L("toggle_cutting"), 40, 200, 18, (200,200,200))
        qtext(out, L("enter_name"), 40, 225, 18, (200,200,200))
        qtext(out, L("play_time") + str(play_minutes) + L("min"), 40, 255, 24, (255,255,0))
        qtext(out, L("set_minutes"), 40, 280, 18, (200,200,200))
        qtext(out, L("start"), 40, 310, 22, (255,255,255))
        qtext(out, L("ranking"), 40, 340, 24, (255,255,0))

        y = 365
        ranks = diet_scores if diet_mode else best_scores
        for i, s in enumerate(ranks[:5]):
            qtext(out, f"{i+1}: {s['team']}  {s['score']}", 40, y, 18, (255,255,255))
            y += 22

        dd_main_rect, dd_option_rects = draw_lang_dropdown(out, w)

        flush_texts(out)
        cv2.imshow("virtual_multi_stickman", out)
        k = cv2.waitKey(1)

        handle_lang_click(dd_main_rect, dd_option_rects)

        if input_mode:
            if k == 13:
                input_mode = False
            elif k == 8:
                team_name = team_name[:-1]
            elif k != -1:
                try:
                    team_name += chr(k)
                except:
                    pass
            continue

        if k == ord('d'):
            diet_mode = not diet_mode
        if k == ord('k'):
            seated_mode = not seated_mode
        if k == ord('j'):
            cutting_mode = not cutting_mode
        if k == ord('t'):
            team_name = ""
            input_mode = True
        if k in [ord(str(i)) for i in range(1,10)]:
            play_minutes = int(chr(k))
        if k == ord('s') and team_name != "":
            mode = "count"
            count_start = time.time()
        continue

    if mode == "count":
        sec = 5 - int(time.time() - count_start)
        if sec <= 0:
            if seated_mode:
                bgm_process = subprocess.Popen(["python", "bgm.py"])
            elif not diet_mode:
                bgm_process = subprocess.Popen(["python", "bgm.py"])
            elif diet_mode:
                bgm_process = subprocess.Popen(["python", "diet_bgm.py"])
            mode = "game"
            apples = []
            bombs = []
            greens = []
            angles = []
            slices = []
            person_last_pos = {}
            pose_hold = {}
            team_score = 0
            start = time.time()
            last_apple_time = time.time()
        else:
            cv2.putText(out, str(sec), (w//2-40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 8)
            cv2.imshow("virtual_multi_stickman", out)
            cv2.waitKey(1)
        continue

    if mode == "result":
        qtext(out, L("result"), w//2-120, 100, 46, (255,255,255))
        qtext(out, (L("name_label") if diet_mode else L("team_label")) + team_name, 40, 160, 28, (255,255,255))

        ranks = diet_scores if diet_mode else best_scores
        best = ranks[0]["score"] if ranks else 0

        qtext(out, L("best") + str(best), 40, 220, 28, (255,255,0))
        qtext(out, L("score") + str(final_score), 40, 280, 28, (255,255,255))
        qtext(out, L("home"), 40, 340, 22, (200,200,200))

        flush_texts(out)
        cv2.imshow("virtual_multi_stickman", out)
        k = cv2.waitKey(1)
        if k == ord('h'):
            mode = "menu"
        continue

    results = pose(frame, stream=True)

    speed_mul = 3.5 if diet_mode else 0.6
    if seated_mode:
        speed_mul = speed_mul * 0.3

    spawn_x0 = seated_x0 if seated_mode else 50
    spawn_x1 = seated_x1 if seated_mode else w-50

    if random.random() < 0.01:
        apples.append([random.randint(spawn_x0, spawn_x1), 0, random.randint(5,10)*speed_mul])
        angles.append(random.randint(0,360))

    if diet_mode and time.time() - last_apple_time > 1:
        apples.append([random.randint(spawn_x0, spawn_x1), 0, random.randint(5,10)*speed_mul])
        angles.append(random.randint(0,360))
        last_apple_time = time.time()

    if not diet_mode and time.time() - last_apple_time > 2.5:
        apples.append([random.randint(spawn_x0, spawn_x1), 0, random.randint(5,10)*speed_mul])
        angles.append(random.randint(0,360))
        last_apple_time = time.time()

    if not seated_mode and random.random() < 0.005:
        bombs.append([random.randint(spawn_x0, spawn_x1), 0, random.randint(5,10)*speed_mul])
        angles.append(random.randint(0,360))
    if random.random() < 0.0005:
        greens.append([random.randint(spawn_x0, spawn_x1), 0, random.randint(5,10)*speed_mul])
        angles.append(random.randint(0,360))

    for i, a in enumerate(apples):
        a[1] += a[2]
        angles[i] += 5
        draw_png(out, apple_img, int(a[0]-20), int(a[1]-20), 0.15, angles[i])

    for i, b in enumerate(bombs):
        b[1] += b[2]
        angles[i] += 7
        draw_png(out, bomb_img, int(b[0]-25), int(b[1]-25), 0.18, angles[i])

    for i, g in enumerate(greens):
        g[1] += g[2]
        angles[i] += 4
        draw_png(out, green_img, int(g[0]-22), int(g[1]-22), 0.16, angles[i])

    if seated_mode:
        cv2.line(out, (seated_x0, 0), (seated_x0, h), (0,255,0), 2)
        cv2.line(out, (seated_x1, 0), (seated_x1, h), (0,255,0), 2)

    for r in results:
        if r.keypoints is None:
            continue
        kps_all = r.keypoints.xy.cpu().numpy()
        if diet_mode and len(kps_all) > 1:
            kps_all = kps_all[:1]
        for kps in kps_all:
            cx = (kps[11][0] + kps[12][0]) / 2
            cy = (kps[11][1] + kps[12][1]) / 2
            if seated_mode and (cx < seated_x0 or cx > seated_x1):
                continue
            pid = find_person_id(cx, cy)
            if pid is None:
                pid = len(person_colors) + 1
                person_colors[pid] = (random.randint(80,255), random.randint(80,255), random.randint(80,255))
            person_last_pos[pid] = (cx, cy)
            col = person_colors[pid]
            if pid not in pose_hold:
                pose_hold[pid] = 0
            if is_star_pose(kps):
                pose_hold[pid] += 1/30
            else:
                pose_hold[pid] = 0
            scale = 1.0
            if pose_hold[pid] >= 5:
                scale = 1.8
            kps_scaled = []
            for x, y in kps:
                kps_scaled.append((cx + (x - cx) * scale, cy + (y - cy) * scale))
            kps_scaled = np.array(kps_scaled)
            hx = (kps_scaled[5][0] + kps_scaled[6][0]) / 2
            hy = (kps_scaled[5][1] + kps_scaled[6][1]) / 2 - 40 * scale
            cv2.circle(out, (int(hx), int(hy)), int(18*scale), col, int(3*scale))
            for a, b in pairs:
                x1, y1 = kps_scaled[a]
                x2, y2 = kps_scaled[b]
                cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), col, int(2*scale), cv2.LINE_AA)

            lex, ley = kps_scaled[7]
            rex, rey = kps_scaled[8]
            lwx, lwy = kps_scaled[9]
            rwx, rwy = kps_scaled[10]

            cv2.circle(out, (int(lwx), int(lwy)), int(10*scale), (255,255,255), -1)
            cv2.circle(out, (int(rwx), int(rwy)), int(10*scale), (255,255,255), -1)

            ltx, lty = lwx, lwy
            rtx, rty = rwx, rwy
            if cutting_mode:
                ldx = lwx - lex
                ldy = lwy - ley
                ll = math.hypot(ldx, ldy)
                if ll > 0:
                    ltx = lwx + ldx/ll*sword_len*scale
                    lty = lwy + ldy/ll*sword_len*scale
                rdx = rwx - rex
                rdy = rwy - rey
                rl = math.hypot(rdx, rdy)
                if rl > 0:
                    rtx = rwx + rdx/rl*sword_len*scale
                    rty = rwy + rdy/rl*sword_len*scale
                cv2.line(out, (int(lwx), int(lwy)), (int(ltx), int(lty)), (200,200,255), int(4*scale))
                cv2.line(out, (int(rwx), int(rwy)), (int(rtx), int(rty)), (200,200,255), int(4*scale))
                cv2.circle(out, (int(lwx), int(lwy)), int(8*scale), (0,0,180), -1)
                cv2.circle(out, (int(rwx), int(rwy)), int(8*scale), (0,0,180), -1)

            for a in apples:
                if cutting_mode:
                    hit = point_seg_dist(a[0], a[1], lwx, lwy, ltx, lty) < 25*scale or point_seg_dist(a[0], a[1], rwx, rwy, rtx, rty) < 25*scale
                else:
                    hit = math.hypot(a[0] - lwx, a[1] - lwy) < 30*scale or math.hypot(a[0] - rwx, a[1] - rwy) < 30*scale
                if hit:
                    if cutting_mode and apple_left is not None:
                        add_slice(apple_left, apple_right, a[0], a[1], 0.15)
                    team_score += a[2]
                    a[1] = h + 200
                    last_apple_time = time.time()

            for g in greens:
                if cutting_mode:
                    hit = point_seg_dist(g[0], g[1], lwx, lwy, ltx, lty) < 30*scale or point_seg_dist(g[0], g[1], rwx, rwy, rtx, rty) < 30*scale
                else:
                    hit = math.hypot(g[0] - lwx, g[1] - lwy) < 35*scale or math.hypot(g[0] - rwx, g[1] - rwy) < 35*scale
                if hit:
                    if cutting_mode and green_left is not None:
                        add_slice(green_left, green_right, g[0], g[1], 0.16)
                    start += 30
                    g[1] = h + 200

            for b in bombs:
                hit = False
                if cutting_mode:
                    if point_seg_dist(b[0], b[1], lwx, lwy, ltx, lty) < 30*scale or point_seg_dist(b[0], b[1], rwx, rwy, rtx, rty) < 30*scale:
                        hit = True
                if not hit:
                    for (bx, by) in kps_scaled:
                        if math.hypot(b[0] - bx, b[1] - by) < 40 * scale:
                            hit = True
                            break
                if hit:
                    add_explosion(b[0], b[1])
                    team_score -= b[2]
                    b[1] = h + 200
    for s in slices[:]:
        s["x"] += s["vx"]
        s["y"] += s["vy"]
        s["vy"] += 0.6
        s["angle"] += s["va"]
        s["frame"] += 1
        alpha = max(0.0, 1.0 - s["frame"] / 40.0)
        img = s["img"]
        if img is not None:
            hh, ww = img.shape[:2]
            nw = max(1, int(ww * s["scale"]))
            nh = max(1, int(hh * s["scale"]))
            draw_png_alpha(out, img, int(s["x"] - nw/2), int(s["y"] - nh/2), s["scale"], s["angle"], alpha)
        if s["frame"] >= 40 or s["y"] > h + 100:
            slices.remove(s)

    for e in explosions[:]:
        alpha = max(0.0, 1.0 - e["frame"] / 20.0)
        scale = e["scale"]
        draw_png_alpha(
            out,
            explosion_img,
            int(e["x"] - 128),
            int(e["y"] - 128),
            scale,
            e["angle"],
            alpha
        )
        e["frame"] += 1
        if e["frame"] >= 10000:
            explosions.remove(e)
    apples = [a for a in apples if a[1] < h + 50]
    bombs = [b for b in bombs if b[1] < h + 50]
    greens = [g for g in greens if g[1] < h + 50]

    limit = play_minutes * 60
    t = int(limit - (time.time() - start))
    if t <= 0:
        if bgm_process is not None:
            bgm_process.kill()
            bgm_process = None
        t = 0

    qtext(out, (L("name_label") if diet_mode else L("team_label")) + team_name, 20, 25, 22, (255,255,255))
    qtext(out, L("time") + str(t), 20, 55, 20, (255,255,255))
    qtext(out, L("score") + str(team_score), 20, 85, 20, (255,255,255))
    qtext(out, L("diet_short") + (L("on") if diet_mode else L("off")), 20, 115, 20, (0,255,255))
    qtext(out, L("seated_short") + (L("on") if seated_mode else L("off")), 20, 145, 20, (0,255,0))
    qtext(out, L("cutting_short") + (L("on") if cutting_mode else L("off")), 20, 175, 20, (0,180,255))

    flush_texts(out)
    cv2.imshow("リンゴキャッチゲーム", out)
    k = cv2.waitKey(1)

    if k == ord('r') or t == 0:
        final_score = team_score
        if diet_mode:
            diet_scores = [s for s in diet_scores if s["team"] != team_name]
            diet_scores.append({"team": team_name, "score": final_score})
            diet_scores = sorted(diet_scores, key=lambda x: x["score"], reverse=True)[:10]
            save_diet_rank()
        else:
            best_scores = [s for s in best_scores if s["team"] != team_name]
            best_scores.append({"team": team_name, "score": final_score})
            best_scores = sorted(best_scores, key=lambda x: x["score"], reverse=True)[:10]
            save_rank()
        mode = "result"

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
