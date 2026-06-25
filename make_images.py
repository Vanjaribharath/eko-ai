from PIL import Image, ImageDraw, ImageFont
import os

OUT = "/home/claude/eko-prototype/sample-images"
os.makedirs(OUT, exist_ok=True)

NAVY = (15, 27, 45)
GOLD = (201, 162, 75)
WHITE = (255, 255, 255)
LIGHT = (235, 238, 244)
LINE = (180, 188, 204)
TEXT = (40, 48, 64)

def font(size, bold=False):
    try:
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def box(draw, xy, text, fill=LIGHT, outline=NAVY, text_color=TEXT, f=None, center=True):
    draw.rectangle(xy, fill=fill, outline=outline, width=2)
    f = f or font(16)
    x0, y0, x1, y1 = xy
    if center:
        bbox = draw.textbbox((0,0), text, font=f)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.text(((x0+x1)/2 - w/2, (y0+y1)/2 - h/2), text, fill=text_color, font=f)
    else:
        draw.text((x0+12, y0+10), text, fill=text_color, font=f)

# ---------- IMAGE 1: Escalation Matrix Diagram ----------
img = Image.new("RGB", (1000, 700), WHITE)
d = ImageDraw.Draw(img)
d.rectangle([0,0,1000,70], fill=NAVY)
d.text((30, 20), "Escalation Matrix — IT and Security Incidents", fill=WHITE, font=font(24, bold=True))

levels = [
    ("Level 1", "IT Helpdesk / Service Desk", "First point of contact for all access, asset, and general IT issues. Target response: 4 business hours."),
    ("Level 2", "Tower IT Lead / Team Lead", "Escalate if unresolved after 1 business day, or for tower-specific tooling issues."),
    ("Level 3", "InfoSec Helpdesk", "For any suspected security incident, data breach, or confidentiality violation. Target response: immediate (within 1 hour)."),
    ("Level 4", "CISO Office", "Critical incidents only — major breach, multi-system compromise, or client-notified incidents."),
]
y = 110
for lvl, owner, desc in levels:
    box(d, (40, y, 230, y+90), lvl, fill=GOLD, outline=NAVY, text_color=NAVY, f=font(20, bold=True))
    box(d, (250, y, 560, y+90), owner, fill=LIGHT, outline=NAVY, f=font(15, bold=True))
    d.rectangle((580, y, 960, y+90), outline=LINE, width=1)
    d.text((595, y+12), desc, fill=TEXT, font=font(13))
    if y > 110:
        d.line((135, y-20, 135, y), fill=NAVY, width=2)
    y += 130

img.save(os.path.join(OUT, "escalation-matrix-diagram.png"))
print("Saved escalation-matrix-diagram.png")

# ---------- IMAGE 2: New Joiner Journey Flow ----------
img2 = Image.new("RGB", (1100, 500), WHITE)
d2 = ImageDraw.Draw(img2)
d2.rectangle([0,0,1100,70], fill=NAVY)
d2.text((30, 20), "New Joiner Journey — Day 1 to Week 1", fill=WHITE, font=font(24, bold=True))

steps = ["Offer\n& BGV", "Laptop\nDispatch", "Day 1\nWelcome", "IT Setup\n& SSO", "Mandatory\nTrainings", "Project\nAllocation", "Week 1\nComplete"]
n = len(steps)
box_w, box_h = 130, 110
gap = (1100 - 40 - n*box_w) / (n-1)
y0 = 200
x = 30
for i, s in enumerate(steps):
    fill = GOLD if i == n-1 else LIGHT
    box(d2, (x, y0, x+box_w, y0+box_h), s, fill=fill, outline=NAVY, f=font(15, bold=True))
    if i < n-1:
        ax0 = x + box_w
        ax1 = x + box_w + gap
        d2.line((ax0, y0+box_h/2, ax1, y0+box_h/2), fill=NAVY, width=3)
        d2.polygon([(ax1, y0+box_h/2-7), (ax1, y0+box_h/2+7), (ax1+10, y0+box_h/2)], fill=NAVY)
    x += box_w + gap

img2.save(os.path.join(OUT, "new-joiner-journey-flow.png"))
print("Saved new-joiner-journey-flow.png")

# ---------- IMAGE 3: Tower Org Structure ----------
img3 = Image.new("RGB", (1000, 600), WHITE)
d3 = ImageDraw.Draw(img3)
d3.rectangle([0,0,1000,70], fill=NAVY)
d3.text((30, 20), "Delivery Tower Structure", fill=WHITE, font=font(24, bold=True))

box(d3, (380, 100, 620, 160), "Delivery Organization", fill=NAVY, outline=NAVY, text_color=WHITE, f=font(16, bold=True))

towers = ["AuthEnq", "EIS", "ITS", "Digital", "Operations"]
tw = 170
total_w = tw*5 + 20*4
start_x = (1000 - total_w)/2
y = 280
for i, t in enumerate(towers):
    x0 = start_x + i*(tw+20)
    box(d3, (x0, y, x0+tw, y+90), t, fill=GOLD, outline=NAVY, text_color=NAVY, f=font(17, bold=True))
    mid_x = x0 + tw/2
    d3.line((mid_x, 160, mid_x, y), fill=NAVY, width=2)
    d3.line((500, 160, mid_x, 160), fill=NAVY, width=2)

img3.save(os.path.join(OUT, "tower-org-structure.png"))
print("Saved tower-org-structure.png")

print("All images created.")
