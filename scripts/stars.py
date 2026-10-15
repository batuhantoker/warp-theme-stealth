import sys, numpy as np
from PIL import Image, ImageFilter

W, H = 3840, 2400
rng = np.random.default_rng(11)
def hx(h): return np.array([int(h[i:i+2], 16) for i in (1, 3, 5)], np.float32) / 255
def blur(a, s):
    r = int(s * 3) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2).astype(np.float32); k /= k.sum()
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        out += np.roll(a, i - r, axis=0) * w
    a, out = out, np.zeros_like(a)
    for i, w in enumerate(k):
        out += np.roll(a, i - r, axis=1) * w
    return out

# base matches the theme's background gradient so opacity only fades the sky
t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
img = np.broadcast_to(hx('#101114') + (hx('#050506') - hx('#101114')) * t, (H, W, 3)).copy()

# diagonal Milky Way band: weight by distance from a tilted line
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
ang = np.deg2rad(-24)
dist = ((xx - W * 0.62) * np.sin(ang) - (yy - H * 0.42) * np.cos(ang))
band = np.exp(-(dist / (H * 0.16)) ** 2)

def noise(scale, seed):
    g = np.random.default_rng(seed).random((H // scale + 2, W // scale + 2)).astype(np.float32)
    return np.asarray(Image.fromarray(g).resize((W, H), Image.BICUBIC), np.float32)
haze = sum(noise(s, i) * w for i, (s, w) in enumerate([(420, .55), (180, .3), (70, .15)]))
haze = np.clip(haze - 0.35, 0, 1) * band
img += hx('#8fb6dc') * (haze * 0.075)[..., None]

tints = np.stack([hx('#ffffff'), hx('#cdeeff'), hx('#a9dcff'), hx('#ffe9d2')])
def scatter(n, weight=None):
    if weight is None:
        ys, xs = rng.integers(0, H, n), rng.integers(0, W, n)
    else:
        p = (weight[::4, ::4] / weight[::4, ::4].sum()).ravel()
        idx = rng.choice(p.size, n, p=p)
        ys = np.clip(idx // (W // 4) * 4 + rng.integers(0, 4, n), 0, H - 1)
        xs = np.clip(idx % (W // 4) * 4 + rng.integers(0, 4, n), 0, W - 1)
    tint = tints[rng.choice(4, n, p=[.5, .3, .12, .08])]
    return ys, xs, tint

pts = np.zeros((H, W, 3), np.float32)
glow = np.zeros((H, W, 3), np.float32)
# field stars: mostly faint, a few bright
ys, xs, tint = scatter(2300)
mag = rng.random(2300) ** 4.5
np.add.at(pts, (ys, xs), tint * (0.16 + 0.85 * mag)[:, None])
bright = mag > 0.55
np.add.at(glow, (ys[bright], xs[bright]), tint[bright] * mag[bright, None])
# dust of tiny stars inside the band
ys, xs, tint = scatter(5000, band)
np.add.at(pts, (ys, xs), tint * (0.07 + 0.24 * rng.random(5000) ** 3)[:, None])

img += blur(pts, 0.75) * 3.2
img += blur(glow, 4.0) * 9.0 * 0.35

# corner galaxies: log-spiral arms on an exponential disk, projected by inclination
def galaxy(cx, cy, r0, incl, pa, arms, pitch, seed, gain, dust=0.0):
    gr = np.random.default_rng(seed)
    n = int(r0 * 2.8) // 2 * 2
    x0, y0 = int(cx) - n // 2, int(cy) - n // 2
    v, u = np.mgrid[0:n, 0:n].astype(np.float32) - n / 2
    c, s = np.cos(pa), np.sin(pa)
    xp, yp = u * c + v * s, -u * s + v * c
    yd = yp / np.cos(incl)
    r = np.hypot(xp, yd) + 1e-3
    th = np.arctan2(yd, xp)
    def lnoise(scale):
        g = gr.random((n // scale + 2, n // scale + 2)).astype(np.float32)
        return np.asarray(Image.fromarray(g).resize((n, n), Image.BICUBIC), np.float32)
    clump = 0.55 * lnoise(max(4, n // 40)) + 0.45 * lnoise(max(3, n // 90))
    phase = arms * (th - np.log(r / (0.08 * r0)) / np.tan(pitch)) + 1.4 * (lnoise(max(6, n // 12)) - 0.5)
    # arms only beyond the core so the centre reads as a bulge, not a whorl
    am = np.clip((r / r0 - 0.07) / 0.18, 0, 1)
    arm = ((np.cos(phase) + 1) / 2) ** 2.2 * am + 0.35 * (1 - am)
    disk = np.exp(-r / (0.30 * r0)) * np.clip(1.15 - r / r0, 0, 1)
    armlight = disk * (0.22 + 1.1 * arm * (0.4 + 1.2 * clump))
    rb = np.hypot(xp, yp / np.cos(incl) ** 0.4)
    bulge = np.exp(-(rb / (0.075 * r0)) ** 1.3)
    if dust:
        lane = np.exp(-((yp + 0.05 * r0 * np.sign(np.cos(incl))) / (0.035 * r0)) ** 2) * np.clip(1 - r / (0.9 * r0), 0, 1)
        armlight *= 1 - dust * lane
        bulge *= 1 - dust * 0.8 * lane
    col = (hx('#b4d2ff')[None, None] * armlight[..., None] * 0.55
           + hx('#ffe2bf')[None, None] * bulge[..., None] * 1.0
           + hx('#e8eeff')[None, None] * (disk * 0.25)[..., None])
    # resolved stars along the arms
    p = (armlight / armlight.sum()).ravel()
    idx = gr.choice(p.size, int(r0 * 2.2), p=p)
    dots = np.zeros((n, n, 3), np.float32)
    np.add.at(dots, (idx // n, idx % n), hx('#d6e8ff') * (0.15 + 0.6 * gr.random(idx.size) ** 4)[:, None])
    col = blur(col, 1.1) + blur(dots, 0.7) * 3.0
    a, b = max(0, -y0), max(0, -x0)
    sl = img[max(0, y0):y0 + n, max(0, x0):x0 + n]
    sl += col[a:a + sl.shape[0], b:b + sl.shape[1]] * gain

galaxy(W * 0.115, H * 0.150, 190, np.deg2rad(40), 0.9, 2, np.deg2rad(24), 1, 0.5)
galaxy(W * 0.875, H * 0.175, 330, np.deg2rad(32), -0.4, 2, np.deg2rad(22), 2, 0.72)
galaxy(W * 0.120, H * 0.840, 300, np.deg2rad(84), -0.35, 2, np.deg2rad(18), 3, 0.75, dust=0.75)
galaxy(W * 0.870, H * 0.830, 290, np.deg2rad(58), 0.55, 2, np.deg2rad(26), 4, 0.7, dust=0.45)

# fine film grain; also dithers away gradient banding
img += rng.normal(0, 1.6 / 255, (H, W, 1)).astype(np.float32)

img = np.clip(img, 0, 1)
Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(sys.argv[1], quality=94, subsampling=0, optimize=True)
lum = img @ np.array([.2126, .7152, .0722], np.float32)
print('mean', lum.mean().round(4), 'p99.9', np.percentile(lum, 99.9).round(3), 'max', lum.max().round(3))
