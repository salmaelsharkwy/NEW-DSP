# video_codec.py  –  basic video compressor (I-frames + P-frames)
import numpy as np
import cv2
import pickle
from scipy.fft import dctn, idctn
from collections import Counter
import heapq

_Q_BASE = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], np.float32)

_ZZ = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),(2,1),(3,0),
    (4,0),(3,1),(2,2),(1,3),(0,4),(0,5),(1,4),(2,3),(3,2),(4,1),
    (5,0),(6,0),(5,1),(4,2),(3,3),(2,4),(1,5),(0,6),(0,7),(1,6),
    (2,5),(3,4),(4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),(7,2),(7,3),
    (6,4),(5,5),(4,6),(3,7),(4,7),(5,6),(6,5),(7,4),(7,5),(6,6),
    (5,7),(6,7),(7,6),(7,7),
]

def _qmat(quality=75):
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip((_Q_BASE * s + 50) / 100, 1, 255)

_M  = np.float32([[0.299,  0.587,  0.114],
                   [-0.169, -0.331,  0.500],
                   [0.500, -0.419, -0.081]])
_Mi = np.linalg.inv(_M)

def rgb2yuv(frame):
    yuv = frame.astype(np.float32) @ _M.T
    yuv[:, :, 1:] += 128
    return np.clip(yuv, 0, 255).astype(np.uint8)

def yuv2rgb(yuv):
    f = yuv.astype(np.float32)
    f[:, :, 1:] -= 128
    return np.clip(f @ _Mi.T, 0, 255).astype(np.uint8)

def _dct8(blk, Q):
    return np.round(dctn(blk.astype(float) - 128.0, norm='ortho') / Q).astype(np.int16)

def _idct8(qb, Q):
    return np.clip(idctn(qb.astype(float) * Q, norm='ortho') + 128.0, 0, 255)

def _rle(arr):
    out, z = [], 0
    for v in arr:
        if v == 0: z += 1
        else:
            out.append((z, int(v))); z = 0
    out.append((-1, 0))
    return out

def _rle_dec(rle):
    out = []
    for z, v in rle:
        if z == -1: break
        out.extend([0] * z); out.append(v)
    out += [0] * (64 - len(out))
    return out[:64]

def _huff_build(data):
    freq = Counter(data)
    heap = [[w, [s, ""]] for s, w in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1: return {heap[0][1][0]: "0"}
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        for p in a[1:]: p[1] = "0" + p[1]
        for p in b[1:]: p[1] = "1" + p[1]
        heapq.heappush(heap, [a[0] + b[0]] + a[1:] + b[1:])
    return {s: c for s, c in heap[0][1:]}

def _huff_enc(flat, cb):
    bits = "".join(cb[v] for v in flat)
    pad  = (-len(bits)) % 8
    bits += "0" * pad
    enc  = bytes(int(bits[i: i + 8], 2) for i in range(0, len(bits), 8))
    return enc, pad

def _huff_dec(enc, rev_cb, pad, count):
    bits = "".join(f"{b:08b}" for b in enc)
    if pad: bits = bits[:-pad]
    out, buf = [], ""
    for b in bits:
        buf += b
        if buf in rev_cb:
            out.append(rev_cb[buf]); buf = ""
        if len(out) == count: break
    out += [0] * (count - len(out))
    return out[:count]

def gen_video(path, n=30, h=128, w=160, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:, :, 0] = np.tile(np.linspace(30, 180, w, dtype=np.uint8), (h, 1))
        frame[:, :, 1] = 60
        cx = int(20 + (w - 40) * i / (n - 1))
        cv2.circle(frame, (cx, h // 2), 18, (220, 200, 80), -1)
        cv2.rectangle(frame, (4, 4), (w - 4, h - 4), (100, 150, 200), 2)
        out.write(frame)
    out.release()

def _enc_iframe(y, Q):
    h, w = y.shape; blocks = []
    for r in range(0, h, 8):
        for c in range(0, w, 8):
            if r + 8 > h or c + 8 > w: continue
            qb = _dct8(y[r: r+8, c: c+8].astype(float), Q)
            zz = [int(qb[ri, ci]) for ri, ci in _ZZ]
            blocks.append(_rle(zz))
    return blocks

def _dec_iframe(blocks, h, w, Q):
    y = np.zeros((h, w), np.uint8); idx = 0
    for r in range(0, h, 8):
        for c in range(0, w, 8):
            if r + 8 > h or c + 8 > w or idx >= len(blocks): continue
            zz = _rle_dec(blocks[idx]); idx += 1
            qb = np.zeros((8, 8), np.int16)
            for k, (ri, ci) in enumerate(_ZZ): qb[ri, ci] = zz[k]
            y[r: r+8, c: c+8] = _idct8(qb.astype(float), Q)
    return y

def _motion_est(curr, ref, bsize=16, sr=4):
    h, w = curr.shape; mvs, residuals = [], []
    for r in range(0, h, bsize):
        for c in range(0, w, bsize):
            if r + bsize > h or c + bsize > w: continue
            blk = curr[r: r+bsize, c: c+bsize].astype(np.int16)
            best_mv, best_sad = (0, 0), float('inf')  # SAD = Sum of Absolute Differences
            for dy in range(-sr, sr + 1):
                for dx in range(-sr, sr + 1):
                    rr, cc = r + dy, c + dx
                    if 0 <= rr and rr+bsize <= h and 0 <= cc and cc+bsize <= w:
                        sad = int(np.sum(np.abs(blk - ref[rr: rr+bsize, cc: cc+bsize])))
                        if sad < best_sad: best_sad = sad; best_mv = (dy, dx)
            dy, dx = best_mv
            res = (blk - ref[r+dy: r+dy+bsize, c+dx: c+dx+bsize].astype(np.int16))
            mvs.append(best_mv); residuals.append(res.flatten().tolist())
    return mvs, residuals

def _dec_pframe(mvs, flat, ref_y, bsize, n_blocks):
    h, w = ref_y.shape; y = ref_y.copy().astype(np.int16); nc = w // bsize
    for i, (dy, dx) in enumerate(mvs):
        r = (i // nc) * bsize; c = (i % nc) * bsize
        if r + bsize > h or c + bsize > w: continue
        seg = flat[i * bsize * bsize: (i+1) * bsize * bsize]
        res = np.array(seg, np.int16).reshape(bsize, bsize)
        rr, cc = r + dy, c + dx
        if 0 <= rr and rr+bsize <= h and 0 <= cc and cc+bsize <= w:
            y[r: r+bsize, c: c+bsize] = (
                ref_y[rr: rr+bsize, cc: cc+bsize].astype(np.int16) + res)
    return np.clip(y, 0, 255).astype(np.uint8)

def _collect_bgr_frames(vid_path):
    MAX_FRAMES = 90; MAX_W, MAX_H = 480, 272
    def resize_bgr(bgr):
        th = min(bgr.shape[0], MAX_H); tw = min(bgr.shape[1], MAX_W)
        th = max((th // 16) * 16, 16); tw = max((tw // 16) * 16, 16)
        return cv2.resize(bgr, (tw, th))
    cap = cv2.VideoCapture(vid_path, cv2.CAP_ANY)
    if not cap.isOpened(): cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + vid_path + "  Tip: convert to H.264 .mp4 with VLC.")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 9999
    step  = max(1, total // MAX_FRAMES)
    frames, fi = [], 0
    while len(frames) < MAX_FRAMES:
        ret, bgr = cap.read()
        if not ret: break
        fi += 1
        if (fi - 1) % step != 0: continue
        if bgr is None or bgr.size == 0: continue
        frames.append(resize_bgr(bgr))
    cap.release()
    if not frames:
        raise RuntimeError("No frames decoded. Convert to H.264 .mp4 with VLC and retry.")
    return frames, fps

def encode_video(vid_path, gop=10, quality=75, progress_cb=None):
    bgr_frames, fps = _collect_bgr_frames(vid_path)
    Q = _qmat(quality); bsize = 16
    raw_frames = []; orig_frames = []; ref_y = None
    total = len(bgr_frames)
    for i, bgr in enumerate(bgr_frames):
        if progress_cb:
            progress_cb(i + 1, total)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_frames.append(rgb)
        yuv = rgb2yuv(rgb); h, w = yuv.shape[:2]
        y_ch = yuv[:, :, 0]
        u_s = yuv[::2, ::2, 1]; v_s = yuv[::2, ::2, 2]
        if i % gop == 0:
            blocks = _enc_iframe(y_ch, Q)
            raw_frames.append(('I', blocks, u_s, v_s)); ref_y = y_ch.copy()
        else:
            # small search radius = faster encoding
            mvs, residuals = _motion_est(y_ch, ref_y, bsize=bsize, sr=1)
            flat = [v for res in residuals for v in res]; nb = len(mvs)
            raw_frames.append(('P', mvs, flat, bsize, nb, u_s, v_s))
            ref_y = _dec_pframe(mvs, flat[:], ref_y, bsize, nb)
    # collect all residual values and build one Huffman table
    all_res = []
    all_mvs = []
    for fd in raw_frames:
        if fd[0] == 'P':
            all_res.extend(fd[2])
            for mv in fd[1]:
                all_mvs.extend(mv)
    global_cb = _huff_build(all_res) if all_res else {0: "0"}
    mv_cb     = _huff_build(all_mvs) if all_mvs else {0: "0"}

    frames_data = []
    for fd in raw_frames:
        if fd[0] == 'I':
            frames_data.append(fd)
        else:
            _, mvs, flat, bsize_f, nb, u_s, v_s = fd
            # Huffman encode residuals and motion vectors
            enc_res, pad_res = _huff_enc(flat, global_cb)
            mvs_flat = []
            for mv in mvs:
                mvs_flat.extend(mv)
            enc_mv, pad_mv = _huff_enc(mvs_flat, mv_cb)
            frames_data.append(('P', enc_mv, pad_mv, enc_res, pad_res,
                                 bsize_f, nb, u_s, v_s))

    h, w = orig_frames[0].shape[:2]
    payload = {'fps': fps, 'h': h, 'w': w, 'Q': Q, 'gop': gop,
               'global_cb': global_cb, 'mv_cb': mv_cb, 'frames': frames_data}
    bs = pickle.dumps(payload, protocol=4)
    orig_bytes = len(orig_frames) * h * w * 3
    return bs, orig_frames, orig_bytes

def decode_video(bs):
    data = pickle.loads(bs)
    Q = data['Q']; h, w = data['h'], data['w']
    rev_cb    = {v: k for k, v in data['global_cb'].items()}
    rev_mv_cb = {v: k for k, v in data['mv_cb'].items()}
    recon = []; ref_y = None
    for fd in data['frames']:
        if fd[0] == 'I':
            _, blocks, u_s, v_s = fd; y_ch = _dec_iframe(blocks, h, w, Q)
        else:
            _, enc_mv, pad_mv, enc_res, pad_res, bsize, nb, u_s, v_s = fd
            # Huffman-decode motion vectors then re-pair into (dy, dx) tuples
            mvs_flat = _huff_dec(enc_mv, rev_mv_cb, pad_mv, nb * 2)
            mvs = [(mvs_flat[i * 2], mvs_flat[i * 2 + 1]) for i in range(nb)]
            # Huffman-decode residuals
            flat = _huff_dec(enc_res, rev_cb, pad_res, nb * bsize * bsize)
            y_ch = _dec_pframe(mvs, flat, ref_y, bsize, nb)
        u_full = np.repeat(np.repeat(u_s, 2, axis=0), 2, axis=1)[:h, :w]
        v_full = np.repeat(np.repeat(v_s, 2, axis=0), 2, axis=1)[:h, :w]
        recon.append(yuv2rgb(np.stack([y_ch, u_full, v_full], axis=2)))
        ref_y = y_ch.copy()
    return recon, data['fps']
