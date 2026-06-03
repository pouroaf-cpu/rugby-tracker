"""
reanchor_gui.py - Manual re-anchor / track-cleanup tool for rugby-tracker CSVs.

Operates on a tracks CSV (frame, track_id, x1, y1, x2, y2, confidence) over the
source video. Lets you consolidate a player onto a canonical ID and delete the
fragments/false-positives the tracker leaves behind. Nothing is written until
you press 's'.

CANONICAL TARGET
  Your player is consolidated onto one ID (default 1, set with --id). Merges
  relabel an errant track onto the target.

MOUSE
  Left-click a bbox        arm that track ID (briefly highlighted)
  Left-click PLAY button   play / pause (button top-left of the video)
  Right-click a bbox       pop a "Delete ID" button; left-click it to nuke that track

KEYS
  <-  / ->  or  , / .      step -1 / +1 frame
  [  /  ]                  step -30 / +30 frames
  space                    play / pause
  digits                   type a merge destination ID (shown in status bar)
  Enter                    MERGE armed ID -> typed ID (or target if none typed),
                           from current frame forward until the next >30f gap
  d                        DELETE FORWARD: armed ID from current frame forward
                           until the next >30f gap   (addendum)
  D (shift+d)              NUKE: delete ALL rows for the armed ID    (addendum)
  i                        MANAGE IDs dialog (Tkinter bulk editor)   (addendum)
  t                        set target = armed ID
  u                        undo last change (merge / delete / dialog-apply)
  s                        save CSV (no autosave)
  Esc                      clear armed ID / merge-input buffer
  q                        quit

YOUR RE-ANCHOR WORKFLOW (e.g. "34 is following me but it's actually 35"):
  click the box that is really you  -> Enter   (merge correct segment onto 1)
  click the stray box               -> d       (delete its future)
  ... scrub on, repeat at the next swap. 'u' reverses any step.

Usage:
  python scripts\\reanchor_gui.py --video input\\chunk_028.mp4 --csv output\\chunk_028_tracks.csv
  python scripts\\reanchor_gui.py --csv output\\chunk_028_tracks.csv --selftest   # headless logic test
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

FPS_ASSUMED = 29.97
GAP = 30  # frames; a forward segment ends at the first gap larger than this


# ===========================================================================
# Data model + undo stack  (pure logic, no GUI -- covered by --selftest)
# ===========================================================================
class TrackStore:
    """In-memory tracks with stable row ids and a reversible op log."""

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.records = {}          # rid -> dict(frame,tid,x1,y1,x2,y2,conf)
        self._next_rid = 0
        self.undo_stack = []       # list of undo records
        self.merges = 0
        self.deletes = 0           # counts tracks deleted (segments/tracks)
        self._load()
        self._reindex()

    # ---- load / save ----
    def _load(self):
        with open(self.csv_path, newline="") as f:
            for r in csv.DictReader(f):
                rid = self._next_rid; self._next_rid += 1
                self.records[rid] = {
                    "frame": int(r["frame"]), "tid": int(r["track_id"]),
                    "x1": float(r["x1"]), "y1": float(r["y1"]),
                    "x2": float(r["x2"]), "y2": float(r["y2"]),
                    "conf": float(r["confidence"]),
                }

    def save(self, out_path=None):
        out = Path(out_path) if out_path else self.csv_path
        rows = sorted(self.records.values(), key=lambda d: (d["frame"], d["tid"]))
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "confidence"])
            for d in rows:
                w.writerow([d["frame"], d["tid"], f"{d['x1']:.2f}", f"{d['y1']:.2f}",
                            f"{d['x2']:.2f}", f"{d['y2']:.2f}", f"{d['conf']:.4f}"])
        return out

    # ---- indices ----
    def _reindex(self):
        self.by_frame = defaultdict(list)        # frame -> [rid]
        self.by_id = defaultdict(list)           # tid -> [(frame, rid)]
        for rid, d in self.records.items():
            self.by_frame[d["frame"]].append(rid)
            self.by_id[d["tid"]].append((d["frame"], rid))
        for tid in self.by_id:
            self.by_id[tid].sort()

    def unique_ids(self):
        return sorted({d["tid"] for d in self.records.values()})

    def id_stats(self, tid):
        frames = [fr for fr, _ in self.by_id.get(tid, [])]
        if not frames:
            return {"count": 0, "first": 0, "last": 0, "dur": 0.0}
        return {"count": len(frames), "first": min(frames), "last": max(frames),
                "dur": (max(frames) - min(frames) + 1) / FPS_ASSUMED}

    # ---- segment scoping (shared by merge-forward and delete-forward) ----
    def forward_segment(self, tid, start_frame, gap=GAP):
        """rids of `tid` from start_frame forward, stopping before the first
        absence gap larger than `gap` frames."""
        appics = [(fr, rid) for fr, rid in self.by_id.get(tid, []) if fr >= start_frame]
        if not appics:
            return []
        seg = [appics[0][1]]
        prev = appics[0][0]
        for fr, rid in appics[1:]:
            if fr - prev > gap:
                break
            seg.append(rid); prev = fr
        return seg

    # ---- mutations (each pushes one undo record) ----
    def merge_forward(self, src_id, dst_id, start_frame):
        seg = self.forward_segment(src_id, start_frame)
        if not seg:
            return 0
        changes = [(rid, self.records[rid]["tid"]) for rid in seg]
        for rid in seg:
            self.records[rid]["tid"] = dst_id
        self.undo_stack.append({"kind": "relabel", "data": changes, "counter": "merges"})
        self.merges += 1
        self._reindex()
        return len(seg)

    def _remove_rids(self, rids, counter, n_tracks):
        removed = [(rid, dict(self.records[rid])) for rid in rids if rid in self.records]
        for rid, _ in removed:
            del self.records[rid]
        self.undo_stack.append({"kind": "remove", "data": removed,
                                "counter": counter, "n": n_tracks})
        if counter == "deletes":
            self.deletes += n_tracks
        self._reindex()
        return len(removed)

    def delete_forward(self, tid, start_frame):
        seg = self.forward_segment(tid, start_frame)
        return self._remove_rids(seg, "deletes", 1) if seg else 0

    def nuke(self, tid):
        rids = [rid for _, rid in self.by_id.get(tid, [])]
        return self._remove_rids(rids, "deletes", 1) if rids else 0

    def bulk_delete(self, ids):
        ids = set(ids)
        rids = [rid for rid, d in self.records.items() if d["tid"] in ids]
        return self._remove_rids(rids, "deletes", len(ids)) if rids else 0

    def undo(self):
        if not self.undo_stack:
            return False
        rec = self.undo_stack.pop()
        if rec["kind"] == "relabel":
            for rid, old in rec["data"]:
                if rid in self.records:
                    self.records[rid]["tid"] = old
            if rec.get("counter") == "merges":
                self.merges = max(0, self.merges - 1)
        elif rec["kind"] == "remove":
            for rid, data in rec["data"]:
                self.records[rid] = data
            if rec.get("counter") == "deletes":
                self.deletes = max(0, self.deletes - rec.get("n", 1))
        self._reindex()
        return True


# ===========================================================================
# Headless self-test (no GUI)
# ===========================================================================
def selftest(csv_path):
    print("[selftest] loading", csv_path)
    s = TrackStore(csv_path)
    ids = s.unique_ids()
    print(f"[selftest] {len(s.records)} rows, {len(ids)} unique ids")
    assert ids, "no ids loaded"

    # pick a long-lived track to exercise scoping
    tid = max(ids, key=lambda t: s.id_stats(t)["count"])
    st = s.id_stats(tid)
    print(f"[selftest] busiest id={tid} count={st['count']} "
          f"frames {st['first']}..{st['last']} dur={st['dur']:.1f}s")

    # forward_segment must be contiguous (no internal gap > GAP)
    mid = (st["first"] + st["last"]) // 2
    seg = s.forward_segment(tid, mid)
    seg_frames = sorted(s.records[r]["frame"] for r in seg)
    assert all(seg_frames[i+1] - seg_frames[i] <= GAP for i in range(len(seg_frames)-1)), \
        "segment has an internal gap > GAP"
    assert min(seg_frames) >= mid, "segment leaked before start_frame"
    print(f"[selftest] forward_segment(mid={mid}) -> {len(seg)} rows, "
          f"frames {seg_frames[0]}..{seg_frames[-1]}  OK")

    # merge_forward then undo restores exactly
    before = {rid: d["tid"] for rid, d in s.records.items()}
    dst = 1
    n = s.merge_forward(tid, dst, mid)
    assert n == len(seg)
    assert all(s.records[r]["tid"] == dst for r in seg), "merge didn't relabel"
    assert any(rid not in s.records or True for rid in seg)
    s.undo()
    after = {rid: d["tid"] for rid, d in s.records.items()}
    assert before == after, "undo(merge) did not restore tids"
    print(f"[selftest] merge_forward({tid}->{dst}) relabelled {n}; undo restored  OK")

    # delete_forward then undo restores row set
    total_before = len(s.records)
    n = s.delete_forward(tid, mid)
    assert len(s.records) == total_before - n, "delete count mismatch"
    s.undo()
    assert len(s.records) == total_before, "undo(delete) did not restore rows"
    print(f"[selftest] delete_forward removed {n}; undo restored {total_before}  OK")

    # nuke then undo
    cnt = s.id_stats(tid)["count"]
    n = s.nuke(tid)
    assert n == cnt and tid not in s.unique_ids(), "nuke incomplete"
    s.undo()
    assert tid in s.unique_ids(), "undo(nuke) did not restore id"
    print(f"[selftest] nuke({tid}) removed {n}; undo restored  OK")

    # bulk_delete (single undo) then undo
    victims = ids[:3]
    total_before = len(s.records)
    n = s.bulk_delete(victims)
    assert all(v not in s.unique_ids() for v in victims), "bulk_delete incomplete"
    assert len(s.undo_stack) >= 1
    s.undo()
    assert len(s.records) == total_before, "undo(bulk) did not restore"
    assert all(v in s.unique_ids() for v in victims), "undo(bulk) lost ids"
    print(f"[selftest] bulk_delete({victims}) removed {n} rows as ONE undo; restored  OK")

    print("[selftest] ALL PASSED")


# ===========================================================================
# GUI  (OpenCV main window + Tkinter modal dialog)
# ===========================================================================
def run_gui(video_path, store, target):
    import cv2
    import numpy as np

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_ASSUMED
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win = "reanchor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    state = {
        "frame": 1, "playing": False, "armed": None, "armed_box": None,
        "input": "", "target": target, "decoded": -1, "img": None,
        "click_hl": 0, "tb_guard": False, "status": "", "menu": None,
    }
    PLAY_BTN = (8, 6, 150, 44)   # x1,y1,x2,y2 of the play/pause button

    # ---- Tkinter modal "Manage IDs" dialog ----
    tk_root = {"root": None}

    def ensure_tk():
        import tkinter as tk
        if tk_root["root"] is None:
            r = tk.Tk(); r.withdraw()
            tk_root["root"] = r
        return tk_root["root"]

    def manage_ids_dialog():
        import tkinter as tk
        from tkinter import ttk, messagebox
        root = ensure_tk()
        dlg = tk.Toplevel(root)
        dlg.title("Manage IDs")
        dlg.geometry("520x640")
        dlg.transient(root); dlg.grab_set()

        result = {"applied": False}
        ticks = {}            # tid -> tk.BooleanVar
        sort_mode = tk.StringVar(value="id")

        # cache stats
        ids = store.unique_ids()
        stats = {t: store.id_stats(t) for t in ids}
        for t in ids:
            ticks[t] = tk.BooleanVar(value=True)

        # --- top: sort radios ---
        top = ttk.Frame(dlg); top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Sort:").pack(side="left")
        for label, val in [("By ID", "id"), ("By frame count", "count"),
                           ("By first appearance", "first")]:
            ttk.Radiobutton(top, text=label, value=val, variable=sort_mode,
                            command=lambda: rebuild_rows()).pack(side="left", padx=4)

        # --- bulk action buttons ---
        bar = ttk.Frame(dlg); bar.pack(fill="x", padx=8)
        def tick_all():   [v.set(True) for v in ticks.values()]
        def untick_all(): [v.set(False) for v in ticks.values()]
        def untick_short():
            for t, v in ticks.items():
                if stats[t]["count"] < 30:
                    v.set(False)
        def untick_except_target():
            for t, v in ticks.items():
                v.set(t == state["target"])
        ttk.Button(bar, text="Tick all", command=tick_all).pack(side="left", padx=2, pady=4)
        ttk.Button(bar, text="Untick all", command=untick_all).pack(side="left", padx=2)
        ttk.Button(bar, text="Untick short (<30)", command=untick_short).pack(side="left", padx=2)
        ttk.Button(bar, text="Untick all except target",
                   command=untick_except_target).pack(side="left", padx=2)

        # --- scrollable list ---
        mid = ttk.Frame(dlg); mid.pack(fill="both", expand=True, padx=8, pady=6)
        canvas = tk.Canvas(mid, highlightthickness=0)
        sb = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta/120), "units"))

        header = ttk.Frame(inner); header.grid(row=0, column=0, sticky="w")
        ttk.Label(header, text=f"{'keep':<5}{'ID':>5}{'frames':>9}"
                               f"{'   span':>14}{'   dur(s)':>9}", font=("Consolas", 9, "bold")
                  ).pack(anchor="w")

        def sorted_ids():
            if sort_mode.get() == "count":
                return sorted(ids, key=lambda t: -stats[t]["count"])
            if sort_mode.get() == "first":
                return sorted(ids, key=lambda t: stats[t]["first"])
            return sorted(ids)

        rows_frame = {"f": None}
        def rebuild_rows():
            if rows_frame["f"] is not None:
                rows_frame["f"].destroy()
            rf = ttk.Frame(inner); rf.grid(row=1, column=0, sticky="w")
            rows_frame["f"] = rf
            for t in sorted_ids():
                s_ = stats[t]
                is_target = (t == state["target"])
                star = "★ " if is_target else "  "
                txt = (f"{star}{t:>4}{s_['count']:>9}"
                       f"  {s_['first']:>5}->{s_['last']:<5}{s_['dur']:>8.1f}")
                row = ttk.Frame(rf); row.pack(anchor="w", fill="x")
                cb = ttk.Checkbutton(row, variable=ticks[t])
                cb.pack(side="left")
                lbl = tk.Label(row, text=txt, font=("Consolas", 9,
                               "bold" if is_target else "normal"),
                               fg="#b8860b" if is_target else "black")
                lbl.pack(side="left")
        rebuild_rows()

        # --- apply / cancel ---
        def on_apply():
            unticked = [t for t in ids if not ticks[t].get()]
            if not unticked:
                result["applied"] = False; dlg.destroy(); return
            # guardrail: deleting the active target
            if state["target"] in unticked:
                if not messagebox.askyesno("Confirm", "Delete your active target ID?"):
                    return
            # guardrail: >50% of rows
            total = len(store.records)
            doomed = sum(stats[t]["count"] for t in unticked)
            if total and doomed / total > 0.5:
                pct = round(100 * doomed / total)
                if not messagebox.askyesno(
                        "Confirm", f"This will remove {pct}% of all tracking data. Continue?"):
                    return
            result["applied"] = True
            result["unticked"] = unticked
            result["rows"] = doomed
            dlg.destroy()

        def on_cancel():
            result["applied"] = False
            dlg.destroy()

        btns = ttk.Frame(dlg); btns.pack(fill="x", padx=8, pady=8)
        ttk.Button(btns, text="Apply", command=on_apply).pack(side="right", padx=4)
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right")
        dlg.protocol("WM_DELETE_WINDOW", on_cancel)   # X button == cancel

        root.wait_window(dlg)   # modal: blocks here until dialog closes

        if result.get("applied"):
            ids_removed = result["unticked"]
            n = store.bulk_delete(ids_removed)
            print(f"Removed {len(ids_removed)} IDs ({n} rows)")
        return

    # ---- frame decode (cache; only seek when frame changes) ----
    def get_frame(idx):
        idx = max(1, min(n_frames, idx))
        if state["decoded"] != idx or state["img"] is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
            ok, img = cap.read()
            if not ok:
                img = state["img"]
            state["img"] = img; state["decoded"] = idx
        return None if state["img"] is None else state["img"].copy()

    def color_for(tid):
        return (int((tid*47) % 256), int((tid*97) % 256), int((tid*151) % 256))

    # ---- mouse helpers ----
    def hit_test(x, y):
        """Smallest-area bbox containing the point (most specific hit), or None."""
        best = None; best_area = None
        for rid in store.by_frame.get(state["frame"], []):
            d = store.records[rid]
            if d["x1"] <= x <= d["x2"] and d["y1"] <= y <= d["y2"]:
                a = (d["x2"]-d["x1"])*(d["y2"]-d["y1"])
                if best_area is None or a < best_area:
                    best_area = a; best = d
        return best

    def _in(rect, x, y):
        return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    def _arm(d):
        state["armed"] = d["tid"]
        state["armed_box"] = (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
        state["click_hl"] = 8

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 1) a context "Delete ID" button is open?
            menu = state["menu"]
            if menu is not None:
                if _in(menu["rect"], x, y):
                    tid = menu["tid"]
                    n = store.nuke(tid)
                    state["status"] = f"Nuked ID {tid} entirely ({n:,} rows)"
                    print("[reanchor]", state["status"])
                state["menu"] = None          # any click dismisses the menu
                return
            # 2) the play/pause button?
            if _in(PLAY_BTN, x, y):
                state["playing"] = not state["playing"]
                return
            # 3) arm a bbox
            best = hit_test(x, y)
            if best is not None:
                _arm(best)
                state["status"] = f"armed ID {best['tid']}"
        elif event == cv2.EVENT_RBUTTONDOWN:
            best = hit_test(x, y)
            if best is None:
                state["menu"] = None
                return
            _arm(best)
            bw, bh = 168, 32
            mx = min(x, width - bw - 2); my = min(y, height - bh - 2)
            state["menu"] = {"tid": best["tid"], "rect": (mx, my, mx + bw, my + bh)}
            state["status"] = f"right-click ID {best['tid']} -> click Delete"
    cv2.setMouseCallback(win, on_mouse)

    # ---- trackbar seek ----
    def on_tb(pos):
        if state["tb_guard"]:
            return
        state["frame"] = max(1, min(n_frames, pos + 1))
    cv2.createTrackbar("frame", win, 0, max(0, n_frames - 1), on_tb)

    def render():
        img = get_frame(state["frame"])
        if img is None:
            return None
        for rid in store.by_frame.get(state["frame"], []):
            d = store.records[rid]
            p1 = (int(d["x1"]), int(d["y1"])); p2 = (int(d["x2"]), int(d["y2"]))
            if d["tid"] == state["target"]:
                col, th = (0, 215, 255), 3        # target: gold, thick
            elif d["tid"] == state["armed"]:
                col, th = (0, 0, 255), 3          # armed: red
            else:
                col, th = color_for(d["tid"]), 1
            cv2.rectangle(img, p1, p2, col, th)
            cv2.putText(img, str(d["tid"]), (p1[0], max(12, p1[1]-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        if state["click_hl"] > 0 and state["armed_box"]:
            x1, y1, x2, y2 = state["armed_box"]
            cv2.rectangle(img, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 255), 2)
            state["click_hl"] -= 1
        # --- control bar (taller, bigger antialiased text) ---
        BARH = 48
        cv2.rectangle(img, (0, 0), (img.shape[1], BARH), (0, 0, 0), -1)

        # play / pause button with a glyph
        px1, py1, px2, py2 = PLAY_BTN
        cv2.rectangle(img, (px1, py1), (px2, py2), (70, 70, 70), -1)
        cv2.rectangle(img, (px1, py1), (px2, py2), (210, 210, 210), 1)
        cy = (py1 + py2) // 2
        if state["playing"]:
            cv2.rectangle(img, (px1 + 13, py1 + 9), (px1 + 20, py2 - 9), (255, 255, 255), -1)
            cv2.rectangle(img, (px1 + 25, py1 + 9), (px1 + 32, py2 - 9), (255, 255, 255), -1)
            btxt = "PAUSE"
        else:
            tri = np.array([[px1 + 14, py1 + 9], [px1 + 14, py2 - 9],
                            [px1 + 32, cy]], np.int32)
            cv2.fillConvexPoly(img, tri, (0, 230, 0))
            btxt = "PLAY"
        cv2.putText(img, btxt, (px1 + 44, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # status text
        bar = (f"frame {state['frame']}/{n_frames}   target {state['target']}   "
               f"Merges {store.merges}   Deletes {store.deletes}   armed {state['armed']}")
        if state["input"]:
            bar += f"   merge>{state['input']}_"
        if state["status"]:
            bar += f"   [{state['status']}]"
        cv2.putText(img, bar, (px2 + 16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # context "Delete ID" button (from right-click)
        if state["menu"] is not None:
            mx1, my1, mx2, my2 = state["menu"]["rect"]
            cv2.rectangle(img, (mx1, my1), (mx2, my2), (40, 40, 200), -1)
            cv2.rectangle(img, (mx1, my1), (mx2, my2), (255, 255, 255), 1)
            cv2.putText(img, f"Delete ID {state['menu']['tid']}", (mx1 + 8, my2 - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    print(__doc__.split("Usage:")[0])
    print(f"[reanchor] {n_frames} frames @ {fps:.2f}fps | target ID = {state['target']}")

    while True:
        img = render()
        if img is not None:
            cv2.imshow(win, img)
            state["tb_guard"] = True
            cv2.setTrackbarPos("frame", win, state["frame"] - 1)
            state["tb_guard"] = False

        key = cv2.waitKeyEx(15 if state["playing"] else 20)

        if state["playing"]:
            state["frame"] = min(n_frames, state["frame"] + 1)
            if state["frame"] >= n_frames:
                state["playing"] = False

        if key == -1:
            # window closed?
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        # normalize
        k = key & 0xFF
        state["status"] = ""

        if key in (2424832,) or k == ord(","):        # left
            state["frame"] = max(1, state["frame"] - 1); state["playing"] = False
        elif key in (2555904,) or k == ord("."):      # right
            state["frame"] = min(n_frames, state["frame"] + 1); state["playing"] = False
        elif k == ord("["):
            state["frame"] = max(1, state["frame"] - 30); state["playing"] = False
        elif k == ord("]"):
            state["frame"] = min(n_frames, state["frame"] + 30); state["playing"] = False
        elif k == ord(" "):
            state["playing"] = not state["playing"]
        elif 48 <= k <= 57:                            # digit -> merge-dest buffer
            state["input"] += chr(k)
        elif k == 8:                                   # backspace
            state["input"] = state["input"][:-1]
        elif key in (13, 10):                          # Enter -> merge
            if state["armed"] is None:
                state["status"] = "no ID armed"
            else:
                dst = int(state["input"]) if state["input"] else state["target"]
                n = store.merge_forward(state["armed"], dst, state["frame"])
                state["status"] = f"merged ID {state['armed']} -> {dst} ({n} rows fwd)"
                print(state["status"]); state["input"] = ""
        elif k == ord("d"):                            # delete forward
            if state["armed"] is None:
                state["status"] = "no ID armed"
            elif state["armed"] == state["target"] and \
                    not _confirm_cv("Delete your own target ID? (y/n)", cv2, win):
                state["status"] = "delete cancelled"
            else:
                tid = state["armed"]; fr = state["frame"]
                n = store.delete_forward(tid, fr)
                state["status"] = f"Deleted ID {tid} from frame {fr} onward ({n:,} rows)"
                print(state["status"])
        elif k == ord("D"):                            # nuke entire track
            if state["armed"] is None:
                state["status"] = "no ID armed"
            elif state["armed"] == state["target"] and \
                    not _confirm_cv("Delete your own target ID? (y/n)", cv2, win):
                state["status"] = "nuke cancelled"
            else:
                tid = state["armed"]
                n = store.nuke(tid)
                state["status"] = f"Nuked ID {tid} entirely ({n:,} rows)"
                print(state["status"])
        elif k == ord("i"):                            # manage IDs dialog
            manage_ids_dialog()
        elif k == ord("t"):                            # set target
            if state["armed"] is not None:
                state["target"] = state["armed"]
                state["status"] = f"target = {state['target']}"
        elif k == ord("u"):                            # undo
            ok = store.undo()
            state["status"] = "undo" if ok else "nothing to undo"
            print("[reanchor]", state["status"])
        elif k == ord("s"):                            # save
            out = store.save()
            state["status"] = f"saved -> {out}"
            print("[reanchor]", state["status"])
        elif k == 27:                                  # esc
            state["armed"] = None; state["input"] = ""; state["status"] = "cleared"
        elif k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if tk_root["root"] is not None:
        try:
            tk_root["root"].destroy()
        except Exception:
            pass


def _confirm_cv(prompt, cv2, win):
    """Tiny y/n confirm overlaid on the OpenCV window."""
    print("[reanchor]", prompt)
    while True:
        k = cv2.waitKeyEx(0) & 0xFF
        if k in (ord("y"), ord("Y")):
            return True
        if k in (ord("n"), ord("N"), 27):
            return False


def main():
    ap = argparse.ArgumentParser(description="Manual re-anchor / track cleanup GUI")
    ap.add_argument("--video", help="source video (required for GUI)")
    ap.add_argument("--csv", required=True, help="tracks CSV from track.py")
    ap.add_argument("--id", type=int, default=1, help="canonical target track ID (default 1)")
    ap.add_argument("--selftest", action="store_true", help="run headless logic test and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.csv)
        return

    if not args.video:
        sys.exit("--video is required to launch the GUI (or use --selftest)")
    store = TrackStore(args.csv)
    run_gui(args.video, store, args.id)


if __name__ == "__main__":
    main()
