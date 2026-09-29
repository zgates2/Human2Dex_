#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageTk

try:
    import tkinter as tk
except Exception as exc:  # pragma: no cover
    tk = None
    TK_IMPORT_ERROR = exc
else:
    TK_IMPORT_ERROR = None

from common import (
    clip_bbox,
    list_episode_dirs,
    list_images,
    load_annotations,
    normalize_bbox,
    parse_bbox,
    save_annotations,
)


class RoiAnnotator:
    def __init__(
        self,
        image_path: Path,
        existing_bbox=None,
        existing_exclude_bboxes=None,
        max_display_size=980,
    ):
        if tk is None:
            raise RuntimeError(f"tkinter is not available: {TK_IMPORT_ERROR}")
        self.image_path = image_path
        self.image = Image.open(image_path).convert("RGB")
        self.width, self.height = self.image.size
        scale = min(max_display_size / self.width, max_display_size / self.height, 1.0)
        self.scale = scale
        display_size = (int(self.width * scale), int(self.height * scale))
        self.display = self.image.resize(display_size, Image.Resampling.LANCZOS)

        self.root = tk.Tk()
        self.root.title(f"ROI: {image_path.parent.parent.name}")
        self.photo = ImageTk.PhotoImage(self.display)
        self.canvas = tk.Canvas(
            self.root, width=display_size[0], height=display_size[1], cursor="crosshair"
        )
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.status = tk.StringVar()
        self.status.set(
            "Mode=glove. Drag main glove box. e=exclude mode, g=glove mode, Enter=save."
        )
        label = tk.Label(self.root, textvariable=self.status, anchor="w")
        label.pack(fill="x")

        self.start = None
        self.active_rect_id = None
        self.main_rect_id = None
        self.bbox = normalize_bbox(existing_bbox) if existing_bbox else None
        self.exclude_bboxes = [
            normalize_bbox(b) for b in (existing_exclude_bboxes or [])
        ]
        self.exclude_rect_ids = []
        self.mode = "glove"
        self.result = None
        self.quit_requested = False

        if self.bbox:
            self._draw_main_bbox(self.bbox)
        for exclude_bbox in self.exclude_bboxes:
            self._draw_exclude_bbox(exclude_bbox)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.root.bind("<Return>", self._on_accept)
        self.root.bind("r", self._on_reset)
        self.root.bind("e", self._on_exclude_mode)
        self.root.bind("g", self._on_glove_mode)
        self.root.bind("u", self._on_undo_exclude)
        self.root.bind("s", self._on_skip)
        self.root.bind("q", self._on_quit)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

    def _to_image_coords(self, x, y):
        return int(round(x / self.scale)), int(round(y / self.scale))

    def _to_canvas_coords(self, bbox):
        x0, y0, x1, y1 = bbox
        return [v * self.scale for v in (x0, y0, x1, y1)]

    def _draw_main_bbox(self, bbox):
        if self.main_rect_id is not None:
            self.canvas.delete(self.main_rect_id)
        self.main_rect_id = self.canvas.create_rectangle(
            *self._to_canvas_coords(bbox), outline="#00ffff", width=2
        )

    def _draw_exclude_bbox(self, bbox):
        rect_id = self.canvas.create_rectangle(
            *self._to_canvas_coords(bbox), outline="#ffb000", width=2
        )
        self.exclude_rect_ids.append(rect_id)

    def _set_status(self):
        self.status.set(
            f"Mode={self.mode}. main={self.bbox}, excludes={len(self.exclude_bboxes)}. "
            "g=glove, e=exclude, u=undo exclude, r=reset, Enter=save, s=skip, q=quit."
        )

    def _on_press(self, event):
        self.start = (event.x, event.y)
        if self.active_rect_id is not None:
            self.canvas.delete(self.active_rect_id)
            self.active_rect_id = None
        if self.main_rect_id is not None and self.mode == "glove":
            self.canvas.delete(self.main_rect_id)
            self.main_rect_id = None
        self.active_rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#00ffff", width=2
        )
        if self.mode == "exclude":
            self.canvas.itemconfig(self.active_rect_id, outline="#ffb000")

    def _on_drag(self, event):
        if self.start is None or self.active_rect_id is None:
            return
        self.canvas.coords(
            self.active_rect_id, self.start[0], self.start[1], event.x, event.y
        )

    def _on_release(self, event):
        if self.start is None:
            return
        x0, y0 = self._to_image_coords(*self.start)
        x1, y1 = self._to_image_coords(event.x, event.y)
        bbox = clip_bbox((x0, y0, x1, y1), self.width, self.height)
        if self.mode == "exclude":
            self.exclude_bboxes.append(bbox)
            self.exclude_rect_ids.append(self.active_rect_id)
            self.active_rect_id = None
        else:
            self.bbox = bbox
            if self.active_rect_id is not None:
                self.canvas.delete(self.active_rect_id)
                self.active_rect_id = None
            self._draw_main_bbox(self.bbox)
        self._set_status()
        self.start = None

    def _on_accept(self, _event=None):
        if not self.bbox:
            self.status.set("No bbox yet. Drag a box around the glove first.")
            return
        self.result = {"bbox": self.bbox, "exclude_bboxes": self.exclude_bboxes}
        self.root.destroy()

    def _on_reset(self, _event=None):
        self.bbox = None
        if self.active_rect_id is not None:
            self.canvas.delete(self.active_rect_id)
            self.active_rect_id = None
        if self.main_rect_id is not None:
            self.canvas.delete(self.main_rect_id)
            self.main_rect_id = None
        for rect_id in self.exclude_rect_ids:
            self.canvas.delete(rect_id)
        self.exclude_rect_ids = []
        self.exclude_bboxes = []
        self._set_status()

    def _on_exclude_mode(self, _event=None):
        self.mode = "exclude"
        self._set_status()

    def _on_glove_mode(self, _event=None):
        self.mode = "glove"
        self._set_status()

    def _on_undo_exclude(self, _event=None):
        if self.exclude_rect_ids:
            self.canvas.delete(self.exclude_rect_ids.pop())
        if self.exclude_bboxes:
            self.exclude_bboxes.pop()
        self._set_status()

    def _on_skip(self, _event=None):
        self.result = None
        self.root.destroy()

    def _on_quit(self, _event=None):
        self.quit_requested = True
        self.result = None
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.result, self.quit_requested


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Annotate one glove ROI bbox on the first frame of each episode."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input dataset root.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to roi_annotations.json.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-annotate episodes already in JSON."
    )
    parser.add_argument("--limit", type=int, default=None, help="Annotate at most N episodes.")
    parser.add_argument(
        "--episode",
        default=None,
        help="Only annotate one episode name, e.g. demo_20260521_171105_ep0006.",
    )
    parser.add_argument(
        "--bbox",
        default=None,
        help="Non-GUI mode: assign x0,y0,x1,y1 to selected episodes.",
    )
    parser.add_argument(
        "--exclude-bbox",
        action="append",
        default=[],
        help="Non-GUI mode: exclude x0,y0,x1,y1. Repeat for multiple boxes.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    data = load_annotations(args.output)
    episodes = list_episode_dirs(args.input)
    if args.episode:
        episodes = [ep for ep in episodes if ep.name == args.episode]
        if not episodes:
            raise SystemExit(f"Episode not found: {args.episode}")
    if args.limit is not None:
        episodes = episodes[: args.limit]

    if args.bbox:
        bbox = parse_bbox(args.bbox)
        exclude_bboxes = [parse_bbox(v) for v in args.exclude_bbox]
        for ep in episodes:
            images = list_images(ep)
            if not images:
                print(f"skip {ep.name}: no images")
                continue
            width, height = Image.open(images[0]).size
            clipped = clip_bbox(bbox, width, height)
            clipped_excludes = [clip_bbox(b, width, height) for b in exclude_bboxes]
            data["episodes"][ep.name] = {
                "bbox": list(clipped),
                "exclude_bboxes": [list(b) for b in clipped_excludes],
                "image": str(images[0].relative_to(args.input)),
                "width": width,
                "height": height,
            }
            print(f"{ep.name}: {clipped}, excludes={clipped_excludes}")
        save_annotations(args.output, data)
        return

    for idx, ep in enumerate(episodes, start=1):
        if ep.name in data["episodes"] and not args.overwrite:
            print(f"[{idx}/{len(episodes)}] skip annotated {ep.name}")
            continue
        images = list_images(ep)
        if not images:
            print(f"[{idx}/{len(episodes)}] skip {ep.name}: no images")
            continue
        existing_item = data["episodes"].get(ep.name, {})
        existing = existing_item.get("bbox")
        existing_excludes = existing_item.get("exclude_bboxes", [])
        print(f"[{idx}/{len(episodes)}] annotate {ep.name}: {images[0]}")
        annotator = RoiAnnotator(
            images[0],
            existing_bbox=existing,
            existing_exclude_bboxes=existing_excludes,
        )
        result, quit_requested = annotator.run()
        if quit_requested:
            save_annotations(args.output, data)
            print("quit requested; saved current annotations")
            return
        if result is None:
            print(f"skipped {ep.name}")
            continue
        bbox = result["bbox"]
        exclude_bboxes = result["exclude_bboxes"]
        width, height = Image.open(images[0]).size
        data["episodes"][ep.name] = {
            "bbox": list(clip_bbox(bbox, width, height)),
            "exclude_bboxes": [
                list(clip_bbox(b, width, height)) for b in exclude_bboxes
            ],
            "image": str(images[0].relative_to(args.input)),
            "width": width,
            "height": height,
        }
        save_annotations(args.output, data)
        print(f"saved {ep.name}: {bbox}, excludes={len(exclude_bboxes)}")

    save_annotations(args.output, data)
    print(f"done: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
