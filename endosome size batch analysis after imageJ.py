"""
Endosome Batch Analyzer GUI with original/dilated mask outlines
---------------------------
Use with the Fiji macro output folder (fiji_batch_neurite_line_rois.ijm) containing:
- metadata_per_neurite.csv
- * __endosome.tif
- * __mask.tif
- optionally *__cytoskeleton.tif

What it does:
1) Loads up to N ROI crops for side-by-side preview.
2) Applies ONE shared parameter set to all preview ROIs, so settings are comparable.
3) Restricts endosome segmentation to the neurite mask from Fiji.
4) Allows to adjust parameters and visually inspect best settings before
4) Applies final parameters to all ROI crops.
5) Exports:
   - summary_per_neurite.csv
   - all_endosome_measurements_long.csv
   - endosome_sizes_by_original_image.csv
   - optional label masks per ROI

Install:
    pip install numpy pandas scipy scikit-image tifffile matplotlib pillow

Run:
    python endosome_batch_gui.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image, ImageTk
from scipy import ndimage as ndi
from skimage import exposure, feature, filters, measure, morphology, segmentation


def read_gray(path: Path) -> np.ndarray:
    arr = tiff.imread(long_path_str(path))
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # If RGB-like, use first channel. Fiji macro should save single-channel TIFFs.
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        else:
            arr = arr[0]
    return arr.astype(np.float32)


def robust_norm(img: np.ndarray, p1=1, p99=99.7) -> np.ndarray:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(finite, [p1, p99])
    if hi <= lo:
        hi = lo + 1
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return out.astype(np.float32)




def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first existing column name from a list of aliases."""
    cols = {str(c).strip(): c for c in df.columns}
    for name in candidates:
        if name in cols:
            return cols[name]
    raise KeyError(
        "None of these required metadata columns were found: "
        + ", ".join(candidates)
        + "\nAvailable columns: "
        + ", ".join(map(str, df.columns))
    )



def long_path_str(path: Path | str) -> str:
    """Return a string path that is safer for very long Windows paths."""
    p = Path(path)
    s = str(p)
    if os.name == "nt":
        s = s.replace("/", "\\")
        # Add Windows extended-length prefix for long absolute paths.
        if len(s) >= 240 and not s.startswith("\\\\?\\"):
            if s.startswith("\\\\"):
                # UNC path: \\server\share -> \\?\UNC\server\share
                s = "\\\\?\\UNC\\" + s.lstrip("\\")
            else:
                s = "\\\\?\\" + s
    return s

def _norm_name(s: str) -> str:
    """Normalize filenames for fuzzy matching when paths were moved/renamed."""
    return (str(s)
            .replace("\\", "/")
            .split("/")[-1]
            .replace(" ", "")
            .replace("__", "_")
            .lower())


def resolve_file(base_folder: Path, value) -> Path:
    """Resolve a file path from the Fiji metadata CSV.

    Tries, in order:
    1) exact path from the CSV,
    2) selected output folder + filename,
    3) recursive search for the exact filename below selected output folder,
    4) recursive fuzzy filename match allowing moved folders, extra spaces, or double underscores.
    """
    if pd.isna(value):
        raise FileNotFoundError("Missing file path in metadata CSV")

    raw = str(value).strip().strip('"').strip("'")
    path = Path(raw)

    # 1. Exact path from CSV.
    if path.exists():
        return path

    # 2. Same filename inside the selected Fiji output folder.
    filename = path.name
    candidate = base_folder / filename
    if candidate.exists():
        return candidate

    # 3. Exact filename search anywhere under the selected Fiji output folder.
    matches = list(base_folder.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple files named {filename} found under {base_folder}; metadata path is ambiguous."
        )

    # 4. Fuzzy fallback: tolerate double underscores, spaces, and moved folders.
    target = _norm_name(filename)
    fuzzy_matches = []
    for f in base_folder.rglob("*.tif"):
        if _norm_name(f.name) == target:
            fuzzy_matches.append(f)
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0]
    if len(fuzzy_matches) > 1:
        raise FileNotFoundError(
            f"Multiple fuzzy matches for {filename} found under {base_folder}; metadata path is ambiguous."
        )

    raise FileNotFoundError(
        "Could not locate file from metadata CSV.\n"
        f"CSV value: {raw}\n"
        f"Selected folder: {base_folder}\n"
        f"Expected filename: {filename}"
    )


def segment_endosomes(
    img: np.ndarray,
    mask: np.ndarray,
    detection_method: str,
    bg_sigma: float,
    smooth_sigma: float,
    dog_low: float,
    dog_high: float,
    peak_percentile: float,
    min_distance: int,
    fg_percentile: float,
    min_size: int,
    max_size: int,
    mask_dilation_px: int,
    seed_dilation_px: int,
    foreground_open_px: int,
    foreground_close_px: int,
    watershed_compactness: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, peak coords, measurement dataframe, spot-enhancement image, foreground mask, analysis mask."""
    mask_bool = mask > 0
    if int(mask_dilation_px) > 0:
        mask_bool = morphology.binary_dilation(mask_bool, morphology.disk(int(mask_dilation_px)))
    if mask_bool.shape != img.shape:
        raise ValueError(f"Mask shape {mask_bool.shape} does not match image shape {img.shape}")

    work = img.copy().astype(np.float32)
    work[~mask_bool] = np.nanmedian(work[mask_bool]) if np.any(mask_bool) else 0

    if bg_sigma > 0:
        bg = filters.gaussian(work, sigma=bg_sigma, preserve_range=True)
        work = work - bg
    if smooth_sigma > 0:
        work = filters.gaussian(work, sigma=smooth_sigma, preserve_range=True)

    # Spot-enhancement image used for peak detection, foreground thresholding, and watershed.
    # DoG = Difference of Gaussians.
    # LoG = Laplacian of Gaussian; for bright blobs we invert the response so spots are positive.
    method = str(detection_method).strip().lower()
    if method == "log":
        dog = -ndi.gaussian_laplace(work, sigma=max(0.1, float(dog_low)))
    else:
        dog = filters.difference_of_gaussians(work, low_sigma=dog_low, high_sigma=dog_high)
    dog[~mask_bool] = np.min(dog[mask_bool]) if np.any(mask_bool) else 0

    vals = dog[mask_bool]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        labels = np.zeros_like(img, dtype=np.int32)
        return labels, np.empty((0, 2), int), pd.DataFrame(), dog, np.zeros_like(mask_bool), mask_bool

    peak_thr = np.percentile(vals, peak_percentile)
    coords = feature.peak_local_max(
        dog,
        min_distance=max(1, int(min_distance)),
        threshold_abs=peak_thr,
        labels=mask_bool.astype(np.uint8),
        exclude_border=False,
    )

    markers = np.zeros(img.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i

    fg_thr = np.percentile(vals, fg_percentile)
    foreground = (dog > fg_thr) & mask_bool
    foreground = morphology.remove_small_objects(foreground, min_size=max(1, int(min_size)))
    if int(foreground_open_px) > 0:
        foreground = morphology.binary_opening(foreground, morphology.disk(int(foreground_open_px)))
    if int(foreground_close_px) > 0:
        foreground = morphology.binary_closing(foreground, morphology.disk(int(foreground_close_px)))
    foreground = foreground & mask_bool

    if coords.size == 0 or not foreground.any():
        labels = np.zeros_like(img, dtype=np.int32)
    else:
        # Seeded watershed on -spot-enhancement image, restricted to foreground.
        # Seed dilation ensures pixels immediately surrounding each detected peak are part of the object region.
        seed_radius = max(0, int(seed_dilation_px))
        seed_zone = morphology.binary_dilation(markers > 0, morphology.disk(seed_radius)) if seed_radius > 0 else (markers > 0)
        foreground = foreground | (seed_zone & mask_bool)
        labels = segmentation.watershed(
            -dog,
            markers=markers,
            mask=foreground,
            compactness=max(0.0, float(watershed_compactness)),
        )

    # Remove too-small/too-large labels and relabel sequentially.
    if labels.max() > 0:
        props_tmp = measure.regionprops(labels)
        keep = np.zeros(labels.max() + 1, dtype=bool)
        for p in props_tmp:
            keep[p.label] = (p.area >= min_size) and (p.area <= max_size)
        labels = keep[labels] * labels
        labels = measure.label(labels > 0, connectivity=1).astype(np.int32)

    props = measure.regionprops_table(
        labels,
        intensity_image=img,
        properties=(
            "label", "area", "centroid", "mean_intensity", "max_intensity",
            "min_intensity", "equivalent_diameter", "major_axis_length", "minor_axis_length"
        ),
    )
    df = pd.DataFrame(props)
    return labels, coords, df, dog, foreground, mask_bool


class ZoomViewer:
    """Scrollable zoom window for inspecting one ROI segmentation overlay."""
    def __init__(self, parent, title: str, images: dict[str, np.ndarray], initial: str = "Overlay"):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("1100x850")
        self.images = images
        self.photo = None

        controls = ttk.Frame(self.top)
        controls.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        ttk.Label(controls, text="View:").pack(side=tk.LEFT)
        self.view_var = tk.StringVar(value=initial if initial in images else list(images.keys())[0])
        view_menu = ttk.OptionMenu(controls, self.view_var, self.view_var.get(), *images.keys(), command=lambda _: self.redraw())
        view_menu.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(controls, text="Zoom:").pack(side=tk.LEFT)
        self.zoom_var = tk.DoubleVar(value=3.0)
        zoom = ttk.Scale(controls, from_=0.5, to=12.0, variable=self.zoom_var, command=lambda _: self.redraw())
        zoom.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.zoom_label = ttk.Label(controls, text="3.0×", width=8)
        self.zoom_label.pack(side=tk.LEFT)

        ttk.Button(controls, text="Fit", command=self.fit).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="100%", command=lambda: self.set_zoom(1.0)).pack(side=tk.LEFT, padx=4)

        canvas_frame = ttk.Frame(self.top)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="black")
        self.hbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.redraw()

    def set_zoom(self, value: float):
        self.zoom_var.set(float(value))
        self.redraw()

    def fit(self):
        img = self.images[self.view_var.get()]
        h, w = img.shape[:2]
        cw = max(1, self.canvas.winfo_width() - 20)
        ch = max(1, self.canvas.winfo_height() - 20)
        self.set_zoom(max(0.1, min(cw / w, ch / h)))

    def on_mousewheel(self, event):
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        new_zoom = min(12.0, max(0.5, self.zoom_var.get() * factor))
        self.set_zoom(new_zoom)

    def redraw(self):
        img = self.images[self.view_var.get()]
        if img.dtype != np.uint8:
            if img.ndim == 2:
                img = (robust_norm(img) * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img)
        z = float(self.zoom_var.get())
        new_size = (max(1, int(pil.width * z)), max(1, int(pil.height * z)))
        pil = pil.resize(new_size, Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, new_size[0], new_size[1]))
        self.zoom_label.configure(text=f"{z:.1f}×")


class BatchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch endosome size analyzer v9 - mask outlines")
        self.root.geometry("1500x900")

        self.folder: Path | None = None
        self.meta: pd.DataFrame | None = None
        self.preview_rows: pd.DataFrame | None = None
        self.preview_results = []
        self.photo_refs = []

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.Frame(main, width=360)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Button(left, text="Load Fiji output folder", command=self.load_folder).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Update preview", command=self.update_preview).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Run all + export tables", command=self.run_all).pack(fill=tk.X, pady=3)

        self.vars = {}
        self._entry(left, "Preview ROI count", "preview_count", "5")
        self._entry(left, "Pixel size (µm/px)", "pixel_size", "0.119701")
        ttk.Separator(left).pack(fill=tk.X, pady=8)
        ttk.Label(left, text="Segmentation parameters", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(left, text="Detection filter").pack(anchor="w", pady=(5, 0))
        self.detection_method = tk.StringVar(value="DoG")
        ttk.OptionMenu(left, self.detection_method, "DoG", "DoG", "LoG").pack(fill=tk.X)
        self._entry(left, "Background sigma", "bg_sigma", "8")
        self._entry(left, "Smooth sigma", "smooth_sigma", "0.5")
        self._entry(left, "DoG low sigma / LoG sigma", "dog_low", "1.0")
        self._entry(left, "DoG high sigma (ignored for LoG)", "dog_high", "3.0")
        self._entry(left, "Peak percentile", "peak_percentile", "98.5")
        self._entry(left, "Peak min distance (px)", "min_distance", "3")
        self._entry(left, "Foreground percentile", "fg_percentile", "95")
        self._entry(left, "Min object size (px²)", "min_size", "3")
        self._entry(left, "Max object size (px²)", "max_size", "300")
        ttk.Separator(left).pack(fill=tk.X, pady=8)
        ttk.Label(left, text="Mask / watershed controls", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self._entry(left, "Dilate neurite mask (px)", "mask_dilation_px", "2")
        self._entry(left, "Seed dilation radius (px)", "seed_dilation_px", "1")
        self._entry(left, "Foreground opening radius (px)", "foreground_open_px", "0")
        self._entry(left, "Foreground closing radius (px)", "foreground_close_px", "0")
        self._entry(left, "Watershed compactness", "watershed_compactness", "0")

        ttk.Separator(left).pack(fill=tk.X, pady=8)
        self.save_masks = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Save label masks", variable=self.save_masks).pack(anchor="w")
        self.status = tk.StringVar(value="Load Fiji output folder containing neurite_metadata.csv")
        ttk.Label(left, textvariable=self.status, wraplength=340).pack(anchor="w", pady=8)

        # scrollable preview area
        canvas = tk.Canvas(right, bg="white")
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=canvas.yview)
        self.preview_frame = ttk.Frame(canvas)
        self.preview_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.preview_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _entry(self, parent, label, key, default):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(5, 0))
        v = tk.StringVar(value=str(default))
        self.vars[key] = v
        ttk.Entry(parent, textvariable=v).pack(fill=tk.X)

    def params(self):
        return dict(
            detection_method=self.detection_method.get(),
            pixel_size=float(self.vars["pixel_size"].get()),
            bg_sigma=float(self.vars["bg_sigma"].get()),
            smooth_sigma=float(self.vars["smooth_sigma"].get()),
            dog_low=float(self.vars["dog_low"].get()),
            dog_high=float(self.vars["dog_high"].get()),
            peak_percentile=float(self.vars["peak_percentile"].get()),
            min_distance=int(float(self.vars["min_distance"].get())),
            fg_percentile=float(self.vars["fg_percentile"].get()),
            min_size=int(float(self.vars["min_size"].get())),
            max_size=int(float(self.vars["max_size"].get())),
            mask_dilation_px=int(float(self.vars["mask_dilation_px"].get())),
            seed_dilation_px=int(float(self.vars["seed_dilation_px"].get())),
            foreground_open_px=int(float(self.vars["foreground_open_px"].get())),
            foreground_close_px=int(float(self.vars["foreground_close_px"].get())),
            watershed_compactness=float(self.vars["watershed_compactness"].get()),
        )

    def load_folder(self):
        folder = filedialog.askdirectory(title="Choose Fiji output folder")
        if not folder:
            return
        self.folder = Path(folder)

        # Accept both the older Python-script name and the Fiji macro name.
        candidates = [self.folder / "neurite_metadata.csv", self.folder / "metadata_per_neurite.csv"]
        meta_path = next((x for x in candidates if x.exists()), None)
        if meta_path is None:
            messagebox.showerror(
                "Missing metadata",
                "Could not find neurite_metadata.csv or metadata_per_neurite.csv in this folder."
            )
            return

        self.meta = pd.read_csv(meta_path).dropna(how="all")
        self.meta.columns = [str(c).strip() for c in self.meta.columns]

        # Column aliases: the Fiji macro currently writes *_crop_file; older scripts used shorter names.
        try:
            self.endosome_col = first_existing_column(self.meta, ["endosome_crop_file", "endosome_file", "Endosomes", "endosomes_file"])
            self.mask_col = first_existing_column(self.meta, ["mask_file", "neurite_mask_file", "mask_crop_file"])
            self.prph_col = first_existing_column(self.meta, ["prph_crop_file", "prph_file", "PRPH_file", "cytoskeleton_file"])
        except KeyError as e:
            messagebox.showerror("Metadata column error", str(e))
            return

        if "neurite_id" in self.meta.columns:
            self.meta["neurite_id"] = self.meta["neurite_id"].astype(str).str.extract(r"(\d+)", expand=False).fillna(self.meta["neurite_id"].astype(str)).str.zfill(3)
        else:
            self.meta["neurite_id"] = [str(i + 1).zfill(3) for i in range(len(self.meta))]

        n = min(int(float(self.vars["preview_count"].get())), len(self.meta))
        self.preview_rows = self.meta.head(n).copy()
        self.status.set(f"Loaded {len(self.meta)} neurite ROIs from {meta_path.name}")
        self.update_preview()

    def update_preview(self):
        if self.meta is None or self.folder is None:
            return
        try:
            p = self.params()
        except Exception as e:
            messagebox.showerror("Parameter error", str(e))
            return
        n = min(int(float(self.vars["preview_count"].get())), len(self.meta))
        self.preview_rows = self.meta.head(n).copy()
        for child in self.preview_frame.winfo_children():
            child.destroy()
        self.photo_refs.clear()
        self.preview_results = []

        total_spots = 0
        for idx, row in self.preview_rows.iterrows():
            frame = ttk.LabelFrame(self.preview_frame, text=f"{row['original_image']} — neurite {row['neurite_id']}")
            frame.pack(fill=tk.X, padx=6, pady=6)
            try:
                endo = read_gray(resolve_file(self.folder, row[self.endosome_col]))
                mask = read_gray(resolve_file(self.folder, row[self.mask_col]))
                labels, coords, df, dog, fg, analysis_mask = segment_endosomes(
                    endo, mask, p["detection_method"],
                    p["bg_sigma"], p["smooth_sigma"], p["dog_low"], p["dog_high"],
                    p["peak_percentile"], p["min_distance"], p["fg_percentile"],
                    p["min_size"], p["max_size"],
                    p["mask_dilation_px"], p["seed_dilation_px"],
                    p["foreground_open_px"], p["foreground_close_px"],
                    p["watershed_compactness"]
                )
                total_spots += len(df)
                rgb = self.make_overlay(endo, mask, analysis_mask, labels, coords)
                zoom_images = self.make_zoom_images(endo, mask, analysis_mask, labels, coords)
                title = f"{row['original_image']} — neurite {row['neurite_id']}"
                self.preview_results.append((title, zoom_images))
                result_index = len(self.preview_results) - 1

                imgtk = self.to_tk(rgb, max_w=900, max_h=260)
                self.photo_refs.append(imgtk)
                ttk.Label(frame, image=imgtk).pack(side=tk.LEFT, padx=5, pady=5)
                mean_area = df["area"].mean() if len(df) else np.nan
                eq = df["equivalent_diameter"].mean() if len(df) else np.nan
                txt = f"Objects: {len(df)}\nMean area: {mean_area:.2f} px²\nMean eq. diameter: {eq:.2f} px"
                info = ttk.Frame(frame)
                info.pack(side=tk.LEFT, padx=8, anchor="n")
                ttk.Label(info, text=txt).pack(anchor="w")
                ttk.Button(info, text="Open zoom viewer", command=lambda i=result_index: self.open_zoom_viewer(i)).pack(anchor="w", pady=(8, 0))
            except Exception as e:
                ttk.Label(frame, text=f"Error: {e}").pack(anchor="w", padx=8, pady=8)
        self.status.set(f"Preview updated: {n} ROIs, {total_spots} detected objects total.")



    def open_zoom_viewer(self, index: int):
        if index < 0 or index >= len(self.preview_results):
            return
        title, images = self.preview_results[index]
        ZoomViewer(self.root, title, images)

    def colorize_labels(self, labels: np.ndarray) -> np.ndarray:
        """Return deterministic RGB colors for final watershed labels. Background is black."""
        h, w = labels.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        max_label = int(labels.max())
        if max_label == 0:
            return (rgb * 255).astype(np.uint8)

        # Deterministic pseudo-random colors generated from label IDs.
        for lab in range(1, max_label + 1):
            m = labels == lab
            if not np.any(m):
                continue
            r = ((lab * 37) % 255) / 255.0
            g = ((lab * 91) % 255) / 255.0
            b = ((lab * 53) % 255) / 255.0
            # Avoid very dark colors.
            rgb[m, 0] = 0.35 + 0.65 * r
            rgb[m, 1] = 0.35 + 0.65 * g
            rgb[m, 2] = 0.35 + 0.65 * b
        return (rgb * 255).astype(np.uint8)

    def draw_peak_crosses(self, rgb: np.ndarray, coords: np.ndarray, radius: int = 2) -> np.ndarray:
        """Draw red crosses at seed coordinates."""
        out = rgb.copy()
        if out.dtype != np.float32:
            out = out.astype(np.float32) / 255.0
        for y, x in coords:
            y = int(y); x = int(x)
            if y < 0 or x < 0 or y >= out.shape[0] or x >= out.shape[1]:
                continue
            y0, y1 = max(0, y-radius), min(out.shape[0], y+radius+1)
            x0, x1 = max(0, x-radius), min(out.shape[1], x+radius+1)
            out[y, x0:x1, 0] = 1; out[y, x0:x1, 1:] = 0
            out[y0:y1, x, 0] = 1; out[y0:y1, x, 1:] = 0
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)

    def make_overlay(self, img, original_mask, analysis_mask, labels, coords):
        """Main preview: raw endosome image with FINAL watershed particle boundaries.

        Red crosses = seeds/local maxima.
        Green outlines = final watershed objects that are measured/exported.
        Blue outline = original Fiji neurite mask.
        Yellow/cyan outline = dilated analysis mask actually used.
        """
        base = robust_norm(img)
        rgb = np.dstack([base, base, base])

        # Original Fiji neurite mask boundary in blue.
        edge_original = morphology.binary_dilation(original_mask > 0) ^ morphology.binary_erosion(original_mask > 0)
        rgb[edge_original] = [0.0, 0.35, 1.0]

        # Dilated analysis mask boundary in yellow/cyan. This is the mask used for detection/measurement.
        edge_analysis = morphology.binary_dilation(analysis_mask > 0) ^ morphology.binary_erosion(analysis_mask > 0)
        # Show only the expanded/different part prominently when possible.
        rgb[edge_analysis] = [1.0, 1.0, 0.0]

        # Final watershed label boundaries in green. These are the measured particles.
        edges = segmentation.find_boundaries(labels, mode="outer") if labels.max() else np.zeros_like(original_mask, dtype=bool)
        rgb[edges, 1] = 1.0
        rgb[edges, 0] *= 0.2
        rgb[edges, 2] *= 0.2

        return self.draw_peak_crosses((rgb * 255).astype(np.uint8), coords, radius=2)

    def make_watershed_overlay(self, img, original_mask, analysis_mask, labels, coords, alpha: float = 0.45):
        """Colored final watershed labels over raw image."""
        base = robust_norm(img)
        raw_rgb = np.dstack([base, base, base])
        colored = self.colorize_labels(labels).astype(np.float32) / 255.0
        label_pixels = labels > 0
        out = raw_rgb.copy()
        out[label_pixels] = (1 - alpha) * out[label_pixels] + alpha * colored[label_pixels]

        # Strong final label boundaries.
        edges = segmentation.find_boundaries(labels, mode="outer") if labels.max() else np.zeros_like(original_mask, dtype=bool)
        out[edges] = [0.0, 1.0, 0.0]

        # Original Fiji neurite mask boundary in blue.
        edge_original = morphology.binary_dilation(original_mask > 0) ^ morphology.binary_erosion(original_mask > 0)
        out[edge_original] = [0.0, 0.35, 1.0]

        # Dilated analysis mask boundary in yellow.
        edge_analysis = morphology.binary_dilation(analysis_mask > 0) ^ morphology.binary_erosion(analysis_mask > 0)
        out[edge_analysis] = [1.0, 1.0, 0.0]

        return self.draw_peak_crosses((np.clip(out, 0, 1) * 255).astype(np.uint8), coords, radius=2)

    def make_zoom_images(self, img, original_mask, analysis_mask, labels, coords):
        """Images available in the zoom viewer."""
        overlay = self.make_overlay(img, original_mask, analysis_mask, labels, coords)
        watershed_overlay = self.make_watershed_overlay(img, original_mask, analysis_mask, labels, coords)
        raw = (robust_norm(img) * 255).astype(np.uint8)
        raw_rgb = np.dstack([raw, raw, raw])

        mask_u8 = ((original_mask > 0) * 255).astype(np.uint8)
        mask_rgb = np.dstack([mask_u8, mask_u8, mask_u8])
        analysis_mask_u8 = ((analysis_mask > 0) * 255).astype(np.uint8)
        analysis_mask_rgb = np.dstack([analysis_mask_u8, analysis_mask_u8, analysis_mask_u8])
        mask_comparison = np.zeros((*original_mask.shape, 3), dtype=np.uint8)
        mask_comparison[analysis_mask > 0] = [255, 255, 0]  # analysis/dilated mask
        mask_comparison[original_mask > 0] = [0, 90, 255]   # original Fiji mask

        colored_labels = self.colorize_labels(labels)

        boundaries = np.zeros_like(raw_rgb)
        if labels.max() > 0:
            edges = segmentation.find_boundaries(labels, mode="outer")
            boundaries[edges] = [0, 255, 0]

        return {
            "Watershed overlay": watershed_overlay,
            "Final boundaries + seeds": overlay,
            "Colored labels only": colored_labels,
            "Label boundaries only": boundaries,
            "Raw endosomes": raw_rgb,
            "Original Fiji mask": mask_rgb,
            "Dilated analysis mask": analysis_mask_rgb,
            "Mask comparison": mask_comparison,
        }

    def to_tk(self, rgb, max_w=900, max_h=260):
        im = Image.fromarray(rgb)
        im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(im)

    def run_all(self):
        if self.meta is None or self.folder is None:
            return
        try:
            p = self.params()
        except Exception as e:
            messagebox.showerror("Parameter error", str(e))
            return

        out_dir = self.folder / "python_endosome_results"
        out_dir.mkdir(exist_ok=True)
        labels_dir = out_dir / "label_masks"
        if self.save_masks.get():
            labels_dir.mkdir(exist_ok=True)

        all_rows = []
        summary_rows = []
        pixel_size = p["pixel_size"]

        for _, row in self.meta.iterrows():
            try:
                endo = read_gray(resolve_file(self.folder, row[self.endosome_col]))
                mask = read_gray(resolve_file(self.folder, row[self.mask_col]))
                labels, coords, df, dog, fg, analysis_mask = segment_endosomes(
                    endo, mask, p["detection_method"],
                    p["bg_sigma"], p["smooth_sigma"], p["dog_low"], p["dog_high"],
                    p["peak_percentile"], p["min_distance"], p["fg_percentile"],
                    p["min_size"], p["max_size"],
                    p["mask_dilation_px"], p["seed_dilation_px"],
                    p["foreground_open_px"], p["foreground_close_px"],
                    p["watershed_compactness"]
                )
                df = df.copy()
                if len(df):
                    df["area_um2"] = df["area"] * pixel_size ** 2
                    df["equivalent_diameter_um"] = df["equivalent_diameter"] * pixel_size
                    df["centroid_x_um"] = df["centroid-1"] * pixel_size
                    df["centroid_y_um"] = df["centroid-0"] * pixel_size
                    df["original_image"] = row["original_image"]
                    df["neurite_id"] = row["neurite_id"]
                    # Keep columns tidy
                    for col in ["prph_mean", "prph_integrated", "prph_area_um2", "prph_area_px", "mean_Peripherin", "integrated_density_Peripherin", "area_mask_um2"]:
                        if col in row.index:
                            df[col] = row[col]
                    all_rows.append(df)

                avg_size = df["area_um2"].mean() if len(df) else np.nan
                median_size = df["area_um2"].median() if len(df) else np.nan
                avg_diam = df["equivalent_diameter_um"].mean() if len(df) else np.nan
                summary_rows.append({
                    "original_image": row["original_image"],
                    "neurite_id": row["neurite_id"],
                    "prph_mean": row.get("prph_mean", row.get("mean_Peripherin", np.nan)),
                    "prph_integrated": row.get("prph_integrated", row.get("integrated_density_Peripherin", np.nan)),
                    "prph_area_um2": row.get("prph_area_um2", row.get("area_mask_um2", np.nan)),
                    "n_endosomes": len(df),
                    "average_size_endosomes_um2": avg_size,
                    "median_size_endosomes_um2": median_size,
                    "average_equivalent_diameter_um": avg_diam,
                    "endosome_crop_file": str(row[self.endosome_col]),
                    "mask_file": str(row[self.mask_col]),
                })

                if self.save_masks.get():
                    stem = resolve_file(self.folder, row[self.endosome_col]).stem.replace("__endosome", "").replace("_Endosomes", "")
                    tiff.imwrite(long_path_str(labels_dir / f"{stem}__labels.tif"), labels.astype(np.uint16))
            except Exception as e:
                summary_rows.append({
                    "original_image": row.get("original_image", ""),
                    "neurite_id": row.get("neurite_id", ""),
                    "error": str(e),
                })

        summary = pd.DataFrame(summary_rows)
        long = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

        summary_path = out_dir / "summary_per_neurite.csv"
        long_path = out_dir / "all_endosome_measurements_long.csv"
        wide_path = out_dir / "endosome_sizes_by_original_image.csv"
        params_path = out_dir / "parameters_used.csv"

        summary.to_csv(summary_path, index=False)
        long.to_csv(long_path, index=False)

        # Wide table: each column = original image name, populated with all endosome area_um2 values from that image, across all neurites.
        if len(long):
            wide_dict = {}
            for img_name, sub in long.groupby("original_image"):
                wide_dict[str(img_name)] = sub["area_um2"].reset_index(drop=True)
            wide = pd.DataFrame(wide_dict)
        else:
            wide = pd.DataFrame()
        wide.to_csv(wide_path, index=False)
        pd.DataFrame([p]).to_csv(params_path, index=False)

        self.status.set(f"Done. Results saved in {out_dir}")
        messagebox.showinfo("Done", f"Saved:\n{summary_path}\n{long_path}\n{wide_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BatchGUI(root)
    root.mainloop()
