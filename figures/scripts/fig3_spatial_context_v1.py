"""Linked 3D and orthogonal sections from the current complete predictor.

Question: where are the predicted oil morphologies within a fixed real volume?
The pre-existing WI validation interval, midpoint and central ROI are retained.
This is a spatial application view, not a comparative accuracy experiment.
All input files are read-only; no model inference or training is performed.
"""
from __future__ import annotations
from figure_io import load_snapshot, asset_directory

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image
from skimage.measure import marching_cubes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRED = ROOT / "实验/body_inference_locked_20260930/outputs/val_seed42"
RECORDS = ROOT / "实验/native_detail_decoder_20260930/cache/records.json"
STYLE = json.loads((HERE / "visual_style.json").read_text(encoding="utf-8"))
COLORS = STYLE["semantic_colors"]
NAMES = ["Background", "Rock", "Water", "Film oil", "Pore-corner", "Droplet", "Columnar", "Clustered"]
ROI = (384, 640, 384, 640)
FIRST, LAST, MID = 591, 759, 675
PITCH_MM = .01325
WIDTH_MM, HEIGHT_MM = 114.0, 88.0
INK = "#26313A"
PLATE = "#F6F7F9"
CUT = "#304658"
from audit_panel_alignment import require_matplotlib_panel_alignment


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_volume():
    return load_snapshot("spatial")


def add_mesh(ax, binary, color, alpha):
    if not binary.any() or binary.all():
        return dict(voxels=int(binary.sum()), vertices=0, faces=0)
    verts, faces, _, _ = marching_cubes(binary.astype(np.uint8), level=.5,
                                       spacing=(2*PITCH_MM,)*3, allow_degenerate=False)
    # Display-grid samples represent native voxel centres at indices 0,2,4,... .
    xyz = verts[:, [2, 1, 0]] + .5*PITCH_MM
    ax.add_collection3d(Poly3DCollection(
        xyz[faces], facecolors=color, alpha=alpha,
        shade=True, lightsource=LightSource(azdeg=315, altdeg=40),
        linewidth=0, rasterized=True))
    return dict(voxels=int(binary.sum()), vertices=len(verts), faces=len(faces))


def cut_face_quads(labels):
    """Rock context on the three exposed faces, using the fixed half-volume cut."""
    nz, ny, nx = labels.shape
    iz, iy, ix = nz//2, ny//2, nx//2
    step = 2*PITCH_MM
    zc, yc, xc = np.array([iz, iy, ix])*step + .5*PITCH_MM
    quads = []
    for y, xx in np.argwhere(labels[iz-1, :iy, ix:] == 1):
        x = xx+ix
        quads.append([[x*step, y*step, zc], [(x+1)*step, y*step, zc],
                      [(x+1)*step, (y+1)*step, zc], [x*step, (y+1)*step, zc]])
    for zz, y in np.argwhere(labels[iz:, :iy, ix-1] == 1):
        z = zz+iz
        quads.append([[xc, y*step, z*step], [xc, (y+1)*step, z*step],
                      [xc, (y+1)*step, (z+1)*step], [xc, y*step, (z+1)*step]])
    for zz, xx in np.argwhere(labels[iz:, iy, ix:] == 1):
        z, x = zz+iz, xx+ix
        quads.append([[x*step, yc, z*step], [(x+1)*step, yc, z*step],
                      [(x+1)*step, yc, (z+1)*step], [x*step, yc, (z+1)*step]])
    if quads:
        result = np.asarray(quads)
        extent_xyz = np.array([256, 256, 169])*PITCH_MM
        result = np.minimum(result, extent_xyz)
    else:
        result = np.empty((0, 4, 3))
    return result, (xc, yc, zc)


def volume_panel(ax, labels):
    display = labels[::2, ::2, ::2]
    iz, iy, ix = np.array(display.shape)//2
    open_labels = display.copy()
    open_labels[iz:, :iy, ix:] = 0
    quads, (xc, yc, zc) = cut_face_quads(display)
    if len(quads):
        ax.add_collection3d(Poly3DCollection(quads, facecolors=COLORS[1],
                            alpha=.82, linewidth=0, shade=True,
                            lightsource=LightSource(azdeg=315, altdeg=40), rasterized=True))
    layers = {"Water": add_mesh(ax, open_labels == 2, COLORS[2], .22)}
    for cid in range(7, 2, -1):
        layers[NAMES[cid]] = add_mesh(ax, open_labels == cid, COLORS[cid], .88)
    lx, ly, lz = np.asarray(labels.shape)[[2, 1, 0]]*PITCH_MM
    for z in (0, lz):
        ax.plot([0, lx, lx, 0, 0], [0, 0, ly, ly, 0], [z]*5,
                color="#AAB3BE", linewidth=.5, alpha=.75)
    for x in (0, lx):
        for y in (0, ly):
            ax.plot([x,x], [y,y], [0,lz], color="#AAB3BE", linewidth=.5, alpha=.75)
    outlines = [
        [[xc,0,zc],[lx,0,zc],[lx,yc,zc],[xc,yc,zc],[xc,0,zc]],
        [[xc,0,zc],[xc,yc,zc],[xc,yc,lz],[xc,0,lz],[xc,0,zc]],
        [[xc,yc,zc],[lx,yc,zc],[lx,yc,lz],[xc,yc,lz],[xc,yc,zc]],
    ]
    for points in outlines:
        v = np.asarray(points)
        ax.plot(v[:,0],v[:,1],v[:,2], color=CUT, linewidth=.55, alpha=.65)
    ax.set(xlim=(0,lx), ylim=(0,ly), zlim=(0,lz))
    ax.set_proj_type("ortho")
    ax.set_box_aspect((lx,ly,lz), zoom=1.05)
    ax.view_init(elev=24, azim=-55)
    ax.set_axis_off()
    ax.set_facecolor(PLATE)
    # Translate the exact orthographic projection of a 0.5-mm x segment into
    # the clear margin. Translation does not change its calibrated length.
    projected = np.array([proj3d.proj_transform(x, 0, 0, ax.get_proj())[:2]
                          for x in (.20, .70)])
    screen = ax.transAxes.inverted().transform(ax.transData.transform(projected))
    delta = screen[1]-screen[0]
    start = np.array([.10, .12])
    stop = start+delta
    ax.add_artist(Line2D([start[0],stop[0]], [start[1],stop[1]],
                        transform=ax.transAxes, color=INK, linewidth=1.7,
                        solid_capstyle="butt"))
    ax.text2D((start[0]+stop[0])/2, min(start[1],stop[1])-.025, "0.5 mm",
              transform=ax.transAxes, color=INK, fontsize=7.0, ha="center", va="top")
    return dict(display_stride_zyx=[2,2,2], iso_level=.5, layers=layers,
                rock_face_quads=len(quads), camera=dict(elevation=24, azimuth=-55, projection="orthographic"),
                removed_octant="z >= middle, y < middle, x >= middle (display only)",
                cut_planes_xyz_mm=[xc,yc,zc], physical_z_exaggeration=1.0)


def overlay(raw, labels):
    gray = np.repeat((raw.astype(float)/255)[...,None], 3, axis=-1)
    for cid in range(3,8):
        inside = labels == cid
        gray[inside] = .78*np.array(to_rgb(COLORS[cid])) + .22*gray[inside]
    return gray


def section(ax, rgb, width, height):
    ax.imshow(rgb, extent=(0,width,0,height), origin="lower", interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#8D969F")
        spine.set_linewidth(.45)
    x, y = width*.07, height*.10
    ax.plot([x,x+.5], [y,y], color="black", lw=2.8, solid_capstyle="butt")
    ax.plot([x,x+.5], [y,y], color="white", lw=1.35, solid_capstyle="butt")
    ax.text(x+.25, y+.07, "0.5 mm", color="white", fontsize=7.0,
            ha="center", va="bottom", bbox=dict(facecolor="black", edgecolor="none", pad=.3))




