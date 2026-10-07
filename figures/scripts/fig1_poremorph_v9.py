"""PoreMorph, drawn at the PR 114-mm text width from retained real assets.

The private concept study informs only the stacked composition. It supplies
no image, feature, label, topology, numerical result or final artwork pixel.
"""
from __future__ import annotations
from figure_io import load_snapshot, asset_directory

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.path import Path as MplPath
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ASSETS = asset_directory("fig1_real_assets")
WIDTH, HEIGHT = 114., 175.
INK, MUTED = "#26343A", "#58666B"
GLOBAL, PRIOR, NATIVE, BODY = "#A84F79", "#237E74", "#3C8D82", "#685185"
COLORS = json.loads((HERE / "visual_style.json").read_text(encoding="utf-8"))["semantic_colors"]
MASK = ListedColormap(COLORS)
# Activations retain their measured spatial pattern. Decorative branch colours
# never recolour raw CT, response values, semantic labels or retained 3D pixels.
GLOBAL_MAP = NATIVE_MAP = "gray"
from audit_panel_alignment import require_matplotlib_panel_alignment


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "fig1_poremorph_v9"
    meta = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    assert (meta["sample"], meta["split"], meta["slice_id"]) == ("WI", "val", 675)
    raw, low = [np.load(ASSETS/f"{n}.npy") for n in ("input_full", "input_low")]
    gm, nm, fm = [np.load(ASSETS/f"{n}.npy") for n in ("global_mask", "native_mask", "final_mask")]
    assert np.array_equal(nm >= 3, fm >= 3)
    assert np.array_equal(nm[nm < 3], fm[nm < 3])
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(WIDTH/25.4, HEIGHT/25.4), dpi=300)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0), aspect="equal")
    ax.axis("off")
    objects, edges, text_artists = [], [], []

    def text(x, y, value, color=INK, size=7, bold=False, ha="center", rotation=0):
        artist = ax.text(x, y, value, fontsize=size, color=color, ha=ha, va="center",
                         fontweight="bold" if bold else "normal", linespacing=1.20,
                         rotation=rotation, zorder=30)
        text_artists.append(artist)
        return artist

    def obj(key, kind, bounds, role, source=""):
        objects.append(dict(id=key, kind=kind, bounds_mm=list(bounds), role=role, source=source))

    def route(key, source, target, points, color=GLOBAL, dashed=False, width=.65):
        path = MplPath(points, [MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=4.7,
                                     lw=width, color=color, linestyle="--" if dashed else "-",
                                     shrinkA=0, shrinkB=0, zorder=9))
        edges.append(dict(id=key, source=source, target=target, points_mm=points,
                          direction="forward", style="dashed" if dashed else "solid"))

    def tile(key, array, x, y, side, kind="image", color=GLOBAL, stack=False, source=""):
        if stack:
            for offset in (.8, .4):
                ax.add_patch(Rectangle((x+offset, y-offset), side, side, ec=color,
                                       fc="white", lw=.32, zorder=2))
        if kind == "feature":
            lo, hi = np.percentile(array, (2, 99))
            display = np.clip((array-lo)/max(hi-lo, 1e-8), 0, 1)
            cmap, vmin, vmax = (NATIVE_MAP if color == NATIVE else GLOBAL_MAP), 0, 1
        elif kind == "mask":
            display, cmap, vmin, vmax = array, MASK, -.5, 7.5
        else:
            display, cmap, vmin, vmax = array, "gray", 0, 255
        ax.imshow(display, extent=(x, x+side, y+side, y), origin="upper",
                  interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, zorder=5)
        ax.add_patch(Rectangle((x,y),side,side,ec=color,fc="none",lw=.48,zorder=6))
        obj(key, "image", (x,y,x+side,y+side), kind, source)

    def feature(key, x, y, side, color=GLOBAL):
        name = key+"_energy_full" if key.startswith("global_") else key
        tile(key, np.load(ASSETS/f"{name}.npy"), x,y,side,"feature",color,True,name+".npy")

    def box(key,x,y,w,h,label,color=PRIOR):
        ax.add_patch(Rectangle((x,y),w,h,fc="white",ec=color,lw=.6,zorder=10))
        text(x+w/2,y+h/2,label,color)
        obj(key,"operation",(x,y,x+w,y+h),label)

    # Both modalities start at the same height. I is a named raw-image port.
    text(13,2.2,"Micro-CT image",GLOBAL,7.7,True)
    tile("image",raw,3,5,21,source="input_full.npy")
    text(14,28.3,"1024² · I",MUTED)
    y0,y1,x0,x1=meta["roi_native_y0y1x0x1"]
    ax.add_patch(Rectangle((3+x0*21/1024,5+y0*21/1024),(x1-x0)*21/1024,
                           (y1-y0)*21/1024,fc="none",ec="#F4CF59",lw=.7,zorder=8))
    text(89.5,2.2,"Sample prior",PRIOR,7.7,True)
    box("prior",68,5,43,17,"WI flooding stage\nDolomite rock\nDeionized water",PRIOR)
    tile("low",low,28,10,12,source="input_low.npy")
    text(34,24,"512²",MUTED)
    route("resize","image","low",[(24,15.5),(28,15.5)])
    box("text_encoder",94,28,17,9,"Text encoder\n2 blocks",PRIOR)
    route("prior_encode","prior","text_encoder",[(102.5,22),(102.5,28)],PRIOR)

    # Image tokens are separate from convolutional lateral features.
    tile("patches",low,28,28,8,source="input_low.npy")
    for q in range(1,4):
        ax.plot((28+2*q,28+2*q),(28,36),color="white",lw=.25,zorder=7)
        ax.plot((28,36),(28+2*q,28+2*q),color="white",lw=.25,zorder=7)
    text(32,39,"Patches",MUTED)
    route("patch_input","low","patches",[(40,16),(43,16),(43,26),(32,26),(32,28)])
    for k,x in enumerate((43,55,67,79),1):
        feature(f"global_block{k}",x,28,7)
        text(x+3.5,24.8,f"B{k}",GLOBAL,7,True)
    route("patch_embed","patches","global_block1",[(36,32),(43,32)])
    for k,x in enumerate((50,62,74),1):
        route(f"token_{k}_{k+1}",f"global_block{k}",f"global_block{k+1}",[(x,31.5),(x+5,31.5)])
    text(64.5,38.5,"Visual tokens · 4 blocks",GLOBAL)

    # Matched columns retain the true four decoder resolutions.
    columns=(8.,31.,54.,77.)
    for k,(x,res) in enumerate(zip(columns,(512,256,128,64)),1):
        feature(f"global_stem{k}",x,44,9)
        feature(f"global_dec{k}",x,65,9)
        text(x+4.5,41.5,f"{res}²",MUTED)
        if k>1:
            route(f"stem_{k-1}_{k}",f"global_stem{k-1}",f"global_stem{k}",
                  [(columns[k-2]+9,48.5),(x,48.5)])
            route(f"dec_{k}_{k-1}",f"global_dec{k}",f"global_dec{k-1}",
                  [(x,69.5),(columns[k-2]+9,69.5)])
        if k==1:
            route("lateral1","global_stem1","global_dec1",[(12.5,53),(12.5,65)])
        else:
            cx=x+4.5
            route(f"lat{k}_in",f"global_stem{k}",f"sum{k}",[(cx,53),(cx,57.7)])
            ax.add_patch(Circle((cx,59.2),1.5,fc="white",ec=GLOBAL,lw=.5,zorder=11))
            text(cx,59.2,"+",GLOBAL,7)
            obj(f"sum{k}","operation",(cx-1.5,57.7,cx+1.5,60.7),"scaled visual-token addition")
            token={2:1,3:3,4:4}[k]
            text(cx+6.2,59.2,f"B{token}",GLOBAL,7,True)
            route(f"token_to_lat{k}",f"global_block{token}",f"sum{k}",[(cx+3.5,59.2),(cx+1.5,59.2)])
            route(f"lat{k}_out",f"sum{k}",f"global_dec{k}",[(cx,60.7),(cx,65)])
    route("conv_input","low","global_stem1",[(28,16),(26,16),(26,33),(5,33),(5,48.5),(8,48.5)])
    text(45,78.1,"Conditioned decoding",GLOBAL)

    feature("global_attention",95,44,10)
    text(109,49,"Cross-attn",PRIOR,rotation=90)
    route("Q","global_block4","global_attention",[(86,31.5),(91,31.5),(91,49),(95,49)])
    text(88,40.5,"Q",GLOBAL)
    route("KV","text_encoder","global_attention",[(102.5,37),(102.5,44)],PRIOR)
    text(108,40.5,"K,V",PRIOR)
    feature("global_context_pool",95,61,10)
    text(109,66,"Context",GLOBAL,rotation=90)
    route("fusion_pool","global_attention","global_context_pool",[(100,54),(100,61)])
    feature("global_bridge",95,79,9)
    text(107,83.5,"32²",MUTED)
    route("pool_bridge","global_context_pool","global_bridge",[(100,71),(100,79)])
    route("bridge_decode","global_bridge","global_dec4",[(95,83.5),(81.5,83.5),(81.5,74)])
    tile("global_map",gm,5,81,15,"mask",GLOBAL,source="global_mask.npy")
    text(34,85,"Global map",GLOBAL,7,True)
    text(38,89,"8 logits → native grid",MUTED)
    route("global_predict","global_dec1","global_map",[(12.5,74),(12.5,81)])

    # Original image residual with native encoder/decoder skips.
    text(31,96,"Native detail",NATIVE,7.7,True)
    tile("native_image",raw[y0:y1,x0:x1],5,101,14,"image",NATIVE,source="input_full.npy:central ROI")
    text(12,118,"Native ROI",MUTED)
    route("native_I","image","native_image",[(3,15.5),(1,15.5),(1,108),(5,108)],NATIVE)
    nkeys=("native_fine","native_middle","native_deep","native_up_middle","native_up_fine")
    for key,x,ch in zip(nkeys,(27,40,53,66,79),(16,32,64,32,16)):
        feature(key,x,103,8,NATIVE)
        text(x+4,114,f"{ch} ch",MUTED)
    route("detail_stem","native_image","native_fine",[(19,107),(27,107)],NATIVE)
    for i in range(4):
        x=27+i*13
        route(f"native_{i}_{i+1}",nkeys[i],nkeys[i+1],[(x+8,107),(x+13,107)],NATIVE)
    route("fine_skip","native_fine","native_up_fine",[(31,103),(31,100),(83,100),(83,103)],NATIVE,width=.45)
    route("middle_skip","native_middle","native_up_middle",[(44,103),(44,101.5),(70,101.5),(70,103)],NATIVE,width=.45)
    ax.add_patch(Circle((97,107),2.2,fc="white",ec=NATIVE,lw=.7,zorder=11))
    text(97,107,"+",NATIVE,9)
    obj("native_add","operation",(94.8,104.8,99.2,109.2),"interpolated Global plus eight-channel native residual")
    route("residual","native_up_fine","native_add",[(87,107),(94.8,107)],NATIVE)
    text(90.5,98,"Residual",NATIVE)
    route("sample_logits","global_map","native_add",[(20,88.5),(22,88.5),(22,93),(97,93),(97,104.8)])
    tile("native_map",nm,93,121,16,"mask",NATIVE,source="native_mask.npy")
    text(101,118,"Refined map",NATIVE,7,True)
    route("native_predict","native_add","native_map",[(99.2,107),(111,107),(111,129),(109,129)],NATIVE)

    # The body inset and samples reuse exactly the previous figure's example.
    labels,_=ndimage.label(nm>=3,ndimage.generate_binary_structure(2,1))
    body=labels==166
    yy,xx=np.nonzero(body)
    assert len(yy)==389
    top,left,side=497,533,72
    tile("body_points",raw[top:top+side,left:left+side],73,121,14,"image",BODY,source="input_full.npy:fixed body inset")
    ax.contour(np.linspace(73,87,side),np.linspace(121,135,side),body[top:top+side,left:left+side],
               levels=[.5],colors=[BODY],linewidths=.55,zorder=8)
    adjacency=ndimage.convolve((nm==1).astype(np.uint8),np.array([[0,1,0],[1,0,1],[0,1,0]],np.uint8),mode="constant")>0
    for ys,xs,color in ((yy,xx,BODY),(yy[adjacency[yy,xx]],xx[adjacency[yy,xx]],NATIVE)):
        indices=np.linspace(0,len(ys)-1,min(7,len(ys)),dtype=int)
        ax.scatter(73+(xs[indices]-left+.5)*14/side,121+(ys[indices]-top+.5)*14/side,
                   s=4,c=color,edgecolors="white",linewidths=.2,zorder=10)
    text(79.5,118,"Body + contacts",BODY)
    text(80,138,"4-connected oil",BODY)
    route("make_body","native_map","body_points",[(93,128),(87,128)],BODY)
    # A named raw-image port supplies intensity; fine features join here too.
    text(71,121,"I",NATIVE)
    route("body_I","image","body_points",[(71,124),(73,124)],NATIVE,width=.45)
    route("body_feature","native_up_fine","body_points",[(83,111),(89,111),(89,120),(85,120),(85,121)],NATIVE,width=.45)
    box("body_attention",45,122,23,9,"Within-type\nattention ×2",BODY)
    route("body_embed","body_points","body_attention",[(73,128),(68,128)],BODY)
    box("type_pools",45,135,23,8,"Type means\n+ body summary",BODY)
    route("attention_pool","body_attention","type_pools",[(56.5,131),(56.5,135)],BODY)
    route("summary","body_points","type_pools",[(73,133),(70.3,133),(70.3,139),(68,139)],BODY,width=.45)
    text(71.8,134.9,"s",BODY)
    box("body_update",19,135,22,8,"5-way\nlogit update",BODY)
    route("pool_update","type_pools","body_update",[(45,139),(41,139)],BODY)
    route("upstream_body_probs","native_map","body_update",[(101,137),(101,144.5),(30,144.5),(30,143)],BODY,width=.5)
    text(74,143,"q",BODY)
    tile("final_map",fm,4,149,18,"mask",BODY,source="final_mask.npy")
    text(13,146,"Final map",BODY,7,True)
    route("assign_body","body_update","final_map",[(19,139),(1.8,139),(1.8,158),(4,158)],BODY)
    route("copy_support","native_map","final_map",[(109,133),(113,133),(113,173),(13,173),(13,167)],MUTED,True,.45)

    for name,x,y in (("final_prev",34,152),("final_mid",37,150),("final_next",40,148)):
        array=fm if name=="final_mid" else np.load(ASSETS/f"{name}.npy")
        tile(name,array,x,y,12,"mask",BODY,source=("final_mask" if name=="final_mid" else name)+".npy")
    text(43,168,"Ordered slices",BODY)
    route("stack_outputs","final_map","final_prev",[(22,158),(34,158)],BODY)
    volume=plt.imread(ASSETS/"wi_val_169_3d_oil.png")
    ax.imshow(volume,extent=(78,104,171,145),zorder=5)
    obj("volume","image",(78,145,104,171),"retained 3D oil render","wi_val_169_3d_oil.png")
    text(91,142,"3D oil volume",BODY,7,True)
    route("volume_stack","final_next","volume",[(52,154),(65,154),(65,158),(78,158)],BODY)
    text(63,170,"Fixed support",MUTED)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=stem.with_suffix(".alignment.json"),strict=True)
    renderer=fig.canvas.get_renderer()
    for i,artist in enumerate(text_artists):
        bounds=artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        x1,y1,x2,y2=bounds.extents
        obj(f"text_{i}","text",(x1,min(y1,y2),x2,max(y1,y2)),artist.get_text())
    fig.savefig(stem.with_suffix(".pdf"),facecolor="white")
    fig.savefig(stem.with_suffix(".svg"),facecolor="white")
    fig.savefig(stem.with_suffix(".png"),dpi=450,facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"),dpi=600,facecolor="white",pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
    geometry={"width_mm":WIDTH,"height_mm":HEIGHT,"minimum_font_pt":7,"objects":objects,"edges":edges}
    stem.with_suffix(".geometry.json").write_text(json.dumps(geometry,indent=2),encoding="utf-8")
    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ASSETS.iterdir() if p.is_file()}
    stem.with_suffix(".source.json").write_text(json.dumps({"sample":meta,"inputs":inputs,
        "source_script":Path(__file__).name,"concept_pixels_in_final":False},indent=2),encoding="utf-8")
    print(stem)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,default=HERE)
    main(parser.parse_args().out_dir)
