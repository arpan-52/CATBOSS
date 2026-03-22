"""
CATBOSS Plotting - Diagnostic viewers for POOH and NIMKI.

POOH viewer:
- Per-field self-contained HTML, clickable baseline sidebar
- Left panel: raw dynamic spectra (inferno colourmap, PIL-rendered → base64 PNG embedded inline)
- Right panel: same + flag overlay — existing=red (#FF4444), new=cyan (#00FFFF)
- No external files, no disk-saved PNGs, no matplotlib figure overhead

NIMKI viewer:
- Per (field, spw) Bokeh HTML (fully interactive — zoom/pan/hover)
- Left:  UV vs amplitude — clean input data (existing flags already applied)
- Right: same + new NIMKI outliers in magenta (#FF2D95) + Gabor fit line
- Correlation tabs via Bokeh Tabs widget

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
import os
import json
import base64
import io
from typing import List, Dict, Any, Optional

# ── colour constants ────────────────────────────────────────────────────────────
COLOR_EXISTING_HEX = '#FF4444'   # red    — existing flags (POOH)
COLOR_NEW_HEX      = '#00FFFF'   # cyan   — new POOH flags
COLOR_NIMKI_HEX    = '#FF2D95'   # magenta — new NIMKI flags

# ── inferno LUT (built once) ────────────────────────────────────────────────────
_INFERNO_LUT: Optional[np.ndarray] = None


def _get_inferno_lut() -> np.ndarray:
    global _INFERNO_LUT
    if _INFERNO_LUT is not None:
        return _INFERNO_LUT
    try:
        import matplotlib.cm as cm
        lut = cm.get_cmap('inferno')(np.linspace(0.0, 1.0, 256))[:, :3]
        _INFERNO_LUT = (lut * 255).astype(np.uint8)
    except Exception:
        _INFERNO_LUT = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            _INFERNO_LUT[i] = (
                int(min(255, t * 2 * 255)),
                int(max(0, (t - 0.5) * 2 * 255)),
                int(max(0, (1.0 - t * 2) * 200)),
            )
    return _INFERNO_LUT


# ── spectrogram PIL renderer → inline base64 ───────────────────────────────────

def _render_spectra(
    amp: np.ndarray,
    existing_flags: np.ndarray,
    new_flags: np.ndarray,
    show_flags: bool = True,
):
    """Render a (n_time, n_freq) amplitude array as a PIL Image."""
    from PIL import Image

    lut   = _get_inferno_lut()
    exist = np.asarray(existing_flags, dtype=bool)
    new   = np.asarray(new_flags, dtype=bool)

    any_flag  = exist | new
    unflagged = amp[~any_flag] if any_flag.any() else amp.ravel()
    if len(unflagged) > 10:
        vmin = float(np.percentile(unflagged, 2))
        vmax = float(np.percentile(unflagged, 98))
    else:
        vmin, vmax = float(np.nanmin(amp)), float(np.nanmax(amp))
    if vmin == vmax:
        vmax = vmin + 1.0

    norm    = np.clip((amp - vmin) / (vmax - vmin), 0.0, 1.0)
    indices = (norm * 255).astype(np.uint8)
    rgb     = lut[indices].copy()

    if show_flags:
        rgb[exist] = (255,  68,  68)   # red
        rgb[new]   = (  0, 255, 255)   # cyan

    rgb = rgb[::-1]   # time increases upward

    nt, nf = rgb.shape[:2]
    scale  = max(1, min(8, 320 // max(nt, 1), 1200 // max(nf, 1)))
    img    = Image.fromarray(rgb, 'RGB')
    if scale > 1:
        img = img.resize((nf * scale, nt * scale), Image.NEAREST)
    return img


def _to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=False)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _encode_baseline(
    amp: np.ndarray,
    existing_flags: np.ndarray,
    new_flags: np.ndarray,
):
    """Return (b64_raw, b64_flagged) inline-embeddable PNG strings."""
    return (
        _to_b64(_render_spectra(amp, existing_flags, new_flags, show_flags=False)),
        _to_b64(_render_spectra(amp, existing_flags, new_flags, show_flags=True)),
    )


# ── POOH field viewer ───────────────────────────────────────────────────────────

def create_field_viewer(
    baseline_data: List[Dict[str, Any]],
    field_id: int,
    field_name: str,
    output_path: str,
    logger=None,
) -> Optional[str]:
    """
    Create self-contained HTML viewer for all baselines in a field.

    All spectrogram images are base64-encoded and embedded inline —
    no external files, no PIL on disk, no matplotlib.

    Args:
        baseline_data: list of dicts with keys
            baseline, corr_label, amp, existing_flags, new_flags,
            pct_existing, pct_new, pct_total
        field_id:    integer field index
        field_name:  field name string
        output_path: path for the output HTML file
        logger:      optional logger
    Returns:
        path to saved HTML, or None on failure
    """
    if not baseline_data:
        return None

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    meta   = []
    images = {}

    for bl_info in baseline_data:
        bl   = bl_info['baseline']
        corr = bl_info['corr_label']
        uid  = f"bl{bl[0]}-{bl[1]}_{corr}"

        b64_raw, b64_flag = _encode_baseline(
            bl_info['amp'],
            bl_info['existing_flags'],
            bl_info['new_flags'],
        )

        meta.append({
            'id':           uid,
            'label':        f"BL {bl[0]}-{bl[1]} | {corr}",
            'pct_existing': f"{bl_info['pct_existing']:.1f}",
            'pct_new':      f"{bl_info['pct_new']:.1f}",
            'pct_total':    f"{bl_info['pct_total']:.1f}",
        })
        images[uid] = {'raw': b64_raw, 'flagged': b64_flag}

    html = _generate_viewer_html(field_id, field_name, meta, images)

    with open(output_path, 'w') as f:
        f.write(html)

    if logger:
        logger.info(f"  Viewer: {output_path} ({len(meta)} baselines)")

    return output_path


def _generate_viewer_html(
    field_id: int,
    field_name: str,
    meta: List[Dict],
    images: Dict[str, Dict],
) -> str:
    meta_str   = json.dumps(meta)
    images_str = json.dumps(images)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CATBOSS — Field {field_id}: {field_name}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      background:#0d0d0d;color:#e0e0e0;min-height:100vh}}
.header{{background:#161616;padding:16px 28px;border-bottom:1px solid #222;
         display:flex;align-items:center;gap:18px}}
.header h1{{font-size:18px;font-weight:600;color:#fff;letter-spacing:-0.3px}}
.header .sep{{color:#333;font-size:18px}}
.header .sub{{color:#555;font-size:13px}}
.wrap{{display:flex;height:calc(100vh - 57px)}}
/* sidebar */
.sidebar{{width:240px;background:#111;border-right:1px solid #1e1e1e;
          overflow-y:auto;padding:10px;flex-shrink:0}}
.sidebar::-webkit-scrollbar{{width:4px}}
.sidebar::-webkit-scrollbar-thumb{{background:#2a2a2a;border-radius:2px}}
.sb-head{{font-size:9px;text-transform:uppercase;letter-spacing:1.5px;
          color:#333;margin-bottom:10px;padding:0 4px}}
.btn{{display:block;width:100%;padding:9px 11px;margin-bottom:4px;
      background:#181818;border:1px solid #222;border-radius:6px;
      color:#777;font-size:12px;text-align:left;cursor:pointer;
      transition:background 0.1s,border-color 0.1s,color 0.1s}}
.btn:hover{{background:#1f1f1f;color:#bbb;border-color:#2a2a2a}}
.btn.active{{background:#0f2238;border-color:#1a4a7a;color:#e8f4ff}}
.btn .bl{{font-weight:500;font-size:12px}}
.btn .st{{font-size:10px;color:#3a3a3a;margin-top:2px}}
.btn.active .st{{color:#2a5a8a}}
/* main */
.main{{flex:1;padding:20px 24px;overflow-y:auto;background:#0a0a0a;min-width:0}}
.top{{display:flex;justify-content:space-between;align-items:center;
      margin-bottom:18px;gap:12px;flex-wrap:wrap}}
.title{{font-size:15px;font-weight:500;color:#ccc}}
.stats{{display:flex;gap:20px}}
.stat{{text-align:right}}
.sv{{font-size:20px;font-weight:700;letter-spacing:-0.5px}}
.sv.ex{{color:{COLOR_EXISTING_HEX}}}
.sv.nw{{color:{COLOR_NEW_HEX}}}
.sv.tot{{color:#4fc3f7}}
.sl{{font-size:9px;color:#333;text-transform:uppercase;letter-spacing:0.8px;margin-top:1px}}
/* panels */
.panels{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.panel{{background:#141414;border:1px solid #1e1e1e;border-radius:8px;overflow:hidden}}
.panel-head{{padding:9px 14px;background:#181818;border-bottom:1px solid #1e1e1e;
             display:flex;align-items:center;gap:8px}}
.panel-head h3{{font-size:11px;font-weight:500;color:#555;text-transform:uppercase;
                letter-spacing:0.8px}}
.dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
.dot.raw{{background:#444}}
.dot.ex{{background:{COLOR_EXISTING_HEX}}}
.dot.nw{{background:{COLOR_NEW_HEX}}}
.panel img{{width:100%;display:block;image-rendering:pixelated}}
.panel .empty{{padding:40px 20px;color:#2a2a2a;font-size:12px;text-align:center}}
/* legend + nav */
.legend{{display:flex;gap:18px;margin-top:14px}}
.li{{display:flex;align-items:center;gap:6px;font-size:11px;color:#444}}
.lc{{width:10px;height:10px;border-radius:2px;flex-shrink:0}}
.lc.ex{{background:{COLOR_EXISTING_HEX}}}
.lc.nw{{background:{COLOR_NEW_HEX}}}
.nav{{display:flex;gap:8px;margin-top:14px}}
.nb{{padding:7px 18px;background:#181818;border:1px solid #222;border-radius:5px;
     color:#666;font-size:12px;cursor:pointer;transition:all 0.1s}}
.nb:hover:not(:disabled){{background:#1f1f1f;color:#bbb;border-color:#2a2a2a}}
.nb:disabled{{opacity:0.2;cursor:not-allowed}}
.placeholder{{color:#2a2a2a;font-size:13px;padding:60px 0;text-align:center}}
</style>
</head>
<body>
<div class="header">
  <h1>CATBOSS</h1>
  <span class="sep">|</span>
  <span class="sub">Field {field_id}: {field_name} &nbsp;&mdash;&nbsp; {len(meta)} baselines</span>
</div>
<div class="wrap">
  <div class="sidebar">
    <div class="sb-head">Baselines</div>
    <div id="list"></div>
  </div>
  <div class="main">
    <div class="top">
      <div class="title" id="title">Select a baseline</div>
      <div class="stats">
        <div class="stat">
          <div class="sv ex" id="se">—</div>
          <div class="sl">Existing</div>
        </div>
        <div class="stat">
          <div class="sv nw" id="sn">—</div>
          <div class="sl">New</div>
        </div>
        <div class="stat">
          <div class="sv tot" id="st">—</div>
          <div class="sl">Total</div>
        </div>
      </div>
    </div>
    <div class="panels">
      <div class="panel">
        <div class="panel-head">
          <span class="dot raw"></span>
          <h3>Dynamic Spectra</h3>
        </div>
        <img id="img-raw" src="" alt="" style="display:none">
        <div class="empty" id="raw-empty">No baseline selected</div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <span class="dot ex"></span>
          <span class="dot nw"></span>
          <h3>Flagged</h3>
        </div>
        <img id="img-flag" src="" alt="" style="display:none">
        <div class="empty" id="flag-empty">No baseline selected</div>
      </div>
    </div>
    <div class="legend">
      <div class="li"><div class="lc ex"></div><span>Existing flags</span></div>
      <div class="li"><div class="lc nw"></div><span>New flags</span></div>
    </div>
    <div class="nav">
      <button class="nb" id="prev" onclick="nav(-1)" disabled>&#8592; Prev</button>
      <button class="nb" id="next" onclick="nav(1)"  disabled>Next &#8594;</button>
    </div>
  </div>
</div>
<script>
const M = {meta_str};
const I = {images_str};
let cur = 0;

function init() {{
  const L = document.getElementById('list');
  M.forEach((m, i) => {{
    const b = document.createElement('button');
    b.className = 'btn' + (i === 0 ? ' active' : '');
    b.innerHTML = `<div class="bl">${{m.label}}</div>`
                + `<div class="st">Exist ${{m.pct_existing}}% &nbsp;|&nbsp; New ${{m.pct_new}}%</div>`;
    b.onclick = () => show(i);
    L.appendChild(b);
  }});
  if (M.length > 0) show(0);
}}

function show(i) {{
  cur = i;
  const m = M[i];
  document.querySelectorAll('.btn').forEach((b, j) => b.classList.toggle('active', j === i));
  document.getElementById('title').textContent = m.label;
  document.getElementById('se').textContent = m.pct_existing + '%';
  document.getElementById('sn').textContent = m.pct_new + '%';
  document.getElementById('st').textContent = m.pct_total + '%';
  document.getElementById('prev').disabled = i === 0;
  document.getElementById('next').disabled = i === M.length - 1;
  document.querySelectorAll('.btn')[i].scrollIntoView({{behavior:'smooth',block:'nearest'}});

  const imgs = I[m.id];
  const imgRaw  = document.getElementById('img-raw');
  const imgFlag = document.getElementById('img-flag');
  const rawEmpty  = document.getElementById('raw-empty');
  const flagEmpty = document.getElementById('flag-empty');

  if (imgs) {{
    imgRaw.src  = 'data:image/png;base64,' + imgs.raw;
    imgFlag.src = 'data:image/png;base64,' + imgs.flagged;
    imgRaw.style.display  = 'block';
    imgFlag.style.display = 'block';
    rawEmpty.style.display  = 'none';
    flagEmpty.style.display = 'none';
  }} else {{
    imgRaw.style.display  = 'none';
    imgFlag.style.display = 'none';
    rawEmpty.style.display  = 'block';
    flagEmpty.style.display = 'block';
  }}
}}

function nav(d) {{ const n = cur + d; if (n >= 0 && n < M.length) show(n); }}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   nav(-1);
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown')  nav(1);
}});

init();
</script>
</body>
</html>'''


# ── master index ────────────────────────────────────────────────────────────────

def create_master_index(
    field_viewers: Dict[int, str],
    field_names: Dict[int, str],
    stats: Dict[str, Any],
    output_dir: str,
    logger=None,
) -> Optional[str]:
    """Create master index HTML linking all field viewers."""
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, 'index.html')

    fields_html = ''
    for fid in sorted(field_viewers.keys()):
        fname = os.path.basename(field_viewers[fid])
        name  = field_names.get(fid, f'Field {fid}')
        fields_html += (
            f'<a href="{fname}" class="fc">'
            f'<div class="fn">{name}</div>'
            f'<div class="fi">Field {fid}</div>'
            f'</a>\n'
        )

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CATBOSS Results</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      background:#0d0d0d;color:#e0e0e0;min-height:100vh;padding:40px}}
h1{{font-size:22px;font-weight:600;letter-spacing:-0.5px;margin-bottom:4px}}
.sub{{color:#444;margin-bottom:28px;font-size:13px}}
.stats{{display:flex;gap:12px;margin-bottom:36px;flex-wrap:wrap}}
.sc{{background:#141414;padding:16px 22px;border-radius:8px;border:1px solid #1e1e1e}}
.sv{{font-size:22px;font-weight:700;color:#4fc3f7;letter-spacing:-0.5px}}
.sl{{font-size:9px;color:#333;text-transform:uppercase;letter-spacing:1px;margin-top:3px}}
h2{{font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:#333;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}}
.fc{{display:block;background:#141414;padding:16px 18px;border-radius:8px;
     border:1px solid #1e1e1e;text-decoration:none;color:#bbb;transition:all 0.1s}}
.fc:hover{{background:#1a1a1a;border-color:#2a2a2a;color:#fff}}
.fn{{font-size:14px;font-weight:500;color:#fff}}
.fi{{font-size:11px;color:#333;margin-top:3px}}
</style>
</head>
<body>
<h1>CATBOSS Results</h1>
<p class="sub">Click a field to view baseline diagnostics</p>
<div class="stats">
  <div class="sc">
    <div class="sv">{stats.get("baselines_processed", 0)}</div>
    <div class="sl">Baselines</div>
  </div>
  <div class="sc">
    <div class="sv">{stats.get("new_flags", 0):,}</div>
    <div class="sl">New Flags</div>
  </div>
  <div class="sc">
    <div class="sv">{stats.get("new_percent_flagged", 0):.2f}%</div>
    <div class="sl">New Flagged</div>
  </div>
  <div class="sc">
    <div class="sv">{stats.get("total_processing_time", 0):.1f}s</div>
    <div class="sl">Time</div>
  </div>
</div>
<h2>Fields</h2>
<div class="grid">
{fields_html}</div>
</body>
</html>'''

    with open(index_path, 'w') as f:
        f.write(html)

    if logger:
        logger.info(f"  Index: {index_path}")

    return index_path


# ── NIMKI viewer (Bokeh scatter) ────────────────────────────────────────────────

def _style_nimki_fig(p):
    """Dark theme for a Bokeh figure."""
    p.background_fill_color = '#141414'
    p.border_fill_color     = '#0d0d0d'
    p.grid.grid_line_color  = '#1e1e1e'
    p.axis.axis_line_color          = '#333'
    p.axis.major_tick_line_color    = '#333'
    p.axis.minor_tick_line_color    = '#222'
    p.axis.major_label_text_color   = '#777'
    p.axis.axis_label_text_color    = '#555'
    p.title.text_color     = '#999'
    p.title.text_font_size = '12px'
    p.toolbar.logo         = None


def create_nimki_viewer(
    plot_data_list: List[Dict[str, Any]],
    output_path: str,
    corr_labels: List[str],
    logger=None,
) -> Optional[str]:
    """
    Create Bokeh HTML viewer for NIMKI UV-plane diagnostics.

    Each entry in plot_data_list needs:
        uv, amp, predicted, outliers, corr, field_id, chunk_idx, spw,
        mad_sigma, n_components

    Output: one HTML per (field_id, spw) with correlation tabs.
    """
    if not plot_data_list:
        return None

    try:
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import (ColumnDataSource, HoverTool,
                                  TabPanel, Tabs)
        from bokeh.layouts import row
    except ImportError:
        if logger:
            logger.warning("Bokeh not available — skipping NIMKI plots")
        return None

    from collections import defaultdict

    output_dir = (output_path if os.path.isdir(output_path)
                  else os.path.dirname(output_path) or '.')
    os.makedirs(output_dir, exist_ok=True)

    # Group by (field_id, spw); include chunk in tab label
    groups: Dict = defaultdict(list)
    for pd in plot_data_list:
        groups[(pd['field_id'], pd['spw'])].append(pd)

    saved = []

    for (fid, spw), entries in groups.items():
        out_file = os.path.join(output_dir, f"nimki_field{fid}_spw{spw}.html")
        output_file(out_file, title=f"NIMKI Field {fid} SPW {spw}")

        tabs = []
        for pd in entries:
            corr      = pd['corr']
            chunk_idx = pd.get('chunk_idx', 0)
            corr_name = corr_labels[corr] if corr < len(corr_labels) else f'Corr{corr}'
            tab_label = f"{corr_name} C{chunk_idx}"

            uv        = np.asarray(pd['uv'],        dtype=np.float64)
            amp       = np.asarray(pd['amp'],       dtype=np.float64)
            predicted = np.asarray(pd['predicted'], dtype=np.float64)
            outliers  = np.asarray(pd['outliers'],  dtype=bool)
            mad_sigma = float(pd['mad_sigma'])
            n_comp    = int(pd.get('n_components', 0))

            good      = ~outliers
            sort_idx  = np.argsort(uv)
            pct_out   = 100.0 * np.sum(outliers) / max(len(uv), 1)

            tools = "pan,wheel_zoom,box_zoom,reset,save"

            # ── left: input data ───────────────────────────────────────────
            src_all = ColumnDataSource({'uv': uv, 'amp': amp})
            p_left  = figure(
                width=560, height=380,
                title=f"Input data — {corr_name} chunk {chunk_idx}",
                x_axis_label='UV distance (λ)',
                y_axis_label='Amplitude',
                tools=tools,
            )
            _style_nimki_fig(p_left)
            p_left.scatter('uv', 'amp', source=src_all,
                           size=2, alpha=0.35, color='#5599cc', line_color=None)
            p_left.add_tools(HoverTool(
                tooltips=[('UV', '@uv{0.0}'), ('Amp', '@amp{0.0000}')]))

            # ── right: flagged overlay ─────────────────────────────────────
            src_good = ColumnDataSource({
                'uv': uv[good], 'amp': amp[good]})
            src_bad  = ColumnDataSource({
                'uv': uv[outliers], 'amp': amp[outliers]})
            p_right  = figure(
                width=560, height=380,
                title=(f"NIMKI flags — {corr_name} "
                       f"({pct_out:.1f}% flagged  |  MADσ={mad_sigma:.3g})"),
                x_axis_label='UV distance (λ)',
                y_axis_label='Amplitude',
                tools=tools,
                x_range=p_left.x_range,
                y_range=p_left.y_range,
            )
            _style_nimki_fig(p_right)
            p_right.scatter('uv', 'amp', source=src_good,
                            size=2, alpha=0.35, color='#5599cc',
                            line_color=None, legend_label='Clean')
            if np.any(outliers):
                p_right.scatter('uv', 'amp', source=src_bad,
                                size=5, alpha=0.85, color=COLOR_NIMKI_HEX,
                                line_color=None, legend_label='Flagged')
            # Gabor fit
            p_right.line(
                uv[sort_idx], predicted[sort_idx],
                line_width=2, color='#ffffff', alpha=0.55,
                legend_label=f'Gabor ({n_comp} comp)',
            )
            p_right.legend.click_policy        = 'hide'
            p_right.legend.location            = 'top_right'
            p_right.legend.background_fill_color = '#1a1a1a'
            p_right.legend.border_line_color   = '#2a2a2a'
            p_right.legend.label_text_color    = '#aaa'
            p_right.legend.label_text_font_size = '11px'
            p_right.add_tools(HoverTool(
                tooltips=[('UV', '@uv{0.0}'), ('Amp', '@amp{0.0000}')]))

            layout = row(p_left, p_right)
            tabs.append(TabPanel(child=layout, title=tab_label))

        if tabs:
            root = Tabs(tabs=tabs) if len(tabs) > 1 else tabs[0].child
            save(root)
            saved.append(out_file)
            if logger:
                logger.info(f"  NIMKI viewer: {out_file}")

    return saved[0] if saved else None
