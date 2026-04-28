"""Static assets for the HTML report."""

from __future__ import annotations


REPORT_CSS = """
body{font-family:Arial, sans-serif; margin:0; background:#fafafa;}
header{padding:16px 20px; background:#111; color:#fff;}
main{display:flex; gap:16px; padding:16px 20px;}
nav{width:260px; position:sticky; top:16px; align-self:flex-start; background:#fff; border:1px solid #ddd; border-radius:10px; padding:12px;}
.card{background:#fff; border:1px solid #ddd; border-radius:12px; padding:14px; margin-bottom:14px;}
.muted{color:#666; font-size:13px;}
.carousel{border:1px solid #eee; border-radius:12px; padding:10px; margin-top:10px;}
.tabs{display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px;}
.tab{border:1px solid #ccc; background:#f6f6f6; border-radius:15px; padding:6px 12px; cursor:pointer; font-size:12px;}
.slide img{max-width:100%; border-radius:10px;}
img.inline{max-width:100%;height:auto;border-radius:10px;}
table.pairs{width:100%;border-collapse:collapse;font-size:12px;}
table.pairs th, table.pairs td{border:1px solid #ddd;padding:6px;}
table.pairs th{background:#f5f5f5;text-align:left;}
table.matrix{border-collapse:collapse; font-size:11px; width:100%; overflow-x:auto; display:block;}
table.matrix th, table.matrix td{border:1px solid #eee; padding:4px 6px; text-align:right;}
table.matrix th{background:#f9f9f9; text-align:center;}
.grid{display:grid; grid-template-columns:repeat(auto-fit, minmax(220px,1fr)); gap:8px; margin-top:10px;}
.meta{margin-top:8px; padding:8px 10px; border:1px dashed #ddd; border-radius:10px; font-size:12px; color:#444; background:#fcfcfc;}
.grid2{display:grid; grid-template-columns:1fr 1fr; gap:12px;}
.scroll{max-height:420px; overflow:auto; border:1px solid #ddd; border-radius:10px; padding:8px; background:#fff;}
.scroll table{width:100%; border-collapse:collapse; font-size:12px;}
.scroll th, .scroll td{border-bottom:1px solid #eee; padding:4px 6px; text-align:left;}
.mono{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size:12px;}
"""
