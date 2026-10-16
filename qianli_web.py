# -*- coding: utf-8 -*-
"""千里本地网页控制器：页面按钮调用本机 Python 筛选，不显示黑色命令行窗口。"""
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qianli  # noqa: E402

PORT = 8765


def page():
    path = HERE / "qianli_latest.html"
    if path.exists():
        content = path.read_text(encoding="utf-8")
    else:
        content = "<html><body><h2>还没有筛选结果，请点击筛选</h2></body></html>"
    style = """
<style>
.qianli-action{margin:16px 0;padding:14px 18px;background:#eff6ff;border-radius:12px;display:flex;align-items:center;gap:14px}
.qianli-action button{border:0;border-radius:9px;background:#2563eb;color:#fff;padding:11px 28px;font-size:16px;font-weight:700;cursor:pointer}
.qianli-action button:disabled{background:#94a3b8;cursor:wait}.qianli-status{color:#475569;font-size:14px}
</style>
"""
    action = """
<div class="qianli-action"><button id="qianliScan" onclick="runQianli()">筛选</button><span id="qianliStatus">点击后获取当天数据并更新数据集</span></div>
<script>
async function runQianli(){
 const b=document.getElementById('qianliScan'), s=document.getElementById('qianliStatus');
 b.disabled=true; s.textContent='正在获取数据、保存数据集并筛选，请稍候……';
 try{ const r=await fetch('/scan',{method:'POST'}); const x=await r.json();
   if(!r.ok) throw new Error(x.message||'筛选失败');
   s.textContent='完成，正在刷新结果……'; location.reload();
 }catch(e){ b.disabled=false; s.textContent='失败：'+e.message; }
}
</script>
"""
    content = content.replace("</style>", style + "</style>", 1)
    # 把按钮放在页面顶部统计卡片下面、结果表格前面，打开页面即可看到。
    marker = '<section class="panel">'
    if marker in content:
        return content.replace(marker, action + marker, 1)
    return content.replace("</section></main>", action + "</section></main>", 1)


class Handler(BaseHTTPRequestHandler):
    def reply(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            raw = page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/scan":
            self.reply({"message": "地址不存在"}, 404)
            return
        try:
            qianli.main(open_report=False)
            self.reply({"ok": True})
        except Exception as exc:
            self.reply({"message": f"{type(exc).__name__}: {exc}"}, 500)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    webbrowser.open(f"http://127.0.0.1:{PORT}/")
    server.serve_forever()
