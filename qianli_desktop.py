# -*- coding: utf-8 -*-
"""千里桌面版：点击按钮后运行 qianli.py，并自动打开可视化结果。"""
import os
import threading
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "qianli_latest.html"


class App:
    def __init__(self, root):
        self.root = root
        root.title("千里 · 底部烂板筛选")
        root.geometry("430x250")
        root.resizable(False, False)
        root.configure(bg="#f3f6fb")

        tk.Label(root, text="千里 · 底部烂板筛选", font=("Microsoft YaHei", 18, "bold"),
                 bg="#f3f6fb", fg="#172033").pack(pady=(25, 8))
        tk.Label(root, text="点击一次：获取数据 → 保存数据集 → 生成并打开可视化页面",
                 font=("Microsoft YaHei", 10), bg="#f3f6fb", fg="#64748b").pack()
        self.button = tk.Button(root, text="开始筛选", command=self.start,
                                font=("Microsoft YaHei", 14, "bold"),
                                bg="#2563eb", fg="white", activebackground="#1d4ed8",
                                relief="flat", width=18, height=2)
        self.button.pack(pady=22)
        self.status = tk.Label(root, text="准备就绪", font=("Microsoft YaHei", 10),
                               bg="#f3f6fb", fg="#64748b")
        self.status.pack()

    def start(self):
        self.button.config(state="disabled", text="筛选中，请稍候")
        self.status.config(text="正在获取数据并保存数据集……", fg="#2563eb")
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        try:
            import qianli
            qianli.main()
        except Exception as exc:
            self.root.after(0, lambda: self.failed(exc))
            return
        self.root.after(0, self.done)

    def done(self):
        self.button.config(state="normal", text="开始筛选")
        self.status.config(text="完成，已打开可视化页面", fg="#15803d")

    def failed(self, exc):
        self.button.config(state="normal", text="开始筛选")
        self.status.config(text="筛选失败，请查看提示", fg="#dc2626")
        messagebox.showerror("千里筛选失败", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
