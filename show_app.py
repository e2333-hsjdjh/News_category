import os
# 启用 MPS 回退机制，解决部分算子在 MPS 上未实现的问题
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import tkinter as tk
from tkinter import ttk, messagebox
from model_utils import EnsemblePredictor

class NewsClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("News Category Classifier (Transformer)")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("TButton", font=("Helvetica", 12, "bold"))
        
        # 初始化预测器
        self.status_var = tk.StringVar()
        self.status_var.set("Loading models...")
        self.root.update()
        
        try:
            self.predictor = EnsemblePredictor()
            self.status_var.set("Models loaded successfully. Ready.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}\nPlease ensure .pt files are in the same directory.")
            self.status_var.set("Error loading models.")
            self.predictor = None

        self.create_widgets()

    def create_widgets(self):
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="News Classifier", font=("Helvetica", 20, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Headline Input
        ttk.Label(main_frame, text="Headline (新闻标题):").pack(anchor=tk.W)
        self.headline_text = tk.Text(main_frame, height=3, font=("Helvetica", 11))
        self.headline_text.pack(fill=tk.X, pady=(5, 15))
        
        # Description Input
        ttk.Label(main_frame, text="Short Description (新闻摘要):").pack(anchor=tk.W)
        self.desc_text = tk.Text(main_frame, height=5, font=("Helvetica", 11))
        self.desc_text.pack(fill=tk.X, pady=(5, 15))
        
        # Classify Button
        self.classify_btn = ttk.Button(main_frame, text="Classify News", command=self.classify)
        self.classify_btn.pack(pady=10, ipadx=20, ipady=5)
        
        # Result Area
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Result", padding="10")
        result_frame.pack(fill=tk.X, pady=20)
        
        self.result_label = ttk.Label(result_frame, text="Category: -", font=("Helvetica", 16, "bold"), foreground="#007AFF")
        self.result_label.pack()
        
        self.confidence_label = ttk.Label(result_frame, text="Confidence: -", font=("Helvetica", 10))
        self.confidence_label.pack()
        
        # Status Bar
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, font=("Helvetica", 10))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def classify(self):
        if not self.predictor:
            messagebox.showerror("Error", "Models are not loaded.")
            return
            
        headline = self.headline_text.get("1.0", tk.END).strip()
        description = self.desc_text.get("1.0", tk.END).strip()
        
        if not headline and not description:
            messagebox.showwarning("Input Required", "Please enter at least a Headline or Description.")
            return
            
        try:
            category, confidence = self.predictor.predict(headline, description)
            self.result_label.config(text=f"Category: {category}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2%}")
            self.status_var.set("Classification complete.")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NewsClassifierApp(root)
    root.mainloop()
