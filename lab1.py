import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from threading import Thread, Event

class VideoAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Аннотатор видео")
        self.root.configure(bg='#2e2e2e')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Стиль (тёмная тема)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2e2e2e')
        style.configure('TLabel', background='#2e2e2e', foreground='white', font=('Segoe UI', 11))
        style.configure('TButton', background='#444444', foreground='white', font=('Segoe UI', 10), borderwidth=1)
        style.map('TButton', background=[('active', '#5a5a5a')])

        # Заголовок
        self.header = ttk.Label(root, text="Аннотатор видео", font=('Segoe UI', 16, 'bold'))
        self.header.pack(pady=10)

        # Холст для видео
        self.canvas = tk.Canvas(root, bg='black', width=640, height=480, highlightthickness=0)
        self.canvas.pack(padx=10, pady=5)
        self.canvas.bind("<Button-1>", self.on_mouse_click)

        # Панель с кнопками
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)

        self.btn_clear = ttk.Button(btn_frame, text="🧹 Очистить", command=self.clear_rectangles)
        self.btn_clear.grid(row=0, column=0, padx=5)

        self.btn_quit = ttk.Button(btn_frame, text="❌ Выход", command=self.on_closing)
        self.btn_quit.grid(row=0, column=1, padx=5)

        # Параметры
        self.rectangles = []            # список прямоугольников
        self.rect_size = 20
        self.camera_index = 3           # твой индекс DroidCam Source
        self.cap = None
        self.running = False
        self.stop_event = Event()
        self.current_frame = None

        # Запускаем камеру автоматически после построения окна
        self.root.after(100, self.start_video)

    def start_video(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Ошибка: не удалось открыть камеру с индексом {self.camera_index}")
            return
        ret, frame = self.cap.read()
        if ret:
            h, w = frame.shape[:2]
            self.canvas.config(width=w, height=h)
        else:
            self.cap.release()
            return
        self.running = True
        self.stop_event.clear()
        self.video_thread = Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        self._update_canvas()

    def _video_loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            for (x1, y1, x2, y2) in self.rectangles:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.current_frame = frame
            self.stop_event.wait(0.03)
        self.running = False

    def _update_canvas(self):
        if self.current_frame is not None:
            img_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            self.photo = ImageTk.PhotoImage(image=img_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.current_frame = None
        if self.running:
            self.root.after(30, self._update_canvas)

    def on_mouse_click(self, event):
        x, y = event.x, event.y
        self.rectangles.append((x - self.rect_size, y - self.rect_size,
                                x + self.rect_size, y + self.rect_size))

    def clear_rectangles(self):
        self.rectangles.clear()

    def on_closing(self):
        self.stop_event.set()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotator(root)
    root.mainloop()