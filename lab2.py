import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from threading import Thread, Event

class ScreenReplacerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Замена экрана")
        self.root.configure(bg='#2e2e2e')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Стиль (тёмная тема)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2e2e2e')
        style.configure('TLabel', background='#2e2e2e', foreground='white', font=('Segoe UI', 11))
        style.configure('TButton', background='#444444', foreground='white', font=('Segoe UI', 10), borderwidth=1)
        style.map('TButton', background=[('active', '#5a5a5a')])
        style.configure('TEntry', fieldbackground='#444444', foreground='white')
        style.configure('TRadiobutton', background='#2e2e2e', foreground='white', font=('Segoe UI', 10))

        # Переменные
        self.mode = tk.StringVar(value="video")  # "video" или "photo"
        self.cap_main = None
        self.cap_overlay = None
        self.video_thread = None
        self.stop_event = Event()
        self.current_frame = None
        self.last_screen_points = None
        self.writer = None
        self.fps = 30.0
        self.frame_size = (640, 480)
        self.playing = False

        # Заголовок
        header = ttk.Label(root, text="Вставка контента на экран", font=('Segoe UI', 16, 'bold'))
        header.pack(pady=10)

        # Выбор режима
        mode_frame = ttk.Frame(root)
        mode_frame.pack(pady=5)
        ttk.Label(mode_frame, text="Режим:").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(mode_frame, text="Видео", variable=self.mode, value="video",
                        command=self.on_mode_change).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(mode_frame, text="Фото", variable=self.mode, value="photo",
                        command=self.on_mode_change).grid(row=0, column=2, padx=5)

        # Фрейм выбора файлов (будет меняться)
        self.file_frame = ttk.Frame(root)
        self.file_frame.pack(pady=5)

        # Основной файл (видео или фото)
        ttk.Label(self.file_frame, text="Основной файл:").grid(row=0, column=0, padx=5, sticky='e')
        self.entry_main = ttk.Entry(self.file_frame, width=40)
        self.entry_main.grid(row=0, column=1, padx=5)
        self.btn_main = ttk.Button(self.file_frame, text="Обзор...", command=self.select_main_file)
        self.btn_main.grid(row=0, column=2, padx=5)

        # Накладываемый файл
        ttk.Label(self.file_frame, text="Накладываемый файл:").grid(row=1, column=0, padx=5, sticky='e')
        self.entry_overlay = ttk.Entry(self.file_frame, width=40)
        self.entry_overlay.grid(row=1, column=1, padx=5)
        self.btn_overlay = ttk.Button(self.file_frame, text="Обзор...", command=self.select_overlay_file)
        self.btn_overlay.grid(row=1, column=2, padx=5)

        # Холст
        self.canvas = tk.Canvas(root, bg='black', width=640, height=480, highlightthickness=0)
        self.canvas.pack(padx=10, pady=5)

        # Кнопка Старт (и опционально Выход, но выход через закрытие)
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)
        self.btn_start = ttk.Button(btn_frame, text="▶ Старт", command=self.start_processing)
        self.btn_start.grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="❌ Выход", command=self.on_closing).grid(row=0, column=1, padx=5)

        # Строка состояния
        self.status_var = tk.StringVar(value="Выберите режим и файлы, затем нажмите Старт")
        self.status_label = ttk.Label(root, textvariable=self.status_var, wraplength=500)
        self.status_label.pack(pady=5)

        # Привязка клавиши q
        self.root.bind('q', lambda e: self.on_closing())
        self.canvas.focus_set()

        # Установим фильтры для режима по умолчанию
        self.on_mode_change()

    # ---------- Смена режима ----------
    def on_mode_change(self):
        """Обновляет надписи и типы файлов в зависимости от режима"""
        if self.mode.get() == "video":
            self.file_frame.winfo_children()[0].config(text="Основное видео:")
            self.file_frame.winfo_children()[2].config(text="Накладываемое видео:")
            self.status_var.set("Режим Видео: выберите два видеофайла")
        else:
            self.file_frame.winfo_children()[0].config(text="Основное фото:")
            self.file_frame.winfo_children()[2].config(text="Накладываемое фото:")
            self.status_var.set("Режим Фото: выберите два изображения")

    # ---------- Выбор файлов ----------
    def select_main_file(self):
        if self.mode.get() == "video":
            filename = filedialog.askopenfilename(
                title="Выберите основное видео",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
            )
        else:
            filename = filedialog.askopenfilename(
                title="Выберите основное изображение",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
        if filename:
            self.entry_main.delete(0, tk.END)
            self.entry_main.insert(0, filename)

    def select_overlay_file(self):
        if self.mode.get() == "video":
            filename = filedialog.askopenfilename(
                title="Выберите накладываемое видео",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
            )
        else:
            filename = filedialog.askopenfilename(
                title="Выберите накладываемое изображение",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
        if filename:
            self.entry_overlay.delete(0, tk.END)
            self.entry_overlay.insert(0, filename)

    # ---------- Запуск обработки ----------
    def start_processing(self):
        main_path = self.entry_main.get().strip()
        overlay_path = self.entry_overlay.get().strip()
        if not main_path or not overlay_path:
            messagebox.showerror("Ошибка", "Укажите оба файла")
            return

        if self.mode.get() == "video":
            self.start_video_processing(main_path, overlay_path)
        else:
            self.start_photo_processing(main_path, overlay_path)

    def start_video_processing(self, main_path, overlay_path):
        if self.playing:
            messagebox.showwarning("Подождите", "Обработка видео уже запущена")
            return

        self.cap_main = cv2.VideoCapture(main_path)
        self.cap_overlay = cv2.VideoCapture(overlay_path)
        if not self.cap_main.isOpened() or not self.cap_overlay.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть видео")
            return

        self.fps = self.cap_main.get(cv2.CAP_PROP_FPS)
        w = int(self.cap_main.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap_main.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (w, h)
        self.canvas.config(width=w, height=h)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter("result_video.mp4", fourcc, self.fps, self.frame_size)

        self.playing = True
        self.stop_event.clear()
        self.last_screen_points = None
        self.btn_start.config(state='disabled')
        self.status_var.set("Идёт обработка видео...")

        self.video_thread = Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        self._update_canvas()

    def _video_loop(self):
        while not self.stop_event.is_set():
            ret_main, frame_main = self.cap_main.read()
            if not ret_main:
                break

            ret_over, frame_overlay = self.cap_overlay.read()
            if not ret_over:
                frame_overlay = None

            screen_points = self.detect_tv_screen(frame_main)
            if screen_points is None:
                screen_points = self.last_screen_points
            else:
                self.last_screen_points = screen_points

            if frame_overlay is not None and screen_points is not None:
                frame_main = self.warp_image(frame_main, frame_overlay, screen_points)

            self.current_frame = frame_main
            if self.writer:
                self.writer.write(frame_main)

        # Завершение
        if self.cap_main:
            self.cap_main.release()
        if self.cap_overlay:
            self.cap_overlay.release()
        if self.writer:
            self.writer.release()
        self.playing = False
        self.root.after(0, self._on_video_done)

    def _on_video_done(self):
        self.btn_start.config(state='normal')
        self.status_var.set("Готово. Результат сохранён в result_video.mp4")
        self.current_frame = None

    def start_photo_processing(self, main_path, overlay_path):
        try:
            img_main = cv2.imread(main_path)
            img_overlay = cv2.imread(overlay_path)
            if img_main is None or img_overlay is None:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображения")
                return

            screen_points = self.detect_tv_screen(img_main)
            if screen_points is None:
                messagebox.showwarning("Внимание", "Экран не найден, результат не изменён")
                result = img_main
            else:
                result = self.warp_image(img_main, img_overlay, screen_points)

            # Отображение
            self._show_image(result)
            # Сохранение
            cv2.imwrite("result_photo.jpg", result)
            self.status_var.set("Готово. Результат сохранён в result_photo.jpg")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обработки: {e}")

    def _show_image(self, img):
        """Отображает массив OpenCV на холсте"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.photo = ImageTk.PhotoImage(image=img_pil)
        self.canvas.config(width=img.shape[1], height=img.shape[0])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.current_frame = None   # чтобы не мешало обновлению видео

    # ---------- Компьютерное зрение ----------
    def detect_tv_screen(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        screen = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                screen = approx
                max_area = area
        if screen is not None:
            pts = screen.reshape(4, 2).astype(np.float32)
            return self.order_points(pts)
        return None

    def order_points(self, points):
        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        return rect

    def warp_image(self, base, overlay, dst_points):
        h, w = overlay.shape[:2]
        src_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        H, _ = cv2.findHomography(src_points, dst_points)
        warped = cv2.warpPerspective(overlay, H, (base.shape[1], base.shape[0]))
        mask = np.zeros(base.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_points.astype(int), 255)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = np.where(mask == 255, warped, base)
        return result

    # ---------- Обновление холста для видео ----------
    def _update_canvas(self):
        if self.current_frame is not None:
            img_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            self.photo = ImageTk.PhotoImage(image=img_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.current_frame = None
        if self.playing:
            interval = int(1000 / self.fps) if self.fps else 30
            self.root.after(interval, self._update_canvas)

    def on_closing(self):
        self.stop_event.set()
        if self.cap_main:
            self.cap_main.release()
        if self.cap_overlay:
            self.cap_overlay.release()
        if self.writer:
            self.writer.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenReplacerApp(root)
    root.mainloop()