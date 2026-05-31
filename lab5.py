import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
from threading import Thread, Event
from ultralytics import YOLO

class PeopleCrossingCounter:
    def __init__(self, root):
        self.root = root
        self.root.title("Подсчёт людей, пересекающих линию")
        self.root.configure(bg='#2e2e2e')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Стилизация интерфейса
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2e2e2e')
        style.configure('TLabel', background='#2e2e2e', foreground='white', font=('Segoe UI', 11))
        style.configure('TButton', background='#444444', foreground='white', font=('Segoe UI', 10), borderwidth=1)
        style.map('TButton', background=[('active', '#5a5a5a')])

        # Параметры видео и потоков
        self.video_path = None
        self.cap = None
        self.writer = None
        self.running = False
        self.paused = False
        self.selecting_line = False
        
        self.line_pts = []          
        self.line = None            
        self.model = None 
        
        # Счётчики и память трекера
        self.track_last_side = {}
        self.track_last_cross_frame = {}  
        self.counted_ids = set()
        self.count_in = 0
        self.count_out = 0
        
        self.current_frame = None
        self.frame_to_show = None
        self.frame_idx = 0
        self.stop_event = Event()
        self.video_thread = None

        # Константы
        self.conf_threshold = 0.5
        self.model_path = "yolo11n.pt"  
        self.cooldown_frames = 35       

        # Макет интерфейса
        main_frame = ttk.Frame(root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Видео холст
        self.canvas = tk.Canvas(main_frame, bg='black', width=800, height=450)
        self.canvas.pack(side=tk.LEFT, padx=(0,10))
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Панель управления
        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        ttk.Label(ctrl_frame, text="Управление", font=('Segoe UI', 14, 'bold')).pack(anchor=tk.W, pady=(0,10))

        # Выбор видео
        ttk.Button(ctrl_frame, text="📁 Выбрать видео", command=self.choose_video).pack(anchor=tk.W, pady=2)
        self.lbl_video = ttk.Label(ctrl_frame, text="Файл не выбран", wraplength=200)
        self.lbl_video.pack(anchor=tk.W, pady=(0,10))

        # Кнопки управления
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(anchor=tk.W, pady=5)
        self.btn_play = ttk.Button(btn_frame, text="▶ Старт", command=self.toggle_play_pause, state='disabled')
        self.btn_play.pack(side=tk.LEFT, padx=2)
        self.btn_stop = ttk.Button(btn_frame, text="⏹ Стоп", command=self.stop_video, state='disabled')
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        
        self.btn_line = ttk.Button(ctrl_frame, text="📏 Задать линию", command=self.start_line_selection, state='disabled')
        self.btn_line.pack(anchor=tk.W, pady=2)

        ttk.Button(ctrl_frame, text="🔄 Сброс счётчиков", command=self.reset_counts).pack(anchor=tk.W, pady=2)

        self.btn_snapshot = ttk.Button(ctrl_frame, text="📸 Сохранить кадр", command=self.save_snapshot, state='disabled')
        self.btn_snapshot.pack(anchor=tk.W, pady=2)

        # Статистика
        ttk.Separator(ctrl_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(ctrl_frame, text="Статистика", font=('Segoe UI', 14, 'bold')).pack(anchor=tk.W, pady=(0,5))
        self.lbl_up = ttk.Label(ctrl_frame, text="Внутрь (IN): 0", foreground='lightgreen')
        self.lbl_up.pack(anchor=tk.W)
        self.lbl_down = ttk.Label(ctrl_frame, text="Наружу (OUT): 0", foreground='lightcoral')
        self.lbl_down.pack(anchor=tk.W)
        self.lbl_total = ttk.Label(ctrl_frame, text="Всего уникальных: 0", foreground='white')
        self.lbl_total.pack(anchor=tk.W)

        self.btn_quit = ttk.Button(ctrl_frame, text="❌ Выход", command=self.on_closing)
        self.btn_quit.pack(anchor=tk.W, pady=(20,0))

    def choose_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.video_path = path
            self.lbl_video.config(text=Path(path).name)
            
            temp_cap = cv2.VideoCapture(path)
            ret, first_frame = temp_cap.read()
            temp_cap.release()
            
            if ret:
                orig_h, orig_w = first_frame.shape[:2]
                self.display_w = 800
                self.display_h = int(800 * orig_h / orig_w)
                self.canvas.config(width=self.display_w, height=self.display_h)
                
                self.current_frame = cv2.resize(first_frame, (self.display_w, self.display_h))
                self._show_frame(self.current_frame)
                
                self.btn_line.config(state='normal')
                self.btn_play.config(state='normal')
                self.btn_stop.config(state='disabled')
                self.reset_counts()

    def toggle_play_pause(self):
        if not self.running:
            self.start_video()
        else:
            self.paused = not self.paused
            self.btn_play.config(text="▶ Продолжить" if self.paused else "⏸ Пауза")

    def start_video(self):
        if self.running:
            return
        
        if self.line is None:
            messagebox.showwarning("Линия не задана", "Пожалуйста, сначала задайте линию для подсчета.")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", f"Не удалось открыть видео-файл: {self.video_path}")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 25

        self.track_last_side.clear()
        self.track_last_cross_frame.clear()
        self.counted_ids.clear()
        self.frame_idx = 0

        self.running = True
        self.paused = False
        self.stop_event.clear()
        
        self.btn_play.config(text="⏸ Инициализация ИИ...")
        self.btn_stop.config(state='normal')
        self.btn_snapshot.config(state='normal')
        self.btn_line.config(state='disabled')

        self.video_thread = Thread(target=self._process_video, daemon=True)
        self.video_thread.start()
        self._update_canvas_loop()

    def stop_video(self):
        self.stop_event.set()
        self.running = False
        self.paused = False
        self.btn_play.config(text="▶ Старт", state='normal')
        self.btn_stop.config(state='disabled')
        self.btn_snapshot.config(state='disabled')
        self.btn_line.config(state='normal')
        
        if self.cap: 
            self.cap.release()
            self.cap = None
        if self.writer: 
            self.writer.release()
            self.writer = None

    def start_line_selection(self):
        if self.current_frame is None:
            messagebox.showwarning("Файл не выбран", "Сначала выберите видеофайл.")
            return
        self.selecting_line = True
        self.line_pts = []
        self.line = None
        self._draw_instructions("Кликните ПЕРВУЮ точку на экране")

    def on_canvas_click(self, event):
        if not self.selecting_line:
            return
        x, y = event.x, event.y
        self.line_pts.append((x, y))
        
        if len(self.line_pts) == 1:
            self._draw_instructions("Кликните ВТОРУЮ точку на экране")
        elif len(self.line_pts) == 2:
            self.line = (self.line_pts[0], self.line_pts[1])
            self.selecting_line = False
            
            frame_copy = self.current_frame.copy()
            cv2.line(frame_copy, self.line[0], self.line[1], (0, 255, 255), 3)
            self._show_frame(frame_copy)

    def _draw_instructions(self, text):
        if self.current_frame is None: return
        frame = self.current_frame.copy()
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        for pt in self.line_pts:
            cv2.circle(frame, pt, 5, (0,0,255), -1)
        self._show_frame(frame)

    def _get_line_side(self, point, line_start, line_end):
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        return (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)

    def _is_point_on_segment(self, point, seg_start, seg_end, tol=25):
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end
        if x1 == x2 and y1 == y2:
            return np.hypot(px - x1, py - y1) <= tol
        dx, dy = x2 - x1, y2 - y1
        t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
        if t < 0.0 or t > 1.0:
            return False
        nearest_x = x1 + t*dx
        nearest_y = y1 + t*dy
        return np.hypot(px - nearest_x, py - nearest_y) <= tol

    # ---------------- Поток обработки видео ----------------
    def _process_video(self):
        try:
            # Инициализируем модель заново при каждом старте, гарантируя чистую память трекера
            self.model = YOLO(self.model_path)

            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_name = f"{Path(self.video_path).stem}_tracked.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(output_dir / output_name), fourcc, self.fps, (self.display_w, self.display_h))
            
            if not self.writer.isOpened():
                raise RuntimeError("Не удалось инициализировать запись видео. Проверьте кодеки системы.")

            self.root.after(0, lambda: self.btn_play.config(text="⏸ Пауза"))

            while not self.stop_event.is_set():
                if self.paused:
                    self.stop_event.wait(0.1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, (self.display_w, self.display_h))
                self.frame_idx += 1
                annotated = frame.copy()
                
                cv2.line(annotated, self.line[0], self.line[1], (0, 255, 255), 3)

                # ИСПОЛЬЗУЕМ ВСТРОЕННЫЙ ТРЕКЕР YOLO (BoT-SORT по умолчанию)
                # classes=[0] оставляет только людей, conf фильтрует слабые рамки на уровне ИИ
                results = self.model.track(
                    frame, 
                    persist=True, 
                    conf=self.conf_threshold, 
                    classes=[0], 
                    tracker="botsort.yaml", # Можно заменить на "bytetrack.yaml", если этот будет строже
                    verbose=False
                )[0]

                if results.boxes is not None and results.boxes.id is not None:
                    boxes = results.boxes.xyxy.int().cpu().tolist()
                    ids = results.boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, ids):
                        x1, y1, x2, y2 = box
                        
                        # Ограничиваем рамки краями экрана во избежание сбоев отрисовки
                        x1 = max(0, min(x1, self.display_w))
                        y1 = max(0, min(y1, self.display_h))
                        x2 = max(0, min(x2, self.display_w))
                        y2 = max(0, min(y2, self.display_h))
                        
                        center = (int((x1+x2)/2), int((y1+y2)/2))

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(annotated, center, 4, (0, 0, 255), -1)
                        cv2.putText(annotated, f"ID {track_id}", (x1, max(y1-10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        current_side = self._get_line_side(center, self.line[0], self.line[1])
                        last_cross = self.track_last_cross_frame.get(track_id, -self.cooldown_frames - 1)

                        if track_id not in self.track_last_side:
                            self.track_last_side[track_id] = current_side
                        else:
                            prev_side = self.track_last_side[track_id]
                            crossed = (prev_side < 0 and current_side > 0) or (prev_side > 0 and current_side < 0)
                            
                            if crossed and self._is_point_on_segment(center, self.line[0], self.line[1], tol=25) and (self.frame_idx - last_cross) > self.cooldown_frames:
                                self.counted_ids.add(track_id)
                                self.track_last_cross_frame[track_id] = self.frame_idx
                                
                                if prev_side < 0 and current_side > 0:
                                    self.count_in += 1
                                else:
                                    self.count_out += 1

                            self.track_last_side[track_id] = current_side

                cv2.putText(annotated, f"IN: {self.count_in}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.putText(annotated, f"OUT: {self.count_out}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                self.current_frame = annotated
                self.writer.write(annotated)

            self.running = False
            self.root.after(0, self._on_video_success)

        except Exception as e:
            self.running = False
            self.root.after(0, lambda err=e: self._on_video_error(err))

    def _on_video_success(self):
        self.stop_video()
        messagebox.showinfo("Готово", "Видео успешно обработано.\nРезультат сохранен в папку output/")

    def _on_video_error(self, error):
        self.stop_video()
        messagebox.showerror("Ошибка обработки", f"Произошел сбой при обработке кадра:\n{error}\n\nПроверьте консоль.")

    def _update_canvas_loop(self):
        if self.current_frame is not None:
            self._show_frame(self.current_frame)
            self.lbl_up.config(text=f"Внутрь (IN): {self.count_in}")
            self.lbl_down.config(text=f"Наружу (OUT): {self.count_out}")
            self.lbl_total.config(text=f"Всего уникальных: {len(self.counted_ids)}")
        if self.running:
            self.root.after(30, self._update_canvas_loop)

    def _show_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.photo = ImageTk.PhotoImage(image=img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.frame_to_show = frame

    def reset_counts(self):
        self.count_in = 0
        self.count_out = 0
        self.counted_ids.clear()
        self.track_last_side.clear()
        self.track_last_cross_frame.clear()
        self.lbl_up.config(text="Внутрь (IN): 0")
        self.lbl_down.config(text="Наружу (OUT): 0")
        self.lbl_total.config(text="Всего уникальных: 0")

    def save_snapshot(self):
        if hasattr(self, 'frame_to_show') and self.frame_to_show is not None:
            filename = f"snapshot_{Path(self.video_path).stem}.jpg"
            cv2.imwrite(filename, self.frame_to_show)
            messagebox.showinfo("Сохранено", f"Скриншот сохранен как:\n{filename}")

    def on_closing(self):
        self.stop_video()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PeopleCrossingCounter(root)
    root.mainloop()