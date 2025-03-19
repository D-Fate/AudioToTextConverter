import sys
import re
import time
import threading
import os
import tkinter as tk
from io import StringIO
from pathlib import Path
from typing import List, Dict, Optional
from tkinter import filedialog, messagebox

import customtkinter as ctk
import torch
import whisper
from tkinterdnd2 import TkinterDnD, DND_FILES

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class ProgressParser:
    """Парсит и обрабатывает прогресс транскрипции из stdout"""

    def __init__(self, update_callback):
        self.update_callback = update_callback
        self.buffer = StringIO()
        self.original_stdout = sys.stdout
        self.progress_pattern = re.compile(r"\s(\d+)%")
        self.captured_content = ""

    def __enter__(self):
        sys.stdout = self.buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout

    def read_progress(self) -> float:
        """Обновляет прогресс и возвращает текущее значение"""
        # Сбор данных из буфера
        self.buffer.seek(0)
        new_data = self.buffer.read()
        self.captured_content += new_data
        self.buffer.seek(0)
        self.buffer.truncate(0)

        # Поиск последнего значения прогресса
        if matches := self.progress_pattern.findall(self.captured_content):
            latest_progress = int(matches[-1]) / 100
            self.update_callback(latest_progress)
            return latest_progress
        return 0.0


class AudioConverterApp(ctk.CTk, TkinterDnD.DnDWrapper):
    WINDOW_TITLE = "Audio to Text Converter"
    WINDOW_GEOMETRY = "800x600"
    SUPPORTED_FORMATS = (".wav", ".mp3")
    DEFAULT_STATUS = "Готово"

    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)

        self.title(self.WINDOW_TITLE)
        self.geometry(self.WINDOW_GEOMETRY)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Состояние приложения
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[whisper.Whisper] = None
        self.queue: List[Dict] = []
        self.current_task: Optional[Dict] = None
        self.running = True
        self.progress_parser: Optional[ProgressParser] = None
        self.model_ready = threading.Event()

        # Инициализация GUI
        self._init_ui()
        self._load_model_async()
        self._start_progress_monitoring()

    def _init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._create_control_panel()
        self._create_drop_zone()
        self._create_task_queue()
        self._create_progress_indicator()
        self._create_status_display()

    def _load_model_async(self):
        """Асинхронная загрузка модели Whisper"""

        def model_loader():
            self.model = whisper.load_model("medium", device=self.device)
            self.model_ready.set()
            self.after(0, self._update_status, self.DEFAULT_STATUS)

        threading.Thread(target=model_loader, daemon=True).start()
        self._update_status("Инициализация модели...")

    def _create_control_panel(self):
        """Панель управления с кнопками"""
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=5)

        self.btn_add = ctk.CTkButton(
            self.control_frame,
            text="Добавить файлы",
            command=self._open_file_dialog
        )
        self.btn_add.pack(side=tk.LEFT, padx=5)

    def _create_drop_zone(self):
        """Область для перетаскивания файлов"""
        self.drop_target = ctk.CTkLabel(
            self.main_frame,
            text="Перетащите аудиофайлы сюда",
            fg_color=("gray78", "gray28"),
            height=100
        )
        self.drop_target.pack(fill=tk.X, pady=10)
        self._enable_dnd()

    def _create_task_queue(self):
        """Список задач в очереди"""
        self.lbl_queue = ctk.CTkLabel(self.main_frame, text="Очередь обработки:")
        self.lbl_queue.pack(anchor=tk.W)

        self.task_list = tk.Listbox(
            self.main_frame,
            bg="#343638",
            fg="white",
            selectbackground="#565B5E"
        )
        self.task_list.pack(fill=tk.BOTH, expand=True)

    def _create_progress_indicator(self):
        """Индикатор выполнения"""
        self.progress_bar = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar.pack(fill=tk.X, pady=10)
        self.progress_bar.set(0)

    def _create_status_display(self):
        """Отображение текущего статуса"""
        self.lbl_status = ctk.CTkLabel(self.main_frame, text="Загрузка...")
        self.lbl_status.pack()

    def _enable_dnd(self):
        """Активация Drag-and-Drop"""
        self.drop_target.drop_target_register(DND_FILES)
        self.drop_target.dnd_bind('<<Drop>>', self._handle_dropped_files)

    def _open_file_dialog(self):
        """Открытие диалога выбора файлов"""
        file_types = [("Аудиофайлы", " ".join(f"*{fmt}" for fmt in self.SUPPORTED_FORMATS))]
        selected_files = filedialog.askopenfilenames(filetypes=file_types)
        self._enqueue_files(selected_files)

    def _normalize_path(self, raw_path: str) -> str:
        """Нормализация пути файла"""
        return os.path.normpath(raw_path.strip('{}'))

    def _validate_audio_file(self, file_path: str):
        """Проверка валидности файла"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл отсутствует: {file_path}")
        if not file_path.lower().endswith(self.SUPPORTED_FORMATS):
            raise ValueError(f"Неподдерживаемый формат: {file_path}")

    def _enqueue_files(self, file_paths: List[str]):
        """Добавление файлов в очередь"""
        for path in file_paths:
            try:
                clean_path = self._normalize_path(path)
                self._validate_audio_file(clean_path)

                if not any(t["path"] == clean_path for t in self.queue):
                    self.queue.append({"path": clean_path, "status": "В ожидании"})
                    self.task_list.insert(tk.END, f"{Path(clean_path).name} - В ожидании")
            except Exception as e:
                self._show_error(str(e))

        self._process_next_task()

    def _process_next_task(self):
        """Обработка следующей задачи"""
        if not self.model_ready.wait(timeout=0.1):
            self.after(100, self._process_next_task)
            return

        if self.queue and self.model:
            self.current_task = self.queue.pop(0)
            self._update_status(f"Обработка: {Path(self.current_task['path']).name}")
            self.task_list.delete(0)
            threading.Thread(target=self._execute_transcription, daemon=True).start()

    def _start_progress_monitoring(self):
        """Мониторинг прогресса выполнения"""

        def progress_watcher():
            while self.running:
                if self.progress_parser:
                    try:
                        current_progress = self.progress_parser.read_progress()
                        self.after(0, self._update_progress_ui, current_progress)
                    except Exception as e:
                        print(f"Ошибка отслеживания прогресса: {e}")
                time.sleep(0.05)

        threading.Thread(target=progress_watcher, daemon=True).start()

    def _execute_transcription(self):
        """Выполнение транскрипции аудио"""
        try:
            with ProgressParser(self._handle_progress_update) as parser:
                self.progress_parser = parser
                result = self.model.transcribe(
                    self.current_task["path"],
                    verbose=True,
                    fp16=torch.cuda.is_available(),
                    language="ru"
                )
                self._save_transcription_result(result)
                self._update_status("Транскрипция завершена")
        except Exception as e:
            self._update_status(f"Ошибка: {str(e)}")
            self._show_error(str(e))
        finally:
            self.progress_parser = None
            self.current_task = None
            self.after(0, self._process_next_task)

    def _handle_progress_update(self, value: float):
        """Обновление значения прогресса"""
        self.after(0, self.progress_bar.set, value)
        self.lbl_status.configure(text=f"Прогресс: {int(value * 100)}%")

    def _update_progress_ui(self, value: float):
        """Синхронизация UI с текущим прогрессом"""
        self.progress_bar.set(value)
        self.lbl_status.configure(text=f"Прогресс: {int(value * 100)}%")

    def _save_transcription_result(self, result: dict):
        """Сохранение результата в файл"""
        try:
            source_file = Path(self.current_task["path"])
            output_file = source_file.with_name(f"{source_file.stem}_transcript.txt")

            with output_file.open("w", encoding="utf-8") as f:
                f.write(f"Транскрипция аудио:\n\n{result['text']}")

            messagebox.showinfo("Готово", f"Результат сохранён:\n{output_file}")
        except Exception as e:
            self._show_error(f"Ошибка сохранения: {str(e)}")

    def _update_status(self, text: str):
        """Обновление текста статуса"""
        self.lbl_status.configure(text=text)
        self.update_idletasks()

    def _handle_dropped_files(self, event):
        """Обработка перетащенных файлов"""
        try:
            files = event.data.split() if isinstance(event.data, str) else event.data
            self._enqueue_files(files)
        except Exception as e:
            self._show_error(str(e))

    def _show_error(self, message: str):
        """Отображение сообщения об ошибке"""
        messagebox.showerror("Ошибка", message)
        self._update_status(f"Ошибка: {message}")

    def on_closing(self):
        """Обработка закрытия приложения"""
        self.running = False
        self.model_ready.clear()
        self.destroy()


if __name__ == "__main__":
    app = AudioConverterApp()
    app.mainloop()