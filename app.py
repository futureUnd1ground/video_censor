from __future__ import annotations

import queue
import re
import threading
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tkinter import TclError, filedialog, messagebox, simpledialog
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import customtkinter as ctk
from tkinterdnd2 import COPY, DND_FILES, REFUSE_DROP, TkinterDnD

from engine import CensorError, Hit, detect, load_terms, probe_media, render
from studio import (audio_peaks, chat_activity, parse_chat_log, read_project,
                    render_moment, transcript_to_srt, transcribe_video, visual_highlights, write_project)


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path.home() / ".local" / "share" / "cut-helper"
HISTORY_FILE = DATA_DIR / "history.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
LEGACY_HISTORY_FILE = APP_DIR / "censor_history.json"
MODES = ["Bleep", "Mute", "Fast-Forward", "Cut", "Custom Sound"]
MODEL_OPTIONS = ["small", "medium", "large-v3-turbo"]


def timestamp(value: float) -> str:
    minutes, seconds = divmod(value, 60)
    return f"{int(minutes):02d}:{seconds:06.3f}"


class CensorApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.title("Cut-Helper")
        self.geometry(self._load_geometry() or "980x860")
        self.minsize(800, 700)
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.video = ctk.StringVar()
        self.studio_video = ctk.StringVar()
        self.preview_url = ctk.StringVar()
        self.stream_url = ctk.StringVar()
        self.youtube_result = ctk.StringVar(value="")
        self.output_dir = ctk.StringVar(value=str(Path.home() / "Videos"))
        self.preview_dir = ctk.StringVar(value=str(Path.home() / "Pictures" / "Cut-Helper previews"))
        self.stream_dir = ctk.StringVar(value=str(Path.home() / "Videos" / "Cut-Helper streams"))
        self.mode = ctk.StringVar(value="Bleep")
        self.sound = ctk.StringVar()
        self.padding = ctk.IntVar(value=80)
        self.model = ctk.StringVar(value="small")
        self.studio_model = ctk.StringVar(value="large-v3-turbo")
        self.status = ctk.StringVar(value="Выберите видео для анализа")
        self.hits: list[Hit] = []
        self.transcript: list[dict] = []
        self.transcript_language = ""
        self.studio_moments: list[dict] = []
        self.chat_messages: list[dict] = []
        self.export_profile = ctk.StringVar(value="Original 16:9")
        self.studio_layout = ctk.StringVar(value="Обычный Shorts")
        self.studio_subtitles = ctk.BooleanVar(value=True)
        self.studio_chat_overlay = ctk.BooleanVar(value=False)
        self.studio_watermark = ctk.StringVar()
        self.project_path: Path | None = None
        self.hit_checks: list[ctk.CTkCheckBox] = []
        self.external_lists: list[Path] = []
        self.history: list[dict] = self._load_history()
        self._stream_stop = threading.Event()
        self.events: queue.Queue = queue.Queue()
        self._dnd_ready = False
        self._dnd_widgets: set[str] = set()
        self._build()
        self._geometry_save_job = None
        self.bind("<Configure>", self._on_configure)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        try:
            self.TkdndVersion = TkinterDnD._require(self)
            self._dnd_ready = True
            self._register_drop_targets(self)
            self.drop_hint.configure(text="Можно перетащить видеофайл в окно")
        except (RuntimeError, TclError) as exc:
            self.status.set(f"Drag-and-drop недоступен: {exc}. Выберите видео кнопкой.")
        self.after(100, self._poll)

    def _register_drop_targets(self, widget):
        """Register every visible Tk child so drops work over tabs and controls."""
        if not self._dnd_ready:
            return
        widget_path = str(widget)
        if widget_path in self._dnd_widgets:
            return
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._drop_video)
            self._dnd_widgets.add(widget_path)
        except (AttributeError, RuntimeError, TclError):
            pass
        for child in widget.winfo_children():
            self._register_drop_targets(child)

    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(self, text="Cut-Helper", font=ctk.CTkFont(size=24, weight="bold")).grid(row=0, column=0, padx=22, pady=(18, 12), sticky="w")

        self.tabs = ctk.CTkTabview(
            self,
            anchor="nw",
            segmented_button_fg_color=("#e3e9ed", "#30383f"),
            segmented_button_selected_color=("#287d58", "#287d58"),
            segmented_button_selected_hover_color=("#226b4b", "#226b4b"),
            segmented_button_unselected_color=("#e3e9ed", "#30383f"),
            segmented_button_unselected_hover_color=("#d2dce2", "#414b54"),
            text_color=("#18222a", "#f2f5f6"),
            text_color_disabled=("#777f85", "#858d95"),
            segmented_button_font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.tabs.grid(row=1, column=0, padx=16, pady=4, sticky="nsew")
        censor_tab = self.tabs.add("Цензура")
        studio_tab = self.tabs.add("Студия")
        self.studio_tab = studio_tab
        online_tab = self.tabs.add("Превью и загрузки")
        history_tab = self.tabs.add("История")
        for tab in (censor_tab, studio_tab, online_tab, history_tab):
            tab.grid_columnconfigure(0, weight=1)
        censor_tab.grid_rowconfigure(3, weight=1)
        history_tab.grid_rowconfigure(0, weight=1)
        studio_tab.grid_rowconfigure(3, weight=1)
        online_tab.grid_columnconfigure(0, weight=1)
        online_tab.grid_columnconfigure(2, weight=1)

        source = ctk.CTkFrame(censor_tab, fg_color="transparent")
        source.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        source.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(source, text="Видео").grid(row=0, column=0, padx=(4, 10))
        ctk.CTkEntry(source, textvariable=self.video).grid(row=0, column=1, sticky="ew")
        ctk.CTkButton(source, text="Выбрать…", width=112, command=self._choose_video).grid(row=0, column=2, padx=(8, 0))
        ctk.CTkLabel(source, text="Папка результата").grid(row=1, column=0, padx=(4, 10), pady=(8, 0))
        ctk.CTkEntry(source, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", pady=(8, 0))
        ctk.CTkButton(source, text="Обзор…", width=112, command=self._choose_dir).grid(row=1, column=2, padx=(8, 0), pady=(8, 0))
        self.drop_hint = ctk.CTkLabel(source, text="Подключение drag-and-drop…", text_color=("gray45", "gray65"), anchor="w")
        self.drop_hint.grid(row=2, column=1, sticky="w", pady=(4, 0))

        settings = ctk.CTkFrame(censor_tab)
        settings.grid(row=1, column=0, padx=4, pady=10, sticky="ew")
        settings.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(settings, text="Режим").grid(row=0, column=0, padx=(12, 6), pady=12)
        ctk.CTkOptionMenu(settings, values=MODES, variable=self.mode, width=150, command=self._mode_changed).grid(row=0, column=1, padx=6, pady=12)
        ctk.CTkLabel(settings, text="Модель").grid(row=0, column=2, padx=(12, 6), pady=12)
        ctk.CTkOptionMenu(settings, values=MODEL_OPTIONS, variable=self.model, width=150).grid(row=0, column=3, padx=6, pady=12, sticky="w")
        ctk.CTkLabel(settings, text="Запас").grid(row=1, column=0, padx=(12, 6), pady=(0, 12))
        self.padding_slider = ctk.CTkSlider(settings, from_=0, to=300, number_of_steps=30, variable=self.padding, command=self._padding_changed)
        self.padding_slider.grid(row=1, column=1, sticky="ew", padx=6, pady=(0, 12))
        self.padding_label = ctk.CTkLabel(settings, text="80 мс", width=60)
        self.padding_label.grid(row=1, column=2, padx=6, pady=(0, 12))
        self.sound_entry = ctk.CTkEntry(settings, textvariable=self.sound, placeholder_text="Звук для режима Custom Sound")
        self.sound_entry.grid(row=2, column=0, columnspan=3, padx=(12, 6), pady=(0, 12), sticky="ew")
        self.sound_button = ctk.CTkButton(settings, text="Файл звука…", width=112, command=self._choose_sound)
        self.sound_button.grid(row=2, column=3, padx=6, pady=(0, 12), sticky="w")
        self._mode_changed(self.mode.get())

        toolbar = ctk.CTkFrame(censor_tab, fg_color="transparent")
        toolbar.grid(row=2, column=0, padx=4, pady=(0, 7), sticky="ew")
        self.analyze_button = ctk.CTkButton(toolbar, text="Анализировать видео", command=self._analyze)
        self.analyze_button.pack(side="left")
        self.render_button = ctk.CTkButton(toolbar, text="Рендер", command=self._render, state="disabled")
        self.render_button.pack(side="left", padx=8)
        ctk.CTkButton(toolbar, text="Загрузить blacklist…", width=158, command=self._load_blacklist).pack(side="right")
        ctk.CTkButton(toolbar, text="+ Слово", width=90, command=self._add_word).pack(side="right", padx=8)

        self.results = ctk.CTkScrollableFrame(censor_tab, label_text="Найденные слова · снимите отметку с ложных срабатываний")
        self.results.grid(row=3, column=0, padx=4, pady=4, sticky="nsew")
        self.results.grid_columnconfigure(0, weight=1)

        studio_source = ctk.CTkFrame(studio_tab, fg_color="transparent")
        studio_source.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        studio_source.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(studio_source, text="Видео студии").grid(row=0, column=0, padx=(4, 8))
        ctk.CTkEntry(studio_source, textvariable=self.studio_video,
                     placeholder_text="Выберите отдельный файл для расшифровки и хайлайтов").grid(row=0, column=1, sticky="ew")
        ctk.CTkButton(studio_source, text="Выбрать…", width=112,
                      command=self._choose_studio_video).grid(row=0, column=2, padx=8)
        studio_toolbar = ctk.CTkFrame(studio_tab, fg_color="transparent")
        studio_toolbar.grid(row=1, column=0, padx=4, pady=4, sticky="ew")
        ctk.CTkButton(studio_toolbar, text="Полная расшифровка", command=self._transcribe_studio).pack(side="left")
        ctk.CTkButton(studio_toolbar, text="Импорт чата", command=self._import_chat).pack(side="left", padx=6)
        ctk.CTkButton(studio_toolbar, text="Посмотреть видео", command=self._analyze_video).pack(side="left")
        ctk.CTkButton(studio_toolbar, text="Добавить момент", command=self._add_manual_moment).pack(side="left", padx=6)
        ctk.CTkButton(studio_toolbar, text="Сохранить SRT", command=self._export_srt).pack(side="left", padx=6)
        ctk.CTkLabel(studio_toolbar, text="Модель").pack(side="left", padx=(10, 4))
        ctk.CTkOptionMenu(studio_toolbar, variable=self.studio_model, values=MODEL_OPTIONS, width=150).pack(side="left")
        ctk.CTkButton(studio_toolbar, text="Сохранить проект", command=self._save_project).pack(side="right")
        ctk.CTkButton(studio_toolbar, text="Открыть проект", command=self._load_project).pack(side="right", padx=6)
        self.transcript_search = ctk.StringVar()
        search = ctk.CTkFrame(studio_tab, fg_color="transparent")
        search.grid(row=2, column=0, padx=4, pady=(0, 4), sticky="ew")
        search.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(search, text="Поиск в расшифровке").grid(row=0, column=0, padx=(4, 8))
        ctk.CTkEntry(search, textvariable=self.transcript_search,
                     placeholder_text="слово или фраза").grid(row=0, column=1, sticky="ew")
        ctk.CTkButton(search, text="Найти", width=80, command=self._search_transcript).grid(row=0, column=2, padx=6)
        self.transcript_box = ctk.CTkTextbox(studio_tab, wrap="word", height=260)
        self.transcript_box.grid(row=3, column=0, padx=4, pady=4, sticky="nsew")
        self.transcript_box.configure(state="disabled")
        self.moments_frame = ctk.CTkScrollableFrame(studio_tab, label_text="Моменты и сигналы")
        self.moments_frame.grid(row=4, column=0, padx=4, pady=4, sticky="ew")
        self.moments_frame.grid_columnconfigure(0, weight=1)
        self._show_moments()
        export = ctk.CTkFrame(studio_tab)
        export.grid(row=5, column=0, padx=4, pady=6, sticky="ew")
        ctk.CTkLabel(export, text="Профиль").pack(side="left", padx=(10, 5), pady=8)
        ctk.CTkSegmentedButton(export, variable=self.export_profile,
                               values=["Original 16:9", "YouTube 16:9", "Shorts 9:16", "TikTok 9:16", "Square 1:1", "Audio MP3"],
                               width=520).pack(side="left", padx=5)
        ctk.CTkSegmentedButton(export, variable=self.studio_layout,
                               values=["Обычный Shorts", "Стример сверху + чат снизу", "Игра сверху + чат снизу"],
                               width=360).pack(side="left", padx=5)
        ctk.CTkCheckBox(export, text="Субтитры", variable=self.studio_subtitles).pack(side="left", padx=8)
        ctk.CTkCheckBox(export, text="Чат", variable=self.studio_chat_overlay).pack(side="left", padx=8)
        ctk.CTkEntry(export, textvariable=self.studio_watermark, width=150,
                     placeholder_text="Водяной знак").pack(side="left", padx=8)
        self.export_button = ctk.CTkButton(export, text="Экспорт одобренных", command=self._export_moments,
                                           state="disabled")
        self.export_button.pack(side="right", padx=10)

        youtube = ctk.CTkFrame(online_tab, width=900, height=390)
        youtube.grid(row=0, column=1, padx=10, pady=10, sticky="new")
        youtube.grid_propagate(False)
        self.online_panel = youtube
        youtube.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(youtube, text="Превью").grid(row=0, column=0, padx=(12, 8), pady=9)
        ctk.CTkEntry(youtube, textvariable=self.preview_url, placeholder_text="Ссылка YouTube для превью").grid(row=0, column=1, padx=4, sticky="ew")
        self.youtube_button = ctk.CTkButton(youtube, text="Скачать превью", width=140, fg_color="#287d58", hover_color="#226b4b", command=self._download_youtube_thumbnail)
        self.youtube_button.grid(row=0, column=2, padx=(8, 12))
        self.youtube_result_label = ctk.CTkLabel(youtube, textvariable=self.youtube_result, anchor="w", justify="left")
        self.youtube_result_label.grid(row=1, column=1, padx=4, pady=(0, 8), sticky="ew")
        self.open_thumbnail_button = ctk.CTkButton(youtube, text="Открыть", width=90, state="disabled")
        self.open_thumbnail_button.grid(row=1, column=2, padx=(8, 12), pady=(0, 8))
        ctk.CTkLabel(youtube, text="Стрим / запись").grid(row=2, column=0, padx=(12, 8), pady=(2, 8))
        ctk.CTkEntry(youtube, textvariable=self.stream_url, placeholder_text="Ссылка YouTube или Twitch для видео").grid(row=2, column=1, padx=4, pady=(2, 8), sticky="ew")
        self.stream_button = ctk.CTkButton(youtube, text="Скачать видео", width=140, fg_color="#2674a8", hover_color="#1f638f", command=self._download_stream)
        self.stream_button.grid(row=2, column=2, padx=(8, 12), pady=(2, 8))
        self.stop_stream_button = ctk.CTkButton(youtube, text="Остановить", width=112, fg_color="#a33d3d", hover_color="#873131", state="disabled", command=self._stop_stream)
        self.stop_stream_button.grid(row=3, column=2, padx=(8, 12), pady=(0, 8))
        self.stream_progress = ctk.CTkProgressBar(youtube)
        self.stream_progress.grid(row=3, column=1, padx=4, pady=(0, 8), sticky="ew")
        self.stream_progress.set(0)
        self.stream_status = ctk.CTkLabel(youtube, text="", anchor="w")
        self.stream_status.grid(row=4, column=1, padx=4, pady=(0, 8), sticky="ew")
        ctk.CTkLabel(youtube, text="Папка превью").grid(row=5, column=0, padx=(12, 8), pady=(2, 8))
        ctk.CTkEntry(youtube, textvariable=self.preview_dir).grid(row=5, column=1, padx=4, pady=(2, 8), sticky="ew")
        ctk.CTkButton(youtube, text="Обзор…", width=90, command=lambda: self._choose_download_dir(self.preview_dir)).grid(row=5, column=2, padx=(8, 12), pady=(2, 8))
        ctk.CTkLabel(youtube, text="Папка стримов").grid(row=6, column=0, padx=(12, 8), pady=(2, 8))
        ctk.CTkEntry(youtube, textvariable=self.stream_dir).grid(row=6, column=1, padx=4, pady=(2, 8), sticky="ew")
        ctk.CTkButton(youtube, text="Обзор…", width=90, command=lambda: self._choose_download_dir(self.stream_dir)).grid(row=6, column=2, padx=(8, 12), pady=(2, 8))

        self.history_frame = ctk.CTkScrollableFrame(history_tab, label_text="Зацензуренные видео")
        self.history_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self._show_history()
        self.progress = ctk.CTkProgressBar(self)
        self.progress.grid(row=2, column=0, padx=22, pady=(10, 3), sticky="ew")
        self.progress.set(0)
        ctk.CTkLabel(self, textvariable=self.status, anchor="w").grid(row=3, column=0, padx=22, pady=(2, 12), sticky="ew")

    def _choose_video(self):
        name = filedialog.askopenfilename(title="Выберите видео", filetypes=[("Видео", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"), ("Все файлы", "*.*")])
        if name:
            self._set_video(name)

    def _studio_source(self) -> Path | None:
        path = Path(self.studio_video.get()).expanduser()
        if not path.is_file():
            messagebox.showerror("Нет видео", "Сначала выберите видео в поле «Видео студии».")
            return None
        return path

    def _choose_studio_video(self):
        name = filedialog.askopenfilename(title="Видео для студии", filetypes=[("Видео", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"), ("Все файлы", "*.*")])
        if name:
            self.studio_video.set(name)
            self.status.set(f"Выбрано видео студии: {Path(name).name}")

    def _transcribe_studio(self):
        source = self._studio_source()
        if source is None:
            return
        self.status.set("Готовлю полную расшифровку…")
        self.progress.set(0.05)
        def worker():
            try:
                segments, language = transcribe_video(source, self.studio_model.get(),
                                                      lambda text: self.events.put(("studio_status", text)))
                self.events.put(("transcript_done", (segments, language)))
            except Exception as exc:
                self.events.put(("studio_error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _render_transcript(self, segments=None):
        rows = self.transcript if segments is None else segments
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        for segment in rows:
            self.transcript_box.insert("end", f"{timestamp(float(segment['start']))}  {segment.get('text', '')}\n")
        self.transcript_box.configure(state="disabled")

    def _search_transcript(self):
        query = self.transcript_search.get().strip().casefold()
        self._render_transcript([segment for segment in self.transcript if not query or query in segment.get("text", "").casefold()])
        self.status.set(f"Найдено строк: {sum(1 for segment in self.transcript if not query or query in segment.get('text', '').casefold())}")

    def _import_chat(self):
        name = filedialog.askopenfilename(title="Импорт чата", filetypes=[("Чат", "*.txt *.log *.json"), ("Все файлы", "*.*")])
        if not name:
            return
        try:
            self.chat_messages = parse_chat_log(Path(name))
            self.studio_moments.extend(chat_activity(self.chat_messages))
            self._show_moments()
            self.status.set(f"Импортировано сообщений чата: {len(self.chat_messages)}")
        except Exception as exc:
            messagebox.showerror("Ошибка чата", str(exc))

    def _analyze_video(self):
        source = self._studio_source()
        if source is None:
            return
        self.status.set("Просматриваю кадры и ищу чат/интересные сцены…")
        def worker():
            try:
                moments = visual_highlights(source, self.transcript,
                                            progress=lambda text: self.events.put(("studio_status", text)))
                self.events.put(("moments_done", moments))
            except Exception as exc:
                self.events.put(("studio_error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _add_studio_moment(self, start: float, end: float, label: str = "Ручной момент"):
        self.studio_moments.append({"start": start, "end": end, "score": 1.0,
                                    "source": "manual", "label": label, "enabled": True})
        self._show_moments()

    def _add_manual_moment(self):
        start_text = simpledialog.askstring("Начало момента", "Секунды от начала видео:", parent=self)
        if start_text is None:
            return
        end_text = simpledialog.askstring("Конец момента", "Секунды от начала видео:", parent=self)
        if end_text is None:
            return
        try:
            start, end = float(start_text.replace(",", ".")), float(end_text.replace(",", "."))
            if start < 0 or end <= start:
                raise ValueError
        except ValueError:
            messagebox.showerror("Некорректный момент", "Введите положительные секунды, конец должен быть позже начала.")
            return
        self._add_studio_moment(start, end)
        self.status.set(f"Добавлен момент {timestamp(start)}–{timestamp(end)}")

    def _show_moments(self):
        if not hasattr(self, "moments_frame"):
            return
        for child in self.moments_frame.winfo_children():
            child.destroy()
        if not self.studio_moments:
            ctk.CTkLabel(self.moments_frame, text="Здесь появятся всплески звука, моменты из чата и ручные отметки.", anchor="w").grid(row=0, column=0, padx=10, pady=6, sticky="ew")
            return
        for index, moment in enumerate(self.studio_moments):
            variable = ctk.BooleanVar(value=moment.get("enabled", True))
            description = moment.get("description", "")
            ctk.CTkCheckBox(self.moments_frame,
                            text=f"{timestamp(float(moment['start']))}–{timestamp(float(moment['end']))} · {moment.get('label', moment.get('source', 'момент'))}"
                                 + (f" · {description[:180]}" if description else ""),
                            variable=variable,
                            command=lambda i=index, v=variable: self._toggle_moment(i, v.get())).grid(row=index, column=0, padx=10, pady=3, sticky="w")
            ctk.CTkButton(self.moments_frame, text="Предпросмотр", width=110,
                          command=lambda item=moment: self._preview_moment(item)).grid(row=index, column=1, padx=6, pady=2)
        self.export_button.configure(state="normal" if any(item.get("enabled", True) for item in self.studio_moments) else "disabled")

    def _preview_moment(self, moment):
        source = self._studio_source()
        ffplay = shutil.which("ffplay")
        if source is None or not ffplay:
            self.status.set("Для предпросмотра установите ffmpeg с ffplay.")
            return
        try:
            subprocess.Popen([ffplay, "-hide_banner", "-loglevel", "error", "-ss", str(moment["start"]),
                              "-t", str(min(45, float(moment["end"]) - float(moment["start"]))), str(source)])
            self.status.set(f"Предпросмотр: {timestamp(float(moment['start']))}")
        except OSError as exc:
            messagebox.showerror("Ошибка предпросмотра", str(exc))

    def _toggle_moment(self, index, enabled):
        self.studio_moments[index]["enabled"] = enabled

    def _export_srt(self):
        if not self.transcript:
            messagebox.showinfo("Нет расшифровки", "Сначала запустите полную расшифровку.")
            return
        name = filedialog.asksaveasfilename(title="Сохранить субтитры", defaultextension=".srt",
                                            filetypes=[("SubRip", "*.srt")],
                                            initialfile=f"{Path(self.studio_video.get()).stem}.srt")
        if name:
            Path(name).write_text(transcript_to_srt(self.transcript,
                                                     {hit.word for hit in self.hits if hit.enabled}), encoding="utf-8")
            self.status.set(f"Субтитры сохранены: {name}")

    def _export_moments(self):
        source = self._studio_source()
        selected = [item for item in self.studio_moments if item.get("enabled", True)]
        if source is None or not selected:
            return
        output = Path(self.output_dir.get()).expanduser() / "highlights"
        profile = self.export_profile.get()
        self.export_button.configure(state="disabled")
        def worker():
            paths = []
            try:
                for index, moment in enumerate(selected, 1):
                    destination = output / f"{source.stem}_highlight_{index:02d}.{'mp3' if profile == 'Audio MP3' else 'mp4'}"
                    paths.append(render_moment(source, moment, destination, profile, self.transcript,
                                                self.chat_messages, self.studio_subtitles.get(),
                                                self.studio_chat_overlay.get(), self.studio_watermark.get(),
                                                {hit.word for hit in self.hits if hit.enabled},
                                                self.studio_layout.get(),
                                                lambda text: self.events.put(("studio_status", text))))
                self.events.put(("studio_export_done", [str(path) for path in paths]))
            except Exception as exc:
                self.events.put(("studio_error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _save_project(self):
        source = Path(self.studio_video.get()).expanduser() if self.studio_video.get() else None
        name = filedialog.asksaveasfilename(title="Сохранить проект", defaultextension=".cutproj",
                                            filetypes=[("Cut-Helper project", "*.cutproj")],
                                            initialfile=f"{source.stem if source else 'project'}.cutproj")
        if not name:
            return
        project = {"format": "cut-helper-project-v1", "source": str(source or ""),
                   "transcript": self.transcript, "language": self.transcript_language,
                   "moments": self.studio_moments, "chat": self.chat_messages,
                   "hits": [hit.__dict__ for hit in self.hits], "model": self.studio_model.get()}
        try:
            write_project(Path(name), project)
            self.project_path = Path(name)
            self.status.set(f"Проект сохранён: {name}")
        except OSError as exc:
            messagebox.showerror("Ошибка проекта", str(exc))

    def _load_project(self):
        name = filedialog.askopenfilename(title="Открыть проект", filetypes=[("Cut-Helper project", "*.cutproj")])
        if not name:
            return
        try:
            project = read_project(Path(name))
            self.project_path = Path(name)
            self.studio_video.set(project.get("source", ""))
            self.transcript = project.get("transcript", [])
            self.transcript_language = project.get("language", "")
            self.studio_moments = project.get("moments", [])
            self.chat_messages = project.get("chat", [])
            self.studio_model.set(project.get("model", self.studio_model.get()))
            self.hits = [Hit(str(item.get("word", "")), float(item.get("start", 0)),
                             float(item.get("end", 0)), bool(item.get("enabled", True)))
                         for item in project.get("hits", [])]
            self._render_transcript()
            self._show_moments()
            self._show_hits()
            self.status.set(f"Проект открыт: {Path(name).name}")
        except Exception as exc:
            messagebox.showerror("Ошибка проекта", str(exc))

    def _drop_video(self, event):
        try:
            paths = self.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        normalized = []
        for raw_path in paths:
            value = str(raw_path).strip()
            if value.startswith("file://"):
                value = url2pathname(unquote(urlparse(value).path))
            elif value.startswith("file:"):
                value = url2pathname(unquote(value[5:]))
            if value:
                normalized.append(value)
        if not normalized:
            return REFUSE_DROP
        path = Path(normalized[0]).expanduser()
        supported = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
        if not path.is_file() or path.suffix.casefold() not in supported:
            messagebox.showerror("Неподдерживаемый файл", "Перетащите видео в формате MP4, MKV, MOV, AVI, WebM или M4V.")
            return REFUSE_DROP
        target = getattr(event, "widget", None)
        studio_target = False
        if target is not None and hasattr(self, "studio_tab"):
            current = target
            while current is not None:
                if current == self.studio_tab:
                    studio_target = True
                    break
                try:
                    parent_name = current.winfo_parent()
                    current = current.nametowidget(parent_name) if parent_name else None
                except (AttributeError, TclError):
                    current = None
        if studio_target:
            self.studio_video.set(str(path))
            self.status.set(f"Выбрано видео студии: {path.name}")
        else:
            self._set_video(str(path))
        return COPY

    def _set_video(self, name):
        self.video.set(name)
        self.hits = []
        self.transcript = []
        self.transcript_language = ""
        self.studio_moments = []
        self.chat_messages = []
        if hasattr(self, "transcript_box"):
            self._render_transcript()
            self._show_moments()
        self._show_hits()
        self.render_button.configure(state="disabled")
        self.status.set(f"Выбрано видео: {Path(name).name}")

    def _choose_dir(self):
        name = filedialog.askdirectory(title="Папка результата", initialdir=self.output_dir.get() or str(Path.home()))
        if name:
            self.output_dir.set(name)

    def _choose_download_dir(self, variable):
        name = filedialog.askdirectory(title="Папка загрузки", initialdir=variable.get() or str(Path.home()))
        if name:
            variable.set(name)

    def _choose_sound(self):
        name = filedialog.askopenfilename(title="Выберите звук", filetypes=[("Аудио", "*.wav *.mp3 *.ogg *.flac *.m4a"), ("Все файлы", "*.*")])
        if name:
            self.sound.set(name)

    def _download_youtube_thumbnail(self):
        url = self.preview_url.get().strip()
        parsed = urlparse(url)
        host = (parsed.hostname or "").casefold()
        if parsed.scheme not in ("http", "https") or host not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtube-nocookie.com"}:
            messagebox.showerror("Неверная ссылка", "Вставьте ссылку на видео YouTube.")
            return
        self.youtube_button.configure(state="disabled")
        self.open_thumbnail_button.configure(state="disabled")
        self.youtube_result.set("Получаю название и скачиваю превью…")

        def worker():
            try:
                from yt_dlp import YoutubeDL
                preview_dir = Path(self.preview_dir.get()).expanduser()
                preview_dir.mkdir(parents=True, exist_ok=True)
                template = str(preview_dir / "%(title).150B [%(id)s].%(ext)s")
                options = {
                    "skip_download": True,
                    "writethumbnail": True,
                    "noplaylist": True,
                    "quiet": True,
                    "no_warnings": True,
                    "outtmpl": {"default": template, "thumbnail": template},
                }
                with YoutubeDL(options) as downloader:
                    info = downloader.extract_info(url, download=True)
                video_id = info.get("id", "")
                files = [item for item in preview_dir.iterdir()
                         if item.is_file() and f"[{video_id}]" in item.stem]
                if not files:
                    raise RuntimeError("YouTube не предоставил файл превью для этой ссылки.")
                thumbnail = max(files, key=lambda item: item.stat().st_mtime)
                self.events.put(("youtube_done", (info.get("title", "Видео YouTube"), str(thumbnail))))
            except Exception as exc:
                self.events.put(("youtube_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _download_stream(self):
        url = self.stream_url.get().strip()
        host = (urlparse(url).hostname or "").casefold()
        youtube_hosts = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
        twitch_hosts = {"twitch.tv", "www.twitch.tv", "m.twitch.tv", "clips.twitch.tv"}
        if urlparse(url).scheme not in ("http", "https") or host not in youtube_hosts | twitch_hosts:
            messagebox.showerror("Неверная ссылка", "Вставьте ссылку на публичный стрим или запись YouTube/Twitch.")
            return
        output_dir = Path(self.stream_dir.get()).expanduser()
        if not output_dir.is_dir():
            messagebox.showerror("Папка не найдена", "Выберите существующую папку результата.")
            return
        self._stream_stop.clear()
        self.stream_button.configure(state="disabled")
        self.stop_stream_button.configure(state="normal")
        self.stream_progress.set(0)
        self.stream_status.configure(text="Подключение к потоку…")

        def worker():
            try:
                from yt_dlp import YoutubeDL
                from yt_dlp.utils import DownloadError
                template = str(output_dir / "%(title).150B [%(id)s].%(ext)s")

                def report(data):
                    if self._stream_stop.is_set():
                        raise DownloadError("Остановлено пользователем")
                    state = data.get("status")
                    if state == "downloading":
                        downloaded = data.get("downloaded_bytes", 0)
                        total = data.get("total_bytes") or data.get("total_bytes_estimate")
                        speed = data.get("speed")
                        eta = data.get("eta")
                        percent = downloaded / total if total else None
                        self.events.put(("stream_progress", (percent, speed, eta, downloaded)))
                    elif state == "finished":
                        self.events.put(("stream_status", "Загрузка получена; объединяю аудио и видео…"))

                options = {
                    "format": "bestvideo*+bestaudio/best",
                    "outtmpl": template,
                    "merge_output_format": "mp4",
                    "noplaylist": True,
                    "live_from_start": True,
                    "continuedl": True,
                    "progress_hooks": [report],
                    "quiet": True,
                    "no_warnings": True,
                }
                with YoutubeDL(options) as downloader:
                    info = downloader.extract_info(url, download=True)
                video_id = str(info.get("id", ""))
                files = [item for item in output_dir.iterdir()
                         if item.is_file() and f"[{video_id}]" in item.stem
                         and not item.name.endswith((".part", ".ytdl"))]
                if not files:
                    raise RuntimeError("Загрузка завершилась, но итоговый видеофайл не найден.")
                result = max(files, key=lambda item: item.stat().st_mtime)
                self.events.put(("stream_done", (info.get("title", "Стрим"), str(result))))
            except Exception as exc:
                if self._stream_stop.is_set():
                    self.events.put(("stream_stopped", str(exc)))
                else:
                    self.events.put(("stream_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_stream(self):
        self._stream_stop.set()
        self.stream_status.configure(text="Останавливаю загрузку; временный файл сохранится для продолжения…")

    def _mode_changed(self, _value=None):
        enabled = self.mode.get() == "Custom Sound"
        self.sound_entry.configure(state="normal" if enabled else "disabled")
        self.sound_button.configure(state="normal" if enabled else "disabled")

    def _padding_changed(self, value):
        self.padding_label.configure(text=f"{int(value)} мс")

    def _load_blacklist(self):
        name = filedialog.askopenfilename(title="Загрузить список слов", filetypes=[("Текст", "*.txt"), ("Все файлы", "*.*")])
        if name:
            self.external_lists.append(Path(name))
            self.status.set(f"Добавлен список: {Path(name).name}")

    def _add_word(self):
        word = simpledialog.askstring("Добавить слово", "Точное слово, префикс* или *подстрока*:", parent=self)
        if word and word.strip():
            try:
                with (APP_DIR / "blacklist.txt").open("a", encoding="utf-8") as file:
                    file.write("\n" + word.strip())
                self.status.set(f"Добавлено в blacklist.txt: {word.strip()}")
            except OSError as exc:
                messagebox.showerror("Ошибка", f"Не удалось сохранить слово: {exc}")

    def _analyze(self):
        path = Path(self.video.get()).expanduser()
        if not path.is_file():
            messagebox.showerror("Ошибка", "Выберите существующий видеофайл.")
            return
        self.analyze_button.configure(state="disabled")
        self.render_button.configure(state="disabled")
        self.hits = []
        self._show_hits()
        self.progress.set(0.08)
        self.status.set("Подготовка анализа…")
        def worker():
            try:
                probe_media(path)
                terms = load_terms([APP_DIR / "blacklist.txt", *self.external_lists])
                hits = detect(
                    path, terms, self.model.get(),
                    lambda text: self.events.put(("status", text)),
                    lambda hit: self.events.put(("hit", hit)),
                    lambda current, total: self.events.put(("transcription_progress", (current, total))),
                )
                self.events.put(("analyzed", hits))
            except Exception as exc:
                self.events.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _render(self):
        source = Path(self.video.get()).expanduser()
        selected = [hit for hit in self.hits if hit.enabled]
        if not selected:
            messagebox.showinfo("Нет выбранных слов", "Отметьте хотя бы одно найденное слово.")
            return
        output_dir = Path(self.output_dir.get()).expanduser()
        destination = output_dir / f"{source.stem}_censored.mp4"
        custom = Path(self.sound.get()).expanduser() if self.sound.get() else None
        self.render_button.configure(state="disabled")
        self.analyze_button.configure(state="disabled")
        self.progress.set(0.56)
        def worker():
            try:
                render(source, destination, selected, self.mode.get(), int(self.padding.get()), custom,
                       lambda text: self.events.put(("status", text)))
                self.events.put(("rendered", str(destination)))
            except Exception as exc:
                self.events.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _show_hits(self):
        for child in self.results.winfo_children():
            child.destroy()
        self.hit_checks.clear()
        if not self.hits:
            ctk.CTkLabel(self.results, text="Результаты анализа появятся здесь.", anchor="w").grid(row=0, column=0, padx=10, pady=8, sticky="ew")
        for index, hit in enumerate(self.hits):
            self._add_hit_row(index, hit)
        self.status.set(f"Найдено слов: {len(self.hits)}")

    def _add_hit_row(self, index, hit):
        variable = ctk.BooleanVar(value=hit.enabled)
        checkbox = ctk.CTkCheckBox(
            self.results,
            text=f"{timestamp(hit.start)}  {hit.word}  ({hit.end - hit.start:.2f} с)",
            variable=variable,
            command=lambda i=index, v=variable: self._toggle_hit(i, v.get()),
        )
        checkbox.grid(row=index, column=0, padx=10, pady=4, sticky="w")
        self.hit_checks.append(checkbox)
        self._register_drop_targets(checkbox)

    def _load_history(self):
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            try:
                data = json.loads(LEGACY_HISTORY_FILE.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
            except (OSError, json.JSONDecodeError):
                return []

    def _load_geometry(self):
        try:
            geometry = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("geometry", "")
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
        match = re.fullmatch(r"(\d+)x(\d+)(?:[+-]\d+){0,2}", geometry)
        if not match:
            return None
        width, height = map(int, match.groups())
        if (width >= self.winfo_screenwidth() - 16
                and height >= self.winfo_screenheight() - 24):
            return None
        return geometry

    def _on_configure(self, event):
        if event.widget is not self:
            return
        self._resize_online_panel()
        if self.state() != "normal" or self._is_fullscreen_geometry():
            return
        if self._geometry_save_job is not None:
            self.after_cancel(self._geometry_save_job)
        self._geometry_save_job = self.after(450, self._save_geometry)

    def _resize_online_panel(self):
        if not hasattr(self, "online_panel"):
            return
        available = self.tabs.winfo_width() - 56
        width = max(680, min(1180, available))
        self.online_panel.configure(width=width)

    def _is_fullscreen_geometry(self):
        return (self.winfo_width() >= self.winfo_screenwidth() - 16
                and self.winfo_height() >= self.winfo_screenheight() - 24)

    def _save_geometry(self):
        self._geometry_save_job = None
        if self._is_fullscreen_geometry():
            return
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            temporary = SETTINGS_FILE.with_suffix(".tmp")
            temporary.write_text(json.dumps({"geometry": self.geometry()}, indent=2), encoding="utf-8")
            temporary.replace(SETTINGS_FILE)
        except OSError:
            pass

    def _on_close(self):
        if self._geometry_save_job is not None:
            self.after_cancel(self._geometry_save_job)
        self._save_geometry()
        self.destroy()

    def _show_history(self):
        for child in self.history_frame.winfo_children():
            child.destroy()
        if not self.history:
            ctk.CTkLabel(self.history_frame, text="Готовые видео появятся здесь.", anchor="w").grid(row=0, column=0, padx=10, pady=5, sticky="w")
            return
        for row, record in enumerate(self.history[:20]):
            result = Path(record.get("result", ""))
            source = Path(record.get("source", ""))
            label = f"{record.get('created', '')} · {record.get('mode', '')} · {source.name or 'исходник'} → {result.name or 'файл удалён'}"
            ctk.CTkLabel(self.history_frame, text=label, anchor="w").grid(row=row, column=0, padx=8, pady=3, sticky="w")
            ctk.CTkButton(self.history_frame, text="Видео", width=72,
                          state="normal" if result.is_file() else "disabled",
                          command=lambda path=result: self._open_path(path)).grid(row=row, column=1, padx=4, pady=2)
        self._register_drop_targets(self.history_frame)

    def _open_path(self, path: Path):
        try:
            if sys.platform == "win32":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл: {exc}")

    def _save_history(self, source: Path, destination: Path):
        record = {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "source": str(source), "result": str(destination), "mode": self.mode.get(),
            "words": [{"word": hit.word, "start": hit.start, "end": hit.end} for hit in self.hits if hit.enabled],
        }
        self.history = [record] + [item for item in self.history if item.get("result") != str(destination)]
        self.history = self.history[:50]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        temporary = HISTORY_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(HISTORY_FILE)
        self._show_history()

    def _append_live_hit(self, hit):
        if any(old.word.casefold() == hit.word.casefold() and abs(old.start - hit.start) < 0.12
               for old in self.hits):
            return
        if not self.hits:
            for child in self.results.winfo_children():
                child.destroy()
        self.hits.append(hit)
        self._add_hit_row(len(self.hits) - 1, hit)
        self.status.set(f"Распознано слов: {len(self.hits)}")

    def _toggle_hit(self, index, enabled):
        self.hits[index].enabled = enabled

    def _poll(self):
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "status":
                    self.status.set(payload)
                    if "Извлечение аудио" in payload:
                        self.progress.set(0.15)
                    elif "Транскрибация" in payload:
                        self.progress.set(0.27)
                    elif "GPU недоступна" in payload:
                        self.progress.set(0.27)
                    elif "таймкодов" in payload:
                        self.progress.set(0.58)
                    elif "Рендеринг" in payload:
                        match = re.search(r"(\d+)%", payload)
                        self.progress.set(0.58 + 0.41 * int(match.group(1)) / 100 if match else 0.6)
                elif event == "hit":
                    self._append_live_hit(payload)
                    self.status.set(f"Найден мат: {payload.word} · {timestamp(payload.start)}–{timestamp(payload.end)}")
                elif event == "transcription_progress":
                    current, total = payload
                    fraction = min(1.0, current / total) if total else 0.0
                    self.progress.set(0.27 + 0.3 * fraction)
                    self.status.set(f"Транскрибация: {timestamp(current)} / {timestamp(total)} · найдено {len(self.hits)}")
                elif event == "analyzed":
                    disabled = [(hit.word.casefold(), hit.start) for hit in self.hits if not hit.enabled]
                    self.hits = payload
                    for hit in self.hits:
                        if any(word == hit.word.casefold() and abs(start - hit.start) < 0.12
                               for word, start in disabled):
                            hit.enabled = False
                    self._show_hits()
                    self.progress.set(1)
                    self.analyze_button.configure(state="normal")
                    self.render_button.configure(state="normal" if payload else "disabled")
                elif event == "rendered":
                    try:
                        self._save_history(Path(self.video.get()).expanduser(), Path(payload))
                    except OSError as exc:
                        self.status.set(f"Видео готово, но историю не удалось сохранить: {exc}")
                    self.progress.set(1)
                    if "историю не удалось сохранить" not in self.status.get():
                        self.status.set(f"Готово: {payload}")
                    self.analyze_button.configure(state="normal")
                    messagebox.showinfo("Готово", f"Файл сохранён:\n{payload}")
                elif event == "studio_status":
                    self.status.set(payload)
                elif event == "transcript_done":
                    self.transcript, self.transcript_language = payload
                    self._render_transcript()
                    self.progress.set(1)
                    self.status.set(f"Расшифровка готова · язык: {self.transcript_language or 'не определён'} · строк: {len(self.transcript)}")
                elif event == "moments_done":
                    self.studio_moments.extend(payload)
                    self._show_moments()
                    self.status.set(f"Добавлено звуковых моментов: {len(payload)}")
                elif event == "studio_export_done":
                    self.export_button.configure(state="normal")
                    self.status.set(f"Экспортировано файлов: {len(payload)}")
                    messagebox.showinfo("Экспорт готов", "Сохранено:\n" + "\n".join(payload))
                elif event == "studio_error":
                    self.export_button.configure(state="normal")
                    self.status.set("Ошибка студии")
                    messagebox.showerror("Ошибка студии", payload)
                elif event == "youtube_done":
                    title, path = payload
                    self.youtube_result.set(f"{title}\n{path}")
                    self.open_thumbnail_button.configure(state="normal", command=lambda p=Path(path): self._open_path(p))
                    self.youtube_button.configure(state="normal")
                    self.status.set(f"Превью скачано: {Path(path).name}")
                elif event == "youtube_error":
                    self.youtube_result.set("")
                    self.youtube_button.configure(state="normal")
                    messagebox.showerror("Ошибка YouTube", payload)
                elif event == "stream_progress":
                    percent, speed, eta, downloaded = payload
                    if percent is not None:
                        self.stream_progress.set(min(1.0, max(0.0, percent)))
                        progress_text = f"{percent * 100:.1f}%"
                    else:
                        self.stream_progress.set(0.08 if self.stream_progress.get() < 0.08 else self.stream_progress.get())
                        progress_text = "live"
                    speed_text = f" · {speed / 1024 / 1024:.1f} МБ/с" if speed else ""
                    eta_text = f" · осталось {int(eta)} с" if eta is not None else ""
                    self.stream_status.configure(text=f"{progress_text}{speed_text}{eta_text} · получено {downloaded / 1024 / 1024:.1f} МБ")
                elif event == "stream_status":
                    self.stream_status.configure(text=payload)
                elif event == "stream_done":
                    title, path = payload
                    self.stream_status.configure(text=f"Готово: {title} · {path}")
                    self.stream_progress.set(1)
                    self.stream_button.configure(state="normal")
                    self.stop_stream_button.configure(state="disabled")
                    self._set_video(path)
                elif event == "stream_stopped":
                    self.stream_status.configure(text="Загрузка остановлена; временные данные сохранены.")
                    self.stream_progress.set(0)
                    self.stream_button.configure(state="normal")
                    self.stop_stream_button.configure(state="disabled")
                elif event == "stream_error":
                    self.stream_status.configure(text="Ошибка загрузки")
                    self.stream_button.configure(state="normal")
                    self.stop_stream_button.configure(state="disabled")
                    messagebox.showerror("Ошибка загрузки", payload)
                elif event == "error":
                    self.progress.set(0)
                    self.status.set("Ошибка")
                    self.analyze_button.configure(state="normal")
                    self.render_button.configure(state="normal" if self.hits else "disabled")
                    messagebox.showerror("Ошибка", payload)
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    CensorApp().mainloop()
