import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np


class SimWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Симулятор разностно-дальномерного комплекса (2D)")
        self.root.geometry("850x600")
        self.create_widgets()
        self.setup_treeview_editing()

    def create_widgets(self):
        # Параметры БПЛА и моделирования
        uav_frame = tk.LabelFrame(self.root, text="Параметры моделирования и БПЛА", padx=10, pady=5)
        uav_frame.pack(fill="x", padx=10, pady=5)

        fields = [
            ("Начальная X (м):", "uav_x", "0"),
            ("Начальная Y (м):", "uav_y", "0"),
            ("Скорость X (м/с):", "vel_x", "10"),
            ("Скорость Y (м/с):", "vel_y", "5"),
            ("Время моделирования (с):", "sim_time", "10"),
            ("Период сигнала (мс):", "signal_period_ms", "100"),
        ]

        for i, (label_text, attr_name, default) in enumerate(fields):
            lbl = tk.Label(uav_frame, text=label_text)
            entry = tk.Entry(uav_frame, width=12)
            entry.insert(0, default)
            setattr(self, attr_name, entry)
            lbl.grid(row=i // 3, column=(i % 3) * 2, sticky="e", padx=(10, 2), pady=3)
            entry.grid(row=i // 3, column=(i % 3) * 2 + 1, sticky="w", padx=(2, 10), pady=3)

        # Таблица станций
        station_frame = tk.LabelFrame(self.root, text="Станции приёма", padx=10, pady=5)
        station_frame.pack(fill="both", expand=True, padx=10, pady=5)

        columns = ("ID", "X (м)", "Y (м)")
        self.station_tree = ttk.Treeview(station_frame, columns=columns, show="headings", height=8)
        for col in columns:
            self.station_tree.heading(col, text=col)
            self.station_tree.column(col, width=120, anchor="center")

        vsb = ttk.Scrollbar(station_frame, orient="vertical", command=self.station_tree.yview)
        hsb = ttk.Scrollbar(station_frame, orient="horizontal", command=self.station_tree.xview)
        self.station_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.station_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        station_frame.grid_rowconfigure(0, weight=1)
        station_frame.grid_columnconfigure(0, weight=1)

        # Начальные станции
        for i in range(5):
            self.station_tree.insert("", "end", values=(i + 1, 100 * i, 0))

        # Кнопки управления
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=5)

        self.add_btn = tk.Button(btn_frame, text="Добавить станцию", command=self.add_station)
        self.del_btn = tk.Button(btn_frame, text="Удалить станцию", command=self.del_station)
        self.run_btn = tk.Button(btn_frame, text="Запустить симуляцию", command=self.run_simulation)
        self.export_btn = tk.Button(btn_frame, text="Экспорт конфигурации", command=self.export_config)
        self.import_btn = tk.Button(btn_frame, text="Импорт конфигурации", command=self.import_config)

        self.add_btn.pack(side="left", padx=5)
        self.del_btn.pack(side="left", padx=5)
        self.run_btn.pack(side="left", padx=5)
        self.export_btn.pack(side="left", padx=5)
        self.import_btn.pack(side="left", padx=5)

    # Редактирование ячеек Treeview по двойному клику
    def setup_treeview_editing(self):
        self.edit_popup = None
        self.station_tree.bind("<Double-1>", self._start_editing)

    def _start_editing(self, event):
        if self.edit_popup:
            self._finish_editing()

        row_id = self.station_tree.identify_row(event.y)
        column = self.station_tree.identify_column(event.x)
        if not row_id or not column:
            return

        bbox = self.station_tree.bbox(row_id, column)
        if not bbox:
            return

        x, y, width, height = bbox
        value = self.station_tree.set(row_id, column)

        self.edit_popup = {
            "entry": tk.Entry(self.station_tree),
            "row_id": row_id,
            "column": column,
            "value": value
        }

        entry = self.edit_popup["entry"]
        entry.place(x=x + 1, y=y + 1, width=width - 2, height=height - 2)
        entry.insert(0, value)
        entry.focus()
        entry.select_range(0, tk.END)

        entry.bind("<Return>", self._finish_editing)
        entry.bind("<Escape>", self._cancel_editing)
        entry.bind("<FocusOut>", lambda e: self.root.after(50, self._finish_editing))

    def _finish_editing(self, event=None):
        if not self.edit_popup:
            return

        entry = self.edit_popup["entry"]
        row_id = self.edit_popup["row_id"]
        column = self.edit_popup["column"]
        old_value = self.edit_popup["value"]
        new_value = entry.get().strip()

        col_index = int(column.replace("#", "")) - 1
        if col_index >= 1:  # Редактируем только координаты X/Y
            if new_value == "":
                new_value = "0"
            try:
                float(new_value)
            except ValueError:
                messagebox.showwarning("Ошибка ввода", "Координаты должны быть числами.")
                entry.focus()
                return

        if new_value != old_value:
            self.station_tree.set(row_id, column, new_value)

        self._destroy_popup()

    def _cancel_editing(self, event=None):
        self._destroy_popup()

    def _destroy_popup(self):
        if self.edit_popup:
            self.edit_popup["entry"].destroy()
            self.edit_popup = None

    # Валидация входных данных
    def _parse_float(self, text: str, field_name: str) -> float:
        text = text.strip()
        if not text:
            raise ValueError(f"Поле '{field_name}' не должно быть пустым.")
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"Поле '{field_name}' должно содержать число.")

    def validate_inputs(self):
        x0 = self._parse_float(self.uav_x.get(), "Начальная X")
        y0 = self._parse_float(self.uav_y.get(), "Начальная Y")
        vx = self._parse_float(self.vel_x.get(), "Скорость X")
        vy = self._parse_float(self.vel_y.get(), "Скорость Y")
        T = self._parse_float(self.sim_time.get(), "Время моделирования")
        period_ms = self._parse_float(self.signal_period_ms.get(), "Период сигнала")

        if T <= 0 or period_ms <= 0:
            raise ValueError("Время и период должны быть положительными.")

        uav = {"x0": x0, "y0": y0, "vx": vx, "vy": vy}
        sim = {"duration_sec": T, "signal_period_ms": period_ms}

        stations = []
        seen_ids = set()
        for item_id in self.station_tree.get_children():
            values = self.station_tree.item(item_id)["values"]
            if len(values) < 3:
                continue

            sid = str(values[0]).strip()
            if not sid:
                raise ValueError("ID станции не может быть пустым.")
            if sid in seen_ids:
                raise ValueError(f"Дублирование ID станции: {sid}")
            seen_ids.add(sid)

            try:
                x, y = float(values[1]), float(values[2])
            except (ValueError, TypeError):
                raise ValueError(f"Некорректные координаты станции: {values}")

            stations.append({"id": sid, "x": x, "y": y})

        if not stations:
            raise ValueError("Добавьте хотя бы одну станцию.")

        return uav, sim, stations

    def add_station(self):
        children = self.station_tree.get_children()
        next_id = len(children) + 1
        if children:
            last_values = self.station_tree.item(children[-1])["values"]
            try:
                next_id = int(float(last_values[0])) + 1
            except (ValueError, TypeError):
                pass
        item = self.station_tree.insert("", "end", values=(next_id, 0, 0))
        self.station_tree.selection_set(item)
        self.station_tree.focus(item)
        self.station_tree.see(item)

    def del_station(self):
        items = self.station_tree.selection()
        if items:
            for item in items:
                self.station_tree.delete(item)
        else:
            children = self.station_tree.get_children()
            if children:
                self.station_tree.delete(children[-1])

    def export_config(self):
        try:
            uav, sim, stations = self.validate_inputs()
            config = {"uav": uav, "simulation": sim, "stations": stations}
            path = filedialog.asksaveasfilename(
                title="Сохранить конфигурацию",
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json")]
            )
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Успех", "Конфигурация сохранена!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить конфигурацию:\n{str(e)}")

    def import_config(self):
        try:
            path = filedialog.askopenfilename(title="Загрузить конфигурацию", filetypes=[("JSON Files", "*.json")])
            if not path:
                return

            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)

            required = {"uav", "simulation", "stations"}
            if not required.issubset(config):
                missing = required - set(config.keys())
                raise ValueError(f"Отсутствуют разделы: {missing}")

            uav = config["uav"]
            sim = config["simulation"]
            stations = config["stations"]

            self.uav_x.delete(0, tk.END); self.uav_x.insert(0, uav["x0"])
            self.uav_y.delete(0, tk.END); self.uav_y.insert(0, uav["y0"])
            self.vel_x.delete(0, tk.END); self.vel_x.insert(0, uav["vx"])
            self.vel_y.delete(0, tk.END); self.vel_y.insert(0, uav["vy"])
            self.sim_time.delete(0, tk.END); self.sim_time.insert(0, sim["duration_sec"])
            self.signal_period_ms.delete(0, tk.END); self.signal_period_ms.insert(0, sim["signal_period_ms"])

            for item in self.station_tree.get_children():
                self.station_tree.delete(item)

            for st in stations:
                self.station_tree.insert("", "end", values=(str(st["id"]), st["x"], st["y"]))

            messagebox.showinfo("Успех", "Конфигурация успешно загружена!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить конфигурацию:\n{str(e)}")

    def run_simulation(self):
        try:
            uav, sim, stations = self.validate_inputs()

            x0, y0 = uav["x0"], uav["y0"]
            vx, vy = uav["vx"], uav["vy"]
            T = sim["duration_sec"]
            period_ms = sim["signal_period_ms"]

            dt = period_ms / 1000.0
            times = np.arange(0, T + dt / 2, dt)
            packet_ids = [f"pkt_{i:04d}" for i in range(len(times))]
            uav_positions = np.array([[x0 + vx * t, y0 + vy * t] for t in times])

            # Истинные координаты БПЛА
            uav_truth = {
                pid: {"time": float(t), "x": float(pos[0]), "y": float(pos[1])}
                for pid, t, pos in zip(packet_ids, times, uav_positions)
            }

            result = {
                "uav": uav,
                "simulation": sim,
                "stations": [{"id": st["id"], "x": float(st["x"]), "y": float(st["y"])} for st in stations],
                "uav_truth": uav_truth
            }

            # Расчёт времени прихода сигналов на станции
            c = 299_792_458.0  # скорость света, м/с
            for st in stations:
                station_id = st["id"]
                s_pos = np.array([st["x"], st["y"]])
                station_data = {}
                for i, (t_emit, uav_pos) in enumerate(zip(times, uav_positions)):
                    distance = np.linalg.norm(uav_pos - s_pos)
                    t_receive = t_emit + distance / c
                    station_data[packet_ids[i]] = t_receive
                result[station_id] = station_data

            # Сохранение результатов
            path = filedialog.asksaveasfilename(
                title="Сохранить данные симуляции",
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json")]
            )
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Готово", "Файл с данными успешно сохранён!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при запуске симуляции:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SimWindow(root)
    root.mainloop()