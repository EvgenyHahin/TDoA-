import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.ndimage import uniform_filter1d
import scipy.optimize as opt



class TDoAEquationsWindow:
    def __init__(self, parent, reference_station, stations, station_coords, error_ns,
                 packet_data=None, reception_times=None, station_errors_ns=None,
                 true_uav_coords=None, calculated_coords=None):
        self.window = tk.Toplevel(parent)
        self.window.title(f"Уравнения гипербол TDoA - Пакет {packet_data['id'] if packet_data else 'Общий вид'}")
        self.window.geometry("1200x900")
        self.window.configure(bg="#f0f0f0")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.reference_station = reference_station
        self.stations = stations
        self.station_coords = station_coords
        self.error_ns = error_ns
        self.packet_data = packet_data
        self.reception_times = reception_times or {}
        self.station_errors_ns = station_errors_ns or {}
        self.true_uav_coords = true_uav_coords
        self.calculated_coords = calculated_coords
        self.C = 299792458.0
        self.C_ns = 0.299792458
        self.is_open = True
        self.create_widgets()

    def on_close(self):
        self.is_open = False
        self.window.destroy()

    def destroy(self):
        if self.is_open:
            self.is_open = False
            self.window.destroy()

    def create_widgets(self):
        packet_id = self.packet_data['id'] if self.packet_data else "Общий вид"
        time_val = self.packet_data.get('time')
        time_str = f"{time_val:.6f} с" if time_val is not None else ""
        coords_str = ""
        if self.true_uav_coords:
            x, y = self.true_uav_coords
            coords_str += f" | Истинные: ({x:.2f}, {y:.2f})"
        if self.calculated_coords:
            x_calc, y_calc = self.calculated_coords
            coords_str += f" | Вычисленные: ({x_calc:.2f}, {y_calc:.2f})"
            if self.true_uav_coords:
                x_true, y_true = self.true_uav_coords
                error = math.sqrt((x_calc - x_true) ** 2 + (y_calc - y_true) ** 2)
                coords_str += f" | Ошибка: {error:.2f} м"

        header_frame = tk.Frame(self.window, bg="#2c3e50", height=120)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        title_label = tk.Label(header_frame, text="Уравнения гипербол TDoA", font=('Arial', 18, 'bold'),
                               fg="white", bg="#2c3e50")
        title_label.pack(expand=True, pady=(10, 0))

        subtitle_label = tk.Label(header_frame,
                                  text=f"Пакет: {packet_id} {time_str}{coords_str}",
                                  font=('Arial', 12), fg="#ecf0f1", bg="#2c3e50")
        subtitle_label.pack(expand=True)

        num_errors = len([e for s, e in self.station_errors_ns.items() if s != self.reference_station])
        info_label = tk.Label(header_frame,
                              text=f"Опорная станция: {self.reference_station} | σ ошибки: {self.error_ns} нс | Индивидуальных ε: {num_errors}",
                              font=('Arial', 11), fg="#bdc3c7", bg="#2c3e50")
        info_label.pack(expand=True, pady=(0, 10))

        if self.packet_data and self.reception_times:
            graph_btn = tk.Button(header_frame, text="Построить гиперболы на графике",
                                  command=self.plot_hyperbolas, font=('Arial', 11, 'bold'),
                                  bg="#3498db", fg="white", padx=20, pady=5, relief=tk.RAISED, bd=2)
            graph_btn.pack(pady=(0, 10))

        main_frame = tk.Frame(self.window, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        canvas = tk.Canvas(main_frame, bg="#f0f0f0", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f0f0f0")
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=1160)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.create_equations_content(scrollable_frame)

        button_frame = tk.Frame(self.window, bg="#f0f0f0")
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        close_btn = tk.Button(button_frame, text="Закрыть", command=self.on_close,
                              font=('Arial', 11), bg="#e74c3c", fg="white", padx=30, pady=10)
        close_btn.pack()

    def create_equations_content(self, parent):
        ref_x, ref_y = self.station_coords.get(self.reference_station, (0, 0))
        if self.packet_data:
            moment_frame = self.create_section_frame(parent, f"Данные для пакета {self.packet_data['id']}")
            time_val = self.packet_data.get('time')
            time_str = f"{time_val:.6f} с" if time_val is not None else "N/A"
            info_text = (
                f"Время излучения БПЛА: {time_str}\n"
                f"Стандартное отклонение σ: {self.error_ns:.2f} нс"
            )
            if self.true_uav_coords:
                x, y = self.true_uav_coords
                info_text += f"\nИстинные координаты БПЛА: ({x:.2f}, {y:.2f}) м"
            if self.calculated_coords:
                x_calc, y_calc = self.calculated_coords
                info_text += f"\nВычисленные координаты БПЛА (TDoA): ({x_calc:.2f}, {y_calc:.2f}) м"
                if self.true_uav_coords:
                    x_true, y_true = self.true_uav_coords
                    error = math.sqrt((x_calc - x_true) ** 2 + (y_calc - y_true) ** 2)
                    info_text += f"\nОшибка определения положения: {error:.2f} м"
            tk.Label(moment_frame, text=info_text, font=('Arial', 11, 'bold'),
                     bg="#E8F5E9", justify=tk.LEFT, padx=15, pady=15).pack(fill=tk.X)

        times_frame = self.create_section_frame(parent, "Времена приема и ошибки")
        times_table = tk.Frame(times_frame, bg="#f8f9fa")
        times_table.pack(fill=tk.X, padx=15, pady=10)
        headers = ["Станция", "tᵢ, с", "εᵢ, нс", "εᵢ, м", "Координаты (x, y), м"]
        for i, header in enumerate(headers):
            tk.Label(times_table, text=header, font=('Arial', 10, 'bold'),
                     bg="#3498db", fg="white", padx=10, pady=5).grid(row=0, column=i, sticky="ew", padx=1, pady=1)

        for idx, station in enumerate(self.stations, 1):
            station_bg = "#2ecc71" if station == self.reference_station else "white"
            station_fg = "white" if station == self.reference_station else "black"
            tk.Label(times_table, text=f"Станция {station}",
                     bg=station_bg, fg=station_fg, padx=10, pady=5).grid(row=idx, column=0, sticky="ew", padx=1, pady=1)

            t = self.reception_times.get(station)
            time_text = f"{t:.9f}" if t is not None else "N/A"
            time_bg = "#FFF3CD" if t is not None else "#F5F5F5"
            tk.Label(times_table, text=time_text, bg=time_bg, padx=10, pady=5).grid(row=idx, column=1, sticky="ew",
                                                                                    padx=1, pady=1)

            eps_i = self.station_errors_ns.get(station, 0.0)
            eps_i_str = f"{eps_i:.2f}" if station != self.reference_station else "—"
            tk.Label(times_table, text=eps_i_str, bg="white", padx=10, pady=5).grid(row=idx, column=2, sticky="ew",
                                                                                    padx=1, pady=1)

            eps_i_m = eps_i * self.C_ns
            eps_i_m_str = f"{eps_i_m:.4f}" if station != self.reference_station else "—"
            tk.Label(times_table, text=eps_i_m_str, bg="white", padx=10, pady=5).grid(row=idx, column=3, sticky="ew",
                                                                                      padx=1, pady=1)

            x, y = self.station_coords.get(station, (0, 0))
            tk.Label(times_table, text=f"({x}, {y})", bg="white", padx=10, pady=5).grid(row=idx, column=4, sticky="ew",
                                                                                        padx=1, pady=1)

        for i in range(5):
            times_table.grid_columnconfigure(i, weight=1)

        formula_frame = self.create_section_frame(parent, "Общее уравнение гиперболы")
        tk.Label(formula_frame, text="Для пары станций (опорная и станция i):",
                 font=('Arial', 11, 'bold'), bg="#f8f9fa").pack(anchor=tk.W, padx=15, pady=(10, 5))
        formula_box = tk.Frame(formula_frame, bg="white", bd=2, relief=tk.SUNKEN)
        formula_box.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(formula_box,
                 text="√[(x - xᵢ)² + (y - yᵢ)²] - √[(x - x_ref)² + (y - y_ref)²] = c × (tᵢ - t_ref + εᵢ)",
                 font=('Cambria Math', 14), bg="white", padx=20, pady=15).pack()

        equations_frame = self.create_section_frame(parent,
                                                    f"Конкретные уравнения для пакета {self.packet_data['id'] if self.packet_data else 'Общий вид'}")
        self.hyperbolas_data = []
        for station in self.stations:
            if station == self.reference_station:
                continue
            x_i, y_i = self.station_coords.get(station, (0, 0))
            t_i = self.reception_times.get(station)
            t_ref = self.reception_times.get(self.reference_station)
            eq_card = tk.Frame(equations_frame, bg="white", bd=1, relief=tk.RAISED)
            eq_card.pack(fill=tk.X, padx=15, pady=10)
            header = tk.Frame(eq_card, bg="#9b59b6", height=30)
            header.pack(fill=tk.X)
            header.pack_propagate(False)
            tk.Label(header, text=f"Уравнение для станции {station} (опорная: {self.reference_station})",
                     font=('Arial', 11, 'bold'), fg="white", bg="#9b59b6").pack(expand=True)
            content_frame = tk.Frame(eq_card, bg="white", padx=15, pady=15)
            content_frame.pack(fill=tk.BOTH, expand=True)
            tk.Label(content_frame,
                     text=f"√[(x - {x_i})² + (y - {y_i})²] - √[(x - {ref_x})² + (y - {ref_y})²] =",
                     font=('Cambria Math', 12), bg="white").pack(anchor=tk.W)
            if t_i is not None and t_ref is not None:
                eps_i = self.station_errors_ns.get(station, 0.0)
                delta_t = (t_i - t_ref) + (eps_i * 1e-9)
                delta_d = self.C * delta_t
                distance_between_stations = math.sqrt((x_i - ref_x) ** 2 + (y_i - ref_y) ** 2)
                self.hyperbolas_data.append({
                    'station': station,
                    'x1': ref_x, 'y1': ref_y,
                    'x2': x_i, 'y2': y_i,
                    'delta_d': delta_d,
                    'a': abs(delta_d) / 2,
                    'c': distance_between_stations / 2,
                    'distance_between_stations': distance_between_stations
                })
                right_part = tk.Frame(content_frame, bg="white")
                right_part.pack(anchor=tk.W, pady=10)
                steps = [
                    f"c × (t{station} - t{self.reference_station} + ε{station})",
                    f"= {self.C:.0f} × ({t_i:.9f} - {t_ref:.9f} + {eps_i:.2f}×10⁻⁹)",
                    f"= {self.C:.0f} × ({t_i - t_ref:.9f} + {eps_i * 1e-9:.9f})",
                    f"= {self.C:.0f} × {delta_t:.9f}",
                    f"= {delta_d:.3f} м"
                ]
                for i, step in enumerate(steps):
                    tk.Label(right_part, text=step, font=('Cambria Math', 11 if i == 0 else 10),
                             bg="white", fg="#2c3e50" if i == 0 else "#7f8c8d").pack(anchor=tk.W)
                tk.Label(content_frame,
                         text=f"√[(x - {x_i})² + (y - {y_i})²] - √[(x - {ref_x})² + (y - {ref_y})²] = {delta_d:.3f} м",
                         font=('Cambria Math', 12, 'bold'), bg="#FFF3CD", padx=10, pady=5).pack(anchor=tk.W, pady=10)
                condition_check = f"Условие гиперболы: |Δd| = {abs(delta_d):.3f} м < D = {distance_between_stations:.3f} м"
                if abs(delta_d) < distance_between_stations:
                    condition_color = "#27ae60"
                    condition_status = "✓ Выполняется (это гипербола)"
                else:
                    condition_color = "#e74c3c"
                    condition_status = "✗ Не выполняется (не может быть гиперболой)"
                tk.Label(content_frame, text=f"{condition_check}\n{condition_status}",
                         font=('Arial', 10), bg="white", fg=condition_color).pack(anchor=tk.W, pady=5)
                interp = f"БПЛА ближе к {'опорной станции' if delta_d > 0 else f'станции {station}'} на {abs(delta_d):.1f} м"
                tk.Label(content_frame, text=f"Интерпретация: {interp}",
                         font=('Arial', 10, 'italic'), bg="white", fg="#27ae60").pack(anchor=tk.W, pady=5)
                if self.true_uav_coords:
                    x_true, y_true = self.true_uav_coords
                    dist_to_ref = math.sqrt((x_true - ref_x) ** 2 + (y_true - ref_y) ** 2)
                    dist_to_station = math.sqrt((x_true - x_i) ** 2 + (y_true - y_i) ** 2)
                    actual_delta_d = dist_to_station - dist_to_ref
                    error = abs(actual_delta_d - delta_d)
                    tk.Label(content_frame,
                             text=f"Проверка истинного положения: Δd_ист = {actual_delta_d:.3f} м, расхождение = {error:.3f} м",
                             font=('Arial', 9), bg="white", fg="#8e44ad").pack(anchor=tk.W, pady=5)
            else:
                tk.Label(content_frame, text="Нет данных о времени приема",
                         font=('Arial', 10), bg="white", fg="#e74c3c").pack(anchor=tk.W, pady=10)

        if self.packet_data and any(t is not None for t in self.reception_times.values()):
            geom_frame = self.create_section_frame(parent, "Геометрическая интерпретация")
            geom_text = (
                "Каждое уравнение определяет гиперболу — геометрическое место точек,\n"
                "разность расстояний от которых до двух станций постоянна.\n"
                "Условие гиперболы: |Δd| < D, где D — расстояние между станциями."
            )
            tk.Label(geom_frame, text=geom_text, font=('Arial', 11), bg="#f8f9fa", justify=tk.LEFT).pack(fill=tk.X,
                                                                                                         padx=15,
                                                                                                         pady=10)
            calc_frame = tk.Frame(geom_frame, bg="#E3F2FD", bd=1, relief=tk.SUNKEN)
            calc_frame.pack(fill=tk.X, padx=15, pady=10)
            calc_text = "Δd = c × (tᵢ - t_ref + εᵢ), где εᵢ ~ N(0, σ), независимо для каждой станции."
            tk.Label(calc_frame, text=calc_text, font=('Courier', 10), bg="#E3F2FD", justify=tk.LEFT, padx=10,
                     pady=10).pack(fill=tk.X)

    def plot_hyperbolas(self):
        if not hasattr(self, 'hyperbolas_data') or not self.hyperbolas_data:
            messagebox.showwarning("Внимание", "Нет данных для построения гипербол")
            return
        plot_window = tk.Toplevel(self.window)
        plot_window.title(f"График гипербол - Пакет {self.packet_data.get('id', 'N/A')}")
        plot_window.geometry("1200x900")
        fig = Figure(figsize=(12, 9), dpi=100)
        ax = fig.add_subplot(111)
        all_x = []
        all_y = []
        for station, (x, y) in self.station_coords.items():
            all_x.append(x);
            all_y.append(y)
        if self.true_uav_coords: all_x.append(self.true_uav_coords[0]); all_y.append(self.true_uav_coords[1])
        if self.calculated_coords: all_x.append(self.calculated_coords[0]); all_y.append(self.calculated_coords[1])
        x_min, x_max = min(all_x) - 500, max(all_x) + 500
        y_min, y_max = min(all_y) - 500, max(all_y) + 500
        ax.set_xlim(x_min, x_max);
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X координата, м', fontsize=12);
        ax.set_ylabel('Y координата, м', fontsize=12)
        ax.set_title(f'Гиперболы TDoA для пакета {self.packet_data.get("id", "N/A")}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal', adjustable='datalim')
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig = fig
        self.ax = ax
        self.plot_window = plot_window
        self.zoom_factor = 1.0
        self.pan_start = None
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
        for idx, (station, (x, y)) in enumerate(self.station_coords.items()):
            color = colors[idx % len(colors)]
            marker = 'o' if station == self.reference_station else 's'
            size = 120 if station == self.reference_station else 100
            ax.plot(x, y, color=color, marker=marker, markersize=size / 10, alpha=1.0,
                    label=f'Станция {station}{" (опорная)" if station == self.reference_station else ""}',
                    markeredgecolor='black', markeredgewidth=1.5, zorder=5)
            ax.text(x, y + 40, f'Ст{station}', fontsize=11, ha='center', va='bottom',
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7), zorder=5)
        if self.true_uav_coords:
            x_true, y_true = self.true_uav_coords
            ax.plot(x_true, y_true, color='gold', marker='*', markersize=200 / 10, alpha=1.0,
                    label='Истинное положение БПЛА', markeredgecolor='black', markeredgewidth=2.5, zorder=10)
            ax.text(x_true, y_true + 60, 'БПЛА (ист.)', fontsize=12, ha='center', va='bottom',
                    fontweight='bold', color='darkgoldenrod',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7), zorder=10)
        if self.calculated_coords:
            x_calc, y_calc = self.calculated_coords
            ax.plot(x_calc, y_calc, color='cyan', marker='X', markersize=150 / 10, alpha=1.0,
                    label='Вычисленное положение БПЛА (TDoA)', markeredgecolor='black', markeredgewidth=2.0, zorder=9)
            ax.text(x_calc, y_calc + 60, 'БПЛА (выч.)', fontsize=12, ha='center', va='bottom',
                    fontweight='bold', color='darkcyan',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7), zorder=9)
            ax.plot([x_true, x_calc], [y_true, y_calc], 'r--', linewidth=2, alpha=0.7, label='Ошибка определения')
            error = math.sqrt((x_calc - x_true) ** 2 + (y_calc - y_true) ** 2)
            mid_x = (x_true + x_calc) / 2;
            mid_y = (y_true + y_calc) / 2
            ax.text(mid_x, mid_y, f'Δ={error:.1f} м', fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        problem_stations = []
        hyperbolas_plotted = 0
        for i, hyperbola in enumerate(self.hyperbolas_data):
            color = colors[(i + 2) % len(colors)]
            if abs(hyperbola['delta_d']) >= hyperbola['distance_between_stations']:
                problem_stations.append(hyperbola['station'])
                continue
            a = abs(hyperbola['delta_d']) / 2
            c = hyperbola['distance_between_stations'] / 2
            b = math.sqrt(max(c ** 2 - a ** 2, 0))
            x_center = (hyperbola['x1'] + hyperbola['x2']) / 2
            y_center = (hyperbola['y1'] + hyperbola['y2']) / 2
            dx = hyperbola['x2'] - hyperbola['x1'];
            dy = hyperbola['y2'] - hyperbola['y1']
            angle = math.atan2(dy, dx)
            t = np.linspace(-3, 3, 1000)
            x_right = a * np.cosh(t);
            y_right = b * np.sinh(t)
            x_left = -a * np.cosh(t);
            y_left = b * np.sinh(t)

            def rotate_and_shift(x_arr, y_arr):
                x_rot = x_arr * np.cos(angle) - y_arr * np.sin(angle) + x_center
                y_rot = x_arr * np.sin(angle) + y_arr * np.cos(angle) + y_center
                return x_rot, y_rot

            x_rot_right, y_rot_right = rotate_and_shift(x_right, y_right)
            x_rot_left, y_rot_left = rotate_and_shift(x_left, y_left)
            test_idx = len(t) // 2
            x_test_right, y_test_right = x_rot_right[test_idx], y_rot_right[test_idx]
            dist_to_ref_right = math.sqrt((x_test_right - hyperbola['x1']) ** 2 + (y_test_right - hyperbola['y1']) ** 2)
            dist_to_i_right = math.sqrt((x_test_right - hyperbola['x2']) ** 2 + (y_test_right - hyperbola['y2']) ** 2)
            x_test_left, y_test_left = x_rot_left[test_idx], y_rot_left[test_idx]
            dist_to_ref_left = math.sqrt((x_test_left - hyperbola['x1']) ** 2 + (y_test_left - hyperbola['y1']) ** 2)
            dist_to_i_left = math.sqrt((x_test_left - hyperbola['x2']) ** 2 + (y_test_left - hyperbola['y2']) ** 2)
            if hyperbola['delta_d'] > 0:
                is_right_active = dist_to_ref_right < dist_to_i_right
            else:
                is_right_active = dist_to_i_right < dist_to_ref_right
            if is_right_active:
                ax.plot(x_rot_right, y_rot_right, color=color, linewidth=2.5, alpha=0.8,
                        label=f'Гипербола ст.{hyperbola["station"]} (Δd{"+" if hyperbola["delta_d"] > 0 else "-"})',
                        zorder=3)
                ax.plot(x_rot_left, y_rot_left, color=color, linewidth=1.5, alpha=0.4,
                        linestyle='--', zorder=2)
            else:
                ax.plot(x_rot_left, y_rot_left, color=color, linewidth=2.5, alpha=0.8,
                        label=f'Гипербола ст.{hyperbola["station"]} (Δd{"+" if hyperbola["delta_d"] > 0 else "-"})',
                        zorder=3)
                ax.plot(x_rot_right, y_rot_right, color=color, linewidth=1.5, alpha=0.4,
                        linestyle='--', zorder=2)
            hyperbolas_plotted += 1
            ax.plot([hyperbola['x1'], hyperbola['x2']], [hyperbola['y1'], hyperbola['y2']],
                    color=color, linewidth=1, alpha=0.3, linestyle=':', zorder=1)
        ax.legend(loc='upper right', fontsize=9)
        info_text = f"Пакет: {self.packet_data.get('id', 'N/A')}\n"
        info_text += f"Опорная станция: {self.reference_station}\n"
        info_text += f"Построено гипербол: {hyperbolas_plotted} из {len(self.hyperbolas_data)}"
        if self.true_uav_coords:
            x_true, y_true = self.true_uav_coords
            info_text += f"\nИстинные координаты БПЛА: ({x_true:.1f}, {y_true:.1f})"
        if self.calculated_coords:
            x_calc, y_calc = self.calculated_coords
            info_text += f"\nВычисленные: ({x_calc:.1f}, {y_calc:.1f})"
            if self.true_uav_coords:
                error = math.sqrt((x_calc - x_true) ** 2 + (y_calc - y_true) ** 2)
                info_text += f"\nОшибка: {error:.1f} м"
        if problem_stations:
            info_text += f"\n⚠ Не построены: {', '.join(problem_stations)}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), zorder=10)
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        control_frame = tk.Frame(plot_window)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        reset_btn = tk.Button(control_frame, text="Сбросить вид",
                              command=lambda: self.reset_view(ax, x_min, x_max, y_min, y_max, canvas),
                              bg="#3498db", fg="white", font=('Arial', 10))
        reset_btn.pack(side=tk.LEFT, padx=5)
        save_btn = tk.Button(control_frame, text="Сохранить график как PNG",
                             command=lambda: self.save_plot(fig),
                             bg="#27ae60", fg="white", font=('Arial', 10))
        save_btn.pack(side=tk.LEFT, padx=5)
        zoom_in_btn = tk.Button(control_frame, text="Приблизить (++",
                                command=lambda: self.zoom_in_out(ax, 1.2, canvas),
                                bg="#9b59b6", fg="white", font=('Arial', 10))
        zoom_in_btn.pack(side=tk.LEFT, padx=5)
        zoom_out_btn = tk.Button(control_frame, text="Отдалить (--",
                                 command=lambda: self.zoom_in_out(ax, 0.8, canvas),
                                 bg="#e67e22", fg="white", font=('Arial', 10))
        zoom_out_btn.pack(side=tk.LEFT, padx=5)
        instructions = tk.Label(control_frame,
                                text="Колесико мыши — масштаб, ПКМ — перемещение",
                                font=('Arial', 9), fg="#7f8c8d")
        instructions.pack(side=tk.LEFT, padx=20)
        canvas.mpl_connect('button_press_event', self.on_mouse_press)
        canvas.mpl_connect('button_release_event', self.on_mouse_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_scroll(self, event):
        if event.inaxes == self.ax:
            zoom_factor = 1.1 if event.button == 'up' else 0.9
            xlim = self.ax.get_xlim();
            ylim = self.ax.get_ylim()
            x_center = event.xdata;
            y_center = event.ydata
            x_range = (xlim[1] - xlim[0]) * zoom_factor
            y_range = (ylim[1] - ylim[0]) * zoom_factor
            new_xlim = (x_center - x_range / 2, x_center + x_range / 2)
            new_ylim = (y_center - y_range / 2, y_center + y_range / 2)
            self.ax.set_xlim(new_xlim);
            self.ax.set_ylim(new_ylim)
            self.fig.canvas.draw_idle()

    def on_mouse_press(self, event):
        if event.inaxes == self.ax and event.button == 3:
            self.pan_start = (event.xdata, event.ydata, self.ax.get_xlim(), self.ax.get_ylim())

    def on_mouse_release(self, event):
        self.pan_start = None

    def on_mouse_move(self, event):
        if self.pan_start is not None and event.inaxes == self.ax:
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata
            xlim = self.pan_start[2];
            ylim = self.pan_start[3]
            new_xlim = (xlim[0] + dx, xlim[1] + dx)
            new_ylim = (ylim[0] + dy, ylim[1] + dy)
            self.ax.set_xlim(new_xlim);
            self.ax.set_ylim(new_ylim)
            self.fig.canvas.draw_idle()

    def zoom_in_out(self, ax, factor, canvas):
        xlim = ax.get_xlim();
        ylim = ax.get_ylim()
        x_center = (xlim[0] + xlim[1]) / 2;
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) * factor;
        y_range = (ylim[1] - ylim[0]) * factor
        new_xlim = (x_center - x_range / 2, x_center + x_range / 2)
        new_ylim = (y_center - y_range / 2, y_center + y_range / 2)
        ax.set_xlim(new_xlim);
        ax.set_ylim(new_ylim)
        canvas.draw()

    def reset_view(self, ax, x_min, x_max, y_min, y_max, canvas):
        ax.set_xlim(x_min, x_max);
        ax.set_ylim(y_min, y_max)
        canvas.draw()

    def save_plot(self, fig):
        file_path = filedialog.asksaveasfilename(
            title="Сохранить график",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
        )
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Успех", f"График сохранён в {file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить график:\n{str(e)}")

    def create_section_frame(self, parent, title):
        section_frame = tk.Frame(parent, bg="#f0f0f0")
        section_frame.pack(fill=tk.X, pady=10)
        header = tk.Frame(section_frame, bg="#34495e", height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text=title, font=('Arial', 13, 'bold'), fg="white", bg="#34495e").pack(expand=True)
        content = tk.Frame(section_frame, bg="#f8f9fa", bd=1, relief=tk.SUNKEN)
        content.pack(fill=tk.X)
        return content



class UAVDataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Просмотр данных БПЛА с вычислением разностей")
        self.root.geometry("1400x700")
        self.C = 299792458.0
        self.C_ns = 0.299792458
        self.data = None
        self.stations = []
        self.station_coords = {}
        self.reference_station = None
        self.error_ns = 0.0
        self.num_trials = 100
        self.calculated_positions = {}
        self.equations_window = None
        np.random.seed(42)
        self.create_widgets()

    @staticmethod
    def build_covariance_matrix_Q(sigma_ns, N):
        c = 299792458.0
        sigma_t = sigma_ns * 1e-9
        sigma_d = c * sigma_t
        sigma2 = sigma_d ** 2
        Q = sigma2 * np.eye(N)
        return Q

    def compute_CRLB_for_active_stations(self, x, y, sigma_t_ns, active_stations, reference_station):
        if not active_stations or reference_station is None:
            return np.inf, np.inf
        c = self.C
        sigma_t = sigma_t_ns * 1e-9
        ref_coords = np.array(self.station_coords.get(reference_station, (0.0, 0.0)), dtype=np.float64)
        J = []
        for s in active_stations:
            s_coords = np.array(self.station_coords[s])
            dist_ref = np.linalg.norm([x, y] - ref_coords)
            dist_s = np.linalg.norm([x, y] - s_coords)
            if dist_ref == 0 or dist_s == 0:
                return np.inf, np.inf
            jx = (s_coords[0] - x) / dist_s - (ref_coords[0] - x) / dist_ref
            jy = (s_coords[1] - y) / dist_s - (ref_coords[1] - y) / dist_ref
            J.append([jx, jy])
        J = np.array(J)
        Q = self.build_covariance_matrix_Q(sigma_t_ns, len(active_stations))
        try:
            Q_inv = np.linalg.inv(Q)
            FIM = J.T @ Q_inv @ J
            if np.linalg.det(FIM) < 1e-12:
                return np.inf, np.inf
            CRLB = np.linalg.inv(FIM)
            return np.sqrt(CRLB[0, 0]), np.sqrt(CRLB[1, 1])
        except np.linalg.LinAlgError:
            return np.inf, np.inf

    def generate_GDOP_map(self, grid_size=100, reference_station=None):
        if reference_station is None:
            reference_station = self.reference_station
        if not self.station_coords or reference_station not in self.station_coords or self.error_ns <= 0:
            return None, None, None
        xs = [x for x, _ in self.station_coords.values()]
        ys = [y for _, y in self.station_coords.values()]
        margin = 500
        x_min, x_max = min(xs) - margin, max(xs) + margin
        y_min, y_max = min(ys) - margin, max(ys) + margin
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x, y)
        GDOP = np.full_like(X, np.nan)
        for i in range(grid_size):
            for j in range(grid_size):
                x_ij, y_ij = X[i, j], Y[i, j]
                crlb_x, crlb_y = self.compute_CRLB_for_active_stations(x_ij, y_ij, self.error_ns,
                                                                       [s for s in self.stations if
                                                                        s != reference_station],
                                                                       reference_station)
                if np.isfinite(crlb_x) and np.isfinite(crlb_y):
                    sigma_t = self.error_ns * 1e-9
                    sigma_d = self.C * sigma_t
                    total_error = np.sqrt(crlb_x ** 2 + crlb_y ** 2)
                    gdop_val = total_error / sigma_d
                    if gdop_val > 1000 or not np.isfinite(gdop_val):
                        gdop_val = np.nan
                    GDOP[i, j] = gdop_val
        return X, Y, GDOP

    def calculate_GDOP_statistics(self, reference_station=None):
        if not self.data or "uav_truth" not in self.data or self.error_ns <= 0:
            return None
        if reference_station is None:
            reference_station = self.reference_station
        if reference_station is None:
            return None
        uav_truth = self.data["uav_truth"]
        packet_ids = sorted(uav_truth.keys(), key=lambda pid: uav_truth[pid].get('time', 0))
        gdop_values = []
        for pid in packet_ids:
            truth = uav_truth.get(pid, {})
            tx = truth.get('x');
            ty = truth.get('y')
            if tx is None or ty is None:
                continue
            crlb_x, crlb_y = self.compute_CRLB_for_active_stations(tx, ty, self.error_ns,
                                                                   [s for s in self.stations if s != reference_station],
                                                                   reference_station)
            if np.isinf(crlb_x) or np.isinf(crlb_y):
                continue
            sigma_t = self.error_ns * 1e-9
            sigma_d = self.C * sigma_t
            total_error = np.sqrt(crlb_x ** 2 + crlb_y ** 2)
            gdop = total_error / sigma_d
            gdop_values.append(gdop)
        if not gdop_values:
            return None
        stats = {
            'mean': np.mean(gdop_values), 'max': np.max(gdop_values), 'min': np.min(gdop_values),
            'std': np.std(gdop_values),
            'portion_less_3': sum(1 for g in gdop_values if g < 3) / len(gdop_values) * 100,
            'portion_more_6': sum(1 for g in gdop_values if g > 6) / len(gdop_values) * 100
        }
        return stats

    def show_gdop_map(self):
        if not self.station_coords or self.error_ns <= 0:
            messagebox.showwarning("Внимание", "Сначала загрузите данные и задайте σ.")
            return
        gdop_window = tk.Toplevel(self.root)
        gdop_window.title("Анализ GDOP")
        gdop_window.geometry("1400x850")
        control_frame = tk.Frame(gdop_window, bg="#f0f0f0")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(control_frame, text="Опорная станция:", bg="#f0f0f0").pack(side=tk.LEFT, padx=(5, 2))
        ref_station_var = tk.StringVar()
        station_options = [f"Станция {s}" for s in self.stations]
        ref_station_combo = ttk.Combobox(control_frame, textvariable=ref_station_var,
                                         values=station_options, state="readonly", width=15)
        ref_station_combo.pack(side=tk.LEFT, padx=5)
        if self.reference_station:
            ref_station_var.set(f"Станция {self.reference_station}")
        show_all_var = tk.BooleanVar()
        show_all_check = tk.Checkbutton(control_frame, text="Показать для всех станций", variable=show_all_var,
                                        bg="#f0f0f0")
        show_all_check.pack(side=tk.LEFT, padx=20)
        plot_btn = tk.Button(control_frame, text="Построить карту",
                             command=lambda: self.plot_gdop_map(gdop_window, ref_station_var, show_all_var),
                             bg="#4CAF50", fg="white")
        plot_btn.pack(side=tk.LEFT, padx=5)
        self.gdop_canvas_frame = tk.Frame(gdop_window, bg="#f0f0f0")
        self.gdop_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def plot_gdop_map(self, window, ref_station_var, show_all_var):
        for widget in self.gdop_canvas_frame.winfo_children():
            widget.destroy()
        true_x, true_y = [], []
        if self.data and "uav_truth" in self.data:
            uav_truth = self.data["uav_truth"]
            packet_ids = sorted(uav_truth.keys(), key=lambda pid: uav_truth[pid].get('time', 0))
            for pid in packet_ids:
                truth = uav_truth.get(pid, {})
                tx = truth.get('x')
                ty = truth.get('y')
                if tx is not None and ty is not None:
                    true_x.append(tx)
                    true_y.append(ty)
        if show_all_var.get():
            num_stations = len(self.stations)
            cols = min(3, num_stations)
            rows = (num_stations + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(12, 8), dpi=100)
            if num_stations == 1:
                axes = np.array([[axes]])
            elif rows == 1 or cols == 1:
                axes = np.array(axes).reshape(rows, cols)
            all_gdop_values = []
            for idx, station in enumerate(self.stations):
                row = idx // cols
                col = idx % cols
                ax = axes[row, col]
                X, Y, GDOP = self.generate_GDOP_map(grid_size=50, reference_station=station)
                if X is not None:
                    valid_gdop = GDOP[np.isfinite(GDOP)]
                    if len(valid_gdop) > 0:
                        all_gdop_values.extend(valid_gdop)
            if all_gdop_values:
                vmin = max(0.1, np.min(all_gdop_values))
                vmax = np.max(all_gdop_values)
                for idx, station in enumerate(self.stations):
                    row = idx // cols
                    col = idx % cols
                    ax = axes[row, col]
                    X, Y, GDOP = self.generate_GDOP_map(grid_size=50, reference_station=station)
                    if X is not None:
                        GDOP_plot = np.copy(GDOP)
                        GDOP_plot[np.isnan(GDOP_plot)] = 0
                        im = ax.imshow(GDOP_plot, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower',
                                       cmap='hot_r', aspect='equal',
                                       norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
                        for sid, (sx, sy) in self.station_coords.items():
                            if sid == station:
                                ax.plot(sx, sy, '^', color='blue', markersize=8,
                                        markeredgecolor='black', markeredgewidth=1.5)
                            else:
                                ax.plot(sx, sy, '^', color='black', markersize=6,
                                        markeredgecolor='black', markeredgewidth=1.0)
                        if true_x and true_y:
                            ax.plot(true_x, true_y, 'b-', linewidth=1.5, alpha=0.7)
                        ax.set_title(f'Станция {station}')
                        ax.set_xlabel('X, м')
                        ax.set_ylabel('Y, м')
                        ax.grid(True, alpha=0.3)
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                fig.colorbar(im, cax=cbar_ax, label='GDOP')
                plt.tight_layout(rect=[0, 0, 0.9, 1])
                canvas = FigureCanvasTkAgg(fig, master=self.gdop_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                toolbar_frame = tk.Frame(self.gdop_canvas_frame)
                toolbar_frame.pack(fill=tk.X)
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
                stats_frame = tk.Frame(self.gdop_canvas_frame, bg="#f0f0f0")
                stats_frame.pack(fill=tk.X, padx=10, pady=5)
                tk.Label(stats_frame, text="Статистика GDOP для всех станций:", font=('Arial', 10, 'bold'),
                         bg="#f0f0f0").pack(anchor=tk.W)
                for station in self.stations:
                    gdop_stats = self.calculate_GDOP_statistics(station)
                    if gdop_stats:
                        stats_text = f"Станция {station}: среднее={gdop_stats['mean']:.2f}, max={gdop_stats['max']:.2f}, min={gdop_stats['min']:.2f}"
                        tk.Label(stats_frame, text=stats_text, font=('Arial', 9), bg="#f0f0f0").pack(anchor=tk.W)
        else:
            selected_text = ref_station_var.get()
            if selected_text:
                station_num = selected_text.split()[-1]
                X, Y, GDOP = self.generate_GDOP_map(grid_size=100, reference_station=station_num)
                if X is not None:
                    fig = Figure(figsize=(10, 8), dpi=100)
                    ax = fig.add_subplot(111)
                    GDOP_plot = np.copy(GDOP)
                    GDOP_plot[np.isnan(GDOP_plot)] = 0
                    vmin = np.nanmin(GDOP)
                    vmax = np.nanmax(GDOP)
                    if vmin <= 0: vmin = 1e-3
                    if vmax <= vmin: vmax = vmin * 10
                    im = ax.imshow(GDOP_plot, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower',
                                   cmap='hot_r', aspect='equal', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
                    levels = np.logspace(np.log10(vmin), np.log10(vmax), 10)
                    contour = ax.contour(X, Y, GDOP_plot, levels=levels, colors='black', linewidths=0.5, alpha=0.7)
                    ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')
                    for sid, (sx, sy) in self.station_coords.items():
                        if sid == station_num:
                            ax.plot(sx, sy, '^', color='blue', markersize=10,
                                    markeredgecolor='black', markeredgewidth=1.5, label=f'Станция {sid} (опорная)')
                        else:
                            ax.plot(sx, sy, '^', color='black', markersize=8,
                                    markeredgecolor='black', markeredgewidth=1.0, label=f'Станция {sid}')
                    if true_x and true_y:
                        ax.plot(true_x, true_y, 'b-', linewidth=2.5, alpha=0.8)
                        ax.plot(true_x[0], true_y[0], 'go', markersize=8, markeredgecolor='black')
                        ax.plot(true_x[-1], true_y[-1], 'ro', markersize=8, markeredgecolor='black')
                    ax.set_title(f'GDOP Карта (опорная станция: {station_num})')
                    ax.set_xlabel('X, м')
                    ax.set_ylabel('Y, м')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best', fontsize=8)
                    cbar = fig.colorbar(im, ax=ax, label='GDOP (безразм.)')
                    canvas = FigureCanvasTkAgg(fig, master=self.gdop_canvas_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    toolbar_frame = tk.Frame(self.gdop_canvas_frame)
                    toolbar_frame.pack(fill=tk.X)
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    gdop_stats = self.calculate_GDOP_statistics(station_num)
                    if gdop_stats:
                        stats_frame = tk.Frame(self.gdop_canvas_frame, bg="#f0f0f0")
                        stats_frame.pack(fill=tk.X, padx=10, pady=5)
                        stats_text = f"Статистика GDOP для станции {station_num}:\n"
                        stats_text += f"Среднее: {gdop_stats['mean']:.2f}, Макс: {gdop_stats['max']:.2f}, Мин: {gdop_stats['min']:.2f}\n"
                        stats_text += f"Ст. откл.: {gdop_stats['std']:.2f}, Доля < 3: {gdop_stats['portion_less_3']:.1f}%, Доля > 6: {gdop_stats['portion_more_6']:.1f}%"
                        tk.Label(stats_frame, text=stats_text, font=('Arial', 10), bg="#f0f0f0",
                                 justify=tk.LEFT).pack(anchor=tk.W)
                else:
                    messagebox.showerror("Ошибка", "Не удалось сгенерировать карту GDOP для выбранной станции.")
            else:
                messagebox.showwarning("Внимание", "Выберите опорную станцию.")

    def show_gdop_time_plot(self):
        if not self.data or "uav_truth" not in self.data or self.error_ns <= 0:
            messagebox.showwarning("Внимание", "Сначала загрузите данные, задайте σ.")
            return
        gdop_time_window = tk.Toplevel(self.root)
        gdop_time_window.title("GDOP вдоль траектории")
        gdop_time_window.geometry("1300x800")
        control_frame = tk.Frame(gdop_time_window, bg="#f0f0f0")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(control_frame, text="Опорная станция:", bg="#f0f0f0").pack(side=tk.LEFT, padx=(5, 2))
        ref_station_var = tk.StringVar()
        station_options = [f"Станция {s}" for s in self.stations]
        ref_station_combo = ttk.Combobox(control_frame, textvariable=ref_station_var,
                                         values=station_options, state="readonly", width=15)
        ref_station_combo.pack(side=tk.LEFT, padx=5)
        show_all_var = tk.BooleanVar()
        show_all_check = tk.Checkbutton(control_frame, text="Показать для всех станций", variable=show_all_var,
                                        bg="#f0f0f0")
        show_all_check.pack(side=tk.LEFT, padx=20)
        plot_btn = tk.Button(control_frame, text="Построить график",
                             command=lambda: self.plot_gdop_time(gdop_time_window, ref_station_var, show_all_var),
                             bg="#4CAF50", fg="white")
        plot_btn.pack(side=tk.LEFT, padx=5)
        self.gdop_time_frame = tk.Frame(gdop_time_window, bg="#f0f0f0")
        self.gdop_time_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def plot_gdop_time(self, window, ref_station_var, show_all_var):
        for widget in self.gdop_time_frame.winfo_children():
            widget.destroy()
        uav_truth = self.data["uav_truth"]
        packet_ids = sorted(uav_truth.keys(), key=lambda pid: uav_truth[pid].get('time', 0))
        if show_all_var.get():
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            all_gdop_values = []
            for station in self.stations:
                times = []
                gdop_values = []
                for pid in packet_ids:
                    truth = uav_truth.get(pid, {})
                    tx = truth.get('x')
                    ty = truth.get('y')
                    t = truth.get('time')
                    if tx is None or ty is None or t is None:
                        continue
                    crlb_x, crlb_y = self.compute_CRLB_for_active_stations(tx, ty, self.error_ns,
                                                                           [s for s in self.stations if s != station],
                                                                           station)
                    if np.isfinite(crlb_x) and np.isfinite(crlb_y):
                        sigma_t = self.error_ns * 1e-9
                        sigma_d = self.C * sigma_t
                        total_error = np.sqrt(crlb_x ** 2 + crlb_y ** 2)
                        gdop = total_error / sigma_d
                        if np.isfinite(gdop) and gdop > 0:
                            times.append(t)
                            gdop_values.append(gdop)
                            all_gdop_values.append(gdop)
                if times:
                    ax.plot(times, gdop_values, linewidth=2, label=f'Станция {station}')
            if all_gdop_values:
                max_gdop = np.max(all_gdop_values)
                classifications = [
                    (1, 'Отличная точность', '#2ecc71'),
                    (3, 'Хорошая точность', '#3498db'),
                    (6, 'Удовлетворительная точность', '#f39c12'),
                    (10, 'Низкая точность', '#e74c3c')
                ]
                for threshold, label, color in classifications:
                    ax.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, label=f'{label} (GDOP={threshold})')
                mean_gdop = np.mean(all_gdop_values)
                ax.axhline(y=mean_gdop, color='purple', linestyle=':', alpha=0.8,
                           label=f'Среднее GDOP = {mean_gdop:.2f}')
                max_display = max(15, max_gdop * 1.1) if max_gdop < 50 else max_gdop * 1.1
                ax.set_ylim(0, max_display)
                ax.set_xlabel('Время, с', fontsize=12)
                ax.set_ylabel('GDOP (безразмерный)', fontsize=12)
                ax.set_title('Изменение GDOP вдоль траектории для разных опорных станций', fontsize=14,
                             fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=8)
                canvas = FigureCanvasTkAgg(fig, master=self.gdop_time_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                toolbar_frame = tk.Frame(self.gdop_time_frame)
                toolbar_frame.pack(fill=tk.X)
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
        else:
            selected_text = ref_station_var.get()
            if selected_text:
                station_num = selected_text.split()[-1]
                times = []
                gdop_values = []
                for pid in packet_ids:
                    truth = uav_truth.get(pid, {})
                    tx = truth.get('x')
                    ty = truth.get('y')
                    t = truth.get('time')
                    if tx is None or ty is None or t is None:
                        continue
                    crlb_x, crlb_y = self.compute_CRLB_for_active_stations(tx, ty, self.error_ns,
                                                                           [s for s in self.stations if
                                                                            s != station_num],
                                                                           station_num)
                    if np.isfinite(crlb_x) and np.isfinite(crlb_y):
                        sigma_t = self.error_ns * 1e-9
                        sigma_d = self.C * sigma_t
                        total_error = np.sqrt(crlb_x ** 2 + crlb_y ** 2)
                        gdop = total_error / sigma_d
                        if np.isfinite(gdop) and gdop > 0:
                            times.append(t)
                            gdop_values.append(gdop)
                if not times:
                    messagebox.showwarning("Внимание", "Нет данных для построения графика GDOP(t).")
                    return
                mean_gdop = np.mean(gdop_values)
                max_gdop = np.max(gdop_values)
                min_gdop = np.min(gdop_values)
                std_gdop = np.std(gdop_values)
                fig = Figure(figsize=(10, 6), dpi=100)
                ax = fig.add_subplot(111)
                ax.plot(times, gdop_values, 'b-', linewidth=2, label='GDOP')
                classifications = [
                    (1, 'Отличная точность', '#2ecc71'),
                    (3, 'Хорошая точность', '#3498db'),
                    (6, 'Удовлетворительная точность', '#f39c12'),
                    (10, 'Низкая точность', '#e74c3c')
                ]
                for threshold, label, color in classifications:
                    ax.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, label=f'{label} (GDOP={threshold})')
                ax.axhline(y=mean_gdop, color='purple', linestyle=':', alpha=0.8,
                           label=f'Среднее GDOP = {mean_gdop:.2f}')
                ax.set_xlabel('Время, с', fontsize=12)
                ax.set_ylabel('GDOP (безразмерный)', fontsize=12)
                ax.set_title(f'Изменение GDOP вдоль траектории (опорная станция: {station_num})', fontsize=14,
                             fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=9)
                max_display = max(15, max_gdop * 1.1) if max_gdop < 50 else max_gdop * 1.1
                ax.set_ylim(0, max_display)
                canvas = FigureCanvasTkAgg(fig, master=self.gdop_time_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                toolbar_frame = tk.Frame(self.gdop_time_frame)
                toolbar_frame.pack(fill=tk.X)
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
                right_frame = tk.Frame(self.gdop_time_frame, width=300, bg='white', relief=tk.RAISED, bd=2)
                right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
                right_frame.pack_propagate(False)
                stats_header = tk.Label(right_frame, text="📊 Статистика GDOP",
                                        font=('Arial', 14, 'bold'), bg='#9B59B6', fg='white')
                stats_header.pack(fill=tk.X, pady=(0, 10))
                stats_text = tk.Text(right_frame, font=('Arial', 11), wrap=tk.WORD, bg='white', relief=tk.FLAT,
                                     height=20)
                stats_scroll = tk.Scrollbar(right_frame, command=stats_text.yview)
                stats_text.configure(yscrollcommand=stats_scroll.set)
                stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
                stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
                stats_content = f"""
Основные метрики:
Средний GDOP: {mean_gdop:.2f}
Максимальный GDOP: {max_gdop:.2f}
Минимальный GDOP: {min_gdop:.2f}
СКО GDOP: {std_gdop:.2f}
Распределение по качеству:
Доля точек с GDOP < 3: {(sum(1 for g in gdop_values if g < 3) / len(gdop_values) * 100):.1f}%
Доля точек с GDOP > 6: {(sum(1 for g in gdop_values if g > 6) / len(gdop_values) * 100):.1f}%
Параметры анализа:
σ_TDoA = {self.error_ns:.1f} нс
Опорная станция: Ст{station_num}
Количество станций: {len(self.station_coords)}
Классификация GDOP:
• GDOP < 1 — отличная точность
• GDOP 1–3 — хорошая точность
• GDOP 3–6 — удовлетворительная
• GDOP 6–10 — низкая точность
• GDOP > 10 — неприемлемо
"""
                stats_text.insert(tk.END, stats_content)
                stats_text.config(state=tk.DISABLED)
            else:
                messagebox.showwarning("Внимание", "Выберите опорную станцию.")

    def show_crlb_plot(self):
        if not self.data or self.error_ns <= 0:
            messagebox.showwarning("Внимание", "Сначала задайте σ.")
            return
        crlb_window = tk.Toplevel(self.root)
        crlb_window.title("Анализ CRLB")
        crlb_window.geometry("1300x800")
        control_frame = tk.Frame(crlb_window, bg="#f0f0f0")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(control_frame, text="Опорная станция:", bg="#f0f0f0").pack(side=tk.LEFT, padx=(5, 2))
        ref_station_var = tk.StringVar()
        station_options = [f"Станция {s}" for s in self.stations]
        ref_station_combo = ttk.Combobox(control_frame, textvariable=ref_station_var,
                                         values=station_options, state="readonly", width=15)
        ref_station_combo.pack(side=tk.LEFT, padx=5)
        show_all_var = tk.BooleanVar()
        show_all_check = tk.Checkbutton(control_frame, text="Показать для всех станций", variable=show_all_var,
                                        bg="#f0f0f0")
        show_all_check.pack(side=tk.LEFT, padx=20)
        plot_btn = tk.Button(control_frame, text="Построить график",
                             command=lambda: self.plot_crlb(crlb_window, ref_station_var, show_all_var),
                             bg="#4CAF50", fg="white")
        plot_btn.pack(side=tk.LEFT, padx=5)
        self.crlb_canvas_frame = tk.Frame(crlb_window, bg="#f0f0f0")
        self.crlb_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def plot_crlb(self, window, ref_station_var, show_all_var):
        for widget in self.crlb_canvas_frame.winfo_children():
            widget.destroy()
        uav_truth = self.data.get("uav_truth", {})
        packet_ids = sorted(uav_truth.keys(), key=lambda x: uav_truth[x].get('time', 0))
        if show_all_var.get():
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            all_crlb_values = []
            for station in self.stations:
                times = []
                crlb_total = []
                for pid in packet_ids:
                    truth = uav_truth.get(pid, {})
                    tx = truth.get('x');
                    ty = truth.get('y');
                    t = truth.get('time')
                    if tx is not None and ty is not None and t is not None:
                        crlb_x, crlb_y = self.compute_CRLB_for_active_stations(tx, ty, self.error_ns,
                                                                               [s for s in self.stations if
                                                                                s != station],
                                                                               station)
                        if np.isfinite(crlb_x) and np.isfinite(crlb_y):
                            times.append(t)
                            total_error = np.sqrt(crlb_x ** 2 + crlb_y ** 2)
                            crlb_total.append(total_error)
                            all_crlb_values.append(total_error)
                if times:
                    ax.plot(times, crlb_total, linewidth=2, label=f'Станция {station}')
            if all_crlb_values:
                max_crlb = max(all_crlb_values)
                min_crlb = min(all_crlb_values)
                mean_crlb = np.mean(all_crlb_values)
                ax.axhline(mean_crlb, color='red', linestyle=':', alpha=0.7,
                           label=f'Среднее CRLB = {mean_crlb:.2f} м')
                max_display = max_crlb * 1.2 if max_crlb < 50 else max_crlb * 1.1
                ax.set_ylim(0, max_display)
                ax.set_xlabel('Время, с', fontsize=12)
                ax.set_ylabel('CRLB (суммарная ошибка), м', fontsize=12)
                ax.set_title(f'Сравнение CRLB для разных опорных станций (σ = {self.error_ns} нс)', fontsize=14,
                             fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=9)
                canvas = FigureCanvasTkAgg(fig, master=self.crlb_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                toolbar_frame = tk.Frame(self.crlb_canvas_frame)
                toolbar_frame.pack(fill=tk.X)
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
        else:
            selected_text = ref_station_var.get()
            if selected_text:
                station_num = selected_text.split()[-1]
                times, crlb_x_list, crlb_y_list = [], [], []
                for pid in packet_ids:
                    truth = uav_truth.get(pid, {})
                    tx = truth.get('x');
                    ty = truth.get('y');
                    t = truth.get('time')
                    if tx is not None and ty is not None and t is not None:
                        crlb_x, crlb_y = self.compute_CRLB_for_active_stations(tx, ty, self.error_ns,
                                                                               [s for s in self.stations if
                                                                                s != station_num],
                                                                               station_num)
                        if np.isfinite(crlb_x) and np.isfinite(crlb_y):
                            times.append(t)
                            crlb_x_list.append(crlb_x)
                            crlb_y_list.append(crlb_y)
                if times:
                    fig = Figure(figsize=(10, 6), dpi=100)
                    ax = fig.add_subplot(111)
                    ax.plot(times, crlb_x_list, 'r-', linewidth=2, label=r'$\sigma_x$ (CRLB)')
                    ax.plot(times, crlb_y_list, 'b-', linewidth=2, label=r'$\sigma_y$ (CRLB)')
                    sigma_total = np.sqrt(np.array(crlb_x_list) ** 2 + np.array(crlb_y_list) ** 2)
                    ax.plot(times, sigma_total, 'k--', linewidth=2,
                            label=r'$\sigma_{total} = \sqrt{\sigma_x^2 + \sigma_y^2}$')
                    sigma_x_mean = np.mean(crlb_x_list)
                    sigma_y_mean = np.mean(crlb_y_list)
                    sigma_total_mean = np.mean(sigma_total)
                    sigma_total_max = np.max(sigma_total)
                    ax.axhline(sigma_x_mean, color='r', linestyle=':', alpha=0.5)
                    ax.axhline(sigma_y_mean, color='b', linestyle=':', alpha=0.5)
                    ax.axhline(sigma_total_mean, color='k', linestyle=':', alpha=0.5)
                    ax.set_xlabel('Время, с', fontsize=12);
                    ax.set_ylabel('СКО, м', fontsize=12)
                    ax.set_title(f'Нижняя граница Крамера–Рао (CRLB) для опорной станции {station_num}', fontsize=14,
                                 fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right', fontsize=10)
                    ax.set_xlim(min(times), max(times))
                    if sigma_total_max > 0:
                        ax.set_ylim(0, sigma_total_max * 1.2)
                    canvas = FigureCanvasTkAgg(fig, master=self.crlb_canvas_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    toolbar_frame = tk.Frame(self.crlb_canvas_frame)
                    toolbar_frame.pack(fill=tk.X)
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    sigma_total_min = np.min(sigma_total)
                    sigma_total_std = np.std(sigma_total)
                    stats_frame = tk.Frame(self.crlb_canvas_frame, bg="#f0f0f0")
                    stats_frame.pack(fill=tk.X, padx=10, pady=5)
                    stats_content = f"""
Основная статистика:
Среднее σ_x: {sigma_x_mean:.2f} м
Среднее σ_y: {sigma_y_mean:.2f} м
Среднее σ_total: {sigma_total_mean:.2f} м
Предельные значения:
Максимальное σ_total: {sigma_total_max:.2f} м
Минимальное σ_total: {sigma_total_min:.2f} м
СКО σ_total: {sigma_total_std:.2f} м
Параметры анализа:
σ_TDoA = {self.error_ns:.1f} нс
c = {self.C:.0f} м/с
σ_расстояния = c × σ_TDoA = {self.C * self.error_ns * 1e-9:.1f} м
"""
                    tk.Label(stats_frame, text=stats_content, font=('Arial', 10), bg="#f0f0f0",
                             justify=tk.LEFT).pack(anchor=tk.W)
                else:
                    messagebox.showwarning("Внимание", "Нет данных для построения CRLB для выбранной станции.")
            else:
                messagebox.showwarning("Внимание", "Выберите опорную станцию.")

    def save_figure(self, fig, default_name):
        path = filedialog.asksaveasfilename(
            title="Сохранить график",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if path:
            try:
                fig.savefig(path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Успех", f"Сохранено: {path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

    def create_widgets(self):
        self.control_canvas = tk.Canvas(self.root, height=60, bg="white")
        self.control_scrollbar = ttk.Scrollbar(self.root, orient="horizontal", command=self.control_canvas.xview)
        self.control_canvas.configure(xscrollcommand=self.control_scrollbar.set)
        self.control_frame = tk.Frame(self.control_canvas, bg="white")
        self.control_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        )
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        self.control_canvas.pack(fill=tk.X, padx=10, pady=5)
        self.control_scrollbar.pack(fill=tk.X, padx=10)

        self.load_btn = tk.Button(self.control_frame, text="Загрузить JSON файл",
                                  command=self.load_json_file, bg="#4CAF50", fg="white")
        self.load_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(self.control_frame, text="Опорная станция:").pack(side=tk.LEFT, padx=(20, 5))
        self.ref_station_var = tk.StringVar()
        self.ref_station_combo = ttk.Combobox(self.control_frame, textvariable=self.ref_station_var,
                                              state="readonly", width=15)
        self.ref_station_combo.pack(side=tk.LEFT, padx=5)
        self.ref_station_combo.config(state=tk.DISABLED)
        self.ref_station_combo.bind("<<ComboboxSelected>>", self.on_ref_station_selected)

        tk.Label(self.control_frame, text="σ ошибки (нс):").pack(side=tk.LEFT, padx=(20, 5))
        self.error_var = tk.StringVar(value="0.0")
        self.error_entry = tk.Entry(self.control_frame, textvariable=self.error_var, width=10, justify=tk.CENTER)
        self.error_entry.pack(side=tk.LEFT, padx=5)
        self.error_entry.config(state=tk.DISABLED)

        tk.Label(self.control_frame, text="Число реализаций:").pack(side=tk.LEFT, padx=(20, 5))
        self.num_trials_var = tk.StringVar(value="100")
        self.num_trials_entry = tk.Entry(self.control_frame, textvariable=self.num_trials_var, width=8,
                                         justify=tk.CENTER)
        self.num_trials_entry.pack(side=tk.LEFT, padx=5)
        self.num_trials_entry.config(state=tk.DISABLED)

        self.apply_error_btn = tk.Button(self.control_frame, text="Применить ошибку",
                                         command=self.apply_error, bg="#9C27B0", fg="white")
        self.apply_error_btn.pack(side=tk.LEFT, padx=5)
        self.apply_error_btn.config(state=tk.DISABLED)

        self.calc_tdoa_btn = tk.Button(self.control_frame, text="Вычислить координаты TDoA",
                                       command=self.calculate_tdoa_positions, bg="#00BCD4", fg="white")
        self.calc_tdoa_btn.pack(side=tk.LEFT, padx=5)
        self.calc_tdoa_btn.config(state=tk.DISABLED)

        self.gdop_btn = tk.Button(self.control_frame, text="🗺️ Анализ GDOP",
                                  command=self.show_gdop_map, bg="#673AB7", fg="white")
        self.gdop_btn.pack(side=tk.LEFT, padx=5)
        self.gdop_btn.config(state=tk.DISABLED)

        self.gdop_time_btn = tk.Button(self.control_frame, text="📈 GDOP(t)",
                                       command=self.show_gdop_time_plot, bg="#9B59B6", fg="white")
        self.gdop_time_btn.pack(side=tk.LEFT, padx=5)
        self.gdop_time_btn.config(state=tk.DISABLED)

        self.crlb_btn = tk.Button(self.control_frame, text="📊 Анализ CRLB",
                                  command=self.show_crlb_plot, bg="#3F51B5", fg="white")
        self.crlb_btn.pack(side=tk.LEFT, padx=5)
        self.crlb_btn.config(state=tk.DISABLED)

        self.show_trajectory_btn = tk.Button(self.control_frame, text="Показать траекторию",
                                             command=self.show_trajectory, bg="#FF9800", fg="white")
        self.show_trajectory_btn.pack(side=tk.LEFT, padx=5)
        self.show_trajectory_btn.config(state=tk.DISABLED)

        self.show_errors_btn = tk.Button(self.control_frame, text="Показать ошибки",
                                         command=self.show_errors, bg="#795548", fg="white")
        self.show_errors_btn.pack(side=tk.LEFT, padx=5)
        self.show_errors_btn.config(state=tk.DISABLED)

        self.show_selected_btn = tk.Button(self.control_frame, text="🔍 Уравнения для выбранного пакета",
                                           command=self.show_tdoa_equations_selected,
                                           bg="#E67E22", fg="white", font=('Arial', 10))
        self.show_selected_btn.pack(side=tk.LEFT, padx=5)
        self.show_selected_btn.config(state=tk.DISABLED)

        # НОВАЯ КНОПКА: Анализ опорных станций
        self.analyze_ref_btn = tk.Button(self.control_frame, text="📊 Анализ опорных станций",
                                         command=self.analyze_reference_stations,
                                         bg="#16A085", fg="white")
        self.analyze_ref_btn.pack(side=tk.LEFT, padx=5)
        self.analyze_ref_btn.config(state=tk.DISABLED)

        self.update_btn = tk.Button(self.control_frame, text="Обновить таблицу",
                                    command=self.update_table, bg="#2196F3", fg="white")
        self.update_btn.pack(side=tk.LEFT, padx=5)
        self.update_btn.config(state=tk.DISABLED)

        self.export_btn = tk.Button(self.control_frame, text="Экспорт в CSV",
                                    command=self.export_to_csv, bg="#607D8B", fg="white")
        self.export_btn.pack(side=tk.LEFT, padx=5)
        self.export_btn.config(state=tk.DISABLED)

        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)
        self.file_info_label = tk.Label(self.info_frame, text="Файл не загружен", font=('Arial', 10))
        self.file_info_label.pack(anchor=tk.W)
        self.settings_label = tk.Label(self.info_frame,
                                       text="σ ошибки: 0.0 нс | Опорная станция: не выбрана",
                                       font=('Arial', 10))
        self.settings_label.pack(anchor=tk.W)

        self.tree = ttk.Treeview(self.root)
        vsb = ttk.Scrollbar(self.root, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.root, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.bind('<Double-Button-1>', self.on_row_double_click)

    def on_ref_station_selected(self, event=None):
        selected_text = self.ref_station_var.get()
        if selected_text:
            station_num = selected_text.split()[-1]
            self.reference_station = station_num
            self.num_trials_entry.config(state=tk.NORMAL)
            self.display_data(show_differences=True)
            self.gdop_btn.config(state=tk.NORMAL)
            self.gdop_time_btn.config(state=tk.NORMAL)
            self.crlb_btn.config(state=tk.NORMAL)
            self.update_settings_label()

    def load_json_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите JSON файл",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not file_path: return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.stations = sorted([k for k in self.data.keys() if k.isdigit()], key=int)
            self.station_coords = {}
            if "stations" in self.data:
                for st in self.data["stations"]:
                    self.station_coords[st["id"]] = (st["x"], st["y"])
            else:
                for s in self.stations:
                    self.station_coords[s] = (0.0, 0.0)
            filename = os.path.basename(file_path)
            self.file_info_label.config(text=f"Загружен: {filename} | Станций: {len(self.stations)}")
            station_names = [f"Станция {s}" for s in self.stations]
            self.ref_station_combo.config(values=station_names, state="readonly")
            self.error_entry.config(state=tk.NORMAL)
            self.num_trials_entry.config(state=tk.NORMAL)
            self.apply_error_btn.config(state=tk.NORMAL)
            self.calc_tdoa_btn.config(state=tk.NORMAL)
            self.gdop_btn.config(state=tk.NORMAL)
            self.gdop_time_btn.config(state=tk.NORMAL)
            self.crlb_btn.config(state=tk.NORMAL)
            self.analyze_ref_btn.config(state=tk.NORMAL)
            self.ref_station_var.set("")
            self.reference_station = None
            self.calculated_positions = {}
            if self.equations_window:
                self.equations_window.destroy()
            self.equations_window = None
            self.display_data(show_differences=False)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{e}")

    def apply_error(self):
        try:
            self.error_ns = float(self.error_var.get())
            if self.error_ns < 0:
                raise ValueError
            self.num_trials = int(self.num_trials_var.get())
            if self.num_trials < 1:
                raise ValueError
            self.analyze_ref_btn.config(state=tk.NORMAL)
            self.display_data(show_differences=True)
            self.gdop_btn.config(state=tk.NORMAL)
            self.gdop_time_btn.config(state=tk.NORMAL)
            self.crlb_btn.config(state=tk.NORMAL)
            self.update_settings_label()
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное число ≥ 0 для σ и ≥1 для числа реализаций")
            self.error_var.set("0.0")
            self.num_trials_var.set("100")

    def update_settings_label(self):
        self.settings_label.config(
            text=f"σ ошибки: {self.error_ns} нс | Опорная станция: {self.reference_station or 'не выбрана'} | Реализаций: {self.num_trials}"
        )

    def calculate_tdoa_positions(self):
        if not self.data or "uav_truth" not in self.data:
            return
        if not self.reference_station:
            messagebox.showwarning("Внимание", "Выберите опорную станцию")
            return
        if self.error_ns <= 0:
            messagebox.showwarning("Внимание", "Примените ошибку (σ > 0)")
            return
        try:
            self.num_trials = int(self.num_trials_var.get())
            if self.num_trials < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Число реализаций должно быть целым ≥1")
            return

        uav_truth = self.data["uav_truth"]
        packet_ids = sorted(uav_truth.keys(), key=lambda x: int(x.split('_')[1]))
        self.calculated_positions = {}

        for packet_id in packet_ids:
            truth = uav_truth[packet_id]
            true_x = truth.get('x')
            true_y = truth.get('y')
            if true_x is None or true_y is None:
                continue

            reception_times = {}
            for station in self.stations:
                t = self.data.get(station, {}).get(packet_id)
                if t is not None:
                    try:
                        reception_times[station] = float(t)
                    except:
                        reception_times[station] = None
                else:
                    reception_times[station] = None

            errors = []
            estimates = []
            crlb_totals = []

            for _ in range(self.num_trials):
                local_errors = {}
                for s in self.stations:
                    if s != self.reference_station:
                        local_errors[(packet_id, s)] = np.random.normal(0, self.error_ns)

                noisy_reception = reception_times.copy()
                for s in self.stations:
                    if s != self.reference_station and (packet_id, s) in local_errors:
                        eps = local_errors[(packet_id, s)] * 1e-9
                        if noisy_reception[s] is not None:
                            noisy_reception[s] += eps

                result = self.tdoa_position_estimation_noisy_with_active(noisy_reception)
                if result is None:
                    continue
                calc, active_stations = result

                err = math.sqrt((calc[0] - true_x) ** 2 + (calc[1] - true_y) ** 2)
                errors.append(err)
                estimates.append(calc)

                crlb_x, crlb_y = self.compute_CRLB_for_active_stations(true_x, true_y, self.error_ns,
                                                                       active_stations, self.reference_station)
                if np.isfinite(crlb_x) and np.isfinite(crlb_y):
                    crlb_total = np.sqrt(crlb_x ** 2 + crlb_y ** 2)
                    crlb_totals.append(crlb_total)

            if errors:
                mean_x = np.mean([p[0] for p in estimates])
                mean_y = np.mean([p[1] for p in estimates])
                std_err = np.std(errors)
                mean_crlb_total = np.mean(crlb_totals) if crlb_totals else np.nan
                self.calculated_positions[packet_id] = (mean_x, mean_y, std_err, mean_crlb_total)
            else:
                self.calculated_positions[packet_id] = (None, None, None, None)

        valid_count = sum(1 for v in self.calculated_positions.values() if v[0] is not None)
        if valid_count > 0:
            self.show_trajectory_btn.config(state=tk.NORMAL)
            self.show_errors_btn.config(state=tk.NORMAL)
            messagebox.showinfo("TDoA", f"Обработано {valid_count} пакетов с {self.num_trials} реализациями каждый")
        else:
            messagebox.showwarning("Внимание", "Ни одна позиция не вычислена")
        self.display_data(show_differences=True)

    def tdoa_position_estimation_noisy_with_active(self, reception_times):
        """Возвращает (координаты, список активных станций)"""
        if self.reference_station not in reception_times:
            return None
        t_ref = reception_times[self.reference_station]
        if t_ref is None:
            return None
        try:
            t_ref = float(t_ref)
            if not np.isfinite(t_ref):
                return None
        except:
            return None

        available_stations = []
        time_diffs_list = []
        station_coords_list = []
        ref_coords = np.array(self.station_coords.get(self.reference_station, (0.0, 0.0)), dtype=np.float64)

        for station in self.stations:
            if station == self.reference_station:
                continue
            t_i = reception_times.get(station)
            if t_i is None:
                continue
            try:
                t_i = float(t_i)
                if not np.isfinite(t_i):
                    continue
            except:
                continue

            dt = t_i - t_ref
            if not np.isfinite(dt):
                continue

            sta_coords = np.array(self.station_coords.get(station, (0.0, 0.0)), dtype=np.float64)
            D = np.linalg.norm(sta_coords - ref_coords)
            delta_d = self.C * dt
            if not np.isfinite(delta_d) or abs(delta_d) >= D - 1e-6:
                continue

            available_stations.append(station)
            time_diffs_list.append(dt)
            station_coords_list.append(sta_coords)

        if len(available_stations) < 2:
            return None

        time_diffs = np.array(time_diffs_list, dtype=np.float64)
        station_coords_arr = np.array(station_coords_list, dtype=np.float64)

        if not (np.isfinite(time_diffs).all() and np.isfinite(station_coords_arr).all()):
            return None

        initial_guess = np.mean(np.vstack([ref_coords, station_coords_arr]), axis=0)

        def error_function(pos):
            x, y = pos
            dist_ref = np.hypot(x - ref_coords[0], y - ref_coords[1])
            err = []
            for i in range(len(available_stations)):
                sx, sy = station_coords_arr[i]
                dist_i = np.hypot(x - sx, y - sy)
                delta_actual = dist_i - dist_ref
                delta_meas = self.C * time_diffs[i]
                diff = delta_actual - delta_meas
                if not np.isfinite(diff):
                    return [1e9]
                err.append(diff)
            return np.array(err)

        try:
            result = opt.least_squares(
                error_function,
                initial_guess,
                method='lm',
                ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=1000
            )
            if result.success and np.isfinite(result.x).all():
                return tuple(result.x), available_stations
        except Exception:
            pass

        chan_result = self.chan_algorithm(ref_coords, station_coords_arr, time_diffs, self.error_ns)
        if chan_result is not None:
            return chan_result, available_stations
        return None

    def chan_algorithm(self, ref_coords, station_coords, time_diffs, sigma_ns):
        c = self.C
        N = len(time_diffs)
        if N < 2:
            return None
        try:
            x_ref, y_ref = ref_coords
            x_i = station_coords[:, 0];
            y_i = station_coords[:, 1]
            d_meas = c * time_diffs
            dx = x_i - x_ref;
            dy = y_i - y_ref
            A = np.column_stack([dx, dy, d_meas])
            b = 0.5 * (d_meas ** 2 - dx ** 2 - dy ** 2)
            Q = self.build_covariance_matrix_Q(sigma_ns, N)
            Q_inv = np.linalg.inv(Q)
            ATA = A.T @ (Q_inv @ A)
            ATb = A.T @ (Q_inv @ b)
            x0_vec = np.linalg.solve(ATA, ATb)
        except:
            x0_vec = np.linalg.lstsq(A, b, rcond=None)[0]
        x1, y1, k1 = x0_vec
        dx1 = x1 - x_ref;
        dy1 = y1 - y_ref
        r0 = np.sqrt(dx1 ** 2 + dy1 ** 2 + 1e-10)
        r_hat_ref = k1 + r0
        if r_hat_ref <= 0:
            return None
        alpha = (r_hat_ref - r0) / r0
        x_est = x1 + alpha * dx1;
        y_est = y1 + alpha * dy1
        return float(x_est), float(y_est)

    def show_trajectory(self):
        if not self.calculated_positions:
            messagebox.showwarning("Внимание", "Нет вычисленных координат")
            return
        uav_truth = self.data.get("uav_truth", {})
        if not uav_truth:
            messagebox.showwarning("Внимание", "Нет истинных координат БПЛА")
            return
        packet_ids = sorted(uav_truth.keys(), key=lambda pid: uav_truth[pid].get('time', 0))
        true_x, true_y, calc_x, calc_y, times = [], [], [], [], []
        for pid in packet_ids:
            truth = uav_truth.get(pid, {})
            calc = self.calculated_positions.get(pid)
            tx = truth.get('x')
            ty = truth.get('y')
            t = truth.get('time')
            if calc is not None and calc[0] is not None and tx is not None and ty is not None and t is not None:
                true_x.append(tx)
                true_y.append(ty)
                calc_x.append(calc[0])
                calc_y.append(calc[1])
                times.append(t)
        if not true_x:
            messagebox.showwarning("Внимание", "Нет данных для траектории")
            return

        smooth_x, smooth_y = calc_x.copy(), calc_y.copy()
        if len(calc_x) >= 5:
            try:
                window = min(5, len(calc_x))
                smooth_x = uniform_filter1d(calc_x, size=window, mode='nearest')
                smooth_y = uniform_filter1d(calc_y, size=window, mode='nearest')
            except Exception as e:
                print(f"Ошибка сглаживания: {e}")

        traj_window = tk.Toplevel(self.root)
        traj_window.title("Траектория БПЛА")
        traj_window.geometry("900x700")
        fig = Figure(figsize=(9, 6), dpi=100)
        ax = fig.add_subplot(111)
        for sid, (sx, sy) in self.station_coords.items():
            color = 'red' if sid == self.reference_station else 'black'
            marker = 'D' if sid == self.reference_station else 's'
            ax.plot(sx, sy, marker, color=color, markersize=8,
                    label=f'Станция {sid} (опорная)' if sid == self.reference_station else f'Ст{sid}')
            ax.text(sx, sy + 20, f'Станция {sid}', color=color, ha='center', va='bottom')
        ax.plot(calc_x, calc_y, 'bo-', markersize=4, alpha=0.6, label='TDoA-оценка', linewidth=1, zorder=1)
        ax.plot(smooth_x, smooth_y, 'go-', markersize=4, alpha=0.6, label='Сглаженная (MA)', zorder=2, linewidth=1)
        ax.plot(true_x, true_y, 'r-', linewidth=2, label='Истинная траектория', zorder=3)
        ax.set_xlabel('X, м')
        ax.set_ylabel('Y, м')
        ax.set_title('Траектория БПЛА')
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=traj_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, traj_window)

    def show_errors(self):
        if not self.calculated_positions:
            messagebox.showwarning("Внимание", "Нет вычисленных координат")
            return
        uav_truth = self.data["uav_truth"]
        packet_ids = sorted(uav_truth.keys(), key=lambda x: int(x.split('_')[1]))

        std_errors = []
        for pid in packet_ids:
            if pid in self.calculated_positions:
                data = self.calculated_positions[pid]
                if data[2] is not None:  # std_error
                    std_errors.append(data[2])
                else:
                    std_errors.append(np.nan)

        if not std_errors or all(np.isnan(std_errors)):
            messagebox.showwarning("Внимание", "Нет данных для ошибок")
            return

        packet_indices = list(range(1, len(std_errors) + 1))

        mean_std = np.nanmean(std_errors)
        std_of_std = np.nanstd(std_errors)  # СКО от значений СКО (разброс точности по траектории)

        err_window = tk.Toplevel(self.root)
        err_window.title("СКО ошибки позиционирования по пакетам")
        err_window.geometry("1000x700")

        main_container = tk.Frame(err_window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        fig = Figure(figsize=(7, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(packet_indices, std_errors, 'b-', label='СКО ошибки')
        ax.axhline(mean_std, color='r', linestyle='--', linewidth=2,
                   label=f'Среднее СКО = {mean_std:.2f} м')
        ax.set_xlabel('Время определения координат БПЛА, мс')
        ax.set_ylabel('СКО ошибки, м')
        ax.set_title('СКО ошибки позиционирования для каждого пакета')
        ax.grid(True, alpha=0.6)
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=left_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        NavigationToolbar2Tk(canvas, left_frame)

        right_frame = tk.Frame(main_container, width=300, bg='white', relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)

        stats_header = tk.Label(right_frame, text="📊 Статистика СКО",
                                font=('Arial', 14, 'bold'), bg='#3498db', fg='white')
        stats_header.pack(fill=tk.X, pady=(0, 10))

        stats_content = f"""
    📈 Основные метрики:
    Среднее СКО:       {mean_std:.2f} м
    СКО от СКО:        {std_of_std:.2f} м
    Макс. СКО:         {np.nanmax(std_errors):.2f} м
    Мин. СКО:          {np.nanmin(std_errors):.2f} м

    Количество пакетов: {len(std_errors)}
    Число реализаций:   {self.num_trials}
    σ_TDoA:            {self.error_ns} нс
    """
        stats_text = tk.Text(right_frame, font=('Arial', 11), wrap=tk.WORD, bg='white')
        stats_text.insert(tk.END, stats_content)
        stats_text.config(state=tk.DISABLED)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        close_btn = tk.Button(right_frame, text="Закрыть", command=err_window.destroy,
                              bg="#e74c3c", fg="white", font=('Arial', 10))
        close_btn.pack(pady=10)

    def analyze_reference_stations(self):
        if not self.data or "uav_truth" not in self.data:
            messagebox.showwarning("Внимание", "Сначала загрузите данные")
            return

        try:
            self.num_trials = int(self.num_trials_var.get())
            if self.num_trials < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Число реализаций должно быть целым ≥1")
            return

        if self.error_ns <= 0:
            messagebox.showwarning("Внимание", "Задайте σ > 0")
            return

        results = self.calculate_errors_for_all_references()

        if not results:
            messagebox.showwarning("Внимание", "Не удалось рассчитать статистику")
            return

        self.plot_reference_station_analysis(results)

    def calculate_errors_for_all_references(self):
        uav_truth = self.data["uav_truth"]
        packet_ids = sorted(uav_truth.keys(), key=lambda x: int(x.split('_')[1]))

        results = []

        for ref_station in self.stations:
            print(f"Анализируем опорную станцию {ref_station}...")

            # Временное сохранение текущей опорной станции
            current_ref = self.reference_station
            self.reference_station = ref_station

            station_errors = []

            for packet_id in packet_ids:
                truth = uav_truth[packet_id]
                true_x = truth.get('x')
                true_y = truth.get('y')
                if true_x is None or true_y is None:
                    continue

                reception_times = {}
                for station in self.stations:
                    t = self.data.get(station, {}).get(packet_id)
                    if t is not None:
                        try:
                            reception_times[station] = float(t)
                        except:
                            reception_times[station] = None
                    else:
                        reception_times[station] = None

                errors = []

                for _ in range(min(self.num_trials, 50)):
                    local_errors = {}
                    for s in self.stations:
                        if s != ref_station:
                            local_errors[(packet_id, s)] = np.random.normal(0, self.error_ns)

                    noisy_reception = reception_times.copy()
                    for s in self.stations:
                        if s != ref_station and (packet_id, s) in local_errors:
                            eps = local_errors[(packet_id, s)] * 1e-9
                            if noisy_reception[s] is not None:
                                noisy_reception[s] += eps

                    result = self.tdoa_position_estimation_noisy_with_active(noisy_reception)
                    if result is None:
                        continue

                    calc, _ = result

                    err = math.sqrt((calc[0] - true_x) ** 2 + (calc[1] - true_y) ** 2)
                    errors.append(err)

                if errors:
                    station_errors.extend(errors)

            self.reference_station = current_ref

            if station_errors:
                mean_error = np.mean(station_errors)
                std_error = np.std(station_errors)
                median_error = np.median(station_errors)
                p90_error = np.percentile(station_errors, 90)

                results.append({
                    'station': ref_station,
                    'mean_error': mean_error,
                    'std_error': std_error,
                    'median_error': median_error,
                    'p90_error': p90_error,
                    'num_points': len(station_errors),
                    'num_packets': len([pid for pid in packet_ids
                                        if uav_truth[pid].get('x') is not None
                                        and uav_truth[pid].get('y') is not None])
                })
                print(f"  Станция {ref_station}: среднее СКО = {mean_error:.2f} м")
            else:
                results.append({
                    'station': ref_station,
                    'mean_error': np.nan,
                    'std_error': np.nan,
                    'median_error': np.nan,
                    'p90_error': np.nan,
                    'num_points': 0,
                    'num_packets': 0
                })

        results.sort(key=lambda x: x['mean_error'] if not np.isnan(x['mean_error']) else float('inf'))
        return results

    def plot_reference_station_analysis(self, results):
        # Фильтруем станции с результатами
        valid_results = [r for r in results if not np.isnan(r['mean_error'])]
        if not valid_results:
            messagebox.showwarning("Внимание", "Нет данных для построения графика")
            return

        # Создаем окно для отображения
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Анализ опорных станций")
        analysis_window.geometry("1200x800")

        main_frame = tk.Frame(analysis_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        fig = Figure(figsize=(12, 8), dpi=100)

        # График 1: Среднее СКО по станциям
        ax1 = fig.add_subplot(221)
        stations = [f"Станция{r['station']}" for r in valid_results]
        mean_errors = [r['mean_error'] for r in valid_results]

        bars = ax1.bar(stations, mean_errors, color='skyblue', alpha=0.7)

        if valid_results:
            min_idx = np.argmin(mean_errors)
            bars[min_idx].set_color('green')
            bars[min_idx].set_alpha(1.0)

        ax1.set_xlabel('Опорная станция')
        ax1.set_ylabel('Среднее СКО ошибки, м')
        ax1.set_title('Средняя точность по опорным станциям')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        # График 2: Сравнение статистик для лучших станций
        ax2 = fig.add_subplot(222)
        top_n = min(5, len(valid_results))
        top_results = valid_results[:top_n]

        metrics = ['mean_error', 'median_error', 'p90_error']
        metric_labels = ['Среднее', 'Медиана', '90-й процентиль']
        metric_colors = ['blue', 'orange', 'red']

        x = np.arange(len(top_results))
        width = 0.25

        for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
            values = [r[metric] for r in top_results]
            ax2.bar(x + i * width - width, values, width, label=label, color=color, alpha=0.7)

        ax2.set_xlabel('Опорная станция')
        ax2.set_ylabel('СКО ошибки, м')
        ax2.set_title(f'Сравнение метрик для топ-{top_n} станций')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Станция {r['station']}" for r in top_results])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # График 3: Распределение СКО
        ax3 = fig.add_subplot(223)

        show_stations = min(4, len(valid_results))
        for i, result in enumerate(valid_results[:show_stations]):
            mean = result['mean_error']
            std = result['std_error']
            if not np.isnan(mean) and not np.isnan(std) and std > 0:
                x_vals = np.linspace(mean - 3 * std, mean + 3 * std, 100)
                y_vals = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mean) / std) ** 2)
                ax3.plot(x_vals, y_vals, label=f"Ст{result['station']} (μ={mean:.1f}, σ={std:.1f})", linewidth=2)

        ax3.set_xlabel('СКО ошибки, м')
        ax3.set_ylabel('Плотность вероятности')
        ax3.set_title('Распределение СКО ошибок по станциям')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # График 4: Информационная панель
        ax4 = fig.add_subplot(224)
        ax4.axis('off')

        if valid_results:
            best = valid_results[0]
            worst = valid_results[-1]

            info_text = f"""
📊 Сводка анализа опорных станций

Общие параметры:
• σ_TDoA: {self.error_ns:.1f} нс
• Число реализаций: {self.num_trials}
• Всего станций: {len(self.stations)}
• Анализируемых пакетов: {best['num_packets']}

🏆 Лучшая станция: Ст{best['station']}
• Среднее СКО: {best['mean_error']:.2f} м
• Медианная ошибка: {best['median_error']:.2f} м
• 90-й процентиль: {best['p90_error']:.2f} м

📉 Худшая станция: Ст{worst['station']}
• Среднее СКО: {worst['mean_error']:.2f} м
• Медианная ошибка: {worst['median_error']:.2f} м
• 90-й процентиль: {worst['p90_error']:.2f} м

📈 Диапазон точности:
• Разница лучшая-худшая: {worst['mean_error'] - best['mean_error']:.2f} м
• Улучшение: {(worst['mean_error'] - best['mean_error']) / worst['mean_error'] * 100:.1f}%

Рекомендация:
Использовать станцию Ст{best['station']} как опорную
для достижения наилучшей точности.
"""

            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


        table_data = []
        table_columns = ['Станция', 'Среднее', 'СКО', 'Медиана', '90%', 'Точек']

        for r in valid_results:
            table_data.append([
                f"Ст{r['station']}",
                f"{r['mean_error']:.2f}",
                f"{r['std_error']:.2f}",
                f"{r['median_error']:.2f}",
                f"{r['p90_error']:.2f}",
                str(r['num_points'])
            ])

        fig2 = Figure(figsize=(10, 6), dpi=100)
        ax_table = fig2.add_subplot(111)
        ax_table.axis('tight')
        ax_table.axis('off')

        table = ax_table.table(cellText=table_data,
                               colLabels=table_columns,
                               cellLoc='center',
                               loc='center',
                               colColours=['#DDDDDD'] * len(table_columns))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        fig.tight_layout()

        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(canvas_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        table_frame = tk.Frame(main_frame, width=400, bg='white')
        table_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        table_frame.pack_propagate(False)

        table_label = tk.Label(table_frame, text="📋 Детальные результаты",
                               font=('Arial', 12, 'bold'), bg='#3498db', fg='white')
        table_label.pack(fill=tk.X, pady=(0, 10))

        table_canvas = FigureCanvasTkAgg(fig2, master=table_frame)
        table_canvas.draw()
        table_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        save_btn = tk.Button(button_frame, text="Сохранить результаты",
                             command=lambda: self.save_analysis_results(results),
                             bg="#27ae60", fg="white", font=('Arial', 10))
        save_btn.pack(side=tk.LEFT, padx=5)

        if valid_results:
            best_station = valid_results[0]['station']
            select_btn = tk.Button(button_frame,
                                   text=f"Выбрать лучшую станцию (Ст{best_station})",
                                   command=lambda: self.select_best_station(best_station),
                                   bg="#3498db", fg="white", font=('Arial', 10))
            select_btn.pack(side=tk.LEFT, padx=5)

        close_btn = tk.Button(button_frame, text="Закрыть",
                              command=analysis_window.destroy,
                              bg="#e74c3c", fg="white", font=('Arial', 10))
        close_btn.pack(side=tk.RIGHT, padx=5)

    def save_analysis_results(self, results):
        """Сохраняет результаты анализа в CSV файл"""
        file_path = filedialog.asksaveasfilename(
            title="Сохранить результаты анализа",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )

        if not file_path:
            return

        try:
            import pandas as pd

            data = []
            for r in results:
                data.append({
                    'station': r['station'],
                    'mean_error_m': r['mean_error'],
                    'std_error_m': r['std_error'],
                    'median_error_m': r['median_error'],
                    'p90_error_m': r['p90_error'],
                    'num_points': r['num_points'],
                    'num_packets': r['num_packets']
                })

            df = pd.DataFrame(data)

            df['rank'] = df['mean_error_m'].rank(method='dense').astype(int)

            if file_path.endswith('.xlsx'):
                df.to_excel(file_path, index=False)
            else:
                df.to_csv(file_path, index=False, sep=';', encoding='utf-8-sig')

            messagebox.showinfo("Успех", f"Результаты сохранены в {file_path}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")

    def select_best_station(self, station_num):
        """Выбирает указанную станцию как опорную"""
        self.ref_station_var.set(f"Станция {station_num}")
        self.on_ref_station_selected()
        messagebox.showinfo("Выбор станции",
                            f"Опорная станция установлена: Ст{station_num}\n"
                            f"Для применения обновите расчет TDoA.")

    def show_tdoa_equations_selected(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Внимание", "Выберите пакет")
            return
        if not self.reference_station:
            messagebox.showwarning("Внимание", "Выберите опорную станцию")
            return
        self.show_equations_for_selected_packet()

    def on_row_double_click(self, event):
        self.show_equations_for_selected_packet()

    def show_equations_for_selected_packet(self):
        selected_item = self.tree.selection()
        if not selected_item:
            return
        item = selected_item[0]
        values = self.tree.item(item)['values']
        if not values or len(values) < 2:
            return
        packet_id = values[0]
        uav_truth = self.data.get("uav_truth", {})
        packet_data = uav_truth.get(packet_id, {})
        if not packet_data:
            return

        true_coords = (
            packet_data.get('x'), packet_data.get('y')) if 'x' in packet_data and 'y' in packet_data else None

        calc_data = self.calculated_positions.get(packet_id)
        calculated_coords = (calc_data[0], calc_data[1]) if calc_data and calc_data[0] is not None else None

        reception_times = {}
        for station in self.stations:
            t = self.data.get(station, {}).get(packet_id)
            if t is not None:
                try:
                    reception_times[station] = float(t)
                except:
                    reception_times[station] = None

        station_errors_ns = {}
        for station in self.stations:
            if station != self.reference_station:
                station_errors_ns[station] = np.random.normal(0, self.error_ns)

        if self.equations_window:
            self.equations_window.destroy()
        self.equations_window = None
        try:
            pd = {'id': packet_id, 'time': packet_data.get('time', 0.0)}
            self.equations_window = TDoAEquationsWindow(
                self.root,
                self.reference_station,
                self.stations,
                self.station_coords,
                self.error_ns,
                packet_data=pd,
                reception_times=reception_times,
                station_errors_ns=station_errors_ns,
                true_uav_coords=true_coords,
                calculated_coords=calculated_coords
            )
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть окно уравнений:\n{e}")
            self.equations_window = None

    def update_table(self):
        if self.reference_station:
            self.display_data(show_differences=True)
        else:
            messagebox.showwarning("Внимание", "Выберите опорную станцию")

    def display_data(self, show_differences=True):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if not self.data or "uav_truth" not in self.data:
            return
        uav_truth = self.data["uav_truth"]
        packet_ids = sorted(uav_truth.keys(), key=lambda x: int(x.split('_')[1]))
        columns = ["ID пакета", "Время излучения, с"]
        for station in self.stations:
            columns.append(f"t{station}, с")
        if show_differences and self.reference_station:
            for station in self.stations:
                if station != self.reference_station:
                    columns.append(f"ε{station}, нс")
                    columns.append(f"Δt{station}, с")
        columns.extend(["Истинный x, м", "Истинный y, м", "Средний x, м", "Средний y, м", "СКО ошибки, м", "CRLB, м"])
        self.tree["columns"] = columns
        self.tree["show"] = "headings"
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)
        self.tree.column("ID пакета", width=100)
        self.tree.column("Время излучения, с", width=130)
        self.tree.column("Истинный x, м", width=100)
        self.tree.column("Истинный y, м", width=100)
        self.tree.column("Средний x, м", width=120)
        self.tree.column("Средний y, м", width=120)
        self.tree.column("СКО ошибки, м", width=120)
        self.tree.column("CRLB, м", width=120)

        for packet_id in packet_ids:
            packet_data = uav_truth[packet_id]
            emission_time = packet_data.get("time", 0.0)
            true_x = packet_data.get("x", "N/A");
            true_y = packet_data.get("y", "N/A")
            true_x_str = f"{float(true_x):.2f}" if true_x != "N/A" and true_x is not None else "N/A"
            true_y_str = f"{float(true_y):.2f}" if true_y != "N/A" and true_y is not None else "N/A"

            calc_data = self.calculated_positions.get(packet_id)
            calc_x_str = f"{calc_data[0]:.2f}" if calc_data and calc_data[0] is not None else "N/A"
            calc_y_str = f"{calc_data[1]:.2f}" if calc_data and calc_data[1] is not None else "N/A"
            std_err_str = f"{calc_data[2]:.2f}" if calc_data and calc_data[2] is not None else "N/A"
            crlb_str = f"{calc_data[3]:.2f}" if calc_data and calc_data[3] is not None else "N/A"

            row_data = [packet_id, f"{emission_time:.6f}"]
            reception_times = {}
            for station in self.stations:
                t = self.data.get(station, {}).get(packet_id)
                if t is not None:
                    try:
                        row_data.append(f"{float(t):.9f}")
                    except:
                        row_data.append("N/A")
                else:
                    row_data.append("N/A")

            if show_differences and self.reference_station:
                t_ref = self.data.get(self.reference_station, {}).get(packet_id)
                for station in self.stations:
                    if station == self.reference_station:
                        continue
                    # Для таблицы покажем одну случайную реализацию (для иллюстрации)
                    eps_i = np.random.normal(0, self.error_ns)
                    row_data.append(f"{eps_i:.2f}")
                    t_i = self.data.get(station, {}).get(packet_id)
                    if t_ref is not None and t_i is not None:
                        try:
                            dt = (float(t_i) - float(t_ref)) + eps_i * 1e-9
                            row_data.append(f"{dt:.9f}")
                        except:
                            row_data.append("N/A")
                    else:
                        row_data.append("N/A")

            row_data.extend([true_x_str, true_y_str, calc_x_str, calc_y_str, std_err_str, crlb_str])
            self.tree.insert("", tk.END, values=row_data)
        self.auto_resize_columns()

    def auto_resize_columns(self):
        for col in self.tree["columns"]:
            max_width = 100
            for item in self.tree.get_children():
                value = self.tree.set(item, col)
                if value:
                    width = len(str(value)) * 8 + 20
                    max_width = max(max_width, min(width, 200))
            self.tree.column(col, width=max_width)

    def export_to_csv(self):
        if not self.data:
            messagebox.showwarning("Внимание", "Загрузите данные")
            return
        file_path = filedialog.asksaveasfilename(
            title="Сохранить CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return
        try:
            with open(file_path, 'w', encoding='utf-8-sig') as f:
                header = ["ID пакета", "Время излучения, с"]
                for s in self.stations:
                    header.append(f"t{s}, с")
                if self.reference_station:
                    for s in self.stations:
                        if s != self.reference_station:
                            header.append(f"ε{s}, нс")
                            header.append(f"Δt{s}, с")
                header.extend(
                    ["Истинный x, м", "Истинный y, м", "Средний x, м", "Средний y, м", "СКО ошибки, м", "CRLB, м"])
                f.write(";".join(header) + "\n")
                uav_truth = self.data["uav_truth"]
                packet_ids = sorted(uav_truth.keys(), key=lambda x: int(x.split('_')[1]))
                for packet_id in packet_ids:
                    packet_data = uav_truth[packet_id]
                    row = [packet_id, str(packet_data.get("time", "N/A"))]
                    for s in self.stations:
                        t = self.data.get(s, {}).get(packet_id, "")
                        row.append(str(t))
                    if self.reference_station:
                        t_ref = self.data.get(self.reference_station, {}).get(packet_id)
                        for s in self.stations:
                            if s == self.reference_station:
                                continue
                            eps = np.random.normal(0, self.error_ns)  # для экспорта — одна реализация
                            row.append(str(eps))
                            t_i = self.data.get(s, {}).get(packet_id)
                            if t_ref is not None and t_i is not None:
                                try:
                                    dt = (float(t_i) - float(t_ref)) + eps * 1e-9
                                    row.append(str(dt))
                                except:
                                    row.append("N/A")
                            else:
                                row.append("N/A")
                    true_x = packet_data.get("x", "N/A");
                    true_y = packet_data.get("y", "N/A")
                    calc_data = self.calculated_positions.get(packet_id)
                    calc_x = str(calc_data[0]) if calc_data and calc_data[0] is not None else "N/A"
                    calc_y = str(calc_data[1]) if calc_data and calc_data[1] is not None else "N/A"
                    std_err = str(calc_data[2]) if calc_data and calc_data[2] is not None else "N/A"
                    crlb_val = str(calc_data[3]) if calc_data and calc_data[3] is not None else "N/A"
                    row.extend([str(true_x), str(true_y), calc_x, calc_y, std_err, crlb_val])
                    f.write(";".join(row) + "\n")
            messagebox.showinfo("Успех", f"Файл сохранён: {file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Экспорт не удался:\n{e}")


def main():
    root = tk.Tk()
    app = UAVDataViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()