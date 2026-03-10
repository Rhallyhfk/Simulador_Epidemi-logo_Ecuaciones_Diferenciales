import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SimuladorSIRD_Hibrido:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador Epidemiológico: Agentes vs Ecuaciones Diferenciales")
        self.root.geometry("1350x650") 

        self.bg_color = "#1E1E2E"
        self.panel_color = "#181825"
        self.text_color = "#CDD6F4"
        
        self.colores = {
            0: "#18F404",  # Sanos
            1: "#F10606",  # Infectados
            2: "#0800E9",  # Recuperados
            3: "#F2F2F3"   # Fallecidos
        }

        self.root.configure(bg=self.bg_color)
        
        # Parámetros de la Simulación
        self.num_particulas = 400
        self.radio = 4
        self.w_canvas, self.h_canvas = 550, 550
        self.radio_contagio_sq = (self.radio * 3) ** 2
        
        self.pos = np.random.rand(self.num_particulas, 2) * [self.w_canvas, self.h_canvas]
        angulos = np.random.rand(self.num_particulas) * 2 * np.pi
        self.vel = np.column_stack((np.cos(angulos) * 2.0, np.sin(angulos) * 2.0))
        self.estados = np.zeros(self.num_particulas, dtype=int)
        self.estados[0:3] = 1

        self.historia_s = []
        self.historia_i = []
        self.historia_r = []
        self.historia_d = []
        self.tiempo = 0

        self.crear_interfaz()
        
        self.canvas = tk.Canvas(self.root, width=self.w_canvas, height=self.h_canvas, 
                                bg=self.bg_color, highlightthickness=1, highlightbackground="#313244")
        self.canvas.pack(side=tk.LEFT, padx=20, pady=20)
        
        self.puntos = []
        for i in range(self.num_particulas):
            x, y = self.pos[i]
            r = self.radio
            color = self.colores[self.estados[i]]
            pto = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline="")
            self.puntos.append(pto)

        self.crear_grafica()
        self.calcular_ode_teorico() # Calcular la curva matemática ideal

        self.actualizar_ciclo()

    def crear_interfaz(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=self.panel_color)
        style.configure("TLabel", background=self.panel_color, foreground=self.text_color)
        style.configure("Horizontal.TScale", background=self.panel_color)
        
        sidebar = ttk.Frame(self.root, padding="20", style="TFrame")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(sidebar, text="PARÁMETROS", font=('Helvetica', 12, 'bold')).pack(pady=(0,10))
        self.beta = self.crear_slider(sidebar, "Contagio (β)", 0.8)
        self.gamma = self.crear_slider(sidebar, "Recuperación (γ)", 0.03)
        self.mu = self.crear_slider(sidebar, "Letalidad (μ)", 0.01)

        btn_style = {'bg': '#89B4FA', 'fg': '#11111B', 'font': ('Helvetica', 10, 'bold'), 'relief': 'flat', 'cursor': 'hand2'}
        tk.Button(sidebar, text="Reiniciar Simulación", command=self.limpiar_mundo, **btn_style).pack(pady=30, fill='x')

    def crear_slider(self, parent, label, default):
        frame = ttk.Frame(parent)
        frame.pack(pady=10, fill='x')
        ttk.Label(frame, text=label).pack(side=tk.TOP, anchor="w")
        var = tk.DoubleVar(value=default)
        val_label = ttk.Label(frame, text=f"{default:.2f}", font=('Helvetica', 9))
        val_label.pack(side=tk.RIGHT)
        slider = ttk.Scale(frame, from_=0.0, to=1.0, variable=var, orient=tk.HORIZONTAL,
                           command=lambda v: val_label.config(text=f"{float(v):.2f}"))
        slider.pack(side=tk.LEFT, fill='x', expand=True, padx=(0,10))
        return var

    def crear_grafica(self):
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(5.5, 5), facecolor=self.bg_color)
        self.ax.set_facecolor(self.panel_color)
        self.ax.set_title("Curva Epidemiológica", color=self.text_color)
        self.ax.set_xlabel("Tiempo")
        self.ax.set_ylabel("Población")
        self.ax.set_ylim(0, self.num_particulas + 10)
        
        # Líneas de simulación (Datos reales de las partículas)
        self.linea_s, = self.ax.plot([], [], color=self.colores[0], lw=2, label="Sanos (Sim)")
        self.linea_i, = self.ax.plot([], [], color=self.colores[1], lw=2, label="Infectados (Sim)")
        self.linea_r, = self.ax.plot([], [], color=self.colores[2], lw=2, label="Recup. (Sim)")
        self.linea_d, = self.ax.plot([], [], color=self.colores[3], lw=2, label="Fallec. (Sim)")

        # Líneas de Ecuaciones Diferenciales (Teoría)
        self.linea_ode_s, = self.ax.plot([], [], color=self.colores[0], lw=1.5, linestyle='--', alpha=0.6, label="Sanos (ODE)")
        self.linea_ode_i, = self.ax.plot([], [], color=self.colores[1], lw=1.5, linestyle='--', alpha=0.6, label="Infectados (ODE)")

        self.ax.legend(loc="upper right", fontsize=8, facecolor=self.panel_color, edgecolor=self.bg_color)

        self.canvas_grafica = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_grafica.get_tk_widget().pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH, expand=True)

    def derivadas_sird(self, y, t, N, beta, gamma, mu):
        S, I, R, D = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I - mu * I
        dRdt = gamma * I
        dDdt = mu * I
        return dSdt, dIdt, dRdt, dDdt

    def calcular_ode_teorico(self):
        # Ajustamos ligeramente los parámetros de las ecuaciones para que se asemejen
        # a la probabilidad de colisión del espacio 2D de nuestro canvas.
        factor_colision = 0.08 
        b_ode = self.beta.get() * factor_colision
        g_ode = self.gamma.get() * 0.05
        m_ode = self.mu.get() * 0.05

        N = self.num_particulas
        I0 = 3
        S0 = N - I0
        R0 = 0
        D0 = 0
        y0 = S0, I0, R0, D0
        
        t = np.linspace(0, 1000, 1000)
        ret = odeint(self.derivadas_sird, y0, t, args=(N, b_ode, g_ode, m_ode))
        S, I, R, D = ret.T

        self.linea_ode_s.set_data(t, S)
        self.linea_ode_i.set_data(t, I)

    def limpiar_mundo(self):
        self.pos = np.random.rand(self.num_particulas, 2) * [self.w_canvas, self.h_canvas]
        self.estados = np.zeros(self.num_particulas, dtype=int)
        angulos = np.random.rand(self.num_particulas) * 2 * np.pi
        self.vel = np.column_stack((np.cos(angulos) * 2.0, np.sin(angulos) * 2.0))
        self.estados[0:3] = 1 
        
        self.historia_s.clear()
        self.historia_i.clear()
        self.historia_r.clear()
        self.historia_d.clear()
        self.tiempo = 0
        
        self.calcular_ode_teorico() # Recalcular la teoría con los nuevos sliders
        self.ax.set_xlim(0, 100) # Reiniciar vista del eje X

    def actualizar_ciclo(self):
        vivos = self.estados != 3
        self.pos[vivos] += self.vel[vivos]
        
        rebote_x = (self.pos[:, 0] <= self.radio) | (self.pos[:, 0] >= self.w_canvas - self.radio)
        rebote_y = (self.pos[:, 1] <= self.radio) | (self.pos[:, 1] >= self.h_canvas - self.radio)
        self.vel[rebote_x, 0] *= -1
        self.vel[rebote_y, 1] *= -1
        self.pos[:, 0] = np.clip(self.pos[:, 0], self.radio, self.w_canvas - self.radio)
        self.pos[:, 1] = np.clip(self.pos[:, 1], self.radio, self.h_canvas - self.radio)

        # 3. CONTAGIO
        b = self.beta.get() * 0.1
        g = self.gamma.get() * 0.05
        m = self.mu.get() * 0.05

        idx_infectados = np.where(self.estados == 1)[0]
        idx_sanos = np.where(self.estados == 0)[0]

        if len(idx_infectados) > 0 and len(idx_sanos) > 0:
            pos_inf = self.pos[idx_infectados]
            pos_san = self.pos[idx_sanos]
            dx = pos_san[:, 0:1] - pos_inf[:, 0]
            dy = pos_san[:, 1:2] - pos_inf[:, 1]
            dist_sq = dx**2 + dy**2
            cerca = dist_sq < self.radio_contagio_sq
            
            for i, idx_s in enumerate(idx_sanos):
                if np.any(cerca[i]) and np.random.rand() < b:
                    self.estados[idx_s] = 1

        # 4. RECUPERACIÓN / FALLECIMIENTO
        for idx in idx_infectados:
            suerte = np.random.rand()
            if suerte < m:
                self.estados[idx] = 3
                self.vel[idx] = 0 
            elif suerte < m + g:
                self.estados[idx] = 2

        for i in range(self.num_particulas):
            x, y = self.pos[i]
            r = self.radio
            self.canvas.coords(self.puntos[i], x-r, y-r, x+r, y+r)
            self.canvas.itemconfig(self.puntos[i], fill=self.colores[self.estados[i]])

        # 6. ACTUALIZAR GRÁFICA 
        self.tiempo += 1
        if self.tiempo % 15 == 0:
            valores, conteos = np.unique(self.estados, return_counts=True)
            stats = dict(zip(valores, conteos))
            
            self.historia_s.append(stats.get(0, 0))
            self.historia_i.append(stats.get(1, 0))
            self.historia_r.append(stats.get(2, 0))
            self.historia_d.append(stats.get(3, 0))
            t_data = np.arange(len(self.historia_s)) * 5
            
            self.linea_s.set_data(t_data, self.historia_s)
            self.linea_i.set_data(t_data, self.historia_i)
            self.linea_r.set_data(t_data, self.historia_r)
            self.linea_d.set_data(t_data, self.historia_d)
            
            if t_data[-1] > self.ax.get_xlim()[1]:
                self.ax.set_xlim(0, t_data[-1] + 100)
            
            self.canvas_grafica.draw()

        self.root.after(30, self.actualizar_ciclo)

if __name__ == "__main__":
    root = tk.Tk()
    app = SimuladorSIRD_Hibrido(root)
    root.mainloop()