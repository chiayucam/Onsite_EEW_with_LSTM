import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from tensorflow.keras.models import load_model
from scipy.integrate import cumtrapz
from scipy.signal import iirfilter, sosfilt


def browse_file():
    filename = tk.filedialog.askopenfilename(initialdir="./input/",
                                             title="Select a File",
                                             filetypes=[("Numpy Files", "*.npy*"), ("All Files","*.*")])
    acc = np.load(filename)
    file_label.config(text=filename)
    plot_button.config(state="normal")
    save_button.config(state="normal")
    
def plot_fig(savefile_boolean):
    def data_preprocess(acc, sosfilter):
        if acc.shape[1] == 3:
            vel = sosfilt(sosfilter, cumtrapz(acc, dx=0.01, axis=0, initial=0), axis=0)
            waveform = np.concatenate((acc, vel), axis=1)
            return np.reshape(waveform, (1, waveform.shape[0], waveform.shape[1]))

    def init():
        for idx, line in enumerate(lines):
            line.set_data([], [])
        line_pre[0].set_data([], [])
        return lines+line_pre
    
    def animate(i):
        for idx, line in enumerate(lines):
            line.set_data(t[len(t)-i*5:len(t)], waveform[0,:i*5,idx])
        line_pre[0].set_data(t[len(t)-i*5:len(t)], predict[0,:i*5,0])
        return lines+line_pre
    
    lines = []
    y_labels = ['Z_acc', 'N_acc', 'E_acc', 'Z_vel', 'N_vel', 'E_vel', 'Prob']
    acc = np.load(file_label["text"])
    waveform = data_preprocess(acc, sosfilter)
    predict = model.predict(waveform)
    acc_max = waveform[0,:,:3].max()
    vel_max = waveform[0,:,3:].max()
    
    fig.clear()
    fig.subplots_adjust(hspace=0.1)
    t = np.arange(0, waveform.shape[1] / 100 - 0.00001, 0.01)[::-1]
    
    for idx in range(waveform.shape[2]):
        ax = fig.add_subplot(7, 1, idx+1)
        ax.set_xlim(0, t.max())
        if idx<3:
            ax.set_ylim(-acc_max, acc_max)
        else:
            ax.set_ylim(-vel_max, vel_max)
        ax.tick_params(
            axis='x',          
            which='both',      
            bottom=False,      
            top=False,         
            labelbottom=False)
        ax.invert_xaxis()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(y_labels[idx], rotation=45)
        lines.append(ax.plot([], [])[0])
    
    ax = fig.add_subplot(7, 1, 7)
    ax.set_xlim(0, t.max())
    ax.set_ylim(0, 1)
    ax.tick_params(
        axis='x',          
        which='both',      
        bottom=True,      
        top=False,         
        labelbottom=True)
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(y_labels[6], rotation=45)
    ax.set_xlabel("Time(sec)")
    ax.invert_xaxis()
    ax.plot([0, t.max()], [0.5, 0.5], color='grey', linewidth=1)
    line_pre = ax.plot([], [])
    
    ani = animation.FuncAnimation(fig=fig, func=animate, frames=600, init_func=init, interval=1, blit=True, repeat=False)
    
    if savefile_boolean:
        filename = tk.filedialog.asksaveasfilename(title="Save file as", 
                                                   filetypes=[("GIF Image", "*.gif"), 
                                                              ("All files", "*.*")], 
                                                   defaultextension=".gif")
        if filename:
            print(filename)
            writergif = animation.PillowWriter(fps=30) 
            ani.save(filename, writer=writergif)

testing_model = 'L5U2B512Onadam'
# load model
model = load_model('./'+testing_model+'.h5')
sosfilter = iirfilter(4, 0.075, btype="highpass", output="sos", fs=100)

window = tk.Tk()
window.title("Onsite EEW with LSTM")
file_label = tk.Label(window, text="", width=100, height=2, fg="black")
file_label.pack()

div1 = tk.Frame(window)
div1.pack()

file_button = tk.Button(div1, text="Open file", command=browse_file)
file_button.pack(side="left")

plot_button = tk.Button(div1, text="Start animation", command=lambda : plot_fig(False))
plot_button.config(state="disabled")
plot_button.pack(side="left")

save_button = tk.Button(div1, text="Save animation", command=lambda : plot_fig(True))
save_button.config(state="disable")
save_button.pack(side="right")


fig = plt.figure(figsize=(12,12))
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.titlepad'] = 25
plt.rcParams['axes.labelpad'] = 20

canvas = FigureCanvasTkAgg(fig, window)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

window.mainloop()