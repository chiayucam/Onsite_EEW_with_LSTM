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
    animation_speed_spinbox.config(state="normal")
    
def plot_fig(savefile_boolean):
    def data_preprocess(acc, sosfilter):
        if acc.shape[1] == 3:
            vel = sosfilt(sosfilter, cumtrapz(acc, dx=0.01, axis=0, initial=0), axis=0)
            waveform = np.concatenate((acc, vel), axis=1)
            return np.reshape(waveform, (1, waveform.shape[0], waveform.shape[1]))
    
    def get_gal_gt80_position(waveform):
        positions = np.where(np.absolute(waveform[0,:,:3])>80)
        try:
            return positions[0][0], positions[1][0]
        except IndexError:
            return np.nan, np.nan

    def get_prob_gt50_position(predict):
        positions = np.where(predict[0,:,0]>0.5)
        try:
            return positions[0][0], 6
        except IndexError:
            return np.nan, np.nan

    def init():
        for idx, line in enumerate(lines):
            line.set_data([], [])
        line_pre[0].set_data([], [])
        return lines+line_pre+exceed_lines
    
    def animate(i):
        for idx, line in enumerate(lines):
            line.set_data(t[len(t)-i*s_per_frame:len(t)], waveform[0,:i*s_per_frame,idx])
        line_pre[0].set_data(t[len(t)-i*s_per_frame:len(t)], predict[0,:i*s_per_frame,0])
        for idx, exceed_line in enumerate(exceed_lines):
            exceed_line.set_xdata(exceed_line.get_xdata()+t_per_frame)
        return lines+line_pre+exceed_lines
    
    
    
    lines = []
    exceed_lines = []
    y_labels = ['Z_acc', 'N_acc', 'E_acc', 'Z_vel', 'N_vel', 'E_vel', 'Prob']
    acc = np.load(file_label["text"])
    waveform = data_preprocess(acc, sosfilter)
    predict = model.predict(waveform)
    acc_max = np.absolute(waveform[0,:,:3]).max()
    vel_max = np.absolute(waveform[0,:,3:]).max()
    t = np.arange(0, waveform.shape[1] / 100 - 0.00001, 0.01)[::-1]
    
#   animation parameters   #
    frames_num = 600
    interval_num = 1
    t_max = t.max()
    t_per_frame = t_max/frames_num      # sec/frame
    s_per_frame = int(t_per_frame*100)       # sec/frame * 100Hz sampling rate
############################

    fig.clear()
    fig.subplots_adjust(hspace=0.2)
    
#   features subplot
    for idx in range(waveform.shape[2]):
        ax = fig.add_subplot(7, 1, idx+1)
        ax.set_xlim(0, t_max)
#         ax.minorticks_on()
        ax.invert_xaxis()
        if idx<3:
            ax.set_ylim(-acc_max, acc_max)
            ax.plot([0, t_max], [-80, -80], color='grey', linewidth=0.5)
            ax.plot([0, t_max], [80, 80], color='grey', linewidth=0.5)
        else:
            ax.set_ylim(-vel_max, vel_max)
        ax.tick_params(
            axis='x',          
            which='both',
            direction='in',
            bottom=True,      
            top=False,         
            labelbottom=False)
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(y_labels[idx], rotation=45)
        lines.append(ax.plot([], [])[0])
    
#   probability subplot
    ax = fig.add_subplot(7, 1, 7)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1)
#     ax.minorticks_on()
    ax.invert_xaxis()
    ax.tick_params(
        axis='x',          
        which='both',
        direction='inout',
        bottom=True,      
        top=False,         
        labelbottom=True)
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(y_labels[6], rotation=45)
    ax.set_xlabel("Time(sec)")
    ax.plot([0, t_max], [0.5, 0.5], color='grey', linewidth=0.5)
    line_pre = ax.plot([], [])
    
#   plot exceed lines
    exceed_positions = [get_gal_gt80_position(waveform), get_prob_gt50_position(predict)]
    for exceed_position, axes_num in exceed_positions:
        if not np.isnan(exceed_position):
            exceed_lines.append(fig.axes[axes_num].plot([-exceed_position/100, -exceed_position/100], 
                                        [-acc_max, acc_max], color='red')[0])
    
    ani = animation.FuncAnimation(fig=fig,
                                  func=animate,
                                  frames=frames_num, 
                                  init_func=init, 
                                  interval=interval_num, 
                                  blit=True, 
                                  repeat=False)
    
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
window.iconbitmap("./icon.ico")
window.geometry("1000x900")
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

div2 = tk.Frame(window)
div2.pack()

animation_speed_label = tk.Label(div2, text="Animation speed")
animation_speed_label.pack(side="left")

animation_speed_spinbox = tk.Spinbox(div2, from_=1, to=3, width=3)
animation_speed_spinbox.config(state="disabled")
animation_speed_spinbox.pack(side="right")


fig = plt.figure(figsize=(12,12))
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlepad'] = 25
plt.rcParams['axes.labelpad'] = 20

canvas = FigureCanvasTkAgg(fig, window)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

window.mainloop()
