import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from tensorflow.keras.models import load_model
from scipy.integrate import cumtrapz
from scipy.signal import iirfilter, sosfilt


def data_preprocess(acc, sosfilter):
    if acc.shape[1] == 3:
        vel = sosfilt(sosfilter, cumtrapz(acc, dx=0.01, axis=0, initial=0), axis=0)
        waveform = np.concatenate((acc, vel), axis=1)
        return np.reshape(waveform, (1, waveform.shape[0], waveform.shape[1]))

def browse_file():
    filename = tk.filedialog.askopenfilename(initialdir="./input/",
                                             title="Select a File",
                                             filetypes=[("Numpy Files", "*.npy*"), ("All Files","*.*")])
    acc = np.load(filename)
    time_length = int(acc.shape[0]/100)
    file_label.config(text=filename)
    scale.config(state="normal", to=time_length)
    time_window_spinbox.config(state="normal", to=time_length)
    plot_button.config(state="normal")
    save_button.config(state="normal")
    plot_fig1(acc)

    
def change_start_time(event=None):
    ax = fig1.gca()
    ax.patches[0].set_xy((int(scale.get())*100, -100))
    canvas1.draw()

def change_time_window():
    ax = fig1.gca()
    ax.patches[0].set_width(int(time_window_spinbox.get())*100)
    canvas1.draw()

def plot_fig1(acc):
    fig1.clear()
    ax = fig1.add_subplot(1, 1, 1)
    ax.plot(acc)
    ax.set_xlim(0, len(acc))
    ax.set_ylim(-100, 100)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    rect = Rectangle((int(scale.get())*100, -100), int(time_window_spinbox.get())*100, 200, color="grey", alpha=0.5)
    ax.add_patch(rect)
    canvas1.draw()
    
def plot_fig2(savefile_boolean):
    time_window = int(time_window_spinbox.get())
    start_time = scale.get()
    acc = np.load(file_label["text"])[start_time*100:start_time*100+time_window*100,:]
    waveform = data_preprocess(acc, sosfilter)
    predict = model.predict(waveform)
    
    fig2.clear()
    t1 = np.arange(start_time, start_time+waveform.shape[1] / 100 - 0.00001, 0.01)
    t2 = np.arange(start_time, start_time+predict.shape[1] / 100 - 0.00001, 0.01)
    ax1 = fig2.add_subplot(2, 1, 1)
    ax1.set_xlim(start_time, start_time+time_window)
    ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    ax1.plot(t1, waveform[0,:,:3])
    ax1.set_ylabel("Acceleration (gal)")
    ax1.legend(['Z', 'N', 'E'], prop={'size': 14})

    ax2 = fig2.add_subplot(2, 1, 2)
    ax2.set_xlim(start_time, start_time+time_window)
    ax2.set_ylim(0, 1)
    ax2.plot(t2, predict[0], 'red', label="Probability")
    ax2.set_xlabel("Time (sec)")
    ax2.set_ylabel("Probability")
    ax2.legend(loc=4, prop={'size': 14})

    fig2.subplots_adjust(hspace=0.07)
    canvas2.draw()

    if savefile_boolean:
        filename = tk.filedialog.asksaveasfilename(title="Save file as", filetypes=[("PNG Image", "*.png"), ("JPG Image", "*.jpg")])
        if filename:
            fig2.savefig(filename)

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

plot_button = tk.Button(div1, text="Plot figure", command=lambda : plot_fig2(False))
plot_button.config(state="disabled")
plot_button.pack(side="left")

save_button = tk.Button(div1, text="Save figure", command=lambda : plot_fig2(True))
save_button.config(state="disable")
save_button.pack(side="right")

div2 = tk.Frame(window)
div2.pack()

scale_label = tk.Label(div2, text="start time:")
scale_label.grid(column=0, row=0)

scale = tk.Scale(div2, orient=tk.HORIZONTAL, length=200, command=change_start_time)
scale.config(state="disabled")
scale.grid(column=1, row=0)

time_window_label = tk.Label(div2, text="time window:")
time_window_label.grid(column=0, row=1)

time_window_spinbox = tk.Spinbox(div2, from_=1, to=15, command=change_time_window)
time_window_spinbox.config(state="disabled")
time_window_spinbox.grid(column=1, row=1)


fig1 = plt.figure(figsize=(12,1.5))
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.titlepad'] = 25
plt.rcParams['axes.labelpad'] = 20
canvas1 = FigureCanvasTkAgg(fig1, window)
canvas1.draw()
canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)

fig2 = plt.figure(figsize=(12,7))
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.titlepad'] = 25
plt.rcParams['axes.labelpad'] = 20
canvas2 = FigureCanvasTkAgg(fig2, window)
canvas2.draw()
canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

window.mainloop()