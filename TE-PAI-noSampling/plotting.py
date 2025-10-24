"""
Plotting functions for TE-PAI analyses.
"""
import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from HAMILTONIAN import Hamiltonian
from pathlib import Path
import csv
import ast
#from calculations import calcOverhead, trotter, getTrotterPai, JSONtoDict, DictToArr 
from helpers import saveData, strip_trailing_dot_zero, quimb_to_qiskit, parse_pi_over, group_sum
from main import TE_PAI
from scipy.stats import norm
import ast, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator


# Shared style settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'lines.linewidth': 2,
    'errorbar.capsize': 3,
    'savefig.dpi': 300,
    'figure.autolayout': True,
})

tab_colors = {
    "red":    mcolors.TABLEAU_COLORS["tab:red"],
    "green":  mcolors.TABLEAU_COLORS["tab:green"],
    "blue":   mcolors.TABLEAU_COLORS["tab:blue"],
    "purple":   mcolors.TABLEAU_COLORS["tab:purple"],
    "cyan":   mcolors.TABLEAU_COLORS["tab:cyan"],
    "orange": mcolors.TABLEAU_COLORS["tab:orange"]
}

gate_colors = {
    "h":   (tab_colors["red"],   "white"),
    "z":   (tab_colors["green"], "white"),
    "rxx": (tab_colors["blue"],  "white"),
    "ryy": (tab_colors["purple"],  "white"),
    "rzz": (tab_colors["cyan"],  "white"),
    "zz":  (tab_colors["blue"],  "white"),
    "rz":  (tab_colors["orange"], "black"),
}

# --- Folder-based plotting ---
def plot_data_from_folder(folderpath, ax=None, trotBounds=None):
    quimb_pattern = re.compile(r'N-(\d+)-n-(\d+)-([cp])-(\d+)-Δ-(\w+)-T-([\d\.]+)-q-(\d+)-dT-([\d\.]+)\.csv')
    lie_pattern = re.compile(r'lie-N-(\d+)-T-((?:\d+\.\d+)|(?:\d+))-q-(\d+)\.csv')
    
    quimb_data = []
    lie_data = []
    
    for filename in os.listdir(folderpath):
        print(filename)
        filepath = os.path.join(folderpath, filename)
        
        quimb_match = quimb_pattern.match(filename)
        lie_match = lie_pattern.match(filename)
        
        if lie_match:
            df = pd.read_csv(filepath)
            if df.shape[1] >= 2:
                label = f"N = {lie_match.group(1)}"
                lie_data.append((df.iloc[:, 0], df.iloc[:, 1], label))


        elif quimb_match:
            df = pd.read_csv(filepath)
            if df.shape[1] >= 3:

                char = quimb_match.group(3)
                lab = "ERROR"
                if char == "c":
                    lab = "paralell"
                if char == "p":
                    lab = "pool"

                if(len(lie_data) == 0):
                    label = f"N-{quimb_match.group(1)} {lab}-{quimb_match.group(4)} Δ-{quimb_match.group(5)} T-{quimb_match.group(6)} q-{quimb_match.group(7)} dT-{quimb_match.group(8)}"
                else:
                    delta = quimb_match.group(5)
                    denominator = delta[len("pi_over_"):]
                    power = int(denominator).bit_length() - 1
                    delta = r"\frac{{\pi}}{{2^{{{}}}}}".format(power)
                    label = f"Pool size = {quimb_match.group(4)}, Δ = ${delta}$, dT = {quimb_match.group(8)}"
                quimb_data.append((df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], label))
        

    # prepare axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
        created_fig = True
    
    # plot quimb error-bars
    for x, y, err, label in quimb_data:
        x = x.to_list()
        y = y.to_list()
        err = err.to_list()
        x.insert(0,0)
        y.insert(0,1)
        err.insert(0,0)

        ax.errorbar(
            x, y, yerr=err,
            label=f'TE-PAI ({label})',
            capsize=3, elinewidth=1.5,
            marker='o', markersize=5, color="tab:green"
        )
    
    # plot lie data
    colors = ["gray", "black"]
    linestyles = ["--", "-"]

    if(len(lie_data)==1):
        x, y, label = lie_data[0]
        x = x.to_list()
        y = y.to_list()
        ax.plot(
            x, y,
            label=f'Trotterization ({label})',
            linewidth=2,
            color="black",
        )

    else:
        for i,(x, y, label) in enumerate(lie_data):
            ax.plot(
                x, y,
                label=f'Trotterization ({label})',
                linewidth=2,
                color=colors[i%2],
                linestyle=linestyles[i%2]
            )
    
    ax.grid(True, which='both')
    ax.legend(loc='upper right')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle X_0 \rangle$')
    if created_fig:
        plt.tight_layout()
        plt.show()
    
    return ax

def plot_data_two_folders(folder1, folder2, q, N):
    """
    Plot the contents of two folders side-by-side for direct comparison,
    and add rising Trotter‐error bars to the “Trotterization” curve in the second plot.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # first plot unchanged
    plot_data_from_folder(folder1, ax=axes[0])
    axes[0].text(
        0.05, 0.65, os.path.basename(folder1),
        transform=axes[0].transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    # build your spin-chain Hamiltonian
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    # call l1_norm with T=1 to get ||H||_1, then square for C
    C_bound = hamil.l1_norm(1)**2

    # second plot – draw the curves
    plot_data_from_folder(folder2, ax=axes[1])
    axes[1].text(
        0.05, 0.65, os.path.basename(folder2),
        transform=axes[1].transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    ylims = axes[1].get_ylim()
    # find and decorate the Trotter line
    for line in axes[1].get_lines():
        if line.get_label().startswith("Trotterization"):
            x = line.get_xdata()
            y = line.get_ydata()
            
            trotter_err = (x**2) / (2 * N)#trotter_error_bound(hamil, N, x) #(x**2) / (2 * N) #* C_bound
            axes[1].errorbar(
                x, y,
                yerr=trotter_err,
                fmt='none',
                ecolor='gray',
                alpha=0.7,
                label="Trotter error bound",
                zorder=0
            )
            break
    # Restore y-limits so errorbars don't expand the axis
    
    #axes[1].set_ylim(ylims)
    axes[1].legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("firstTEPAI")
    plt.show()

def plot_data(Ts, averages, stds):
    plt.figure(figsize=(8, 6))
    plt.errorbar(Ts, averages, yerr=stds, fmt='o', capsize=5, capthick=2, elinewidth=1, label="Data with Error Bars")
    plt.xlabel("T values")
    plt.ylabel("Averages")
    plt.title("Plot of Averages vs T with Error Bars")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Helpers ---
def calcOverhead(q, T, Δ):
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    return np.exp(2 * hamil.l1_norm(T) * np.tan(Δ / 2))


# --- High-level dashboards ---
def plotMainCalc2(folder, both=True, justLengths=False):
    trotsim  = [[], []]; paisim  = [[], [], []]
    trotbond = [[], []]; paibond = [[], []]
    trotcost = [[], []]; paicost = [[], []]
    
    params = folder.split("-")
    folder = Path(folder)
    for file in folder.iterdir():
        match file.name:
            case "lengths.csv":
                with file.open() as fp:
                    next(fp)
                    line = next(fp)
                    reader = csv.reader([line.strip()])
                    row = next(reader)
                    trotterLengths = ast.literal_eval(row[0])
                    tePAILengths   = ast.literal_eval(row[2])

            case name if name.startswith("lie"):
                second = name.endswith("fixedCircuit.csv")
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi = map(float, line.split(','))
                        trotsim[0].append(xi); trotsim[1].append(yi)

            case name if name.startswith("N"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paisim[0].append(xi); paisim[1].append(yi); paisim[2].append(zi)

            case name if name.startswith("trotter-bonds"):
                second = name.endswith("fixedCircuit.csv")
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        trotbond[0].append(xi)
                        trotbond[1].append(yi)
                        trotcost[0].append(xi)
                        trotcost[1].append(zi)

            case name if name.startswith("TEPAI-bonds"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paibond[0].append(xi)
                        paibond[1].append(yi)
                        paicost[0].append(xi)
                        paicost[1].append(zi)


    index = -1
    for i,t in enumerate(trotsim[0]):
        if t == paisim[0][0]:
            index = i
    if index == -1:
        print(f"Error: time pai start time {paisim[0][0]} not found in trotter times {trotsim[0]}")
        return

    tePAILengths = [t+trotterLengths[index] for t in tePAILengths]

    if paibond[1][0] != 0:
        paibond[0] = [p+(paibond[0][1]-paibond[0][0]) for p in paibond[0]]
        paibond[0].insert(0,trotbond[0][index])
        paibond[1].insert(0,trotbond[1][index])
    if trotbond[1][0] != 0:
        trotbond[1].insert(0,0)
        trotbond[0].insert(0,0)
    if tePAILengths[0] != 0:
        tePAILengths.insert(0,trotterLengths[index])
    if trotterLengths[0] != 0:
        trotterLengths.insert(0,0)

    paicost[0] = [p + (paicost[0][1]-paicost[0][0]) for p in paicost[0]]
    paicost[0].insert(0,trotcost[0][index])
    trotcost[0].insert(0,0)


    trotcost[1] = np.array(trotterLengths) * np.array(trotbond[1])**3
    if not justLengths:
        paicost[1]   = np.array(tePAILengths) * np.array(paibond[1])**3
    if justLengths:
        paicost[1]   = np.array(tePAILengths) * np.array(trotbond[1][index+1:])**3

    title = f"q={params[6]} | Δ={params[3]}-{params[4]} | N={params[8]} | p={params[14]}" 

    threshold  = paicost[1].max()
    print(threshold)
    print(trotcost[1])
    cutoff_idx = np.argmax(trotcost[1] > threshold)
    cutoff_time = trotcost[0][cutoff_idx]


    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    fig.suptitle(title)

    # 1) Full sim
    ax0 = axes[0]
    ax0.plot(trotsim[0], trotsim[1], color='black', label='Init Trotter')
    ax0.errorbar(paisim[0], paisim[1], yerr=paisim[2],
                 color='tab:green', label='TE-PAI')
    ax0.set_ylabel('Observable')
    ax0.set_title('1) Full Simulation')
    ax0.legend(); ax0.grid(True)

    # 2) Zoom only on TE-PAI window (masking)
    te_start = paisim[0][0]
    te_end = paisim[0][-1]
    ax1 = axes[1]
    t1, y1 = np.array(trotsim[0]), np.array(trotsim[1])
    tp, yp, ep = np.array(paisim[0]), np.array(paisim[1]), np.array(paisim[2])
    m1 = (t1 >= te_start) & (t1 <= te_end)
    mp = (tp >= te_start) & (tp <= te_end)
    ax1.plot(t1[m1], y1[m1], color='black', label='Init Trotter')
    ax1.errorbar(tp[mp], yp[mp], yerr=ep[mp], color='tab:green', label='TE-PAI')
    ax1.set_xlabel('Time')
    ax1.set_title('2) Zoom on TE-PAI Region')
    ax1.legend(); ax1.grid(True)

    # 3) Cutoff & improvement regions
    ax2 = axes[2]
    trotsim[0] = np.array(trotsim[0])
    trotsim[1] = np.array(trotsim[1])
    ax2.plot(trotsim[0][trotsim[0] <= cutoff_time],
             trotsim[1][trotsim[0] <= cutoff_time],
             color='black', label='Trotter (cutoff)')
    ax2.axvline(cutoff_time, linestyle='--', color='black')
    ax2.errorbar(paisim[0], paisim[1], yerr=paisim[2],
                 color='tab:green', label='TE-PAI')
    ax2.axvline(te_end, linestyle='--', color='tab:green')
    ax2.axvspan(te_start, cutoff_time, color='gray', alpha=0.2)
    ax2.axvspan(cutoff_time, te_end, label='TE-PAI advantage',     color='tab:green', alpha=0.2)
    ax2.set_title('3) Cutoff & Improvement') 
    ax2.legend(); ax2.grid(True)

    # 4) Bond sizes
    ax3 = axes[3]
    ax3.plot(trotbond[0], trotbond[1], color='black', label='Init Trotter')
    ax3.plot(paibond[0],  paibond[1],   color='tab:green', label='TE-PAI')
    ax3.set_xlabel('Time'); ax3.set_ylabel('Max bond')
    ax3.set_title('4) Bond Dimension')
    ax3.legend(); ax3.grid(True)

    # 5) Gate counts
    ax4 = axes[4]
    adj_te  = tePAILengths.copy()

    index = -1
    for i,t in enumerate(trotsim[0]):
        if t == paisim[0][0]:
            index = i
    if index == -1:
        print("Error: time pai start time {paisim[0][0]} not found in trotter times {trotsim[0]}")
        return
            
    len_last = trotterLengths[index]
    ax4.plot(trotbond[0], trotterLengths, color='black', label='Init Trotter')
    ax4.plot(paisim[0],   tePAILengths,   color='tab:green', label='TE-PAI')
    #ax4.plot(times,       adj_te,         color="red", label="others")
    ax4.set_xlabel('Time'); ax4.set_ylabel('# gates')
    ax4.set_title('5) Gate Count')
    ax4.legend(); ax4.grid(True)

    # 6) Calculation cost
    #for change,cost in enumerate(paicost[1]):
    #    if cost in trotcost[1]:
    #        break

    ax5 = axes[5]
    ax5.plot(trotcost[0], trotcost[1], color='black', label='Init Trotter')
    ax5.plot(paicost[0],  paicost[1],    color='tab:green', label='TE-PAI')
    ax5.set_xlabel('Time'); ax5.set_ylabel('Cost')
    if not justLengths:
        ax5.set_title('6) Compute cost for individual bond sizes')
    else:
        ax5.set_title('6) Compute cost for shared bond size')
    ax5.legend(); ax5.grid(True)

    plt.tight_layout()
    plt.show()

def plotMainCalcOld(folder, both=True, justLengths=False):
    # Load simulation and bond data
    trotsim, paisim = [[], []], [[], [], []]
    trotbond, paibond = [[], []], [[], []]
    trotcost, paicost = [[], []], [[], []]
    params = folder.split("-")
    folder = Path(folder)
    for file in folder.iterdir():
        name = file.name
        if name == "lengths.csv":
            with file.open() as fp:
                next(fp)
                lengths_line = next(fp).strip()
                row = next(csv.reader([lengths_line]))
                trotterLengths = ast.literal_eval(row[0])
                tePAILengths = ast.literal_eval(row[2])
        elif name.startswith("lie"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x,y = map(float, line.split(',') )
                    trotsim[0].append(x); trotsim[1].append(y)
        elif name.startswith("N"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x,y,z = map(float, line.split(','))
                    paisim[0].append(x); paisim[1].append(y); paisim[2].append(z)
        elif name.startswith("trotter-bonds"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x,y,z = map(float, line.split(','))
                    trotbond[0].append(x); trotbond[1].append(y)
                    trotcost[0].append(x); trotcost[1].append(z)
        elif name.startswith("TEPAI-bonds"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x,y,z = map(float, line.split(','))
                    paibond[0].append(x); paibond[1].append(y)
                    paicost[0].append(x); paicost[1].append(z)

    # Align start of TE-PAI simulation
    idx = next((i for i,t in enumerate(trotsim[0]) if t == paisim[0][0]), -1)
    if idx < 0:
        raise ValueError("Cannot align TE-PAI start time.")
    tePAILengths = [t + trotterLengths[idx] for t in tePAILengths]

    # Insert initial points for alignment
    if paibond[1] and paibond[1][0] != 0:
        shift = paibond[0][1] - paibond[0][0]
        paibond[0] = [p + shift for p in paibond[0]]
        paibond[0].insert(0, trotbond[0][idx]); paibond[1].insert(0, trotbond[1][idx])
    if trotbond[1] and trotbond[1][0] != 0:
        trotbond[0].insert(0, 0); trotbond[1].insert(0, 0)
    if tePAILengths and tePAILengths[0] != 0:
        tePAILengths.insert(0, trotterLengths[idx])
    if trotterLengths and trotterLengths[0] != 0:
        trotterLengths.insert(0, 0)
    # Align cost time arrays
    shift_cost = paicost[0][1] - paicost[0][0]
    paicost[0] = [p + shift_cost for p in paicost[0]]
    paicost[0].insert(0, trotcost[0][idx])
    trotcost[0].insert(0, 0)

    # Compute costs
    trotcost[1] = np.array(trotterLengths) * (np.array(trotbond[1])**3)
    if not justLengths:
        paicost[1] = np.array(tePAILengths) * (np.array(paibond[1])**3)
    else:
        paicost[1] = np.array(tePAILengths) * (np.array(trotbond[1][idx+1:])**3)

    # Compute cutoff time
    threshold = max(paicost[1])
    cutoff_idx = int(np.argmax(np.array(trotcost[1]) > threshold))
    cutoff_time = trotcost[0][cutoff_idx]
    te_start, te_end = paisim[0][0], paisim[0][-1]

    # Convert simulation arrays to numpy
    t_arr = np.array(trotsim[0]); y_arr = np.array(trotsim[1])
    mask_zoom = (t_arr >= te_start) & (t_arr <= te_end)

    # --- Figure 1: Cutoff & Improvement with TE-PAI zoom inset ---
    fig1, ax = plt.subplots(figsize=(8,6))
    fig1.suptitle(f"q={params[6]} | Δ={params[3]}-{params[4]} | N={params[8]} | p={params[14]}")
    mask_cut = t_arr <= cutoff_time
    ax.plot(t_arr[mask_cut], y_arr[mask_cut], color='black', label='Trotterization (cutoff)')
    ax.axvline(cutoff_time, linestyle='--', color='black')
    ax.errorbar(paisim[0], paisim[1], yerr=paisim[2], color='tab:green', label='TE-PAI')
    ax.axvline(te_end, linestyle='--', color='tab:green')
    ax.axvspan(te_start, cutoff_time, color='gray', alpha=0.2)
    ax.axvspan(cutoff_time, te_end, color='tab:green', alpha=0.2, label='TE-PAI advantage')
    y_min_zoom = min(np.min(y_arr[mask_zoom]), np.min(paisim[1]))
    y_max_zoom = max(np.max(y_arr[mask_zoom]), np.max(paisim[1]))
    rect = Rectangle((te_start, y_min_zoom), te_end-te_start, y_max_zoom-y_min_zoom,
                     linestyle='--', edgecolor='black', fill=False)
    ax.add_patch(rect)
    ax.set_xlabel('Time'); ax.set_ylabel(r'$\langle X_0 \rangle$')
    ax.set_title('Cutoff & Improvement with Zoom')
    legend = ax.legend(loc='upper right', framealpha=1)
    legend.get_frame().set_facecolor('white')
    ax.grid(True)
    inset = inset_axes(ax, width='50%', height='40%', loc='lower left')
    inset.plot(t_arr[mask_zoom], y_arr[mask_zoom], color='gray', linestyle='--', label='Trotterization (continued)')
    inset.errorbar(paisim[0], paisim[1], yerr=paisim[2], fmt='o', color='tab:green', label='_nolegend_')
    inset.set_xlim(te_start, te_end)
    inset.set_xticks([]); inset.set_yticks([])
    #inset.text(0.7, 0.4, 'Zoomed in', transform=inset.transAxes,
    #           ha='center', va='bottom', fontsize='small', weight='bold')
    inset_legend = inset.legend(loc='upper right', framealpha=1)
    inset_legend.get_frame().set_facecolor('white')
    inset.grid(True)
    plt.savefig("cutoffImprovement")
    #plt.tight_layout()

    # --- Figure 2: Bond, Gate Count, Compute Cost ---
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax3, ax4, ax5 = axes

    # 1) Bond Dimension
    ax3.plot(trotbond[0], trotbond[1], color='black', label='Trotterization')
    ax3.plot(paibond[0], paibond[1], color='tab:green', label='TE-PAI')
    max_te_pa = max(paibond[1])
    top_y = ax3.get_ylim()[1]
    mask = np.array(trotbond[1]) > max_te_pa
    if mask.any():
        x_min = np.array(trotbond[0])[mask].min()
        x_max = np.array(trotbond[0])[mask].max()
        ax3.add_patch(Rectangle((x_min, max_te_pa), x_max - x_min, top_y - max_te_pa,
                                 color='gray', alpha=0.3, label='Additional cost'))
    ax3.set_ylabel('Dimension')
    ax3.set_title(r"$\mathbf{A}$: Largest bond dimension")
    ax3.grid(True)
    ax3.legend(loc='upper left', framealpha=1)

    # 2) Gate Count
    ax4.plot(trotbond[0], trotterLengths, color='black')
    ax4.plot(paisim[0], tePAILengths, color='tab:green')
    max_te_len = max(tePAILengths)
    top_y = ax4.get_ylim()[1]
    mask = np.array(trotterLengths) > max_te_len
    if mask.any():
        x_min = np.array(trotbond[0])[mask].min()
        x_max = np.array(trotbond[0])[mask].max()
        ax4.add_patch(Rectangle((x_min, max_te_len), x_max - x_min, top_y - max_te_len,
                                 color='gray', alpha=0.3, label='_nolegend_'))
    ax4.set_ylabel('Gates')
    ax4.set_title(r"$\mathbf{B}$: Circuit gate count")
    ax4.grid(True)

    # 3) Compute Cost
    ax5.plot(trotcost[0], trotcost[1], color='black')
    ax5.plot(paicost[0], paicost[1], color='tab:green')
    max_te_cost = max(paicost[1])
    top_y = ax5.get_ylim()[1]
    mask = np.array(trotcost[1]) > max_te_cost
    if mask.any():
        x_min = np.array(trotcost[0])[mask].min()
        x_max = np.array(trotcost[0])[mask].max()
        ax5.add_patch(Rectangle((x_min, max_te_cost), x_max - x_min, top_y - max_te_cost,
                                 color='gray', alpha=0.3, label='_nolegend_'))
    ax5.set_ylabel('Cost ')
    ax5.set_title(r"$\mathbf{C}$: Computational depth")
    ax5.set_xlabel('Time')
    ax5.grid(True)

    for ax in (ax3, ax4, ax5):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel('Time')

    plt.tight_layout()
    plt.savefig("Costs")
    #plt.show()

def plotMainCalc3(folder, both=True, justLengths=False, aligned=False):
    # Load simulation and bond data
    trotsim, paisim = [[], []], [[], [], []]
    trotbond, paibond = [[], []], [[], []]
    trotcost, paicost = [[], []], [[], []]

    import csv, ast, re
    import numpy as np
    from pathlib import Path
    from matplotlib.ticker import MaxNLocator
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # <-- inset back
    import matplotlib.pyplot as plt

    def _sort_unique_xy(*cols):
        if not cols or not cols[0]:
            return [list(c) for c in cols]
        xs = np.asarray(cols[0], float)
        order = np.argsort(xs, kind="mergesort")
        arrs = [np.asarray(c)[order] for c in cols]
        _, first_idx = np.unique(arrs[0], return_index=True)
        last_idx = np.r_[first_idx[1:] - 1, arrs[0].size - 1]  # keep last per x
        return [arr[last_idx].astype(float).tolist() for arr in arrs]

    def _nearest_idx(xs, x0):
        xs = np.asarray(xs, float)
        if xs.size == 0:
            return 0
        j = np.searchsorted(xs, x0, side="left")
        if j == 0: return 0
        if j == xs.size: return xs.size - 1
        return j if abs(xs[j]-x0) < abs(xs[j-1]-x0) else j-1

    folder = Path(folder)
    for file in folder.iterdir():
        name = file.name
        if name == "lengths.csv":
            with file.open() as fp:
                next(fp)
                lengths_line = next(fp).strip()
                row = next(csv.reader([lengths_line]))
                t1 = ast.literal_eval(row[0])
                t2 = ast.literal_eval(row[1])
                trotterLengths = t2 if len(t2) >= len(t1) else t1
                row[2] = re.sub(r"np\.float64\(([^)]+)\)", r"\1", row[2])
                tePAILengths = ast.literal_eval(row[2])

        elif name.startswith("lie"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x, y = map(float, line.split(','))
                    trotsim[0].append(x); trotsim[1].append(y)

        elif name.startswith("N-") and "-dT-" in name and name.endswith(".csv"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x, y, e = map(float, line.split(','))
                    paisim[0].append(x); paisim[1].append(y); paisim[2].append(e)

        elif name.startswith("trotter-bonds"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x, y, z = map(float, line.split(','))
                    trotbond[0].append(x); trotbond[1].append(y)
                    trotcost[0].append(x); trotcost[1].append(z)

        elif name.startswith("TEPAI-bonds"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x, y, z = map(float, line.split(','))
                    paibond[0].append(x); paibond[1].append(y)
                    paicost[0].append(x); paicost[1].append(z)

    # Sort & de-dup
    trotsim  = _sort_unique_xy(*trotsim)
    paisim   = _sort_unique_xy(*paisim)
    trotbond = _sort_unique_xy(*trotbond)
    paibond  = _sort_unique_xy(*paibond)
    trotcost = _sort_unique_xy(*trotcost)
    paicost  = _sort_unique_xy(*paicost)

    # Anchors / timesteps
    te_start = paisim[0][0]
    te_end   = paisim[0][-1]
    t_arr = np.asarray(trotsim[0]); y_arr = np.asarray(trotsim[1])
    idx_bond = _nearest_idx(trotbond[0], te_start) if trotbond[0] else 0

    # cadence for inset alignment shift
    tepa_dt = (np.median(np.diff(paisim[0])) if len(paisim[0]) > 1
               else (np.median(np.diff(t_arr)) if len(t_arr) > 1 else 0.0))

    # Matchable slices
    k_trot = min(len(trotterLengths), len(trotbond[1]))
    trotterLengths_use = np.asarray(trotterLengths[:k_trot], float)
    trotbond_x_use = np.asarray(trotbond[0][:k_trot], float)
    trotbond_y_use = np.asarray(trotbond[1][:k_trot], float)

    k_pai = min(len(tePAILengths), len(paibond[1]))
    tePAILengths_use = np.asarray(tePAILengths[:k_pai], float)
    paibond_x_use = np.asarray(paibond[0][:k_pai], float)
    paibond_y_use = np.asarray(paibond[1][:k_pai], float)

    # --- Trotter continuations for B/C/D ---
    # Bond continuation: hold last value
    if len(trotbond_x_use) >= 2:
        dt_trot = np.median(np.diff(trotbond_x_use))
    else:
        dt_trot = (np.median(np.diff(paibond_x_use)) if len(paibond_x_use) > 1 else 1.0)

    t_trot_ext = np.arange((trotbond_x_use[0] if trotbond_x_use.size else te_start),
                           max(te_end, trotbond_x_use[-1] if trotbond_x_use.size else te_end) + 0.5*dt_trot,
                           dt_trot)
    trot_bond_ext = np.empty_like(t_trot_ext)
    if trotbond_y_use.size:
        nb = min(len(trotbond_y_use), len(t_trot_ext))
        trot_bond_ext[:nb] = trotbond_y_use[:nb]
        if nb < len(t_trot_ext):
            trot_bond_ext[nb:] = trotbond_y_use[nb-1]
    else:
        trot_bond_ext[:] = 0.0

    # Gate continuation: constant Δ
    if trotterLengths_use.size >= 2:
        d_len = float(trotterLengths_use[1] - trotterLengths_use[0])
    else:
        d_len = 0.0
    trot_len_ext = np.empty_like(t_trot_ext)
    n0 = min(len(trotterLengths_use), len(t_trot_ext))
    if n0:
        trot_len_ext[:n0] = trotterLengths_use[:n0]
        if n0 < len(t_trot_ext):
            last_val = trot_len_ext[n0-1]
            steps = len(t_trot_ext) - n0
            trot_len_ext[n0:] = last_val + d_len * np.arange(1, steps+1)
    else:
        trot_len_ext[:] = 0.0

    # --- PREPEND join point to TE-PAI (time, bond, gates) ---
    bond_at_te = float(trotbond[1][idx_bond]) if trotbond[1] else (paibond_y_use[0] if len(paibond_y_use) else 0.0)
    idx_len = min(idx_bond, len(trotterLengths_use)-1) if len(trotterLengths_use) else 0
    gate_at_te = float(trotterLengths_use[idx_len]) if len(trotterLengths_use) else 0.0

    # B: bonds joined
    paibond_x_join = np.r_[te_start, paibond_x_use[:k_pai]]
    paibond_y_join = np.r_[bond_at_te, paibond_y_use[:k_pai]]

    # C: gates joined (trotter_final + TE-PAI increments)
    te_len_times = np.asarray(paisim[0][-len(tePAILengths_use):], float)
    te_len_vals_offset = gate_at_te + np.asarray(tePAILengths_use, float)
    te_len_times_join  = np.r_[te_start, te_len_times[:len(tePAILengths_use)]]
    te_len_vals_join   = np.r_[gate_at_te, te_len_vals_offset[:len(tePAILengths_use)]]

    # --- Canonical costs & old-style advantage threshold for D (and time marker for A) ---
    trotcost_ext_x = t_trot_ext
    trotcost_ext_y = trot_len_ext * (trot_bond_ext ** 3)

    k_cost = min(len(paibond_x_join), len(te_len_vals_join))
    paicost_x_join = paibond_x_join[:k_cost]
    paicost_y_join = te_len_vals_join[:k_cost] * (paibond_y_join[:k_cost] ** 3)

    te_final_cost = float(paicost_y_join[-1]) if len(paicost_y_join) else 0.0
    mask_cost_adv = trotcost_ext_y > te_final_cost
    adv_time = trotcost_ext_x[np.argmax(mask_cost_adv)] if mask_cost_adv.any() else te_end

    # === Plotting ===
    plt.rcParams.update({
        'figure.dpi': 200, 'savefig.dpi': 300,
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.8,
    })

    fig = plt.figure(figsize=(6,4), constrained_layout=True)
    gs  = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0])
    fig.set_constrained_layout_pads(h_pad=0.02, w_pad=0.02, hspace=0.03, wspace=0.04)

    # ===== A: main — ONLY original trotter up to TE-PAI start + TE-PAI points =====
    ax_main = fig.add_subplot(gs[0, :])
    view_start, view_end = 0.0, te_end

    # Plot trotter ONLY up to te_start
    mask_main = (t_arr <= te_start)
    if mask_main.any():
        ax_main.plot(t_arr[mask_main], y_arr[mask_main], color='black', lw=1.0, label='Trotterization')

    # TE-PAI points
    ax_main.errorbar(paisim[0], paisim[1], yerr=paisim[2], fmt='o', ms=2.2, lw=0.9,
                     color='tab:green', label='TE-PAI')
    ax_main.axvline(view_end, linestyle='--', color='tab:green', lw=0.9)

    # Old-style advantage shading: from adv_time (threshold crossing) to end
    if mask_cost_adv.any():
        ax_main.axvline(adv_time, linestyle='--', color='black', lw=0.9)
        ax_main.axvspan(adv_time, view_end, color='tab:green', alpha=0.18, label='TE-PAI advantage')

    # --- Inset (BACK): show continued trotterization only here, shifted left one TE-PAI step ---
    mask_zoom = (t_arr >= te_start) & (t_arr <= te_end)
    if not aligned and mask_zoom.any():
        y_min_zoom = min(np.min(y_arr[mask_zoom]), np.min(paisim[1]))
        y_max_zoom = max(np.max(y_arr[mask_zoom]), np.max(paisim[1]))
        rect = Rectangle((max(te_start, view_start), y_min_zoom),
                         min(te_end, view_end) - max(te_start, view_start),
                         y_max_zoom - y_min_zoom,
                         linestyle='--', linewidth=0.8, edgecolor='black', fill=False)
        ax_main.add_patch(rect)

        inset = inset_axes(ax_main, width='52%', height='36%', loc='lower left')
        cont_x = t_arr[mask_zoom] - tepa_dt   # align start with TE-PAI
        inset.plot(cont_x, y_arr[mask_zoom], color='gray', linestyle='--', lw=0.9,
                   label='Trotterization (continued, shifted)')
        inset.errorbar(paisim[0], paisim[1], yerr=paisim[2], fmt='o', ms=1.9, lw=0.8,
                       color='tab:green', label='_nolegend_')
        inset.set_xlim(max(te_start, view_start), min(te_end, view_end))
        inset.set_xticks([]); inset.set_yticks([])
        il = inset.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), bbox_transform=inset.transAxes,
                          borderaxespad=0.0, frameon=True, framealpha=1.0, fontsize=6,
                          handlelength=1.1, handletextpad=0.4)
        il.get_frame().set_facecolor('white')

    ax_main.set_xlim(view_start, view_end)
    ax_main.set_title(r"$\mathbf{A}$: TE-PAI simulation advantage", fontsize=9, pad=3)
    ax_main.set_xlabel('Time', fontsize=8, labelpad=1)
    ax_main.set_ylabel(r'$\langle X_0 \rangle$', fontsize=8, labelpad=1)
    ax_main.tick_params(labelsize=7, pad=1)
    legend = ax_main.legend(loc='upper right', framealpha=1); legend.get_frame().set_facecolor('white')
    ax_main.grid(True, alpha=0.25)
    ax_main.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_main.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # ===== B: Max bond dimension (joined) =====
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(trotbond[0], trotbond[1], color='black', lw=1.0, label='Trotterization')
    ax3.plot(paibond_x_join, paibond_y_join, color='tab:green', lw=1.0, label='TE-PAI')
    ax3.set_ylabel('Dimension')
    ax3.grid(True, alpha=0.25)
    ax3.set_xlabel('Time')
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # ===== C: Circuit gate count (joined) =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t_trot_ext, trot_len_ext, color='black', lw=1.0, label='Trotterization')
    ax4.plot(te_len_times_join, te_len_vals_join, color='tab:green', lw=1.0, label='TE-PAI')
    ax4.set_ylabel('Gates')
    ax4.grid(True, alpha=0.25)
    ax4.set_xlabel('Time')
    ax4.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax4.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # ===== D: Computational depth (old-style gray region using TE-PAI final cost) =====
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(trotcost_ext_x, trotcost_ext_y, color='black', lw=1.0, label='Trotterization')
    ax5.plot(paicost_x_join, paicost_y_join, color='tab:green', lw=1.0, label='TE-PAI')

    # Gray region where trotter continuation is above TE-PAI final cost
    top_y = ax5.get_ylim()[1]
    if mask_cost_adv.any():
        x_min = trotcost_ext_x[mask_cost_adv].min()
        x_max = trotcost_ext_x[mask_cost_adv].max()
        ax5.add_patch(Rectangle((x_min, te_final_cost),
                                x_max - x_min, top_y - te_final_cost,
                                color='gray', alpha=0.25))

    ax5.set_ylabel('Flops')
    ax5.set_xlabel('Time')
    ax5.grid(True, alpha=0.25)
    ax5.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax5.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Diagnostics (optional)
    if len(paicost_x_join):
        trotter_at_te_end = np.interp(paicost_x_join[-1], trotcost_ext_x, trotcost_ext_y) if trotcost_ext_x.size else np.nan
        print(f"TE-PAI final cost: {te_final_cost}")
        print(f"Trotter-continued @ TE end: {trotter_at_te_end}")
        if trotter_at_te_end:
            print(f"TE-PAI / Trotter-continued @ TE end: {te_final_cost / trotter_at_te_end}")

    for a in (ax3, ax4, ax5):
        a.title.set_fontsize(8)
        a.set_xlabel('Time', fontsize=8, labelpad=1)
        a.set_ylabel(a.get_ylabel(), fontsize=8, labelpad=1)
        a.tick_params(labelsize=7, pad=1)
    fig.set_constrained_layout_pads(w_pad=0.025)
    title_kwargs = dict(fontsize=7, pad=1, loc='left', y=1.1)
    ax3.set_title(r"$\mathbf{B}$: Max bond dimension", **title_kwargs)
    ax4.set_title(r"$\mathbf{C}$: Circuit gate count", **title_kwargs)
    ax5.set_title(r"$\mathbf{D}$: Computational depth", **title_kwargs)

    fig.canvas.draw()

    # Axis-scale suffixing (unchanged behavior)
    for ax, data, base_label in (
        (ax3, paibond_y_join, 'Dimension'),
        (ax4, te_len_vals_join, 'Gates'),
        (ax5, np.r_[trotcost_ext_y, paicost_y_join], 'Flops'),
    ):
        arr = np.asarray(data)
        if arr.size == 0: continue
        maxval = np.nanmax(np.abs(arr))
        exp = int(np.floor(np.log10(maxval))) if maxval > 0 else 0
        if exp > 1:
            ticks = ax.get_yticks()
            scaled = ticks / 10**exp
            ax.set_yticklabels([f"{t:g}" for t in scaled])
            ax.set_ylabel(rf"{base_label} ($\times10^{{{exp}}}$)")
        else:
            ax.set_ylabel(base_label)

    plt.show()
    if aligned:
        plt.savefig("fullCalc_combined")
    else:
        plt.savefig("cutoffImprovement")



def plotManyCalc(folder, justLengths=False):
    param_parts = folder.split("/")[-1].split("-")
    trotterData = [[], [], [0], [0]]
    paiDatas = []
    order = {}

    params = folder.split("-")
    folder = Path(folder)
    for file in folder.iterdir():
        match file.name:
            case name if name.startswith("lie-N"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi = map(float, line.split(','))
                        trotterData[0].append(xi); trotterData[1].append(yi)

            case name if name.startswith("N"):
                Ti = float(name.split('-')[-1].replace('.csv', ''))
                order[Ti] = (len(paiDatas))
                closest_index = min(range(len(trotterData[0])), key=lambda i: abs(trotterData[0][i] - Ti))
                if closest_index != 0:
                    paiDatas.append([[], [], [], trotterData[2][closest_index+1:closest_index+2]])
                else:
                    paiDatas.append([[], [], [], [0]])

                with file.open() as fp:
                    header = next(fp)
                    numbers = header.split(',')[1:]  # Skip the first element, get the numbers
                    numbers = [float(num) for num in numbers]
                    trotterLen = numbers[0]; tepaiLen = numbers[1]
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paiDatas[-1][0].append(xi); paiDatas[-1][1].append(yi); paiDatas[-1][2].append(zi)

            case name if name.startswith("lie-bond"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi, li = map(float, line.split(','))
                        trotterData[2].append(yi)
                        trotterData[3].append(li)
                        
            case name if name.startswith("TEPAI-bonds"):
                Ti = float(name.split('-')[-1].replace('.csv', ''))
                index = order[Ti]
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paiDatas[index][3].append(yi)

    # Calculating costs
    trotterLens = [trotterLen*i for i in range(len(trotterData[0])+1)]
    tepaiLens = []
    tepaiCosts = []
    for i,dataset in enumerate(paiDatas):
        startTime = dataset[0][0]
        closest_index = min(range(len(trotterData[0])), key=lambda i: abs(trotterData[0][i] - startTime))
        if closest_index != 0:
            lengths = np.array([trotterLens[closest_index+1] + i*tepaiLen for i in range(len(dataset[0]))])
        else:
            lengths = np.array([i*tepaiLen for i in range(len(dataset[0]))])
        lengths = np.append(trotterLens[:closest_index], lengths)
        bonds = np.array(dataset[3])
        #bonds = np.append(trotterData[2][:closest_index], bonds)
        #bonds = np.insert(bonds, 0, trotterData[2][closest_index])
        tepaiLens.append(lengths)
        #bonds = np.insert(bonds,0,0)
        tepaiCosts.append(lengths[closest_index:]*bonds**3)

    # Unpack trotter
    t_times, t_vals, t_bonds, t_costs = trotterData

    if justLengths:
        data = paiDatas[0]
        cost = []
        for i,time in enumerate(t_times):
            ind = min(range(len(data[0])), key=lambda i: abs(data[0][i] - time))
            cost.append(trotterLens[i]*data[3][ind]**3) 
        cost.insert(0,0)
        t_costs = cost

    t_times = np.insert(t_times, 0,0)
    t_vals = np.insert(t_vals, 0,1)


 

    # Convert list to dict
    param_dict = {param_parts[i]: param_parts[i+1] for i in range(0, len(param_parts)-1, 2)}

    # Extract required parameters
    q = param_dict.get('q')
    delta = param_dict.get('Δ')
    N = param_dict.get('N')
    p = param_dict.get('p')

    # Format the title
    title = f"q={q} | Δ={delta} | N={N} | p={p}"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title)
    # 1. Values ± std
    ax = axes[0, 0]
    ax.plot(t_times, t_vals, label='Trotter', color="black")
    for i, ds in enumerate(paiDatas):
        p_times, p_vals, p_stds, _ = ds
        ax.errorbar(p_times, p_vals, yerr=p_stds, fmt='x', capsize=3, label=f'TEPAI from T={p_times[0]}')
    ax.set(title='Values vs Time', xlabel='Time', ylabel='Value')
    ax.legend()
    ax.grid(True)

    # 2. Bonds
    ax = axes[0, 1]
    #if t_bonds[0] != 0:
    #    t_bonds = np.insert(t_bonds, 0, 0)
    ax.plot(t_times, t_bonds, label='Trotter', color="black")
    for i, ds in enumerate(paiDatas):
        p_times, _, _, p_bonds = ds
        p_bonds = p_bonds
        diff = len(p_bonds)-len(p_times)
        ax.plot(p_times, p_bonds[len(p_bonds)-len(p_times):], marker='o', linestyle='--', label=f'TEPAI from T={p_times[0]}')
    ax.set(title='Bonds vs Time', xlabel='Time', ylabel='Bonds')
    ax.legend()
    ax.grid(True)

    # 3. Lengths
    ax = axes[1, 0]
    ax.plot(t_times, trotterLens, label='Trotter', color="black")
    for i, ds in enumerate(paiDatas):
        p_times = ds[0]
        ax.plot(p_times, tepaiLens[i][len(tepaiLens[i])-len(p_times):], label=f'TEPAI from T={p_times[0]}')
    ax.set(title='Chain Length vs Time', xlabel='Time', ylabel='Length')
    ax.legend()
    ax.grid(True)

    # 4. Costs
    ax = axes[1, 1]
    #if t_costs[0] != 0:
    #    t_costs = np.insert(t_costs, 0, 0)
    ax.plot(t_times, t_costs, label='Trotter', color="black")
    for i, ds in enumerate(paiDatas):
        p_times = ds[0]
        ax.plot(p_times, tepaiCosts[i][len(tepaiCosts[i])-len(p_times):], label=f'TEPAI from T={p_times[0]}')
    if not justLengths:
        ax.set(title='Cost calculated from individual bonds', xlabel='Time', ylabel='Cost')
    else:
        ax.set(title='Cost calculated from one shared bond', xlabel='Time', ylabel='Cost')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plotManyCalc2(folder, justLengths=False):
    param_parts = folder.split("/")[-1].split("-")
    trotterData = [[], [], [0], [0]]
    paiDatas = []
    order = {}

    params = folder.split("-")
    folder = Path(folder)
    for file in folder.iterdir():
        match file.name:
            case name if name.startswith("lie-N"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi = map(float, line.split(','))
                        trotterData[0].append(xi); trotterData[1].append(yi)

            case name if name.startswith("N"):
                Ti = float(name.split('-')[-1].replace('.csv', ''))
                order[Ti] = (len(paiDatas))
                closest_index = min(range(len(trotterData[0])), key=lambda i: abs(trotterData[0][i] - Ti))
                if closest_index != 0:
                    paiDatas.append([[], [], [], trotterData[2][closest_index+1:closest_index+2]])
                else:
                    paiDatas.append([[], [], [], [0]])

                with file.open() as fp:
                    header = next(fp)
                    numbers = header.split(',')[1:]  # Skip the first element, get the numbers
                    numbers = [float(num) for num in numbers]
                    trotterLen = numbers[0]; tepaiLen = numbers[1]
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paiDatas[-1][0].append(xi); paiDatas[-1][1].append(yi); paiDatas[-1][2].append(zi)

            case name if name.startswith("lie-bond"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi, li = map(float, line.split(','))
                        trotterData[2].append(yi)
                        trotterData[3].append(li)
                        
            case name if name.startswith("TEPAI-bonds"):
                Ti = float(name.split('-')[-1].replace('.csv', ''))
                index = order[Ti]
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paiDatas[index][3].append(yi)

    # Calculating costs
    trotterLens = [trotterLen*i for i in range(len(trotterData[0])+1)]
    tepaiLens = []
    tepaiCosts = []
    for i,dataset in enumerate(paiDatas):
        startTime = dataset[0][0]
        closest_index = min(range(len(trotterData[0])), key=lambda i: abs(trotterData[0][i] - startTime))
        if closest_index != 0:
            lengths = np.array([trotterLens[closest_index+1] + i*tepaiLen for i in range(len(dataset[0]))])
        else:
            lengths = np.array([i*tepaiLen for i in range(len(dataset[0]))])
        lengths = np.append(trotterLens[:closest_index], lengths)
        bonds = np.array(dataset[3])
        #bonds = np.append(trotterData[2][:closest_index], bonds)
        #bonds = np.insert(bonds, 0, trotterData[2][closest_index])
        tepaiLens.append(lengths)
        #bonds = np.insert(bonds,0,0)
        tepaiCosts.append(lengths[closest_index:]*bonds**3)

    # Unpack trotter
    t_times, t_vals, t_bonds, t_costs = trotterData

    if justLengths:
        data = paiDatas[0]
        cost = []
        for i,time in enumerate(t_times):
            ind = min(range(len(data[0])), key=lambda i: abs(data[0][i] - time))
            cost.append(trotterLens[i]*data[3][ind]**3) 
        cost.insert(0,0)
        t_costs = cost

    t_times = np.insert(t_times, 0,0)
    t_vals = np.insert(t_vals, 0,1)

    # Unpack trotter
    t_times, t_vals, t_bonds, t_costs = trotterData
    t_times = np.insert(t_times, 0, 0)
    t_vals = np.insert(t_vals, 0, 1)

    # Convert list to dict
    param_dict = {param_parts[i]: param_parts[i+1] for i in range(0, len(param_parts)-1, 2)}

    # Extract required parameters
    q = param_dict.get('q')
    delta = param_dict.get('Δ')
    N = param_dict.get('N')
    p = param_dict.get('p')

    # Format the title
    title = f"q={q} | Δ={delta} | N={N} | p={p}"

    # --- Begin modified plotting ---
    # Two subplots: Values and Chain Length
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # 1. Values ± std with inset zoom
    ax = axes[0]
    ax.plot(t_times, t_vals, label='Trotter', color='black')
    for ds in paiDatas:
        p_times, p_vals, p_stds, _ = ds
        p_stds = []
        denominator = int(delta.split("pi_over_")[1])
        d =  np.pi / denominator
        for time in p_times:
            p_stds.append(calcOverhead(int(q), time-p_times[0], d))
        ax.errorbar(p_times, p_vals, yerr=p_stds,
                    fmt=' ',               # no marker, no line
                    capsize=5,
                    #ecolor='black',        # optional: sets error bar color
                    elinewidth=3,        # optional: sets error bar thickness
                    label=f'TEPAI from T={p_times[0]}')
    #ax.set(title='Values vs Time', ylabel='Value')
    ax.set_ylabel(r'$\langle X_0 \rangle$')
    ax.legend(loc='upper right')
    ax.grid(True)

    # Determine zoom region: last 0.5 units in t, with padding
    tmax = t_times.max()
    region = 0.5
    pad = 0.05  # padding on x-axis before and after
    x0 = max(0, tmax - region - pad)
    x1 = tmax + pad

    # Gather all y-data in region
    mask_main = (t_times >= x0) & (t_times <= x1)
    y_vals = t_vals[mask_main]
    for ds in paiDatas:
        pts = np.array(ds[0])
        vals = np.array(ds[1])
        mask = (pts >= x0) & (pts <= x1)
        if np.any(mask):
            y_vals = np.concatenate([y_vals, vals[mask]])
    y0, y1 = y_vals.min(), y_vals.max()

    # Add padding on y-axis
    y_pad_frac = 0.1  # 10% padding
    ypad = (y1 - y0) * y_pad_frac
    y0_p = y0 - ypad
    y1_p = y1 + ypad

    # Add dashed rectangle on main axes
    rect = Rectangle((x0, y0_p), x1 - x0, y1_p - y0_p,
                     edgecolor='black', linestyle='--', fill=False)
    ax.add_patch(rect)

    # Create inset axes (lower left corner)
    axins = inset_axes(ax, width='50%', height='40%', loc='lower left', borderpad=2)
    axins.plot(t_times, t_vals, color='black')
    for ds in paiDatas:
        p_times, p_vals, p_stds, _ = ds
        axins.scatter(p_times, p_vals,
                    marker='x', s=100)
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0_p, y1_p)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.legend(title="Mean values")
    # 2. Chain Length vs Time (gate count)
    ax2 = axes[1]
    ax2.plot(t_times, trotterLens, label='Trotter', color='black')
    for i, ds in enumerate(paiDatas):
        p_times = ds[0]
        ax2.plot(p_times, tepaiLens[i][-len(p_times):], label=f'TEPAI from T={p_times[0]}')
    #ax2.set(title='Chain Length vs Time', xlabel='Time', ylabel='Length')
    ax2.set_ylabel('Circuit gate count')
    ax2.set_xlabel('Time')
    ax2.grid(True)

    for ax in (ax, ax2):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(formatter)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("manyCalcPlot", dpi=300)
    plt.show()

    return


    t_vals_quadratic = np.array(t_times)
    # --- Compute constant for Hamiltonian subplot ---
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=int(q))
    hamil = Hamiltonian.spin_chain_hamil(int(q), freqs)
    val = hamil.l1_norm(0.1)  # Evaluate 1-norm at T = 0.1
    print("H(T=0.1) L1 norm =", val)

    # --- Setup the quadratic dependence in time based on val ---
    quad_norm = []
    for t in t_times:
        quad_norm.append(hamil.l1_norm(t)**2)


    #quad_norm = (val * t_vals_quadratic) ** 2

    # --- Create figure with two subplots ---
    fig2, (ax_quad, ax_norm) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Subplot 1: Gate count comparison ---
    # Compute constant c for quadratic gate count: G(T) = c T^2
    T1 = t_times[1]
    G1 = trotterLens[1]
    c = G1 / (T1 ** 2)
    quad_gate_count = c * t_vals_quadratic ** 2

    # First TE-PAI data
    tepai_times = paiDatas[0][0]
    tepai_gates = tepaiLens[0][-len(tepai_times):]

    # Plotting
    ax_quad.plot(t_times, trotterLens, label='Trotter', color='black')
    ax_quad.plot(t_vals_quadratic, quad_gate_count, '--', color='tab:red', label='Gate count with constant error')
    ax_quad.plot(tepai_times, tepai_gates, label=f'TE-PAI from T={tepai_times[0]}', color='tab:blue')

    ax_quad.set_title("Circuit gate count")
    ax_quad.set_ylabel("Gates")
    ax_quad.set_xlabel("Time")
    ax_quad.legend()
    ax_quad.grid(True)

    # --- Subplot 2: Quadratic growth from H-norm ---
    ax_norm.plot(t_vals_quadratic, quad_norm, color='tab:green', linestyle='--', label=fr'TE-PAI at constant error')
    ax_norm.set_title("Circuit shot count")
    ax_norm.set_xlabel("Time")
    ax_norm.set_ylabel("Shots")
    ax_norm.legend()
    ax_norm.grid(True)

    # Format y axes with scientific notation if needed
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    ax_quad.yaxis.set_major_formatter(formatter)
    ax_norm.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig("constantErr")
    plt.show()

def plotTrotterVsTEPAI(folder: str, dataset_index: int = 0, color: str = None):
    """
    Plot a single Trotter line with gray overhead errorbars, and one TE-PAI scatter
    with errorbars equal to the |difference| from the Trotter values at the same times.

    Args:
        folder: path to the results directory (same structure as in plotManyCalc2)
        dataset_index: which TE-PAI dataset to plot (0 = first)
        color: optional matplotlib color for TE-PAI markers/errorbars
    """
    folder = Path(folder)
    param_parts = folder.name.split("-")
    # Parse parameters for title / overhead calc
    param_dict = {param_parts[i]: param_parts[i+1] for i in range(0, len(param_parts)-1, 2)}
    q_str   = param_dict.get('q')
    delta_s = param_dict.get('Δ', '')  # expected like 'pi_over_XX'
    N       = param_dict.get('N')
    p       = param_dict.get('p')

    # Convert Δ to a number d
    d = None
    try:
        if 'pi_over_' in delta_s:
            den = float(delta_s.split('pi_over_')[1])
            d = np.pi / den
        else:
            # fallback: treat Δ as a float (radians) if given plainly
            d = float(delta_s)
    except Exception:
        d = None  # if we can't parse, we’ll skip overhead errorbars

    # --- Load data (Trotter and TE-PAI) ---
    trotter_times, trotter_vals = [], []
    trotter_bonds, trotter_lengths = [0], [0]  # may be unused here but parsed for completeness

    paiDatas = []  # each: [times, vals, stds, bonds]
    order = {}

    trotterLen = None
    tepaiLen   = None

    for file in folder.iterdir():
        name = file.name
        if name.startswith("lie-N"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x, y = map(float, line.split(','))
                    trotter_times.append(x); trotter_vals.append(y)

        elif name.startswith("lie-bond"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x, y, z, l = map(float, line.split(','))
                    trotter_bonds.append(y)
                    trotter_lengths.append(l)

        elif name.startswith("N"):
            Ti = float(name.split('-')[-1].replace('.csv', ''))
            order[Ti] = len(paiDatas)
            with file.open() as fp:
                header = next(fp)
                numbers = [float(num) for num in header.split(',')[1:]]
                trotterLen, tepaiLen = numbers[0], numbers[1]
                # create slot
                paiDatas.append([[], [], [], [0]])  # stds unused here; bonds comes later
                for line in fp:
                    x, y, z = map(float, line.split(','))
                    paiDatas[-1][0].append(x); paiDatas[-1][1].append(y); paiDatas[-1][2].append(z)

        elif name.startswith("TEPAI-bonds"):
            Ti = float(name.split('-')[-1].replace('.csv', ''))
            idx = order[Ti]
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x, y, z = map(float, line.split(','))
                    paiDatas[idx][3].append(y)

    trotter_times = np.array(trotter_times, dtype=float)
    trotter_vals  = np.array(trotter_vals, dtype=float)

    # Insert t=0 point like in your original
    trotter_times = np.insert(trotter_times, 0, 0.0)
    trotter_vals  = np.insert(trotter_vals,  0, 1.0)

    # Grab the requested TE-PAI dataset
    if not paiDatas:
        raise ValueError("No TE-PAI datasets found in folder.")
    if dataset_index < 0 or dataset_index >= len(paiDatas):
        raise IndexError(f"dataset_index {dataset_index} out of range (found {len(paiDatas)} datasets).")

    p_times = np.array(paiDatas[dataset_index][0], dtype=float)
    p_vals  = np.array(paiDatas[dataset_index][1], dtype=float)

    # --- Compute error bars ---
    q = int(q_str) if q_str is not None else None

    # Trotter overhead errorbars via calcOverhead(q, t, d)
    if (q is not None) and (d is not None):
        trotter_yerr = np.array([calcOverhead(q, t, d) for t in trotter_times], dtype=float)
    else:
        trotter_yerr = None  # if we can’t parse inputs, skip trotter errorbars

    # TE-PAI errorbars = |TE-PAI value - Trotter value at same time|
    # Use linear interpolation of Trotter curve at TE-PAI times
    trotter_interp = np.interp(p_times, trotter_times, trotter_vals)
    tepai_yerr = np.abs(p_vals - trotter_interp)

    # --- Plot ---
    # Title
    title = f"q={q_str} | Δ={delta_s} | N={N} | p={p}"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)

    # Trotter: line with gray errorbars
    if trotter_yerr is not None:
        ax.errorbar(
            trotter_times, trotter_vals,
            yerr=trotter_yerr,
            fmt='-', color='black', ecolor='black',
            elinewidth=2, capsize=3,
            label='Trotter ± overhead'
        )
    else:
        ax.plot(trotter_times, trotter_vals, '-', color='black', label='Trotter')

    # TE-PAI: scatter with colored errorbars = deviation from Trotter
    ax.errorbar(
        p_times, p_vals,
        yerr=tepai_yerr,
        fmt='o', linestyle='none',
        capsize=2, elinewidth=2,
        color=color, ecolor=color,
        label='TE-PAI ± |Δ vs Trotter|'
    )

    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle X_0 \rangle$')
    ax.grid(True)
    ax.legend(loc='best')

    # nice scientific notation on y if needed
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    return fig, ax

def getTrotterPai(folder):
    pat_lie = re.compile(r"^lie-N-(?P<N>\d+)-T-(?P<T>[0-9.]+)-q-(?P<q>\d+)(?:-fixedCircuit)?\.csv$")
    pat_tep = re.compile(
        r"^N-(?P<N2>\d+)-n-(?P<n>\d+)-p-(?P<p>\d+)-Δ-pi_over_[^–]+"
        r"-T-(?P<T2>[0-9.]+)-q-(?P<q>\d+)-dT-(?P<dt>[0-9.]+)\.csv$"
    )
    # Match both trotter-bonds and TEPAI-bonds (type captured)
    pat_bonds = re.compile(
        r"^(?P<type>trotter|TE-?PAI)-bonds-N-(?P<N>\d+)-n-(?P<n>\d+)-T-(?P<T>\d+(?:\.\d+)?)-q-(?P<q>\d+)(?:-fixedCircuit)?\.csv$",
        re.IGNORECASE
    )
    
    # Containers for runs
    lie_runs = []
    tep_file = None
    bond_runs = []
    trotterLengths = []
    tepLengths = []

    # Scan directory for matching files
    folder_name = folder
    folder = Path(folder)
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if (m := pat_lie.match(f.name)):
            gd = m.groupdict()
            lie_runs.append((f, float(gd["T"]), int(gd["N"])))
        elif pat_tep.match(f.name):
            tep_file = f
        elif (m := pat_bonds.match(f.name)):
            gd = m.groupdict()
            bond_runs.append((f, gd))

    if tep_file is None:
        raise FileNotFoundError("No TE-PAI file found in folder")
    #if len(bond_runs) != 3:
    #    raise FileNotFoundError(f"Expected 3 bond CSV files (2 trotter + 1 TEPAI) in folder FOUND ONLY {len(bond_runs)}")

    # Load TE-PAI data
    x_t, y_t, e_t = [], [], []
    n_tep = 0
    q = int(tep_file.name.split("-")[11])
    Δ = int(tep_file.name.split("-")[7].split("_")[2])
    p = int(tep_file.name.split("-")[5])
    Δ = np.pi / Δ
    T1= float(folder_name.split("-")[10])
    N1= float(folder_name.split("-")[8])
    T2= float(folder_name.split("-")[16])
    N2= float(folder_name.split("-")[12])
    with tep_file.open() as fp:
        next(fp)
        for line in fp:
            n_tep += 1
            xi, yi, erri = line.strip().split(',')
            x_t.append(float(xi)); y_t.append(float(yi)); e_t.append(float(erri))

    dt2 = (T2 - T1) / n_tep
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)

    NNN = "NNN_" in folder_name
    if not NNN:
        hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    if NNN:
        hamil = Hamiltonian.next_nearest_neighbor_hamil(q, freqs)
    te_pai = TE_PAI(hamil, q, Δ, dt2, 1000, n_tep)
    tep_len = te_pai.expected_num_gates

    return lie_runs, bond_runs, x_t, y_t, e_t, q, N1, N2, p, T1, dt2, tep_len

def plotTrotterPAI(folder):
    lie_runs, bond_runs, x_t, y_t, e_t, q, N1, N2, p, T1, dt2, tep_len = (getTrotterPai(folder))
    # --- Subplot 1: Trotterization + TE-PAI continuation ---
    n1 = 0
    n2 = 0
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    lie_runs.sort(key=lambda t: t[1])
    (f_small, _, N_small), (f_large, _, N_large) = lie_runs
    # smaller-T trotter
    xs, ys = [], []
    with f_small.open() as fp:
        next(fp)
        for line in fp:
            xi, yi = line.strip().split(',')
            xs.append(float(xi)); ys.append(float(yi))
            n2 += 1
    ax1.plot(xs, ys, linestyle='-', color='darkblue', label=f"Trotterization-N-{N_small}")
    # larger-T trotter
    xl, yl = [], []
    with f_large.open() as fp:
        next(fp)
        for line in fp:
            xi, yi = line.strip().split(',')
            xl.append(float(xi)); yl.append(float(yi))
            n1 += 1
    ax1.plot(xl, yl, linestyle='--', color='gray', label=f"Trotterization-N-{N_large}")
    ax1.errorbar(x_t, y_t, yerr=e_t, fmt='o', linestyle='-', color='tab:blue',
                 label=f"TE-PAI continuation-N-{N2}-p-{p}")
    ax1.set_xlabel("time"); ax1.set_ylabel("x expectation value"); ax1.legend()
    
    def TrotterLen(N,q,NNN):
        if not NNN:
            return q*(1+4*N)
        else:
            return q*(1+7*N)

    NNN = "NNN_" in str(folder)
    trotter1_len = TrotterLen(int(N1), q, NNN)
    trotter2_len = TrotterLen(int(N1), q, NNN)

    # Separate trotter vs TEPAI
    trotter_runs = [br for br in bond_runs if br[1]['type'].lower().startswith('trotter')]
    tepai_runs = [br for br in bond_runs if br[1]['type'].lower().replace('-', '') == 'tepai']
    # Sort trotter by T
    trotter_runs.sort(key=lambda t: float(t[1]['T']))
    # First trotter (small)
    f_s, gd_s = trotter_runs[0];  xs2, ys2, zs2 = [], [], []
    with f_s.open() as fp:
        next(fp)
        for line in fp:
            xi, yi, zi = line.strip().split(',')
            xs2.append(float(xi)); ys2.append(float(yi)); zs2.append(float(zi))

    ys2 = np.array(ys2)
    zs2 = [trotter1_len * i * ys**3 for i, ys in enumerate(ys2)]
    ax3.plot(xs2, zs2, linestyle='-', marker='o', label=f"trotter-duration-N-{gd_s['N']}-n-{gd_s['n']}-T-{gd_s['T']}")
    ax2.plot(xs2, ys2, linestyle='-', marker='o', label=f"trotter-bonds-N-{gd_s['N']}-n-{gd_s['n']}-T-{gd_s['T']}")
    # Second trotter (large)
    f_l, gd_l = trotter_runs[1]; xl2, yl2, zl2 = [], [], []
    with f_l.open() as fp:
        next(fp)
        for line in fp:
            xi, yi, zi = line.strip().split(',')
            xl2.append(float(xi)); yl2.append(float(yi)); zl2.append(float(zi))
    yls = np.array(yl2)
    zl2 = [trotter2_len * i * yls**3 for i, yls in enumerate(yl2)]
    ax3.plot(xl2, zl2, linestyle='--', marker='o', label=f"trotter-duration-N-{gd_l['N']}-n-{gd_l['n']}-T-{gd_l['T']}")
    ax2.plot(xl2, yl2, linestyle='--', marker='o', label=f"trotter-bonds-N-{gd_l['N']}-n-{gd_l['n']}-T-{gd_l['T']}")
    # TEPAI-bonds
    if tepai_runs:
        f_t, gd_t = tepai_runs[0]; xt, yt, zt = [], [], []
        with f_t.open() as fp:
            next(fp)
            for line in fp:
                xi, yi, zi = line.strip().split(',')
                xt.append(float(xi)); yt.append(float(yi)); zt.append(float(zi))
        yt = np.array(yt)
        zt = [tep_len * i * yt**3 for i, yt in enumerate(yt)]
        ax3.plot(xt, zt, linestyle='-', marker='x', color='tab:orange', label=f"TEPAI-costs-N-{gd_t['N']}-n-{gd_t['n']}-T-{gd_t['T']}")
        ax2.plot(xt, yt, linestyle='-', marker='x', color='tab:orange', label=f"TEPAI-bonds-N-{gd_t['N']}-n-{gd_t['n']}-T-{gd_t['T']}")
    plt.show()

def plot_trotter_then_tepai(
    Δ_name: str,
    q: int,
    N1: int,
    T1: float,
    N2: int,
    p: int,
    T2: float,
    dt: float,
    base_dir: Path = Path("TE-PAI-noSampling/data/trotterThenTEPAI")
):
    """
    Locate folder q-{q}-N1-{N1}-T1-{T1}-N2-{N2}-p-{p}-T-{T2}-dt-{dt}, then:
      • If 2 lie-*.csv:
         – smaller-T: darkblue solid
         – larger-T: gray dashed
      • If 1 lie-*.csv:
         – that run: darkblue solid
      • TE-PAI: tab:blue errorbars
      • Bond data: two trotter-bonds files plus one TEPAI-bonds file plotted in second subplot
      • Skip first data point for the smaller-T trotter-bonds
      • Skip first data point for the TEPAI-bonds, and shift its x-values by the max x of the first trotter-bonds

    X axis = time, Y axis = x expectation value.
    Title = "Trotterization followed by TE-PAI for {q} qubits for total time {T1+T2}"
    """
    # Build folder path
    folder = base_dir / f"Δ-{Δ_name}-q-{q}-N1-{N1}-T1-{float(T1)}-N2-{N2}-p-{p}-T2-{float(T2)}-dt-{dt}"
    if not folder.is_dir():
        raise FileNotFoundError(f"No such folder: {folder}")

    # Compile filename patterns
    pat_lie = re.compile(r"^lie-N-(?P<N>\d+)-T-(?P<T>[0-9.]+)-q-(?P<q>\d+)(?:-fixedCircuit)?\.csv$")
    pat_tep = re.compile(
        r"^N-(?P<N2>\d+)-n-(?P<n>\d+)-p-(?P<p>\d+)-Δ-pi_over_[^–]+"
        r"-T-(?P<T2>[0-9.]+)-q-(?P<q>\d+)-dT-(?P<dt>[0-9.]+)\.csv$"
    )
    # Accept trotter/TE-PAI bonds, optional -fixedCircuit suffix
    pat_bonds = re.compile(
        r"^(?P<type>trotter|TE-?PAI)-bonds-N-(?P<N>\d+)-n-(?P<n>\d+)-T-(?P<T>\d+(?:\.\d+)?)-q-(?P<q>\d+)(?:-fixedCircuit)?\.csv$",
        re.IGNORECASE
    )

    # Containers
    lie_runs = []
    tep_file = None
    bond_runs = []

    # Scan directory
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if (m := pat_lie.match(f.name)):
            gd = m.groupdict()
            lie_runs.append((f, float(gd["T"]), int(gd["N"])))
        elif pat_tep.match(f.name):
            tep_file = f
        elif (m := pat_bonds.match(f.name)):
            gd = m.groupdict()
            bond_runs.append((f, gd))

    if tep_file is None:
        raise FileNotFoundError("No TE-PAI file found in folder")

    # Load TE-PAI time series (main)
    x_t, y_t, e_t = [], [], []
    with tep_file.open() as fp:
        next(fp)
        for line in fp:
            xi, yi, erri = line.strip().split(',')
            x_t.append(float(xi)); y_t.append(float(yi)); e_t.append(float(erri))

    # --- Subplot 1: Trotterization + TE-PAI continuation ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    if len(lie_runs) == 1:
        file_s, _, N_s = lie_runs[0]
        x_s, y_s = [], []
        with file_s.open() as fp:
            next(fp)
            for line in fp:
                xi, yi = line.strip().split(',')
                x_s.append(float(xi)); y_s.append(float(yi))
        ax1.plot(x_s, y_s, linestyle='-', color='darkblue', label=f"Trotterization-N-{N_s}")
    elif len(lie_runs) >= 2:
        lie_runs.sort(key=lambda t: t[1])
        (f_small, _, N_small), (f_large, _, N_large) = lie_runs[:2]
        # smaller-T trotter
        xs, ys = [], []
        with f_small.open() as fp:
            next(fp)
            for line in fp:
                xi, yi = line.strip().split(',')
                xs.append(float(xi)); ys.append(float(yi))
        ax1.plot(xs, ys, linestyle='-', color='darkblue', label=f"Trotterization-N-{N_small}")
        # larger-T trotter
        xl, yl = [], []
        with f_large.open() as fp:
            next(fp)
            for line in fp:
                xi, yi = line.strip().split(',')
                xl.append(float(xi)); yl.append(float(yi))
        ax1.plot(xl, yl, linestyle='--', color='gray', label=f"Trotterization-N-{N_large}")
    ax1.errorbar(x_t, y_t, yerr=e_t, fmt='o', linestyle='-', color='tab:blue',
                 label=f"TE-PAI continuation-N-{N2}-p-{p}")
    ax1.set_xlabel("time"); ax1.set_ylabel("x expectation value"); ax1.legend()

    # --- Bonds/costs subplots: plot whatever exists, preferring the MAIN TEPAI bonds (not -fixedCircuit) ---
    trotter_runs = [br for br in bond_runs if br[1]['type'].lower().startswith('trotter')]
    tepai_runs_all = [br for br in bond_runs if br[1]['type'].lower().replace('-', '') == 'tepai']

    # Prefer a TEPAI bonds file WITHOUT '-fixedCircuit' in its name; fallback to any if that's all we have
    tepai_main = None
    for fpath, gd in tepai_runs_all:
        if 'fixedCircuit' not in fpath.name:
            tepai_main = (fpath, gd); break
    if tepai_main is None and tepai_runs_all:
        tepai_main = tepai_runs_all[0]

    # Sort trotter by T and take up to two (if present)
    trotter_runs.sort(key=lambda t: float(t[1]['T']))
    trotter_first = trotter_runs[0] if len(trotter_runs) >= 1 else None
    trotter_second = trotter_runs[1] if len(trotter_runs) >= 2 else None

    # First trotter (if present)
    if trotter_first is not None:
        f_s, gd_s = trotter_first; xs2, ys2, zs2 = [], [], []
        with f_s.open() as fp:
            next(fp)
            for line in fp:
                xi, yi, zi = line.strip().split(',')
                xs2.append(float(xi)); ys2.append(float(yi)); zs2.append(float(zi))
        ax3.plot(xs2, zs2, linestyle='-', marker='o', label=f"trotter-costs-N-{gd_s['N']}-n-{gd_s['n']}-T-{gd_s['T']}")
        ax2.plot(xs2, ys2, linestyle='-', marker='o', label=f"trotter-bonds-N-{gd_s['N']}-n-{gd_s['n']}-T-{gd_s['T']}")

    # Second trotter (if present)
    if trotter_second is not None:
        f_l, gd_l = trotter_second; xl2, yl2, zl2 = [], [], []
        with f_l.open() as fp:
            next(fp)
            for line in fp:
                xi, yi, zi = line.strip().split(',')
                xl2.append(float(xi)); yl2.append(float(yi)); zl2.append(float(zi))
        ax3.plot(xl2, zl2, linestyle='--', marker='o', label=f"trotter-costs-N-{gd_l['N']}-n-{gd_l['n']}-T-{gd_l['T']}")
        ax2.plot(xl2, yl2, linestyle='--', marker='o', label=f"trotter-bonds-N-{gd_l['N']}-n-{gd_l['n']}-T-{gd_l['T']}")

    # TEPAI-bonds (MAIN, if present)
    if tepai_main is not None:
        f_t, gd_t = tepai_main; xt, yt, zt = [], [], []
        with f_t.open() as fp:
            next(fp)
            for line in fp:
                xi, yi, zi = line.strip().split(',')
                xt.append(float(xi)); yt.append(float(yi)); zt.append(float(zi))
        ax3.plot(xt, zt, linestyle='-', marker='x', color='tab:orange',
                 label=f"TEPAI-costs-N-{gd_t['N']}-n-{gd_t['n']}-T-{gd_t['T']}")
        ax2.plot(xt, yt, linestyle='-', marker='x', color='tab:orange',
                 label=f"TEPAI-bonds-N-{gd_t['N']}-n-{gd_t['n']}-T-{gd_t['T']}")

    ax2.set_xlabel("time"); ax2.set_ylabel("x expectation value");
    ax2.set_title(f"Bond data for {q} qubits")
    ax2.legend()

    fig.suptitle(f"Trotterization followed by TE-PAI for {q} qubits for total time {T1+T2}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_bond_data(folder_path="TE-PAI-noSampling/data/bonds/plot", out_file="bond_growth.png"):
    """
    Plot maximum bond dimension over time for various snapshots and Trotter parameters.
    """
    import os, re
    import pandas as pd
    import matplotlib.pyplot as plt

    pat_lie = re.compile(r"^lie-bond-N-(?P<N>[^-]+)-T-(?P<T>[^-]+)-q-(?P<q>[^.]+)\.csv$")
    pat_trot = re.compile(r"^trotter-bonds-N-(?P<N>[^-]+)-n-(?P<n>[^-]+)-(?P<T>[^-]+)-q-(?P<q>[^.]+)\.csv$")

    entries = []
    for filename in sorted(os.listdir(folder_path)):
        fp = os.path.join(folder_path, filename)
        m1 = pat_lie.match(filename)
        m2 = pat_trot.match(filename)
        if not (m1 or m2):
            continue
        df = pd.read_csv(fp)
        x = df.iloc[:, 0].tolist()
        y = df.iloc[:, 1].tolist()
        if x and x[0] != 0:
            x.insert(0, 0)
            y.insert(0, 0)
        nval = int(m1.group('q') if m1 else m2.group('n'))
        entries.append({'nval': nval, 'x': x, 'y': y})

    if not entries:
        print(f"No matching CSV files found in {folder_path}")
        return

    entries.sort(key=lambda e: e['nval'])

    # Slightly lower DPI but keep high quality
    plt.figure(figsize=(6, 4.5), dpi=200)
    cmap = plt.get_cmap('RdYlGn_r')
    num = len(entries)

    all_x = []
    max_y = 0

    for idx, entry in enumerate(entries):
        color = cmap(idx / max(1, num - 1))
        plt.plot(entry['x'], entry['y'], linestyle='-', linewidth=2,
                 color=color, label=f"n = {entry['nval']}")
        all_x.extend(entry['x'])
        max_y = max(max_y, max(entry['y']))

    # Axis labels
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Maximum Bond Dimension", fontsize=18)
    plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    plt.yticks(
    [2**k for k in range(1, int(np.log2(max_y)) + 1)],
    [rf"$2^{k}$" for k in range(1, int(np.log2(max_y)) + 1)], fontsize=16
)
    yticks = plt.yticks()[0]  # keep existing tick positions
    plt.yticks(
        yticks,
        [rf"$2^{{{int(np.log2(y))}}}$" if y >= 2**5 and y > 0 and np.log2(y).is_integer() else "" for y in yticks]
    )

    # Legend
    plt.legend(title="Number of qubits", fontsize=14, title_fontsize=16,
               loc='upper left', frameon=True, framealpha=1)

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()


def plot_gate_counts(path, n, bins=10):
    path = strip_trailing_dot_zero(path)
    data_dict = JSONtoDict(path)
    data_arrs, Ts, params, pool = DictToArr(data_dict, True)
    N, _, c, Δ, T, q = params
    q = int(q)
    N = int(N)
    T = round(float(T), 8)
    Δ = parse_pi_over(Δ)
    circuit_pool, sign_pool = data_arrs[0]

    # gather circuit lengths in blocks of size n
    circuit_lengths = [len(c) for c in circuit_pool]
    circuit_lengths = group_sum(circuit_lengths, n)
    experimental_length = np.mean(circuit_lengths)

    # theoretical expected length
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    te_pai = TE_PAI(hamil, q, Δ, T, N, n)
    theoretical_length = te_pai.expected_num_gates
    sigma = np.sqrt(theoretical_length)

    print(f"Theoretical length: {theoretical_length:.4f}, "
          f"Experimental length: {experimental_length:.4f}, "
          f"Rel. diff: {(theoretical_length - experimental_length)/theoretical_length:.4%}")

    # plot
    plt.figure(figsize=(8, 5))
    # histogram
    counts, bins_edges, _ = plt.hist(
        circuit_lengths,
        bins=bins,
        density=True,
        alpha=0.6,
        edgecolor='black',
        label="TE-PAI circuit lengths"
    )

    # normal pdf overlay
    x = np.linspace(bins_edges[0], bins_edges[-1], 1000)
    y = norm.pdf(x, loc=theoretical_length, scale=sigma)
    plt.plot(
        x,
        y,
        'r--',
        linewidth=2,
        label=fr'$\mathcal{{N}}(\nu_\infty,\,\sqrt{{\nu_\infty}})$'
    )

    # theoretical mean line
    plt.axvline(
        theoretical_length,
        color='gray',
        linestyle='dotted',
        linewidth=2,
        label=fr"$\nu_\infty = {theoretical_length:.2f}$"
    )

    plt.xlabel("Circuit length")
    plt.ylabel("Probability density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("TE-PAI-circuit-lengths.png", dpi=300)

    