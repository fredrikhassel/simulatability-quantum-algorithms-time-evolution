from pathlib import Path
import numpy as np
from plotting import plot_data_from_folder, plot_bond_data, plot_trotter_then_tepai, plot_gate_counts, plotTrotterPAI, plot_data_two_folders, plotMainCalc2, plotMainCalc3,plotManyCalc2, plotTrotterVsTEPAI
from calculations import trotter, parse, trotterThenTEPAI, organize_trotter_tepai, trotterComparison, mainCalc, manyCalc, fullCalc, mainCalc2, save_lengths

def TEPAI_from_start(mode,T, N, n, skip_trotter = False,
                     tepaipath = f"TE-PAI-noSampling/NNN_data/circuits/N-100-n-1-p-100-Δ-pi_over_1024-q-10-dT-0.1-T-1.0", 
                     plotpath  = f"TE-PAI-noSampling/NNN_data/fullCalc/Δ-pi_over-1024-q-10-N1-100-T1-1.0-N2-100-p-100-T2-1.0-dt-0.1", 
                     ):
    if mode == 1:
        fullCalc(tepaipath, T=T, N=N, n=n, skip_trotter=skip_trotter)
    if mode == 2:
        plotMainCalc3(plotpath, justLengths=False, aligned=True)
    if mode == 3:
        fullCalc(tepaipath, T=T, N=N, n=n, skip_trotter=skip_trotter)
        plotMainCalc3(plotpath, justLengths=False, aligned=True)

def Trotter_then_TEPAI(mode, n=10, q=10, tepai_dT=0.1, n1=30, n2=10, N1=300, N2=100, Δ=np.pi/(2**10), NNN=True,
        tepaipath = f"TE-PAI-noSampling/NNN_data/circuits/N-100-n-1-p-100-Δ-pi_over_1024-q-10-dT-0.1-T-1.0", 
        plotpath  = f"TE-PAI-noSampling/NNN_data/trotterThenTEPAI/Δ-pi_over-4096-q-10-N1-300-T1-3.0-N2-100-p-100-T2-4.0-dt-0.1", 
):
    if mode == 1:
        mainCalc2(tepaiPath=tepaipath, finalT1=3, N1=300, n1=40, finalT2=4, confirm=True, flip=True)
    if mode == 2:
        save_lengths(n,q,tepai_dT,n1,n2,N1,N2,Δ,NNN,base_dir=plotpath)
        plotMainCalc3(plotpath, justLengths=False, aligned=False)
    if mode == 3:
        mainCalc2(tepaiPath=tepaipath, finalT1=3, N1=300, n1=40, finalT2=4, confirm=True, flip=True)
        save_lengths(n,q,tepai_dT,n1,n2,N1,N2,Δ,NNN,base_dir=plotpath)
        plotMainCalc3(plotpath, justLengths=False, aligned=False)
        
if __name__ == "__main__":
    Trotter_then_TEPAI()

