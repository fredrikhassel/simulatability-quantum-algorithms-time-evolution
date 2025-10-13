from plotting import plot_data_from_folder, plot_bond_data, plot_trotter_then_tepai, plot_gate_counts, plotTrotterPAI, plot_data_two_folders, plotMainCalc2, plotMainCalc3,plotManyCalc2, plotTrotterVsTEPAI
from calculations import trotter, parse, trotterThenTEPAI, organize_trotter_tepai, trotterComparison, mainCalc, manyCalc, fullCalc
import time
import circuitSimulatorMPS



path = "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_1024-q-50-dT-0.01-T-1"
#manyCalc(path, 3, [0, 1, 2], 3000, 12, True)

# TEPAI CIRCUIT DEPTH DECREASE PLOT
#plotManyCalc2("TE-PAI-noSampling/data/manyCalc/N-3000-p-100-Δ-pi_over_1024-T-3-q-10", justLengths=False)
#plotTrotterVsTEPAI("TE-PAI-noSampling/data/manyCalc/N-3000-p-100-Δ-pi_over_1024-T-3-q-10", 0)
#plot_data_from_folder("TE-PAI-noSampling/NNN_data/plotting")
plotTrotterPAI("TE-PAI-noSampling/NNN_data/trotterThenTEPAI/Δ-pi_over-1024-q-6-N1-100-T1-0.1-N2-100-p-100-T2-0.2-dt-0.01")
quit()

# UGLY PLOT FROM DISS NOT IN USE
#plot_data_two_folders("TE-PAI-noSampling/data/plotting/n = 100", "TE-PAI-noSampling/data/plotting/n = 20", 20, 100)

if False:
    trotter(N=1000,
        n_snapshot=50, 
        T=10, 
        q=20, 
        compare=False, 
        save=True, 
        draw=False, 
        flip=True)
    
if True:
    """
    trotterThenTEPAI(path, 
                     saveAndPlot=False,
                     trotterN=100,
                     trottern=10,
                     trotterT=2, 
                     optimize=False, 
                     flip=True,
                     confirm=True) """
    
    path = "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_4096-q-10-dT-0.1-T-1.0"
    #mainCalc2(tepaiPath=path, finalT1=3, N1=4000, n1=40, finalT2=4, confirm=True, flip=True)
    #fullCalc(tepaiPath=path, T=5, N=5000, n=10, flip=True)
    #organize_trotter_tepai()
    plotpath = f"TE-PAI-noSampling/data/trotterThenTEPAI/Δ-pi_over-8192-q-100-N1-1200-T1-4.0-N2-1000-p-100-T2-4.0-dt-0.1"
    plotMainCalc3(plotpath, justLengths=False, aligned=False)
    plotpath = f"TE-PAI-noSampling/data/fullCalc/Δ-pi_over-4096-q-10-N1-5000-T1-5.0-N2-1000-p-100-T2-5.0-dt-0.1"

    # TEPAI FROM START PLOT
    #plotMainCalc3(plotpath, justLengths=False, aligned=True)

if False:
    T = 2
    n = 10
    N = 100
    q = 10
    start=1.2e7
    tol=4e7
    tol = 4
    
    #longer(N, n, T, q, flip=True, path=path, tol = tol)
    #plotLonger(f"TE-PAI-noSampling/data/longerCalc/T-{T}-n-{n}-N-{N}-q-{q}-Δ-pi_over_4096-tol-{tol}")
    #longerCalc(0, T, n, N, q, path, start=start,tol=tol)
    plotTrotterPAI("TE-PAI-noSampling/data/trotterThenTEPAI/Δ-pi_over-1024-q-10-N1-100-T1-2.0-N2-1000-p-100-T2-3.0-dt-0.1")

if False:
    plot_trotter_then_tepai(
        Δ_name='pi_over-8192',
        q = 100,
        N1= 100,
        T1= 3,
        N2= 1000,
        p = 100,
        T2 = 4,
        dt= 0.1)
    

if False:
    trotter(N=10, 
            n_snapshot=10, 
            T=0.1, 
            q=4, 
            compare=False, 
            startTime=0, 
            save=False, 
            draw=True, 
            flip=True, 
            fixedCircuit=None)

if False:
    trotterComparison(100, 10, 6, 4)

if False:
    paths = [   "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_64-q-10-dT-0.1-T-1.0",
                "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_256-q-10-dT-0.1-T-1.0",
                "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_1024-q-10-dT-0.1-T-1",
                "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_4096-q-10-dT-0.1-T-1",
                "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_16384-q-10-dT-0.1-T-1.0",
             ]
    for p in []:
        start_time = time.time()
        parse(p, True, False, False, False, True)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Iteration took {duration:.4f} seconds")
    compareComplexity(paths)


#plot_bond_data("TE-PAI-noSampling/data/bonds/plot")
#plotComplexityFromFolder(path, False)