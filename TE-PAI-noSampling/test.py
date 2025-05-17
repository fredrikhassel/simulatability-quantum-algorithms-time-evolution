#from old import plot_data_from_folder,plotComplexityFromFolder,trotter, parse# plot_bond_data, trotterThenTEPAI, plot_trotter_then_tepai, plot_gate_counts,organize_trotter_tepai,draw_circuits# ,  trotterComparison,show_otimization,calc_optimization,compareComplexity,plotTrotterPAI,mainCalc,plotMainCalc,longerCalc
from circuitSimulatorMPS import plot_data_from_folder,plotComplexityFromFolder,trotter, parse, plot_bond_data, trotterThenTEPAI, plot_trotter_then_tepai, plot_gate_counts,organize_trotter_tepai,draw_circuits , trotterComparison,show_otimization,calc_optimization,compareComplexity,plotTrotterPAI,mainCalc,plotMainCalc

import time

path = "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_4096-q-10-dT-0.1-T-1"
#parse(path, True, False, True, False, True)
#calc_optimization(path)
#show_otimization(path,1)
#draw_circuits(path, variants=False)
#parse(path, True, False, False, False, True)
#organize_trotter_tepai()
#plot_gate_counts(path, n=10, bins=8)
#plot_bond_data()

if False:
    trotter(N=1000,
        n_snapshot=30, 
        T=3, 
        q=10, 
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
    path = "TE-PAI-noSampling/data/trotterThenTEPAI/Δ-pi_over-4096-q-10-N1-100-T1-2.0-N2-1000-p-100-T2-3.0-dt-0.1"
    #mainCalc(path, 2, 100, 10, 100, 1, True, True)
    plotMainCalc(path)

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
        Δ_name='pi_over-1024',
        q = 10,
        N1= 100,
        T1= 2,
        N2= 1000,
        p = 100,
        T2 = 3,
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


#plot_bond_data("TE-PAI-noSampling/data/trotterThenTEPAI/q-20-N1-400-T1-4.0-N2-1000-p-100-T2-6.0-dt-0.2")
#plot_data_from_folder("TE-PAI-noSampling/data/plotting")
#plotComplexityFromFolder(path, False)