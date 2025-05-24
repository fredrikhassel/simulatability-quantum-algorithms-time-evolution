#from old import plot_data_from_folder,plotComplexityFromFolder,trotter, parse# plot_bond_data, trotterThenTEPAI, plot_trotter_then_tepai, plot_gate_counts,organize_trotter_tepai,draw_circuits# ,  trotterComparison,show_otimization,calc_optimization,compareComplexity,plotTrotterPAI,mainCalc,plotMainCalc,longerCalc
from circuitSimulatorMPS import plot_data_from_folder,plotComplexityFromFolder,trotter, parse, plot_bond_data, trotterThenTEPAI, plot_trotter_then_tepai, plot_gate_counts,organize_trotter_tepai,draw_circuits , trotterComparison,show_otimization,calc_optimization,compareComplexity,plotTrotterPAI,mainCalc,plotMainCalc,plot_data_two_folders,manyCalc,plotManyCalc,mainCalc2,plotMainCalc2

import time

#parse(path, True, False, True, False, True)

#calc_optimization(path)
#show_otimization(path,1)
#draw_circuits(path, variants=False)
#parse(path, True, False, False, False, True)
#organize_trotter_tepai()
#plot_gate_counts(path, n=10, bins=8)
#plot_data_from_folder("TE-PAI-noSampling/data/plotting/Nplot")
#plot_data_two_folders("TE-PAI-noSampling/data/plotting/n = 100","TE-PAI-noSampling/data/plotting/n = 4")
#plot_bond_data()

#plot_data_from_folder("TE-PAI-noSampling/data/plotting")
path = "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_1024-q-100-dT-0.1-T-1"
#manyCalc(path, 3, [0, 1, 2], 3000, 12, True)
#plotManyCalc("TE-PAI-noSampling/data/manyCalc/N-100-p-100-Δ-pi_over_1024-T-1-q-4")
#plot_data_from_folder("TE-PAI-noSampling/data/plotting")

if False:
    trotter(N=40,
        n_snapshot=40, 
        T=4, 
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
    
    mainCalc2(tepaiPath=path, finalT1=3, N1=1200, n1=40, finalT2=4, confirm=True, flip=True)
    #organize_trotter_tepai()
    plotpath = f"TE-PAI-noSampling/data/trotterThenTEPAI/Δ-pi_over-1024-q-20-N1-1200-T1-4.0-N2-1000-p-100-T2-4.0-dt-0.1"
    #plotMainCalc2(plotpath)

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


#plot_bond_data("TE-PAI-noSampling/data/trotterThenTEPAI/q-20-N1-400-T1-4.0-N2-1000-p-100-T2-6.0-dt-0.2")
#plotComplexityFromFolder(path, False)