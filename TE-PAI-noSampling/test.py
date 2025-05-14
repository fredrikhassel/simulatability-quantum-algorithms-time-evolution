from circuitSimulatorMPS import plot_data_from_folder,plotComplexityFromFolder,trotter, plot_bond_data, trotterThenTEPAI, plot_trotter_then_tepai, plot_gate_counts,organize_trotter_tepai, parse,draw_circuits, trotterComparison,show_otimization,calc_optimization
path = "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_256-q-10-dT-0.1-T-1"
#calc_optimization(path)
#show_otimization(path,1)
#draw_circuits(path, variants=False)
#parse(path, True, False, False, False, True)
#organize_trotter_tepai()
#plot_gate_counts(path, n=10, bins=8)
#plot_bond_data()

if False:
    trotter(N=100,
        n_snapshot=10, 
        T=10, 
        q=30, 
        compare=False, 
        save=True, 
        draw=False, 
        flip=True)
if True:
    trotterThenTEPAI(path, 
                     saveAndPlot=False,
                     trotterN=100,
                     trottern=10,
                     trotterT=2, 
                     optimize=False, 
                     flip=True,
                     confirm=True)

if True:
    plot_trotter_then_tepai(
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


#plot_bond_data("TE-PAI-noSampling/data/trotterThenTEPAI/q-20-N1-400-T1-4.0-N2-1000-p-100-T2-6.0-dt-0.2")
#plot_data_from_folder("TE-PAI-noSampling/data/plotting")
#plotComplexityFromFolder(path, False)