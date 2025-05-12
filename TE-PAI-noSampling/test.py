from circuitSimulatorMPS import plot_data_from_folder,plotComplexityFromFolder,trotter, plot_bond_data, trotterThenTEPAI, plot_trotter_then_tepai, plot_gate_counts,organize_trotter_tepai, parse,draw_circuits, trotterComparison
path = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-300-Δ-pi_over_1024-q-4-dT-0.1-T-1.0"
#draw_circuits(path, 10)
parse(path, True, False, False, False, True)
#organize_trotter_tepai()
#plot_gate_counts(path, n=10, bins=20)
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
if False:
    trotterThenTEPAI(path, 
                     saveAndPlot=False,
                     trotterN=100,
                     trottern=10,
                     trotterT=2, 
                     optimize=False, 
                     flip=True,
                     confirm=True)

if False:
    plot_trotter_then_tepai(
        q = 20,
        N1= 400,
        T1= 4,
        N2= 1000,
        p = 100,
        T2 = 6,
        dt= 0.2)
    
if False:
    trotter(N=100, 
            n_snapshot=100, 
            T=0.01, 
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
plot_data_from_folder("TE-PAI-noSampling/data/plotting")
#plotComplexityFromFolder(path, False)