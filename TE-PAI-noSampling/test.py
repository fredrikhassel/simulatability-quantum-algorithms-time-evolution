from circuitSimulatorMPS import plot_data_from_folder,plotComplexityFromFolder,trotter, plot_bond_data, trotterThenTEPAI, plot_trotter_then_tepai, plot_gate_counts,organize_trotter_tepai, parse,draw_circuits
path = "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_1024-q-10-dT-0.1-T-1"
draw_circuits(path, 10)
#parse(path, True, True, False, False, True)
#organize_trotter_tepai()
#plot_gate_counts(path, 10, 10)
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
        q = 6,
        N1= 400,
        T1= 4,
        N2= 100,
        p = 100,
        T2 = 6,
        dt= 0.2)

#plot_bond_data("TE-PAI-noSampling/data/trotterThenTEPAI/q-6-N1-400-T1-4.0-N2-100-p-100-T2-6.0-dt-0.2")
#plot_data_from_folder("TE-PAI-noSampling/data/plotting")
#plotComplexityFromFolder(path, False)