from circuitSimulatorMPS import plot_data_from_folder,plotComplexityFromFolder,trotter, plot_bond_data, trotterThenTEPAI
path = "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_1024-q-10-dT-0.1-T-1"
if False:
    trotter(N=100,
        n_snapshot=10, 
        T=3, 
        q=20, 
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
#plot_bond_data()
#plot_data_from_folder("TE-PAI-noSampling/data/plotting")
#plotComplexityFromFolder(path, False)