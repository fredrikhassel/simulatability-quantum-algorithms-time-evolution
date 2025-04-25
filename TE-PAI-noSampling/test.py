from circuitSimulatorMPS import plot_data_from_folder,plotComplexityFromFolder

#plot_data_from_folder("TE-PAI-noSampling/data/plotting")
path = "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_1024-q-100-dT-0.1-T-1.0"
plotComplexityFromFolder(path, False)

