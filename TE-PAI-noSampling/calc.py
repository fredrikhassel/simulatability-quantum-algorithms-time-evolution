from circuitSimulatorMPS import plot_data_from_folder,plotComplexityFromFolder,trotter, parse, plot_bond_data, trotterThenTEPAI, plot_trotter_then_tepai, plot_gate_counts,organize_trotter_tepai,draw_circuits , trotterComparison,show_otimization,calc_optimization,compareComplexity,plotTrotterPAI,mainCalc,plotMainCalc,plot_data_two_folders,manyCalc,plotManyCalc,mainCalc2,plotMainCalc2,plotMainCalc3, plotManyCalc2, fullCalc, showComplexity


path = ""
costs = parse(
    folder=path,
    isJSON=True,
    draw=False,
    saveAndPlot=False,
    optimize=False,
    flip=True
    )
showComplexity(costs, 1, len(costs), path)

trotter(
    N=1000,
    n_snapshot=10, 
    T=10, 
    q=20, 
    compare=False, 
    save=True, 
    draw=False, 
    flip=True)

