def showLonger(folder):
    lie_runs, bond_runs, x_t, y_t, e_t, q, N1, N2, p, T1, dt2, tep_len = (getTrotterPai(folder))


def trotterUpTo(T0, T1, n_snapshot, N, q, path, start=6*1e3, tol=1*1e4, flip=True):
    times = np.linspace(T0, T0+float(T1), int(N))
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    terms = [hamil.get_term(t) for t in times]
    gates = []
    n = int(N / n_snapshot)
    dt = T1/n_snapshot
    
    for i in range(N):
        if i % n == 0:
            gates.append([])
        gates[-1] += [
            (pauli, 2 * coef * T1 / N, ind)
            for (pauli, ind, coef) in terms[i]
        ]

    bonds = []
    costs = []
    res = []
    complexities = []
    times = []
    startCircuit = None
    tmax = 0
    for i, gs in enumerate(gates):
        print(f"Snapshot {i+1} / {len(gates)}")
        circuit = getCircuit(q, flip=flip, mps=True)
        
        for k in range(i + 1):
            applyGates(circuit, gates[k])

        result = measure(circuit, q, False)
        bond, cost = getComplexity(circuit)
        costs.append(cost)
        bonds.append(bond)
        res.append(result)
        times.append((i+1)*dt)
        length = circuit.num_gates
        complexity = length*bond**3
        complexities.append(complexity)
        print(f"Length: {length}, bond: {bond}, complexity: {complexity}")
            
        if complexity < start:
            startCircuit = circuit
            tmax = (i+1)*dt

        if complexity > tol:
            #print(f"Returning circuit with complexity: {startCircuit.num_gates * bond**3}")
            break

    return times, res, bonds, complexities, startCircuit, hamil, tmax

def mes(gates,q, flip):
    quimb = qtn.CircuitMPS(q, cutoff = 1e-12)
    # quimb = qtn.circuit(q, flip)
    quimb.apply_gates(gates)
    return measure(quimb)

def stats_by_timestep(data):
    """
    Compute per-timestep mean and std over an inhomogeneous list of sequences.

    Parameters
    ----------
    data : list of sequences of numbers
        e.g. [[x0_t0, x0_t1, ...], [x1_t0, x1_t1, ...], ...]

    Returns
    -------
    means : list of float
        means[i] is the average of all data[j][i] where that exists.
    stds : list of float
        stds[i] is the population standard deviation of the same values.
    """
    if not data:
        return [], []

    max_t = max(len(seq) for seq in data)

    means = []
    stds = []
    for t in range(max_t):
        # collect all values at timestep t
        vals = [seq[t] for seq in data if len(seq) > t]
        # compute mean and std (population, ddof=0)
        arr = np.array(vals, dtype=float)
        means.append(np.mean(arr))
        stds.append(np.std(arr))
    return means, stds

def longerCalc(T0, T1, n_snapshot, N, q, path, start=6*1e3, tol=2*1e4, flip=True):
    ts, res, bs, complexities, startCircuit, hamil, tmax = trotterUpTo(T0, T1, n_snapshot, N, q, path, start, tol, flip=True)

    paramDict = parse_path(path)
    path = strip_trailing_dot_zero(path)
    data_dict = JSONtoDict(path)
    data_arrs, _, params, pool = DictToArr(data_dict, True)
    N, _, c, Δ, T, q = params
    q = int(q); N = int(N)
    T = round(float(T), 8)
    Δ = paramDict['Δ']
    dT = paramDict['dT']
    circuit_pool, sign_pool = data_arrs[0]
    #circuit_pool = circuit_pool[0:20]
    #sign_pool = sign_pool[0:20]
    results = [[]]; bonds = [[]]; costs = [[]]; times = [[]]

    quimb = startCircuit.copy() #getCircuit(q, flip=flip)
    #gate_seq = startCircuit.gates
    #quimb.apply_gates(gate_seq)
    print(f"init length: {quimb.num_gates}, bond: {getComplexity(quimb)[0]}, complexity: {quimb.num_gates*getComplexity(quimb)[0]**3}")
    for circuit, sign in zip(circuit_pool, sign_pool):
        applyGates(quimb,circuit)
        bond, _ = getComplexity(quimb)
        length = quimb.num_gates
        cost = length*bond**3
        result = mes(quimb.gates, q, flip) #measure(quimb)*sign
        print(f"cost after timestep: {cost} : versus tol {tol} : measured {result}")
        time = dT*len(results[-1])+tmax
        times[-1].append(time)
        results[-1].append(result)
        bonds[-1].append(bond)
        costs[-1].append(cost)

        if cost > tol:
            quimb=startCircuit.copy()
            results.append([]); bonds.append([]); costs.append([]); times.append([])
            counter=0

    """currentQuimb = startCircuit.copy()
    counter = 0
    for circuit, sign in zip(circuit_pool, sign_pool):
        time = dT*len(results[-1])+tmax
        print(f"Calculating for run {len(results[-1])} up to time {time}")
        print(f"currentQuimb has {currentQuimb.num_gates} gates") 
        counter += len(circuit)
        applyGates(currentQuimb,circuit)
        bond, _ = getComplexity(currentQuimb)
        cost = counter*bond**3
        result = measure(currentQuimb)*sign
        print(result)
        times[-1].append(time)
        results[-1].append(result)
        bonds[-1].append(bond)
        costs[-1].append(cost)

        print(f"length : {counter} : cost : {cost} tol : {tol}")
        if cost > tol:
            results.append([]); bonds.append([]); costs.append([]); times.append([])
            counter=0
            currentQuimb=startCircuit.copy() 

    lengths = [len(c) for c in costs]
    median_len = int(np.median(lengths))
    new_costs, new_bonds, new_results, new_times = [], [], [], []
    for c, b, r, t in zip(costs, bonds, results, times):
        if len(c) < median_len:
            # skip any run that’s too short
            continue
        # trim any run that’s too long
        new_costs.append(c[:median_len])
        new_bonds.append(b[:median_len])
        new_results.append(r[:median_len])
        new_times.append(t[:median_len])
    
    costs = new_costs
    bonds = new_bonds
    results = new_results
    times = new_times """

    bonds, _ = stats_by_timestep(bonds)
    results, stds = stats_by_timestep(results)
    times, _ = stats_by_timestep(times)

    #costs = np.mean(costs, axis=0)
    #bonds = np.mean(bonds, axis=0)
    #stds = np.std(results, axis=0)
    #results = np.mean(results, axis=0)
    #times = times[0]

     # --- create a folder named by your parameters ---
    folder_name = f"T-{T1}-n-{n_snapshot}-N-{N}-q-{q}-Δ-{Δ}-start-{start}-tol-{tol}"
    os.makedirs(folder_name, exist_ok=True)

    # --- write longerTrotter.csv with columns x, y, z, l from ts, res, bonds, complexities ---
    with open(os.path.join(folder_name, "longerTrotter.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "l"])
        for x, y, z, l in zip(ts, res, bs, complexities):
            writer.writerow([x, y, z, l])

    # --- write longerTEPAI.csv with columns x, y, z, l, m from times, results, stds, bonds, costs ---
    with open(os.path.join(folder_name, "longerTEPAI.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "l", "m"])
        for x, y, z, l, m in zip(times, results, stds, bonds, costs):
            writer.writerow([x, y, z, l, m])

    # ensure the 'longerCalc' directory exists
    os.makedirs("TE-PAI-noSampling/data/longerCalc", exist_ok=True)

    # build the destination path
    dest = os.path.join("TE-PAI-noSampling/data/longerCalc", folder_name)

    # if it already exists, wipe it out so we can overwrite
    if os.path.exists(dest):
        shutil.rmtree(dest)

    # finally move your newly created folder into longerCalc
    shutil.move(folder_name, dest)

def plotLonger(folder_path):
    """
    Reads four CSVs from folder_path:
      • longerTEPAI.csv     (x, y, z, l, m)
      • longerTrotter.csv   (x, y, z, l)
      • lie-N-100-T-0.1-q-10.csv      (x, y)
      • lie-bond-N-100-T-0.1-q-10.csv (x, y, z, l)

    Produces two stacked subplots:
      1) y vs x: TEPAI (with z errorbars), Trotter, and lie
      2) m vs x (TEPAI), l vs x (Trotter), and l vs y (lie-bond)
    """
    params = folder_path.split("-")
    T = params[3]
    N = params[7]
    q = params[9]

    # build paths
    tepai_file      = os.path.join(folder_path, 'longerTEPAI.csv')
    trotter_file    = os.path.join(folder_path, 'longerTrotter.csv')
    lie_file        = os.path.join(folder_path, f'lie-N-{N}-T-{T}-q-{q}.csv')
    lie_bond_file   = os.path.join(folder_path, f'lie-bond-N-{N}-T-{T}-q-{q}.csv')

    # load
    df_tepai      = pd.read_csv(tepai_file)
    df_trot       = pd.read_csv(trotter_file)
    df_lie        = pd.read_csv(lie_file)
    df_lie_bond   = pd.read_csv(lie_bond_file)

    # make figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # subplot 1: y vs x
    ax1.errorbar(
        df_tepai['x'], df_tepai['y'], yerr=df_tepai['z'],
        fmt='o', capsize=3, label='TEPAI extension', color="tab:green"
    )
    ax1.plot(
        df_trot['x'], df_trot['y'],
        marker='s', linestyle='--', label='Trotter extension', color="tab:red"
    )
    ax1.plot(
        df_lie['x'], df_lie['y'], linestyle='-', label='Initial trotterization', color="black"
    )
    ax1.set_ylabel('y')
    ax1.set_title('y vs x')
    ax1.legend()
    ax1.grid(True)

    # subplot 2: m vs x, l vs x, and l vs y
    ax2.plot(
        df_tepai['x'], df_tepai['m'],
        marker='o', linestyle='-', label='TEPAI extension', color="tab:green"
    )
    ax2.plot(
        df_trot['x'], df_trot['l'],
        marker='o', linestyle='--', label='Trotter extension', color="tab:red"
    )
    # plot lie-bond: l over y
    ax2.plot(
        df_lie_bond['x'], df_lie_bond['l'],
        color="black", linestyle='-', label='lie-bond (l vs y)'
    )
    ax2.set_xlabel('x  (and y for lie-bond)')
    ax2.set_ylabel('m / l')
    ax2.set_title('m & l vs x, plus l vs y for lie-bond')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def longer(N, n, T, q, flip, path, tol):
    ts, rs, cs, circ = trotter(N=N, n_snapshot=n, T=T, q=int(q), compare=False, save=True, draw=False, flip=flip)
    bs, _ = cs
    leng = circ.num_gates
    cos = leng*bs[-1]**3
    max = cos*tol
    paramDict = parse_path(path)
    dT = paramDict['dT']
    Δ = paramDict['Δ']
    print(f"Initial cost: {cos} - max cost: {max}")

    folder_name = f"T-{T}-n-{n}-N-{N}-q-{q}-Δ-{Δ}-tol-{tol}"
    os.makedirs(folder_name, exist_ok=True)
    files_to_move = [
    f"TE-PAI-noSampling/data/plotting/lie-bond-N-{N}-T-{T}-q-{q}.csv",
    f"TE-PAI-noSampling/data/plotting/lie-N-{N}-T-{T}-q-{q}.csv"
    ]
    for src in files_to_move:
        if not os.path.exists(src):
            print(f"⚠️  Source not found: {src}")
            continue
        # destination path inside your newly created folder
        dst = os.path.join(folder_name, os.path.basename(src))
        shutil.move(src, dst)
        print(f"Moved {src} → {dst}")

    path = strip_trailing_dot_zero(path)
    data_dict = JSONtoDict(path)
    data_arrs, _, _, _ = DictToArr(data_dict, True)
    circuit_pool, sign_pool = data_arrs[0]
    circuit_pool = circuit_pool[0:20]
    sign_pool = sign_pool[0:20]
    results = [[]]; bonds = [[]]; costs = [[]]; times = [[]]

    quimb = circ.copy()
    print("New circuit run: ")
    timecounter = 1
    for circuit, sign in zip(circuit_pool, sign_pool):
        time = timecounter*dT+T
        applyGates(quimb,circuit)
        bond, _ = getComplexity(quimb)
        length = quimb.num_gates
        cost = length*bond**3
        result = measure(quimb)#mes(quimb.gates, q, flip) #measure(quimb)*sign
        print(f"cost after timestep: {cost} : versus max {max} : measured {result}")
        #time = dT*len(results)+T
        times[-1].append(time)
        results[-1].append(result)
        bonds[-1].append(bond)
        costs[-1].append(cost)
        timecounter+=1
        if cost > max:
            quimb=circ.copy()
            results.append([]); bonds.append([]); costs.append([]); times.append([])
            print("New circuit run: ")
            timecounter = 1

    costs, _ = stats_by_timestep(costs)
    bonds, _ = stats_by_timestep(bonds)
    results, stds = stats_by_timestep(results)
    times, _ = stats_by_timestep(times)

    with open(os.path.join(folder_name, "longerTEPAI.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "l", "m"])
        for x, y, z, l, m in zip(times, results, stds, bonds, costs):
            writer.writerow([x, y, z, l, m])


    """
    extra = times[-1]-T
    times = np.linspace(T, T+extra, int(N))
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    terms = [hamil.get_term(t) for t in times]
    gates = []
    n = int(N / n)
    dt = extra/n
    
    for i in range(N):
        if i % n == 0:
            gates.append([])
        gates[-1] += [
            (pauli, 2 * coef * extra / N, ind)
            for (pauli, ind, coef) in terms[i]
        ]

    bonds = []
    costs = []
    res = []
    complexities = []
    times = []
    for i, gs in enumerate(gates):
        print(f"Snapshot {i+1} / {len(gates)}")
        circuit = circ.copy()
        
        for k in range(i + 1):
            applyGates(circuit, gates[k])

        result = measure(circuit, q, False)#mes(circuit.gates, q, False)
        bond, cost = getComplexity(circuit)
        costs.append(cost)
        bonds.append(bond)
        res.append(result)
        times.append((i+1)*dt+T)
        length = circuit.num_gates
        complexity = length*bond**3
        complexities.append(complexity)
        print(f"Length: {length}, bond: {bond}, complexity: {complexity}")

        if complexity > max:
            break """
    
    ts, rs, bs, cs, _, _, _ = trotterUpTo(T, times[-1], n, N, q, "", 1, max)

    with open(os.path.join(folder_name, "longerTrotter.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "l"])
        for x, y, z, l in zip(ts, rs, bs, cs):
            writer.writerow([x, y, z, l])
    
    target_root = r"TE-PAI-noSampling\data\longerCalc"
    os.makedirs(target_root, exist_ok=True)
    dest = os.path.join(target_root, folder_name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.move(folder_name, dest)
    print(f"✔️ Moved folder '{folder_name}' into '{target_root}'")
    return

    paramDict = parse_path(path)
    dT = paramDict['dT']
    Δ = paramDict['Δ']
    path = strip_trailing_dot_zero(path)
    data_dict = JSONtoDict(path)
    data_arrs, _, params, pool = DictToArr(data_dict, True)
    circuit_pool, sign_pool = data_arrs[0]
    circuit_pool = circuit_pool[0:20]
    sign_pool = sign_pool[0:20]
    results = [[]]; bonds = [[]]; costs = [[]]; times = [[]]

    quimb = circ.copy()
    print("New circuit run: ")
    for circuit, sign in zip(circuit_pool, sign_pool):
        applyGates(quimb,circuit)
        bond, _ = getComplexity(quimb)
        length = quimb.num_gates
        cost = length*bond**3
        result = mes(quimb.gates, q, flip) #measure(quimb)*sign
        print(f"cost after timestep: {cost} : versus max {max} : measured {result}")
        time = dT*len(results)+T
        times[-1].append(time)
        results[-1].append(result)
        bonds[-1].append(bond)
        costs[-1].append(cost)

        if cost > max:
            quimb=circ.copy()
            results.append([]); bonds.append([]); costs.append([]); times.append([])
            print("New circuit run: ")

    costs, stds = stats_by_timestep(costs)
    bonds, _ = stats_by_timestep(bonds)
    results, _ = stats_by_timestep(results)
    times, _ = stats_by_timestep(times)

    ts2, rs2, bs2, cs2, _, _, _ = trotterUpTo(T, times[-1]*2, 100, 100, q, "", 1, max, flip)
    
    # --- create a folder named by your parameters ---
    folder_name = f"T-{T}-n-{n}-N-{N}-q-{q}-Δ-{Δ}-tol-{tol}"
    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, "longerTEPAI.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "l", "m"])
        for x, y, z, l, m in zip(times, results, stds, bonds, costs):
            writer.writerow([x, y, z, l, m])
    with open(os.path.join(folder_name, "Trotter.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "l"])
        for x, y, z, l in zip(ts, rs, bs, cs):
            writer.writerow([x, y, z, l])
    with open(os.path.join(folder_name, "longerTrotter.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "l"])
        for x, y, z, l in zip(ts2, rs2, bs2, cs2):
            writer.writerow([x, y, z, l])
    os.makedirs("TE-PAI-noSampling/data/longerCalc", exist_ok=True)
    dest = os.path.join("TE-PAI-noSampling/data/longerCalc", folder_name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
