import math
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
import hypernetx

def draw_tensor(rank, draw=False, color=None):
    """
    Create and optionally draw a tensor of the specified rank with a specified color.

    Parameters:
    rank (int): The rank of the tensor to create (0 for scalar, 1 for vector, etc.).
    draw (bool): Whether to draw the tensor. Default is False.
    color (str or list of str, optional): Tag(s) to determine the color of the tensor when drawn.
    """
    if rank == 0:
        # Scalar (Rank-0 Tensor)
        tensor = qtn.Tensor(data=3.14, tags={'Scalar'})
        print("Created a scalar (rank-0 tensor).")
    elif rank == 1:
        # Vector (Rank-1 Tensor)
        tensor = qtn.Tensor(data=np.array([1, 2, 3]), inds=('i',), tags={color})
        print("Created a vector (rank-1 tensor).")
    elif rank == 2:
        # Matrix (Rank-2 Tensor)
        tensor = qtn.Tensor(data=np.array([[1, 2], [3, 4]]), inds=('i', 'j'), tags={color})
        print("Created a matrix (rank-2 tensor).")
    elif rank == 3:
        # Rank-3 Tensor
        tensor = qtn.Tensor(data=np.random.rand(2, 2, 2), inds=('i', 'j', 'k'), tags={color})
        print("Created a rank-3 tensor.")
    elif rank == 4:
        # Rank-4 Tensor
        tensor = qtn.Tensor(data=np.random.rand(2, 2, 2, 2), inds=('i', 'j', 'k', 'l'), tags={color})
        print("Created a rank-4 tensor.")
    else:
        raise ValueError("Unsupported tensor rank. Please choose 0, 1, 2, 3, or 4.")

    if draw:
        # Draw the tensor network with the specified color
        tensor.draw(color=color)
        plt.show()
    
    return tensor

def draw_grid(Lx,Ly,Lz=None,D=2,dim=3,draw=False, color=True):
    if dim == 3:
        tn = qtn.TN3D_rand(Lx, Ly, Lz, D=D)
        # color that tag and each corner of our TN
        if color:
            color = ['CUBE'] + [
                f'I{i},{j},{k}'
                for i in (0, Lx - 1)
                for j in (0, Ly - 1)
                for k in (0, Lz - 1)
                ]   
    if dim == 2:
        tn = qtn.TN2D_rand(Lx, Ly, D=D)
        if color:
            # color that tag and each corner of our TN
            color = ['CUBE'] + [
                f'I{i},{j}'
                for i in (0, Lx - 1)
                for j in (0, Ly - 1)
                ]
    # add the same tag to every tensor
    if draw:
        if color:
            tn.add_tag('CUBE')
            tn.draw(color)
        else:
            print("ding")
            tag = f"I{Lx // 2},{Ly // 2},{Lz // 2}"
            t = tn[tag]
            inds = t.inds
            tn.draw(color=tag, highlight_inds=inds, edge_scale=2)
        plt.show()
    return tn

def get_3d_pos(i, j, k, a=22, b=45, p=0.2):
    return (
        + i * math.cos(math.pi * a / 180) + j * math.cos(math.pi * b / 180) / 2**p,
        - i * math.sin(math.pi * a / 180) + j * math.sin(math.pi * b / 180) / 2**p + k
    )

def draw_manual(Lx=4,Ly=4,Lz=4,D=2):
    tn = qtn.TN3D_rand(Lx, Ly, Lz, D=D)
    pos = {
    f'I{i},{j},{k}': get_3d_pos(i, j, k)
    for i in range(Lx)
    for j in range(Ly)
    for k in range(Lz)
}
    tn.draw(fix=pos, color=pos.keys())

def show_tensors():
    scalar = draw_tensor(rank=0, draw=False, color='Scalar')
    vector = draw_tensor(rank=1, draw=False, color='Vector')
    matrix = draw_tensor(rank=2, draw=False, color='Matrix')
    rank3 = draw_tensor(rank=3, draw=False, color='Rank 3 tensor')
    titles = ['Scalar', 'Vector', 'Matrix', 'Rank 3 tensor']
    node_size_dict = {
        'Scalar': 0.1,
        'Vector': 0.7,
        'Matrix': 1,
        'Rank-3': 1.2,
    }

    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs = axs.flatten()
    for i,t in enumerate([scalar, vector, matrix, rank3]):
        t.draw(ax=axs[i], 
        legend=False,
         node_size=node_size_dict, 
         edge_scale =2, 
         title = titles[i], 
         #color=titles, 
         show_tags=False,
         show_inds = False,
         font_size = 80,
         node_color='tab:green')
    plt.show()

def show_tensors2():
    scalar = draw_tensor(rank=4, draw=False, color='Scalar')
    vector = draw_tensor(rank=4, draw=False, color='Vector')
    matrix = draw_tensor(rank=4, draw=False, color='Matrix')
    rank3 = draw_tensor(rank=4, draw=False, color='Rank-3')
    titles = ['Scalar', 'Vector', 'Matrix', 'Rank-3']

    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    for i,t in enumerate([scalar, vector, matrix, rank3]):
        t.draw(ax=axs[i], 
        legend=False,
         edge_scale =2, 
         title = titles[i], 
         color=titles, 
         show_tags=False,
         edge_color=(0,0,0,1),
         show_inds = False)
    plt.show()

def show_contraction():
    A = qtn.rand_tensor((2,2), [r'μ',r'γ'], tags='B')
    B = qtn.rand_tensor((2,2), [r'γ',r'ν'], tags='C')
    tn = qtn.TensorNetwork([A,B])
    tn.draw(show_inds='all', font_size=20, font_size_inner=20, node_color='tab:green', figsize = (5,5))
    plt.show()

def show_scalarproduct():
    # Define two vectors (tensors) with a common index 'i'
    ket = qtn.Tensor(data=np.array([1, 2, 3]), inds=('α',), tags={'J'})
    bra = qtn.Tensor(data=np.array([4, 5, 6]), inds=('α',), tags={'K'})
    # Create a tensor network from these tensors
    tn = qtn.TensorNetwork([ket, bra])
    # Draw the tensor network to visualize the contraction (both tensors share index 'i')
    tn.draw(show_inds='all', font_size=20, font_size_inner=20, node_color='tab:green', figsize = (5,5))
    plt.show()

def show_complexcontraction():
    E = qtn.rand_tensor((2,2,2,2) , inds=[r'α',r'β',r'γ',r'σ'], tags='E')
    F = qtn.rand_tensor((2,2,2) , inds=[r'β',r'μ',r'ε'], tags='F')
    G = qtn.rand_tensor((2,2,2,2) , inds=[r'γ',r'δ',r'ε', r'ν'], tags='G')
    H = qtn.rand_tensor((2,2,2) , inds=[r'δ',r'ρ',r'α'], tags='H')
    tn = qtn.TensorNetwork([E,F,G,H])
    tn.draw(show_inds='all', font_size=20, font_size_inner=20, node_color='tab:green', figsize = (5,5), layout='kamada_kawai')
    plt.show()

def show_complexSP():
    E = qtn.rand_tensor((2,2,2,2) , inds=[r'α',r'β',r'δ',r'γ'], tags='M')
    F = qtn.rand_tensor((2,2,2) , inds=[r'β',r'γ',r'μ'], tags='N')
    G = qtn.rand_tensor((2,2,2,2) , inds=[r'δ',r'ν',r'μ', r'ω'], tags='O')
    H = qtn.rand_tensor((2,2,2) , inds=[r'ν',r'ω',r'α'], tags='P')
    tn = qtn.TensorNetwork([E,F,G,H])
    tn.draw(show_inds='all', font_size=20, font_size_inner=20, node_color='tab:green', figsize = (5,5), layout='kamada_kawai')
    plt.show()

def show_three():
    A = qtn.rand_tensor((2,2,2), [r'μ',r'α', r'β'], tags='A')
    B = qtn.rand_tensor((2,2,2), [r'ν',r'β', r'γ'], tags='B')
    C = qtn.rand_tensor((2,2,2), [r'ρ',r'γ', r'α'], tags='C')
    node_colors = {0: 'tab:green', 1: 'tab:green', 2: 'tab:green'}
    edge_colors = {'μ':'lightgray','ν':'lightgray','ρ':'lightgray', 'α': 'tab:blue', 'β': 'tab:orange', 'γ': 'tab:cyan'}
    tn = qtn.TensorNetwork([A,B,C])
    H = hypernetx.Hypergraph(tn.ind_map)
    fig, axs = plt.subplots(1, 3)
    hypernetx.draw(H, 
                   pos=tn.draw(get='pos'), 
                   node_radius=6, ax=axs[1], 
                   fill_edges=True,
                   fill_edge_alpha=-0.9, 
                   edge_label_alpha = 0,
                   with_node_labels=False,
                   edges_kwargs={'edgecolors': [edge_colors[edge] for edge in H.edges]},
                   nodes_kwargs={'facecolors': [node_colors[node] for node in H.nodes],
                                 'edgecolors': 'green'},
                    edge_labels_kwargs={'fontsize' : 12, 'color' : 'black', 'weight': 'bold'},
                   )
    tn.draw(show_inds='all', 
            font_size=20, 
            font_size_inner=20, 
            node_color='tab:green', 
            figsize = (5,5), 
            ax=axs[0])
    
    D = qtn.rand_tensor((2,2,2,2), [r'μ', r'ν',r'λ', r'δ'], tags='D')
    C2 = qtn.rand_tensor((2,2,2), [r'ρ',r'δ', r'λ'], tags='C')
    tn2 = qtn.TensorNetwork([C2,D])
    fix = {'D': (0, 0), 'C': (1, 0)}   
    tn2.draw(show_inds='all', fix=fix,
            font_size=20, 
            font_size_inner=20, 
            node_color='tab:green', 
            figsize = (5,5), 
            ax=axs[2])

    axs[1].annotate('A', (0.215, 0.32), fontsize=12, ha='center')
    axs[1].annotate('B', (-0.16, -0.30), fontsize=12, ha='center')
    axs[1].annotate('C', (0.56, -0.31), fontsize=12, ha='center')
    axs[0].text(-0.1, 1, 'A)', fontsize=14, fontweight='bold', transform=axs[0].transAxes)
    axs[1].text(-0.1, 0.75, 'B)', fontsize=14, fontweight='bold', transform=axs[1].transAxes)
    axs[2].text(-0.2, 1.43, 'C)', fontsize=14, fontweight='bold', transform=axs[2].transAxes)
    plt.show()

if __name__ == "__main__":
    #draw_tensor(rank=2, draw=True, color='matrix')
    show_tensors()
    #draw_grid(4,4,4,2,3, draw=True, color=True)
    #draw_manual()
    #show_contraction()
    #show_complexcontraction()
    #show_scalarproduct()
    #show_complexSP()
    #show_three()