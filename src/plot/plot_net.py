from random import sample
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Gold standard data of positive gene functional associations
# from https://www.inetbio.org/wormnet/downloadnetwork.php
# G = nx.read_edgelist("WormNet.v3.benchmark.txt")
def net_fig(txt_file, fig_path):
    G = nx.read_edgelist(txt_file, data=(("weight", float),))
    # G = nx.read_edgelist("pos_homo_pairs.txt")

    # remove randomly selected nodes (to make example fast)
    num_to_remove = int(len(G) / 1.5)
    nodes = sample(list(G.nodes), num_to_remove)
    G.remove_nodes_from(nodes)

    # remove low-degree nodes
    low_degree = [n for n, d in G.degree() if d < 3]
    G.remove_nodes_from(low_degree)
    # print(list(G.edges(data=True)))
    # exit()
    # largest connected component
    components = nx.connected_components(G)
    largest_component = max(components, key=len)
    H = G.subgraph(largest_component)

    # compute centrality
    centrality = nx.betweenness_centrality(H, k=3, endpoints=True)

    # compute community structure
    lpc = nx.community.label_propagation_communities(H)
    community_index = {n: i for i, com in enumerate(lpc) for n in com}

    # ### draw graph ####
    fig, ax = plt.subplots(figsize=(20, 15))
    pos = nx.spring_layout(H, k=0.15, seed=1025)
    node_color = [community_index[n] for n in H]
    node_size = [v * 20000 for v in centrality.values()]
    nx.draw_networkx(
        H,
        pos=pos,
        with_labels=False,
        node_color=node_color,
        node_size=node_size,
        edge_color="gainsboro",
        alpha=0.4,
    )

    # Title/legend
    font = {"color": "k", "fontweight": "bold", "fontsize": 32}
    ax.set_title("Association network", font)
    # Change font color for legend
    font_new = {"color": "r", "fontsize": 18}

    ax.text(
        0.80,
        0.10,
        "node color = community structure",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font_new,
    )
    ax.text(
        0.80,
        0.06,
        "node size = betweeness centrality",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font_new,
    )

    # Resize figure for label readibility
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    # plt.show()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def save_txt(graph_type, net_arr, prob_dict, txt_path):
    prob_arr = np.array(list(prob_dict.values()), dtype=np.float).transpose().mean(axis=1)
    # print(prob_arr)
    # print(len(prob_arr))
    # print(len(net_arr))
    with open(txt_path, 'w') as f:
        for i in range(len(net_arr)):
            if prob_arr[i] > 0.5:
                if graph_type == 'hetero':
                    f.write('A_' + str(int(net_arr[i][0])) + ' B_' + str(int(net_arr[i][1])) + ' ' +
                            str(prob_arr[i]) + '\n')
                else:
                    f.write('A_' + str(int(net_arr[i][0])) + ' A_' + str(int(net_arr[i][1])) + ' ' +
                            str(prob_arr[i]) + '\n')
