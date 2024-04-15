from topology.CommunicationRequest import CommunicationRequest
from topology.SplayNetwork import SplayNetwork
from topology.Node import Node
from algorithms import splaynet
import pandas as pd


def csv_to_sequence(path):
    data = pd.read_csv(path, usecols=['src', 'dst'])
    communication_sequence = [CommunicationRequest(row['src'], row['dst']) for _, row in data.iterrows()]
    all_nodes = pd.concat([data['src'], data['dst']]).unique()
    # nodes = [Node(i) for i in all_nodes] # dieser Schritt erstellt alle Node Instanzen
    return communication_sequence, all_nodes


if __name__ == '__main__':
    filepath = "data/sample.csv"
    sigma, nodes = csv_to_sequence(filepath)
    # print(nodes)
    id_counter = 0
    t = SplayNetwork()

    t.build_network(nodes)
    t.print_tree(t.root)


# if __name__ == '__main__':
#     import pandas as pd
#     import numpy as np
#     from fractions import Fraction
#
#     # CSV-Datei einlesen
#     df = pd.read_csv('data/sample.csv')
#
#     # Eindeutige Knoten extrahieren, um die Größe der Matrix zu bestimmen
#     unique_nodes = pd.unique(df[['src', 'dst']].values.ravel('K'))
#     print("Unique Nodes", unique_nodes)
#     node_index = {node: idx for idx, node in enumerate(unique_nodes)}
#
#     # Matrix initialisieren
#     matrix_size = len(unique_nodes)
#     communication_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
#
#     # Durch die Kommunikationssequenzen iterieren und die Matrix aktualisieren
#     for _, row in df.iterrows():
#         src, dest = row['src'], row['dst']
#         src_idx, dest_idx = node_index[src], node_index[dest]
#         communication_matrix[src_idx, dest_idx] += 1
#
#     # Absolutwerte in Frequenzen umwandeln und als Brüche darstellen
#     total_communications = np.sum(communication_matrix)
#     frequency_matrix = np.empty(communication_matrix.shape, dtype=object)  # dtype=object für Bruchdarstellung
#
#     for i in range(matrix_size):
#         for j in range(matrix_size):
#             frequency_matrix[i, j] = Fraction(communication_matrix[i, j], total_communications).limit_denominator()
#
#     # Ergebnis ausgeben
#     print("Kommunikationsmatrix:")
#     print(communication_matrix)
#     print("Frequenzmatrix als Brüche:")
#     for row in frequency_matrix:
#         print([str(frac) for frac in row])
