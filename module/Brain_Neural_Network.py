from collections import deque
import random
import math
import numpy as np
import time


"""Das Ganze hier funktioniert in diesem rahmen auch nur, weil die aktivierungsfunktionen zentriert sind, glaube ich"""

# Parameter für die Main Datei
ADD_NEURON = 0.3
ADD_WEIGHT = 0.7
CHANGE_WEIGHT = 0.8

BIAS = 0.5
REAC = False

def Sigmoid(x):
    a = 3
    b = 0
    return 1 / (1 + math.exp(-x * a - b))

def ReAc(x):
    return max(0, x)

def actFunc(x):
    return math.tanh(x)

def actSpeed(x):
    return max(0, math.tanh(x))
def actAngle(x):
    return math.tanh(x)

class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs, mutate=2, bias=BIAS):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.bias = bias
        self.neuron_number = self.num_inputs + 1 + self. num_outputs  # Nächste ID für neue Neuronen
        # Eingabe- und Ausgabeneuronen initialisieren
        self.reihnfolge = []
        
        #Initialisiere Input Matrix
        self.Input_Matrix = np.zeros((num_outputs, self.num_inputs + 1)) #no connection exists
        self.Hidden_Matrix = np.zeros((num_outputs, num_outputs)) #also here, no connection exists
        
        for _ in range(mutate):
            self.mutate()
         
    def add_neuron(self):
        """Fügt ein neues Neuron hinzu."""
            #die Matritzen werden aktualisiert
        new_row = np.zeros((1, self.num_inputs + 1))
        self.Input_Matrix = np.vstack((self.Input_Matrix, new_row)) #fügt eine neue null zeile hinzu
            
        new_row = np.zeros((1, self.Hidden_Matrix.shape[0]))
        self.Hidden_Matrix = np.vstack((self.Hidden_Matrix, new_row)) #zeilen dim += 1
            
        new_row = np.zeros((self.Hidden_Matrix.shape[0],1))
        self.Hidden_Matrix = np.hstack((self.Hidden_Matrix, new_row))
        self.neuron_number += 1

    def add_connection(self, source_id, target_id, weight=None):
        """Fügt eine Verbindung zwischen zwei Neuronen hinzu."""
        #source in input, bias, hidden
        #target in output, hidden
        if weight is None:
            weight = random.uniform(-1, 1)
        #if source_id in self.connections:
        if source_id < self.num_inputs+1:
            self.Input_Matrix[target_id-(self.num_inputs+1), source_id] = weight 
        else:
            self.Hidden_Matrix[target_id - (self.num_inputs+1), source_id - (self.num_inputs+1)] = weight

    def forward_vectorized(self, inputs, energy):
        """Führt die Vorwärtspropagation durch."""
        
        # Initialisiere Aktivierungen (inkl. Bias)
        activations = np.zeros(self.num_inputs+1)
        for idx in range(self.num_inputs):
            activations[idx] = inputs[idx]
        activations[-1] = (self.bias * energy)
        
        #first input activation
        activations = np.dot(self.Input_Matrix, activations) 
        
        #checkProgress = np.zeros(ot_hi)
        
        for order in self.reihnfolge:
            activations[order] = actFunc(activations[order] + (self.Hidden_Matrix[order] @ activations))
            
        # Store for visualization
        self.last_inputs = inputs
        self.last_activations = activations.copy() # activations contains hidden + output neurons

        # Rückgabe der Ausgabeneuronen
        return [actSpeed(activations[0] + self.Hidden_Matrix[0] @ activations), actAngle(activations[1] + self.Hidden_Matrix[1] @ activations)]
        
    def mutate(self):
        """Mutiert das Netzwerk durch Hinzufügen von Verbindungen oder Neuronen."""
        in_bi = self.num_inputs + 1
        if random.random() < ADD_NEURON:
            # Neues Neuron hinzufügen
            self.add_neuron()
        
        if random.random() < ADD_WEIGHT:
            # Neue Verbindung hinzufügen
            source = random.choice( (list(range(in_bi)) + list(range(in_bi + self.num_outputs, self.neuron_number))) ) #source in {input, bias, hidden}
            target = random.choice( list(range(in_bi, self.neuron_number)) ) #target in output, hidden
            
            """Diese Schleife lässt sich noch schöner machen!!!!!!"""
            single_conectn = True
            if source < in_bi: #wenn input oder bias
                if self.Input_Matrix[target - in_bi, source] != 0:
                    single_conectn = False
            elif self.Hidden_Matrix[target - in_bi, source - in_bi] != 0:
                single_conectn = False
            
            if single_conectn:
                if not self.infinityLoop(source, target):
                    self.add_connection(source, target)
        
        if random.random() < CHANGE_WEIGHT:
            # Gewicht einer bestehenden Verbindung ändern
            #fehler, falls leeres netzwerk, ohne input, ohne output
            #4 versuche, um eine source mit weight zu finden
            for _ in range(4):
                source = random.choice( (list(range(in_bi)) + list(range(in_bi+self.num_outputs,self.neuron_number))) ) #source in {input, bias, hidden}
                target = random.choice( list(range(in_bi,self.neuron_number)) ) #target in output, hidden
                if source < in_bi: #input matrix wird verändert
                    if self.Input_Matrix[target - in_bi, source] != 0: 
                        self.Input_Matrix[target - in_bi, source] += random.choice([-0.1, 0.1])
                        break
                elif self.Hidden_Matrix[target - in_bi, source - in_bi] != 0:
                    self.Hidden_Matrix[target - in_bi, source - in_bi] += random.choice([-0.1, 0.1])
                    break
                
    def infinityLoop(self, source, target):
        #checks if source and target are hidden layer
        in_bi = self.num_inputs + 1
        in_bi_out = in_bi + self.num_outputs #length: input+hidden+out
        if ((in_bi_out-1) < source) and ((in_bi_out-1) < target): #->we are in the hidden matrix 
            ot_hi = self.neuron_number - in_bi
            
            self.Hidden_Matrix[target - in_bi, source - in_bi] = 1 #dummy wert einfügen,
            
            sampleReihnfolge = []        
            #ich will nur die hiddens in die reihnfolge einpacken
            indexListe = list(range(self.num_outputs, ot_hi))
            #überprüfe erstmal alle nullzeilen
            for i in range(self.num_outputs, ot_hi):
                if np.all(self.Hidden_Matrix[i] == 0):
                    indexListe.remove(i)
                    sampleReihnfolge.append(i)
                    
            while indexListe:
                a = len(indexListe) 
                for i in indexListe:
                    if np.all(self.Hidden_Matrix[i, indexListe] == 0): 
                        indexListe.remove(i)
                        sampleReihnfolge.append(i)
                        #break
                if a == len(indexListe):
                    self.Hidden_Matrix[target - in_bi, source - in_bi] = 0 #ich muss hier eigentlch die länge von input_bias abziehen
                    return True #there is an infinity Loop with this connection
            self.reihnfolge = sampleReihnfolge #sollte länge ot_hi haben
            return False
        else: 
            return False


import networkx as nx
import matplotlib.pyplot as plt

def plot_neural_network(neural_network):
    """
    Stellt ein neuronales Netzwerk mit Networkx dar.

    :param neural_network: Ein Objekt, das folgende Attribute enthält:
                           - input_matrix: Matrix der Dimension (output+hidden) x (inputs+1 bias),
                             wobei Einträge != 0 Verbindungen von j nach i darstellen.
                           - hidden_matrix: Matrix der Dimension (output+hidden)^2, wobei Einträge != 0
                             Verbindungen von j nach i darstellen.
                           - num_inputs: Anzahl der Eingabeneuronen.
                           - num_outputs: Anzahl der Ausgabeneuronen.
                           - total_neurons: Gesamtanzahl der Neuronen (inputs, bias, outputs, hidden).
    """
    input_matrix = neural_network.Input_Matrix
    hidden_matrix = neural_network.Hidden_Matrix
    num_inputs = neural_network.num_inputs
    num_outputs = neural_network.num_outputs
    total_neurons = neural_network.neuron_number

    # Anzahl der Hidden-Neuronen berechnen
    num_hidden = total_neurons - (num_inputs + 1 + num_outputs)  # -1 für Bias

    # Knoten definieren
    input_nodes = [f"Input {i}" for i in range(num_inputs)]
    bias_node = ["Bias"]
    output_nodes = [f"Output {i}" for i in range(num_outputs)]
    hidden_nodes = [f"Hidden {i}" for i in range(num_hidden)]

    all_nodes = input_nodes + bias_node + output_nodes + hidden_nodes

    # Initialisiere den Graphen
    G = nx.DiGraph()

    # Knoten dem Graphen hinzufügen
    G.add_nodes_from(all_nodes)

    # Kanten aus input_matrix hinzufügen
    for i in range(num_outputs + num_hidden):
        for j in range(num_inputs + 1):  # +1 für Bias
            if input_matrix[i, j] != 0:
                G.add_edge(all_nodes[j], all_nodes[num_inputs + 1 + i], weight=f"{input_matrix[i, j]:.2f}")

    # Kanten aus hidden_matrix hinzufügen
    for i in range(num_outputs + num_hidden):
        for j in range(num_outputs + num_hidden):
            if hidden_matrix[i, j] != 0:
                G.add_edge(all_nodes[num_inputs + 1 + j], all_nodes[num_inputs + 1 + i], weight=f"{hidden_matrix[i, j]:.2f}")

    # Positionierung der Knoten
    pos = {}
    # Input-Knoten und Bias links anordnen
    for i, node in enumerate(input_nodes + bias_node):
        pos[node] = (-1, -i)
    # Output-Knoten rechts anordnen
    for i, node in enumerate(output_nodes):
        pos[node] = (1, -i)
    # Hidden-Knoten mittig anordnen
    for i, node in enumerate(hidden_nodes):
        pos[node] = (0+(i**2)/(2*num_hidden**2), -i)

    # Graph zeichnen
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300, font_size=10)
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='red',
        font_size=10,
        label_pos=0.75
    )
    
    plt.title("Neural Network Visualization")
    plt.show()
    
    
# Beispiel für die Initialisierung und Nutzung des Netzwerks
if __name__ == "__main__":
    num_inputs = 6
    nn = NeuralNetwork(num_inputs = num_inputs, num_outputs=2, mutate=10)
    
    #print(nn.Input_Matrix)
    #print(nn.Hidden_Matrix)
    
    
    plot_neural_network(nn)
    
