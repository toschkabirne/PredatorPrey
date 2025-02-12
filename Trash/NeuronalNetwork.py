import random
import math

"""PARAMETER FELD FÜR DIE MAIN DATEI"""
#Mutation Rate
ADD_NEURON = 0.2
ADD_WEIGHT = 0.5
CHANGE_WEIGHT = 0.8

#Neuronal Network
BIAS = 0.5
REAC = False

"""ParameterFeld für diese Datei, essenziell"""
add_random_connection_counter = 3

def change_weight():
    return random.uniform(-0.1, 0.1)
def add_weight():
    return random.uniform(-1,1) # Zufälliges Gewicht zwischen -1 und 1
def Sigmoid(x):
    a = 3; b = 0
    return 1/(1+math.exp(-x*a-b))
def ReAc(x):
    #return max(0,x)
    return x

class Neuron:
    def __init__(self, neuron_id, layer_type):
        """
        :param neuron_id: Einzigartige ID für das Neuron
        :param layer_type: Typ des Neurons ('input', 'hidden', 'output', 'bias')
        """
        self.id = neuron_id
        self.layer_type = layer_type
        self.inputs = []  # Verbindungen, die auf dieses Neuron zeigen (Liste von Tupeln: (Quelle, Gewicht))
        self.outputs = []  # Verbindungen, die von diesem Neuron ausgehen (Liste von Tupeln: (Ziel, Gewicht))

    def __repr__(self):
        return f"Neuron({self.id}, {self.layer_type})"
    
class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs, mutate = 2 ,bias = BIAS):
        self.neurons = {}
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.next_id = 0  # Um Neuronen IDs zu vergeben
        self.bias = bias
        
        # Eingabe- und Ausgabeneuronen initialisieren
        self.input_neurons = [self.add_neuron('input') for _ in range(num_inputs)]
        self.input_neurons.append(self.add_neuron('bias'))
        self.output_neurons = [self.add_neuron('output') for _ in range(num_outputs)]
        for _ in range(mutate):
            self.mutate()
#hier kann ich die anfangsmutation überlegen, wie viel am anfang schon mutiert wird
        
    def calculate(self, inputs, neuron, counter, reac = REAC): #neuron, counter, 
        counter +=1
        if counter > 7:
            return 0
        if neuron.layer_type == 'hidden' and len(neuron.inputs) == 0:
            return 0
        if neuron.layer_type == 'input':
            return inputs[neuron.id]
        if neuron.layer_type == 'bias': 
            return self.bias
        if reac:
            return ReAc(sum(self.calculate(inputs, self.neurons[id_weight[0]],counter)*id_weight[1] for id_weight in neuron.inputs))
        else: 
            return Sigmoid(sum(self.calculate(inputs, self.neurons[id_weight[0]],counter)*id_weight[1] for id_weight in neuron.inputs))
        
    def forward(self, inputs):
        return [self.calculate(inputs, neuron, counter=0, reac = False) for neuron in self.output_neurons]

    def add_neuron(self, layer_type='hidden'):
        """Fügt ein neues Neuron hinzu."""
        neuron = Neuron(self.next_id, layer_type)
        self.neurons[self.next_id] = neuron
        self.next_id += 1
        return neuron

    def add_connection(self, from_id, to_id, weight=None):
        """Fügt eine Verbindung zwischen zwei Neuronen hinzu."""
        #findet beide neuronen, from_neuron.. und fügt entsprechend die input output verbindung hinzu
        if from_id not in self.neurons or to_id not in self.neurons:
            raise ValueError("Ungültige Neuron IDs.")
        if weight is None:
            weight = add_weight()
        from_neuron = self.neurons[from_id] 
        to_neuron = self.neurons[to_id]
        from_neuron.outputs.append((to_id, weight))
        to_neuron.inputs.append((from_id, weight))

#Der Fehler muss hier passiern
    def add_random_connection(self): 
        """Erstellt eine zufällige Verbindung zwischen allen möglichen Neuronen."""
        #bei verweis auf add_connection... kann passieren, dass verbindung nicht erstellt wird, falls verbindung schon vorhanden
        counter = 0
        possible_reciever_ids = list(self.neurons.keys())[(self.num_inputs+1):]
        possible_donater_ids = list(self.neurons.keys())[:(self.num_inputs+1)] + list(self.neurons.keys())[(self.num_inputs+1+self.num_outputs):]
        while counter < add_random_connection_counter: 
            from_id = random.choice(possible_donater_ids)
            to_id = random.choice(possible_reciever_ids)
            if from_id == to_id:
                continue
            #if to_id <= self.num_inputs and from_id <= self.num_inputs:
                #continue
            #if self.num_inputs < to_id <= self.num_inputs+self.num_outputs and self.num_inputs < from_id <= self.num_inputs+self.num_outputs:
                #continue #doesnt connect outputs together
            from_neuron = self.neurons[from_id] 
            #gegen doppelbelegung, natürlicher mechanismus welche neue verbindungen hemmt, je mehr verbindungen, desto unwahrscheinlicher neubelegung
            for connectionOut in from_neuron.outputs: 
                if connectionOut[0] == to_id:
                    counter += 1
                    continue
            if from_neuron.layer_type == 'hidden':    
                for connectionIn in from_neuron.inputs:
                    if connectionIn[0] == to_id:
                        counter +=1
                        continue
            break
        self.add_connection(from_id, to_id)
        
    def mutate(self, ADD_NEURON = ADD_NEURON, ADD_WEIGHT = ADD_WEIGHT, CHANGE_WEIGHT = CHANGE_WEIGHT):
        """Mutiert die Neuronen nach den gegebenen MutationsRaten"""
        #Sinnvolle Mutationsraten hinzufügen
        #1 add empty connected neuron 
        if random.random() < ADD_NEURON:
            self.add_neuron()
        
        #2 add weight/connection
        if random.random() < ADD_WEIGHT: 
            self.add_random_connection()
         #3 mutate just one random weight with weight function change_weight
        potential_ids = []
        for id, neuron in self.neurons.items():
            if neuron.inputs:
                potential_ids.append(id)
        if potential_ids:
            input_neuron = self.neurons[random.choice(potential_ids)] #wählt ein neuron mit inputs
            input_index = random.randint(0,len(input_neuron.inputs)-1) #wählt index aus input liste
            input_neuron.inputs[input_index] = (input_neuron.inputs[input_index][0], input_neuron.inputs[input_index][1]+change_weight()) #verändert den eintrag (id, weight) 
            id_weight = input_neuron.inputs[input_index] #(output_id, weight)
            for index in range(len(self.neurons[id_weight[0]].outputs)): #muss nun richtiges tuple finden und verändern
                if self.neurons[id_weight[0]].outputs[index][0] == input_neuron.id: #von outputs muss nun output id der input id gleichen
                    self.neurons[id_weight[0]].outputs[index] = (input_neuron.id, input_neuron.inputs[input_index][1])
                    break
                
    def __repr__(self):
        return f"NeuralNetwork({len(self.neurons)} Neurons, {self.num_inputs} Inputs, {self.num_outputs} Outputs)"

import networkx as nx
import matplotlib.pyplot as plt

def visualize_network(nn):
    """
    Visualisiert das neuronale Netzwerk mit einer benutzerdefinierten Anordnung.
    :param nn: Ein NeuralNetwork-Objekt.
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    # Erstelle einen gerichteten Graphen
    G = nx.DiGraph()
    
    # Füge Neuronen als Knoten hinzu
    for neuron_id, neuron in nn.neurons.items():
        G.add_node(neuron_id, layer=neuron.layer_type)

    # Füge Verbindungen als gerichtete Kanten hinzu
    for neuron_id, neuron in nn.neurons.items():
        for to_id, weight in neuron.outputs:
            G.add_edge(neuron_id, to_id, weight=round(weight, 2))  # Gewichte runden

    # Benutzerdefiniertes Layout für die Positionierung
    pos = {}
    input_neurons = [n.id for n in nn.neurons.values() if n.layer_type == 'input']
    hidden_neurons = [n.id for n in nn.neurons.values() if n.layer_type == 'hidden']
    output_neurons = [n.id for n in nn.neurons.values() if n.layer_type == 'output']
    bias_neurons = [n.id for n in nn.neurons.values() if n.layer_type == 'bias']

    # Ordne die Input-Neuronen vertikal übereinander (links)
    for i, neuron_id in enumerate(input_neurons):
        pos[neuron_id] = (-1, len(input_neurons) - i - 1)  # Links (x=-1), vertikal geordnet

    # Ordne die Hidden-Neuronen in der Mitte
    for i, neuron_id in enumerate(hidden_neurons):
        pos[neuron_id] = (0, i)  # Mitte (x=0), beliebige vertikale Anordnung

    # Ordne die Output-Neuronen vertikal übereinander (rechts)
    for i, neuron_id in enumerate(output_neurons):
        pos[neuron_id] = (1, len(output_neurons) - i - 1)  # Rechts (x=1), vertikal geordnet

    # Ordne den Bias unterhalb der Inputs
    for i, neuron_id in enumerate(bias_neurons):
        pos[neuron_id] = (-1, -1 - i)  # Unterhalb der Inputs (y negativ)

    # Farbzuordnung für die Neuronen
    color_map = {
        'input': 'lightblue',
        'hidden': 'lightgreen',
        'output': 'lightcoral',
        'bias': 'red'
    }
    node_colors = [color_map[nn.neurons[node].layer_type] for node in G.nodes]

    # Zeichne den Graphen
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1000, font_size=10, font_color='black', edge_color='gray')
    
    # Gewichte der Verbindungen anzeigen
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Neurales Netzwerk")
    plt.show()

def hihi():
    nn = NeuralNetwork(num_inputs=10, num_outputs=2, mutate = 10)
    #for _ in range(30): 
     #   nn.mutate()
    """hidden1 = nn.add_neuron('hidden')
    hidden2 = nn.add_neuron('hidden')
    nn.add_neuron()
    nn.add_neuron()
    nn.add_neuron()
    # Verbindungen hinzufügen
    nn.add_connection(nn.input_neurons[0].id, hidden1.id, 0.5)
    nn.add_connection(hidden1.id, nn.output_neurons[0].id, -0.7)
    for _ in range(20):
        nn.add_random_connection()
    """
    
    # Netzwerk visualisieren
    
    liste = [[random.randint(0,1) for _ in range(nn.num_inputs)] for _ in range(10)]
    for element in liste:
        print(nn.forward(element), end= " ")
    print("")
    visualize_network(nn)

# Beispielnetzwerk erstellen
if __name__ == "__main__":
    hihi()