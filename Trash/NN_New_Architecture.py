import random
import math
import numpy as np
import itertools


# Parameter für die Main Datei
ADD_NEURON = 0.2
ADD_WEIGHT = 0.5
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
    return x

def actSpeed(x):
    return max(0, math.tanh(x))
def actAngle(x):
    return math.tanh(x)

class Neuron:
    def __init__(self, neuron_id, layer_type):
        """
        :param neuron_id: Einzigartige ID für das Neuron
        :param layer_type: Typ des Neurons ('input', 'hidden', 'output', 'bias')
        """
        self.id = neuron_id
        self.layer_type = layer_type

    def __repr__(self):
        return f"Neuron({self.id}, {self.layer_type})"

class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs, mutate=2, bias=BIAS):
        self.neurons = {}  # Enthält alle Neuronen
        self.connections = {}  # {Quelle: [(Ziel, Gewicht)]}
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.bias = bias
        self.next_id = 0  # Nächste ID für neue Neuronen

        # Eingabe- und Ausgabeneuronen initialisieren
        for _ in range(num_inputs):
            self.add_neuron('input')
        self.add_neuron('bias')
        for _ in range(num_outputs):
            self.add_neuron('output')
        for _ in range(mutate):
            self.mutate()
            
    def add_neuron(self, layer_type):
        """Fügt ein neues Neuron hinzu."""
        neuron = Neuron(self.next_id, layer_type)
        self.neurons[self.next_id] = neuron
        self.connections[self.next_id] = []
        self.next_id += 1

    def add_connection(self, source_id, target_id, weight=None):
        """Fügt eine Verbindung zwischen zwei Neuronen hinzu."""
        if weight is None:
            weight = random.uniform(-1, 1)
        #if source_id in self.connections:
        self.connections[source_id].append((target_id, weight)) #der fehler darf eh nicht auftreten

    def forward_vectorized(self, inputs):
        """Führt die Vorwärtspropagation durch."""
        
        # Initialisiere Aktivierungen (inkl. Bias)
        activations = np.zeros(self.num_inputs+1)
        for idx in range(self.num_inputs):
            activations[idx] = inputs[idx]
        activations[-1] = self.bias 
        
        in_bi = self.num_inputs + 1 #length: input+hidden
        ot_hi = len(self.neurons)-in_bi #length: output+hidden
        
        #Initialisiere Input Matrix
        Input_Matrix = np.zeros((in_bi, ot_hi))
        for src, connection in itertools.islice(self.connections.items(), (in_bi)): #slice to
            for target, weight in connection:
                Input_Matrix[src, target - in_bi] = weight
    
        #Initialisiere Hidden Matrix
        Hidden_Matrix = np.zeros((ot_hi, ot_hi))
        #Hidden_Adj = np.zeros((ot_hi, ot_hi))     
        for src, connection in itertools.islice(self.connections.items(), in_bi ,None): #slice from
            for target, weight in connection:
                Hidden_Matrix[src-in_bi, target - in_bi] = weight
                #Hidden_Adj[src-in_bi, target - in_bi] = 1 
        
        Hidden_Matrix = Hidden_Matrix.T
        #Hidden_Adj = Hidden_Adj.T
        
        #first input activation
        activations = np.dot(Input_Matrix.T, activations) 
        
        #checkProgress = np.zeros(ot_hi)
        indexListe = list(range(self.num_outputs, ot_hi))
        for i in range(self.num_outputs, ot_hi):
            if np.all(Hidden_Matrix[i] == 0):
                activations[i] = actFunc(activations[i])
                indexListe.remove(i)
            #Hidden_Adj[:,i]=0
        while indexListe: #skalliert im extremfall schlecht mit neuronen, mit gewichten skaliert besser
            a = len(indexListe)
            for i in indexListe:
                #row == 0?, addiert "aktivierten input" und aktiviert sich selber
                if np.all(Hidden_Matrix[i,indexListe] == 0): #überprüfe nur die entsprehenden relevanten stellen, ob sie null sind oder nicht
                    activations[i] = actFunc(activations[i] + (Hidden_Matrix[i] @ activations))
                    indexListe.remove(i)
                    #Hidden_Adj[:,i]=0 #spalte == null --> neue Aktivierung möglich
                    #break #? 
            #if a == len(indexListe):
                #print("Fatal Error, there seems to be an infinity LooP")
                #return [0 for _ in range(self.num_outputs)]
        #aktuell habe ich keine kontrolle über die letze aktivierungsfunktion
        
        # Rückgabe der Ausgabeneuronen
        #return [actFunc(activations[i] + Hidden_Matrix[i] @ activations) for i in range(self.num_outputs)]
        return [actSpeed(activations[0] + Hidden_Matrix[0] @ activations), actAngle(activations[1] + Hidden_Matrix[1] @ activations)]
        
    def mutate(self):
        """Mutiert das Netzwerk durch Hinzufügen von Verbindungen oder Neuronen."""
        if random.random() < ADD_NEURON:
            # Neues Neuron hinzufügen
            self.add_neuron('hidden')
        
        if random.random() < ADD_WEIGHT:
            # Neue Verbindung hinzufügen
            source = random.choice( (list(range(self.num_inputs+1)) + list(range(self.num_inputs+1+self.num_outputs,len(self.neurons)))) ) 
            target = random.choice( list(range(self.num_inputs+1,len(self.neurons))) )
            single_conectn = True
            for id, _ in self.connections[source]:
                if id == target: 
                    single_conectn = False
                    break
            if single_conectn:
                if not self.infinityLoop(source, target):
                    self.add_connection(source, target)
        
        if random.random() < CHANGE_WEIGHT:
            # Gewicht einer bestehenden Verbindung ändern
            #fehler, falls leeres netzwerk, ohne input, ohne output
            #4 versuche, um eine source mit weight zu finden
            for _ in range(4):
                source = random.randint(0,self.next_id-1)
                if self.connections[source]:
                    target_idx = random.randint(0, len(self.connections[source]) - 1)
                    target, weight = self.connections[source][target_idx]
                    new_weight = weight + random.choice([-0.1, 0.1])
                    self.connections[source][target_idx] = (target, new_weight)
                    break
                
    def infinityLoop(self, source, target):
        #checks if source and target are hidden layer
        in_bi_out = self.num_inputs + 1 + self.num_outputs #length: input+hidden+out
        if ((in_bi_out-1) < source) and ((in_bi_out-1) < target): 
            hi = len(self.neurons)-in_bi_out #length:hidden 
            Hidden_Adj = np.zeros((hi, hi)) #nur dimension hidden*hidden 
            Hidden_Adj[source - in_bi_out, target - in_bi_out] = 1 
            
            for src, connection in itertools.islice(self.connections.items(), in_bi_out ,None): #slice from
                for targ, _ in connection:
                    if targ >= in_bi_out: #only checks connections to other members
                        Hidden_Adj[src - in_bi_out, targ - in_bi_out] = 1 #wenn ein hidden layer output als target, dann ist target id zb 7, in_bi_out = 9 ->> index fehler
                    
            indexListe = list(range(hi)) 
            while indexListe:
                a = len(indexListe) 
                for i in indexListe:
                    if np.all(Hidden_Adj[i] == 0): 
                        indexListe.remove(i) 
                        Hidden_Adj[:,i]=0 
                        #break
                if a == len(indexListe):
                    return True #there is an infinity Loop with this connection
            return False
        else: 
            return False

# Beispiel für die Initialisierung und Nutzung des Netzwerks
if __name__ == "__main__":
    num_inputs = 6
    nn = NeuralNetwork(num_inputs = num_inputs, num_outputs=2, mutate=10)
    
    inputs = [random.uniform(-1,1) for _ in range(num_inputs)]
    outputs = nn.forward_vectorized(inputs)
    if outputs == [0,0]:
        print(inputs)
        print(nn.connections)
    print("Outputs:", outputs)