from collections import deque
from module.Brain_Neural_Network import *

# Beispiel für die Initialisierung und Nutzung des Netzwerks
#if __name__ == "__main__":
  #  num_inputs = 6
 #   nn = NeuralNetwork(num_inputs = num_inputs, num_outputs=2, mutate=20)
    
#    inputs = [random.uniform(-1,1) for _ in range(num_inputs)]
 #   outputs = nn.forward_vectorized(inputs)
  #  if outputs == [0,0]:
   #     print(inputs)
    #print("Outputs:", outputs)
    

# --- Konfiguration ---
NUM_MATRIZES = 40       # Anzahl der Matrizen pro Größe
MATRIX_GROESSEN = [40, 80, 160, 320]  # Größen der Hidden-Matrizen (Anzahl Hidden-Neuronen)
VERBINDUNGS_DICHTE = 0.2  # Wahrscheinlichkeit, dass eine Verbindung existiert

def kahn_zyklus_prüfung(matrix, source_idx, target_idx):
    """Prüft auf Zyklen mit Kahn's Algorithmus (direkt auf der Matrix)."""
    N = matrix.shape[0]
    adj = {i: [] for i in range(N)}
    # Konvertiere Matrix in Adjazenzliste
    for i in range(N):
        for j in range(N):
            if matrix[i, j] != 0:
                adj[j].append(i)  # Verbindung von j → i
    # Füge neue Verbindung hinzu
    adj[source_idx].append(target_idx)
    # Kahn's Algorithmus
    in_degree = [0] * N
    for u in adj:
        for v in adj[u]:
            in_degree[v] += 1
    queue = deque([u for u in range(N) if in_degree[u] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return len(top_order) != N  # True = Zyklus vorhanden

# --- Testlogik ---
def laufzeit_test():
    for größe in MATRIX_GROESSEN:
        original_zeiten = []
        kahn_zeiten = []
        
        for _ in range(NUM_MATRIZES):
            # Generiere Hidden-Matrix
            #matrix = generiere_hidden_matrix(größe, VERBINDUNGS_DICHTE)
            # --- Teste Original-Methode (infinityLoop) ---
            # Initialisiere Netzwerk mit num_inputs=0, num_outputs=0
            nn = NeuralNetwork(num_inputs=10, num_outputs=2, mutate=größe)
            # Füge Hidden-Neuronen hinzu und setze die Matrix
                
            matrix = nn.Hidden_Matrix.copy()
            if nn.num_inputs + 1 + nn.num_outputs == nn.neuron_number:
                continue
            source = random.choice( list(range(nn.num_inputs + 1 + nn.num_outputs, nn.neuron_number)) )
            target = random.choice( list(range(nn.num_inputs + 1 + nn.num_outputs, nn.neuron_number)) )
            # Konvertiere Indizes zu Neuron-IDs (Hidden-Neuronen starten bei ID 1)
            source_id = source - (nn.num_inputs + 1)  # Hidden-Neuronen-IDs: 1, 2, ..., größe
            target_id = target - (nn.num_inputs + 1)
            
            #print(matrix.shape, source, target, source_id, target_id)
            
            start = time.perf_counter()
            hat_zyklus = nn.infinityLoop(source, target)
            end = time.perf_counter()
            original_zeiten.append(end - start)
            
            # --- Teste Kahn's Algorithmus ---
            start = time.perf_counter()
            ergebnis_kahn = kahn_zyklus_prüfung(matrix, source_id, target_id)
            end = time.perf_counter()
            kahn_zeiten.append(end - start)
            
            # Validiere, dass beide Methoden das gleiche Ergebnis liefern
            assert hat_zyklus == ergebnis_kahn, f"Ergebnisse stimmen nicht überein! {hat_zyklus}, und {ergebnis_kahn}, {source} {target}, {nn.Hidden_Matrix}"
        
        # Statistik
        avg_original = sum(original_zeiten) / NUM_MATRIZES
        avg_kahn = sum(kahn_zeiten) / NUM_MATRIZES
        print(f"\n--- Größe {größe}x{größe} (Dichte {VERBINDUNGS_DICHTE}) ---")
        print(f"Original-Methode: {avg_original:.6f} s pro Test")
        print(f"Kahn's Algorithmus: {avg_kahn:.6f} s pro Test")
        print(f"Verbesserung: {avg_original / avg_kahn:.1f}x schneller")

# --- Test ausführen ---
if __name__ == "__main__":
    # Stelle sicher, dass die Original-Klasse verfügbar ist
    #from NN_UPDATE_Archi import NeuralNetwork
    laufzeit_test()