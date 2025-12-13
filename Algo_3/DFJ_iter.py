import pulp
import time


def get_subtours(n, solution_arcs):
    """
    Prend la liste des arcs actifs (ex: [(0,1), (1,0)...]) et retourne
    la liste des cycles (ex: [[0, 1], [2, 3, 4]]).
    """
    # Création d'un dictionnaire de navigation : ville_depart -> ville_arrivee
    next_node = {}
    for (i, j) in solution_arcs:
        next_node[i] = j

    unvisited = set(range(n))
    subtours = []

    while unvisited:
        # On prend une ville au hasard parmi celles non visitées
        start_node = list(unvisited)[0]
        current_cycle = []
        current_node = start_node

        # On suit le chemin jusqu'à revenir au départ
        while True:
            current_cycle.append(current_node)
            unvisited.remove(current_node)

            # Trouver la prochaine ville
            if current_node in next_node:
                next_n = next_node[current_node]

                # Si on boucle (retour au début de ce cycle), c'est fini pour ce sous-tour
                if next_n == start_node:
                    break

                current_node = next_n
            else:
                # Sécurité (ne devrait pas arriver avec contraintes de degré)
                break

        subtours.append(current_cycle)

    return subtours


def solve_dfj_iterative(n, dist_matrix):
    """
    Implémente l'Algorithme 1 : Génération itérative de contraintes.
    """
    # 1. Création du modèle initial
    prob = pulp.LpProblem("TSP_DFJ_Iterative", pulp.LpMinimize)
    cities = range(n)

    # Variables binaires x_ij
    x = pulp.LpVariable.dicts("x",
                              ((i, j) for i in cities for j in cities if i != j),
                              cat='Binary')

    # Objectif
    prob += pulp.lpSum(dist_matrix[i][j] * x[i, j] for i in cities for j in cities if i != j)

    # Contraintes de degré (Base) : 1 entrant, 1 sortant par ville
    for i in cities:
        prob += pulp.lpSum(x[i, j] for j in cities if i != j) == 1
    for j in cities:
        prob += pulp.lpSum(x[i, j] for i in cities if i != j) == 1

    # --- DÉBUT DE LA BOUCLE ITÉRATIVE ---
    iteration = 0
    total_solve_time = 0

    while True:
        iteration += 1

        # MESURE DU TEMPS (Strictement autour du solveur comme demandé [cite: 89])
        start_t = time.time()
        # msg=0 pour cacher le log du solveur
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        end_t = time.time()

        # On cumule le temps
        total_solve_time += (end_t - start_t)

        # Extraction de la solution courante (arcs où x > 0.9)
        solution_arcs = []
        for i in cities:
            for j in cities:
                if i != j and pulp.value(x[i, j]) > 0.9:
                    solution_arcs.append((i, j))

        # Détection des sous-tours
        subtours = get_subtours(n, solution_arcs)

        # Condition d'arrêt : Si 1 seul cycle et qu'il contient toutes les villes
        if len(subtours) == 1 and len(subtours[0]) == n:
            break  # Solution optimale trouvée

        # Sinon, on ajoute les contraintes pour casser les sous-tours trouvés
        # Contrainte : Somme(x_ij pour i,j DANS S) <= |S| - 1
        for S in subtours:
            prob += pulp.lpSum(x[i, j] for i in S for j in S if i != j) <= len(S) - 1

    return prob, total_solve_time, x, iteration


import sys


# --- Nécessaire pour lire le fichier d'instance ---
def read_instance(filename):
    """Lit le fichier d'instance et retourne n et la matrice de distances."""
    try:
        with open(filename, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        n = int(lines[0])
        # La matrice commence après la ligne n et les n lignes de coordonnées
        matrix_start = n + 1

        dist_matrix = []
        for i in range(n):
            line_idx = matrix_start + i
            row_vals = list(map(float, lines[line_idx].split()))
            dist_matrix.append(row_vals)

        return n, dist_matrix
    except Exception as e:
        print(f"Erreur de lecture du fichier : {e}")
        sys.exit(1)


# --- Le Main spécifique pour DFJ Itératif ---
if __name__ == "__main__":
    # Remplacez ceci par le chemin réel de votre fichier instance
    # Assurez-vous que le fichier existe bien à cet emplacement relative
    instance_file = "instances/instance_10_circle_1.txt"

    try:
        # 1. Lecture
        n, dist_matrix = read_instance(instance_file)
        print(f"Instance chargée : {n} villes")

        # 2. Résolution (Attention : DFJ retourne 4 valeurs)
        # prob : le problème résolu
        # duration : le temps cumulé des appels au solveur
        # x_vars : les variables de décision
        # iterations : le nombre de boucles effectuées
        prob, duration, x_vars, iterations = solve_dfj_iterative(n, dist_matrix)

        # 3. Affichage des résultats
        print(f"Status: {pulp.LpStatus[prob.status]}")
        print(f"Objectif: {pulp.value(prob.objective)}")
        print(f"Temps solveur: {duration:.4f} sec")
        print(f"Nombre d'itérations : {iterations}")

        # 4. Affichage des arcs sélectionnés
        print("\n--- Liste des arcs ---")
        solution_arcs = []
        for i in range(n):
            for j in range(n):
                if i != j and pulp.value(x_vars[i, j]) > 0.9:
                    print(f"Arc {i} -> {j}")
                    solution_arcs.append((i, j))

        # 5. Affichage du cycle propre (Optionnel mais recommandé)
        # On réutilise get_subtours pour avoir l'ordre 0 -> 5 -> 2...
        print("\n--- Cycle Ordonné ---")
        cycles = get_subtours(n, solution_arcs)
        if cycles:
            # Affiche le premier cycle trouvé
            print(" -> ".join(map(str, cycles[0])) + " -> " + str(cycles[0][0]))

    except FileNotFoundError:
        print(f"Fichier '{instance_file}' non trouvé. Vérifiez le chemin.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")