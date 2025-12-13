import pulp
import sys
import time


def read_instance(filename):
    """Lit le fichier d'instance et retourne n et la matrice de distances."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Ligne 1 : Nombre de villes n [cite: 38]
    n = int(lines[0].strip())

    # Les n lignes suivantes : Coordonnées (on les ignore car on a la matrice) [cite: 39]
    # On saute donc les lignes 1 à n (indices 1 à n inclus dans le fichier)

    # Les n lignes suivantes : Matrice de distances [cite: 40]
    # Cela commence à la ligne n + 1
    dist_matrix = []
    start_line_matrix = n + 1
    for i in range(n):
        line = lines[start_line_matrix + i].strip().split()
        # Conversion des distances en float
        row = [float(x) for x in line]
        dist_matrix.append(row)

    return n, dist_matrix


def solve_mtz(n, dist_matrix, relax=False):
    """
    Résout le TSP avec la formulation MTZ.
    relax=True résout la relaxation continue (Tâche f=1).
    """

    # 1. Initialisation du problème
    prob = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)

    # Liste des villes (0 à n-1)
    cities = range(n)

    # 2. Variables de décision x_ij
    # x_ij = 1 si l'arc i -> j est utilisé
    # Catégorie : Binary pour entier, Continuous pour relaxation
    cat_type = pulp.LpContinuous if relax else pulp.LpBinary

    x = pulp.LpVariable.dicts("x",
                              ((i, j) for i in cities for j in cities if i != j),
                              lowBound=0,
                              upBound=1,
                              cat=cat_type)

    # 3. Variables auxiliaires u_i pour MTZ
    # u_i représente l'ordre de visite. On le définit pour les villes 1 à n-1.
    # u_0 est implicitement 0 (ou 1), on ne crée pas de variable pour lui.
    u = pulp.LpVariable.dicts("u",
                              (i for i in range(1, n)),
                              lowBound=1,
                              upBound=n - 1,
                              cat=pulp.LpContinuous)  # u peut rester continu même en PLNE

    # 4. Fonction Objectif : Minimiser la distance totale
    prob += pulp.lpSum(dist_matrix[i][j] * x[i, j] for i in cities for j in cities if i != j)

    # 5. Contraintes de flux (Communes à toutes les formulations) [cite: 65]

    # Exactement un arc sortant de chaque ville i
    for i in cities:
        prob += pulp.lpSum(x[i, j] for j in cities if i != j) == 1

    # Exactement un arc entrant dans chaque ville j
    for j in cities:
        prob += pulp.lpSum(x[i, j] for i in cities if i != j) == 1

    # 6. Contraintes MTZ (Élimination des sous-tours)
    # u_i - u_j + n * x_ij <= n - 1
    # Valable pour i, j appartenant à {1, ..., n-1}, i != j
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i, j] <= n - 1

    # 7. Résolution
    # On exclut le temps de modélisation Python, on mesure juste le solveur
    start_time = time.time()
    # Utilisation de CBC (COIN-OR Branch and Cut), solveur par défaut de PuLP
    # msg=0 désactive les logs dans la console
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    solve_time = time.time() - start_time

    return prob, solve_time, x


# Exemple d'utilisation (simulation de l'appel main)
if __name__ == "__main__":
    # Remplacez ceci par le chemin réel de votre fichier instance
    instance_file = "instances/instance_10_circle_1.txt"
    try:
        n, dist_matrix = read_instance(instance_file)
        prob, duration, x_vars = solve_mtz(n, dist_matrix)

        print(f"Status: {pulp.LpStatus[prob.status]}")
        print(f"Objectif: {pulp.value(prob.objective)}")
        print(f"Temps solveur: {duration:.4f} sec")

        # Affichage simple des arcs sélectionnés
        for i in range(n):
            for j in range(n):
                if i != j and pulp.value(x_vars[i, j]) > 0.9:
                    print(f"Arc {i} -> {j}")

    except FileNotFoundError:
        print("Fichier non trouvé. Assurez-vous d'avoir le fichier instance.")