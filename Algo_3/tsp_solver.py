import pulp
import sys
import time


# =============================================================================
# 1. FONCTIONS UTILITAIRES (Lecture, Cycles, Sets)
# =============================================================================

def read_instance(filename):
    """
    Lit le fichier d'instance TSP et retourne le nombre de villes et la matrice de distances.
    
    Le format du fichier attendu est :
    - Ligne 1 : nombre de villes n
    - Lignes 2 à n+1 : coordonnées des villes (ignorées)
    - Lignes n+2 à 2n+1 : matrice de distances n x n
    
    Args:
        filename (str): Chemin vers le fichier d'instance à lire.
        
    Returns:
        tuple: (n, dist_matrix) où :
            - n (int): Nombre de villes dans l'instance, ou None en cas d'erreur
            - dist_matrix (list): Matrice de distances n x n, ou None en cas d'erreur
    """
    try:
        with open(filename, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        # [cite_start]Ligne 1 : Nombre de villes n [cite: 38]
        n = int(lines[0])

        # Les n lignes suivantes sont les coordonnées (on les saute)
        # La matrice commence après la ligne n (indice 0) + n lignes coords
        matrix_start = n + 1

        dist_matrix = []
        for i in range(n):
            line_idx = matrix_start + i
            if line_idx < len(lines):
                row_vals = list(map(float, lines[line_idx].split()))
                dist_matrix.append(row_vals)
            else:
                raise ValueError("Fichier incomplet.")

        return n, dist_matrix
    except Exception as e:
        print(f"Erreur de lecture du fichier : {e}")
        return None, None


def get_subtours(n, solution_arcs):
    """
    Retourne la liste des cycles (sous-tours) présents dans une solution donnée.
    
    Cette fonction analyse les arcs sélectionnés dans une solution TSP et identifie
    tous les cycles présents. Utilisée pour la méthode DFJ itérative (détection
    de sous-tours) et pour l'affichage du cycle final.
    
    Args:
        n (int): Nombre total de villes dans l'instance.
        solution_arcs (list): Liste de tuples (i, j) représentant les arcs
            sélectionnés dans la solution.
            
    Returns:
        list: Liste de cycles, où chaque cycle est une liste de nœuds dans l'ordre
            de parcours. Un cycle hamiltonien complet contiendra n nœuds.
    """
    next_node = {}
    for (i, j) in solution_arcs:
        next_node[i] = j

    unvisited = set(range(n))
    subtours = []

    while unvisited:
        start_node = list(unvisited)[0]
        current_cycle = []
        current_node = start_node

        while True:
            current_cycle.append(current_node)
            unvisited.remove(current_node)

            if current_node in next_node:
                next_n = next_node[current_node]
                if next_n == start_node:
                    break
                if next_n not in unvisited:  # Sécurité
                    break
                current_node = next_n
            else:
                break
        subtours.append(current_cycle)
    return subtours


def get_powerset(n):
    """
    Génère tous les sous-ensembles possibles de l'ensemble {0, 1, ..., n-1}.
    
    Cette fonction est utilisée pour la formulation DFJ énumérative qui nécessite
    toutes les contraintes de sous-tour a priori. Attention : la complexité est
    exponentielle (2^n sous-ensembles), donc cette méthode n'est utilisable que
    pour de petites instances (n <= 15).
    
    Args:
        n (int): Nombre d'éléments dans l'ensemble (nombre de villes).
        
    Returns:
        list: Liste de tous les sous-ensembles possibles, incluant l'ensemble vide.
            Chaque sous-ensemble est représenté comme une liste d'entiers.
    """
    subsets = [[]]
    for city in range(n):
        new_subsets = [subset + [city] for subset in subsets]
        subsets.extend(new_subsets)
    return subsets


def print_solution(prob, duration, x_vars, n, iterations=None):
    """
    Affiche les résultats de la résolution TSP selon le format requis.
    
    Les résultats affichés incluent :
    - La valeur de la fonction objectif (coût total du cycle)
    - Le cycle hamiltonien complet (uniquement pour solutions entières optimales)
    - Le temps de résolution en secondes
    - Le nombre d'itérations (uniquement pour la méthode itérative)
    
    Args:
        prob (pulp.LpProblem): Problème linéaire résolu par PuLP.
        duration (float): Temps de résolution en secondes.
        x_vars (dict): Dictionnaire des variables de décision x[i, j] du problème.
        n (int): Nombre de villes dans l'instance.
        iterations (int, optional): Nombre d'itérations pour les méthodes itératives.
            Par défaut None (non affiché pour les méthodes non-itératives).
    """
    if prob is None:
        print("Erreur: Impossible de résoudre cette instance avec cette méthode.")
        return

    # Valeur de la fonction objectif
    if prob.status == pulp.LpStatusOptimal:
        obj_value = pulp.value(prob.objective)
        print(f"value : {obj_value:.6f}")
    else:
        print(f"Status: {pulp.LpStatus[prob.status]}")
        if prob.status == pulp.LpStatusInfeasible:
            print("Problème infaisable")
        elif prob.status == pulp.LpStatusUnbounded:
            print("Problème non borné")
        return

    # Cycle hamiltonien (uniquement pour solutions entières)
    # On vérifie si c'est une solution entière en regardant les variables
    is_integer_solution = True
    sol_arcs = []
    
    if x_vars is not None:
        for i in range(n):
            for j in range(n):
                if i != j and x_vars.get((i, j)):
                    val = pulp.value(x_vars[i, j])
                    # Vérifier si la valeur est proche de 0 ou 1 (solution entière)
                    if val is not None:
                        if val > 0.1 and val < 0.9:
                            is_integer_solution = False
                        if val > 0.9:
                            sol_arcs.append((i, j))

    # Afficher le cycle hamiltonien seulement pour solutions entières optimales
    if is_integer_solution and sol_arcs and prob.status == pulp.LpStatusOptimal:
        cycles = get_subtours(n, sol_arcs)
        if len(cycles) == 1 and len(cycles[0]) == n:
            # Cycle hamiltonien complet
            cycle = cycles[0]
            path_str = " -> ".join(map(str, cycle)) + " -> " + str(cycle[0])
            print(path_str)
        elif len(cycles) > 0:
            # Afficher quand même le cycle principal si disponible
            cycle = cycles[0]
            path_str = " -> ".join(map(str, cycle)) + " -> " + str(cycle[0])
            print(path_str)

    # Temps de résolution
    print(f"duration:{duration:.6f}")

    # Nombre d'itérations (pour méthode itérative uniquement)
    if iterations is not None:
        print(f"number of iterations:{iterations}")


# =============================================================================
# 2. LES MÉTHODES DE RÉSOLUTION
# =============================================================================

def solve_mtz_relaxation(n, dist_matrix):
    """
    Résout le TSP en utilisant la formulation MTZ (Miller-Tucker-Zemlin) en relaxation continue.
    
    Cette méthode utilise la formulation MTZ avec des variables continues (relaxation)
    au lieu de variables binaires. La solution obtenue est une borne inférieure du
    problème TSP, mais n'est généralement pas une solution entière valide.
    
    La formulation MTZ utilise des variables de position u[i] pour éviter les sous-tours.
    La contrainte MTZ : u[i] - u[j] + n * x[i, j] <= n - 1 pour i, j != 0, i != j.
    
    Args:
        n (int): Nombre de villes dans l'instance.
        dist_matrix (list): Matrice de distances n x n, où dist_matrix[i][j] est la
            distance entre la ville i et la ville j.
            
    Returns:
        tuple: (prob, duration, x) où :
            - prob (pulp.LpProblem): Problème linéaire résolu
            - duration (float): Temps de résolution en secondes
            - x (dict): Dictionnaire des variables de décision x[i, j] (continues)
    """
    prob = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)
    cities = range(n)

    # Variables
    x = pulp.LpVariable.dicts("x",
                              ((i, j) for i in cities for j in cities if i != j),
                              lowBound=0,  # Borne minimale : 0
                              upBound=1,  # Borne maximale : 1
                              cat='Continuous')  # Variable continue (nombres à virgule)
    u = pulp.LpVariable.dicts("u", (i for i in range(1, n)), lowBound=1, upBound=n - 1, cat='Continuous')

    # Objectif
    prob += pulp.lpSum(dist_matrix[i][j] * x[i, j] for i in cities for j in cities if i != j)

    # Contraintes Flux
    for i in cities: prob += pulp.lpSum(x[i, j] for j in cities if i != j) == 1
    for j in cities: prob += pulp.lpSum(x[i, j] for i in cities if i != j) == 1

    # Contraintes MTZ
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i, j] <= n - 1

    start_t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duration = time.time() - start_t

    return prob, duration, x

def solve_mtz(n, dist_matrix):
    """
    Résout le TSP en utilisant la formulation MTZ (Miller-Tucker-Zemlin) avec variables binaires.
    
    Cette méthode utilise la formulation MTZ classique avec des variables binaires x[i, j]
    et des variables continues u[i] pour représenter la position de la ville i dans le tour.
    La contrainte MTZ empêche la formation de sous-tours.
    
    La formulation MTZ utilise des variables de position u[i] pour éviter les sous-tours.
    La contrainte MTZ : u[i] - u[j] + n * x[i, j] <= n - 1 pour i, j != 0, i != j.
    
    Args:
        n (int): Nombre de villes dans l'instance.
        dist_matrix (list): Matrice de distances n x n, où dist_matrix[i][j] est la
            distance entre la ville i et la ville j.
            
    Returns:
        tuple: (prob, duration, x) où :
            - prob (pulp.LpProblem): Problème linéaire résolu
            - duration (float): Temps de résolution en secondes
            - x (dict): Dictionnaire des variables de décision x[i, j] (binaires)
    """
    prob = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)
    cities = range(n)

    # Variables
    x = pulp.LpVariable.dicts("x", ((i, j) for i in cities for j in cities if i != j), cat='Binary')
    u = pulp.LpVariable.dicts("u", (i for i in range(1, n)), lowBound=1, upBound=n - 1, cat='Continuous')

    # Objectif
    prob += pulp.lpSum(dist_matrix[i][j] * x[i, j] for i in cities for j in cities if i != j)

    # Contraintes Flux
    for i in cities: prob += pulp.lpSum(x[i, j] for j in cities if i != j) == 1
    for j in cities: prob += pulp.lpSum(x[i, j] for i in cities if i != j) == 1

    # Contraintes MTZ
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i, j] <= n - 1

    start_t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duration = time.time() - start_t

    return prob, duration, x


def solve_dfj_enum(n, dist_matrix):
    """
    Résout le TSP en utilisant la formulation DFJ (Dantzig-Fulkerson-Johnson) énumérative.
    
    Cette méthode génère a priori toutes les contraintes de sous-tour possibles pour
    tous les sous-ensembles de villes. La solution est garantie d'être optimale mais
    la méthode est exponentielle en complexité (2^n contraintes).
    
    Pour des raisons de performance, cette méthode est limitée aux instances avec
    n <= 15. Pour des instances plus grandes, utilisez solve_dfj_iterative.
    
    La contrainte DFJ pour un sous-ensemble S : sum(x[i, j] pour i, j dans S) <= |S| - 1
    
    Args:
        n (int): Nombre de villes dans l'instance (doit être <= 15).
        dist_matrix (list): Matrice de distances n x n, où dist_matrix[i][j] est la
            distance entre la ville i et la ville j.
            
    Returns:
        tuple: (prob, duration, x) où :
            - prob (pulp.LpProblem): Problème linéaire résolu, ou None si n > 15
            - duration (float): Temps de résolution en secondes (0 si n > 15)
            - x (dict): Dictionnaire des variables de décision x[i, j] (binaires),
                ou None si n > 15
    """
    # Sécurité pour éviter le crash sur grandes instances
    if n > 15:
        return None, 0, None

    prob = pulp.LpProblem("TSP_DFJ_Enum", pulp.LpMinimize)
    cities = range(n)
    x = pulp.LpVariable.dicts("x", ((i, j) for i in cities for j in cities if i != j), cat='Binary')

    prob += pulp.lpSum(dist_matrix[i][j] * x[i, j] for i in cities for j in cities if i != j)

    for i in cities: prob += pulp.lpSum(x[i, j] for j in cities if i != j) == 1
    for j in cities: prob += pulp.lpSum(x[i, j] for i in cities if i != j) == 1

    # Génération brute de toutes les contraintes
    all_subsets = get_powerset(n)
    for subset in all_subsets:
        k = len(subset)
        if 2 <= k <= n - 1:
            prob += pulp.lpSum(x[i, j] for i in subset for j in subset if i != j) <= k - 1

    start_t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duration = time.time() - start_t

    return prob, duration, x




def solve_dfj_enum_relax(n, dist_matrix):
    """
    Résout le TSP en utilisant la formulation DFJ énumérative en relaxation continue.
    
    Cette méthode génère a priori toutes les contraintes de sous-tour possibles pour
    tous les sous-ensembles de villes, mais utilise des variables continues au lieu
    de variables binaires. La solution obtenue est une borne inférieure du problème
    TSP, mais n'est généralement pas une solution entière valide.
    
    Pour des raisons de performance, cette méthode est limitée aux instances avec
    n <= 15. Pour des instances plus grandes, utilisez solve_dfj_iterative.
    
    La contrainte DFJ pour un sous-ensemble S : sum(x[i, j] pour i, j dans S) <= |S| - 1
    
    Args:
        n (int): Nombre de villes dans l'instance (doit être <= 15).
        dist_matrix (list): Matrice de distances n x n, où dist_matrix[i][j] est la
            distance entre la ville i et la ville j.
            
    Returns:
        tuple: (prob, duration, x) où :
            - prob (pulp.LpProblem): Problème linéaire résolu, ou None si n > 15
            - duration (float): Temps de résolution en secondes (0 si n > 15)
            - x (dict): Dictionnaire des variables de décision x[i, j] (continues),
                ou None si n > 15
    """
    # Sécurité pour éviter le crash sur grandes instances
    if n > 15:
        return None, 0, None

    prob = pulp.LpProblem("TSP_DFJ_Enum", pulp.LpMinimize)
    cities = range(n)
    x = pulp.LpVariable.dicts("x",
                              ((i, j) for i in cities for j in cities if i != j),
                              lowBound=0,  # Borne minimale : 0
                              upBound=1,  # Borne maximale : 1
                              cat='Continuous')  # Variable continue (nombres à virgule)

    prob += pulp.lpSum(dist_matrix[i][j] * x[i, j] for i in cities for j in cities if i != j)

    for i in cities: prob += pulp.lpSum(x[i, j] for j in cities if i != j) == 1
    for j in cities: prob += pulp.lpSum(x[i, j] for i in cities if i != j) == 1

    # Génération brute de toutes les contraintes
    all_subsets = get_powerset(n)
    for subset in all_subsets:
        k = len(subset)
        if 2 <= k <= n - 1:
            prob += pulp.lpSum(x[i, j] for i in subset for j in subset if i != j) <= k - 1

    start_t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duration = time.time() - start_t

    return prob, duration, x


def solve_dfj_iterative(n, dist_matrix):
    """
    Résout le TSP en utilisant la formulation DFJ itérative (génération de contraintes à la volée).
    
    Cette méthode résout le problème TSP en ajoutant itérativement les contraintes
    de sous-tour uniquement lorsque nécessaire. À chaque itération :
    1. Résout le problème relaxé
    2. Détecte les sous-tours dans la solution
    3. Ajoute des contraintes pour éliminer ces sous-tours
    4. Répète jusqu'à obtenir un cycle hamiltonien complet
    
    Cette approche est beaucoup plus efficace que la méthode énumérative car elle
    n'ajoute que les contraintes nécessaires, permettant de traiter des instances
    plus grandes.
    
    Args:
        n (int): Nombre de villes dans l'instance.
        dist_matrix (list): Matrice de distances n x n, où dist_matrix[i][j] est la
            distance entre la ville i et la ville j.
            
    Returns:
        tuple: (prob, duration, x, iteration) où :
            - prob (pulp.LpProblem): Problème linéaire résolu
            - duration (float): Temps total de résolution en secondes (somme de toutes les itérations)
            - x (dict): Dictionnaire des variables de décision x[i, j] (binaires)
            - iteration (int): Nombre d'itérations effectuées avant convergence
    """
    prob = pulp.LpProblem("TSP_DFJ_Iter", pulp.LpMinimize)
    cities = range(n)
    x = pulp.LpVariable.dicts("x", ((i, j) for i in cities for j in cities if i != j), cat='Binary')

    prob += pulp.lpSum(dist_matrix[i][j] * x[i, j] for i in cities for j in cities if i != j)

    for i in cities: prob += pulp.lpSum(x[i, j] for j in cities if i != j) == 1
    for j in cities: prob += pulp.lpSum(x[i, j] for i in cities if i != j) == 1

    iteration = 0
    total_time = 0
    max_iterations = 1000  # Limite de sécurité pour éviter les boucles infinies

    while iteration < max_iterations:
        iteration += 1
        start_t = time.time()
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        total_time += (time.time() - start_t)

        # Vérifier le statut du solveur
        if prob.status != pulp.LpStatusOptimal:
            # Si le problème devient infaisable ou non borné, arrêter
            break

        # Extraction solution
        sol_arcs = []
        for i in cities:
            for j in cities:
                if i != j and pulp.value(x[i, j]) > 0.9:
                    sol_arcs.append((i, j))

        subtours = get_subtours(n, sol_arcs)

        # Condition d'arrêt : 1 seul cycle couvrant tout n
        if len(subtours) == 1 and len(subtours[0]) == n:
            break

        # Ajout Lazy Constraints pour chaque sous-tour détecté
        for S in subtours:
            if len(S) >= 2:  # S'assurer que le sous-ensemble a au moins 2 éléments
                prob += pulp.lpSum(x[i, j] for i in S for j in S if i != j) <= len(S) - 1

    return prob, total_time, x, iteration


# =============================================================================
# 3. MAIN (Interface en ligne de commande)
# =============================================================================

if __name__ == "__main__":
    # Vérification des arguments en ligne de commande
    if len(sys.argv) != 3:
        print("Usage: python3 tsp_solver.py <fichier_instance> <f>")
        print("  <fichier_instance>: Nom du fichier contenant l'instance")
        print("  <f>: Paramètre de formulation (0-4)")
        print("    0: MTZ (formulation entière)")
        print("    1: MTZ (relaxation continue)")
        print("    2: DFJ avec toutes les contraintes a priori (formulation entière)")
        print("    3: DFJ avec toutes les contraintes a priori (relaxation continue)")
        print("    4: DFJ avec génération itérative de contraintes")
        sys.exit(1)

    instance_file = sys.argv[1]
    
    try:
        f = int(sys.argv[2])
        if f not in [0, 1, 2, 3, 4]:
            print(f"Erreur: Le paramètre f doit être entre 0 et 4, reçu: {f}")
            sys.exit(1)
    except ValueError:
        print(f"Erreur: Le paramètre f doit être un entier, reçu: {sys.argv[2]}")
        sys.exit(1)

    # Lecture de l'instance
    n, dist_matrix = read_instance(f"instances/{instance_file}")
    
    if n is None or dist_matrix is None:
        print(f"Erreur: Impossible de lire le fichier {instance_file}")
        sys.exit(1)

    # Sélection et exécution de la méthode selon f
    prob = None
    duration = 0
    x_vars = None
    iterations = None

    try:
        if f == 0:
            # MTZ (formulation entière)
            prob, duration, x_vars = solve_mtz(n, dist_matrix)
            
        elif f == 1:
            # MTZ (relaxation continue)
            prob, duration, x_vars = solve_mtz_relaxation(n, dist_matrix)
            
        elif f == 2:
            # DFJ avec toutes les contraintes a priori (formulation entière)
            if n > 15:
                print(f"Erreur: L'instance est trop grande (n={n}) pour la méthode DFJ énumérative.")
                print("Cette méthode nécessite n <= 15. Utilisez f=4 pour la méthode itérative.")
                sys.exit(1)
            prob, duration, x_vars = solve_dfj_enum(n, dist_matrix)
            
        elif f == 3:
            # DFJ avec toutes les contraintes a priori (relaxation continue)
            if n > 15:
                print(f"Erreur: L'instance est trop grande (n={n}) pour la méthode DFJ énumérative.")
                print("Cette méthode nécessite n <= 15. Utilisez f=4 pour la méthode itérative.")
                sys.exit(1)
            prob, duration, x_vars = solve_dfj_enum_relax(n, dist_matrix)
            
        elif f == 4:
            # DFJ avec génération itérative de contraintes
            prob, duration, x_vars, iterations = solve_dfj_iterative(n, dist_matrix)
            
    except Exception as e:
        print(f"Erreur lors de la résolution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Vérification que la résolution a réussi
    if prob is None:
        print("Erreur: La résolution a échoué (prob est None)")
        sys.exit(1)

    # Affichage des résultats
    print_solution(prob, duration, x_vars, n, iterations)