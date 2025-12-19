import os
import csv
from Tsp_solver import (
    read_instance,
    solve_mtz,
    solve_mtz_relaxation,
    solve_dfj_enum,
    solve_dfj_enum_relax,
    solve_dfj_iterative
)
import pulp


def get_problem_stats(prob):
    """Extrait le nombre de variables et de contraintes d'un problème PuLP."""
    if prob is None:
        return 0, 0
    num_variables = len(prob.variables())
    num_constraints = len(prob.constraints)
    return num_variables, num_constraints


def get_objective_value(prob):
    """Extrait la valeur de la fonction objectif."""
    if prob is None:
        return None
    try:
        if prob.status == pulp.LpStatusOptimal:
            return pulp.value(prob.objective)
        else:
            return None
    except Exception:
        return None


def run_benchmark_on_instance(instance_file):
    """Exécute toutes les méthodes sur une instance et retourne les résultats."""
    print(f"\n{'='*60}")
    print(f"Traitement de l'instance: {instance_file}")
    print(f"{'='*60}")
    
    # Lecture de l'instance
    full_path = f"instances/{instance_file}"
    n, dist_matrix = read_instance(full_path)
    
    if n is None or dist_matrix is None:
        print(f"Erreur: Impossible de lire {instance_file}")
        return None
    
    print(f"Instance chargée: {n} villes")
    
    results = {
        'instance': instance_file,
        'n': n
    }
    
    # 1. MTZ (f=0)
    print("\n>>> Lancement MTZ...")
    try:
        prob, duration, x_vars = solve_mtz(n, dist_matrix)
        obj_value = get_objective_value(prob)
        num_vars, num_constraints = get_problem_stats(prob)
        results['MTZ'] = {
            'time': duration,
            'objective': obj_value,
            'variables': num_vars,
            'constraints': num_constraints,
            'status': pulp.LpStatus[prob.status] if prob else 'ERROR'
        }
        print(f"  Temps: {duration:.4f}s, Obj: {obj_value}, Vars: {num_vars}, Contr: {num_constraints}")
    except Exception as e:
        print(f"  Erreur: {e}")
        results['MTZ'] = {'time': None, 'objective': None, 'variables': 0, 'constraints': 0, 'status': 'ERROR'}
    
    # 2. MTZ Relaxation (f=1)
    print("\n>>> Lancement MTZ Relaxation...")
    try:
        prob, duration, x_vars = solve_mtz_relaxation(n, dist_matrix)
        obj_value = get_objective_value(prob)
        num_vars, num_constraints = get_problem_stats(prob)
        results['MTZ_Relax'] = {
            'time': duration,
            'objective': obj_value,
            'variables': num_vars,
            'constraints': num_constraints,
            'status': pulp.LpStatus[prob.status] if prob else 'ERROR'
        }
        print(f"  Temps: {duration:.4f}s, Obj: {obj_value}, Vars: {num_vars}, Contr: {num_constraints}")
    except Exception as e:
        print(f"  Erreur: {e}")
        results['MTZ_Relax'] = {'time': None, 'objective': None, 'variables': 0, 'constraints': 0, 'status': 'ERROR'}
    
    # 3. DFJ Itératif (f=4)
    print("\n>>> Lancement DFJ Itératif...")
    try:
        prob, duration, x_vars, iterations = solve_dfj_iterative(n, dist_matrix)
        obj_value = get_objective_value(prob)
        num_vars, num_constraints = get_problem_stats(prob)
        results['DFJ_Iter'] = {
            'time': duration,
            'objective': obj_value,
            'variables': num_vars,
            'constraints': num_constraints,
            'iterations': iterations,
            'status': pulp.LpStatus[prob.status] if prob else 'ERROR'
        }
        print(f"  Temps: {duration:.4f}s, Obj: {obj_value}, Vars: {num_vars}, Contr: {num_constraints}, Iter: {iterations}")
    except Exception as e:
        print(f"  Erreur: {e}")
        results['DFJ_Iter'] = {'time': None, 'objective': None, 'variables': 0, 'constraints': 0, 'iterations': 0, 'status': 'ERROR'}
    
    # 4. DFJ Enum Relax (f=3)
    print("\n>>> Lancement DFJ Enum Relax...")
    try:
        if n > 15:
            print(f"  Ignoré: n={n} > 15")
            results['DFJ_Enum_Relax'] = {'time': None, 'objective': None, 'variables': 0, 'constraints': 0, 'status': 'SKIPPED'}
        else:
            prob, duration, x_vars = solve_dfj_enum_relax(n, dist_matrix)
            obj_value = get_objective_value(prob)
            num_vars, num_constraints = get_problem_stats(prob)
            results['DFJ_Enum_Relax'] = {
                'time': duration,
                'objective': obj_value,
                'variables': num_vars,
                'constraints': num_constraints,
                'status': pulp.LpStatus[prob.status] if prob else 'ERROR'
            }
            print(f"  Temps: {duration:.4f}s, Obj: {obj_value}, Vars: {num_vars}, Contr: {num_constraints}")
    except Exception as e:
        print(f"  Erreur: {e}")
        results['DFJ_Enum_Relax'] = {'time': None, 'objective': None, 'variables': 0, 'constraints': 0, 'status': 'ERROR'}
    
    # 5. DFJ Énumératif (f=2) - Seulement si n <= 15
    print("\n>>> Lancement DFJ Énumératif...")
    try:
        if n > 15:
            print(f"  Ignoré: n={n} > 15")
            results['DFJ_Enum'] = {'time': None, 'objective': None, 'variables': 0, 'constraints': 0, 'status': 'SKIPPED'}
        else:
            prob, duration, x_vars = solve_dfj_enum(n, dist_matrix)
            obj_value = get_objective_value(prob)
            num_vars, num_constraints = get_problem_stats(prob)
            results['DFJ_Enum'] = {
                'time': duration,
                'objective': obj_value,
                'variables': num_vars,
                'constraints': num_constraints,
                'status': pulp.LpStatus[prob.status] if prob else 'ERROR'
            }
            print(f"  Temps: {duration:.4f}s, Obj: {obj_value}, Vars: {num_vars}, Contr: {num_constraints}")
    except Exception as e:
        print(f"  Erreur: {e}")
        results['DFJ_Enum'] = {'time': None, 'objective': None, 'variables': 0, 'constraints': 0, 'status': 'ERROR'}
    
    return results


def format_value(val, decimals=6):
    """Fonction helper pour formater les valeurs numériques."""
    if val is None:
        return ''
    try:
        if isinstance(val, (int, float)):
            if isinstance(val, float):
                return f"{val:.{decimals}f}"
            return str(val)
        return str(val)
    except:
        return ''


def calculate_integrality_gap(integer_obj, relax_obj):
    """Calcule l'integrality gap en pourcentage."""
    if integer_obj is None or relax_obj is None:
        return ''
    if relax_obj == 0:
        return ''
    if integer_obj == 0: # Attention à la division par zéro
        return ''
    
    try:
        gap = (integer_obj - relax_obj) / integer_obj
        return f"{gap:.6f}"
    except:
        return ''


def write_results_to_csv(all_results, filename='results.csv'):
    """Écrit tous les résultats dans un fichier CSV avec une ligne par formulation."""
    if not all_results:
        print("Aucun résultat à écrire.")
        return
    
    # Définition des colonnes selon les consignes
    fieldnames = [
        'Nom de l\'instance',
        'Formulation utilisée',
        'Valeur objective de la solution entière',
        'Temps de résolution du programme linéaire en nombres entiers (en secondes)',
        'Valeur objective de la relaxation continue (si applicable)',
        'Temps de résolution de la relaxation continue (en secondes, si applicable)',
        'Integrality gap (pour MTZ et DFJ_enum)',
        'Nombre de variables',
        'Nombre de contraintes (pour DFJ itérative: nombre final)'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            instance_name = result['instance']
            
            # Ligne pour MTZ
            mtz_int_obj = result['MTZ']['objective']
            mtz_relax_obj = result['MTZ_Relax']['objective']
            mtz_gap = calculate_integrality_gap(mtz_int_obj, mtz_relax_obj)
            
            row_mtz = {
                'Nom de l\'instance': instance_name,
                'Formulation utilisée': 'MTZ',
                'Valeur objective de la solution entière': format_value(mtz_int_obj),
                'Temps de résolution du programme linéaire en nombres entiers (en secondes)': format_value(result['MTZ']['time']),
                'Valeur objective de la relaxation continue (si applicable)': format_value(mtz_relax_obj),
                'Temps de résolution de la relaxation continue (en secondes, si applicable)': format_value(result['MTZ_Relax']['time']),
                'Integrality gap (pour MTZ et DFJ_enum)': mtz_gap,
                'Nombre de variables': result['MTZ']['variables'],
                'Nombre de contraintes (pour DFJ itérative: nombre final)': result['MTZ']['constraints']
            }
            writer.writerow(row_mtz)
            
            # Ligne pour DFJ_enum (seulement si n <= 15)
            if result['n'] <= 15 and result['DFJ_Enum']['status'] != 'SKIPPED':
                dfj_enum_int_obj = result['DFJ_Enum']['objective']
                dfj_enum_relax_obj = result['DFJ_Enum_Relax']['objective']
                dfj_enum_gap = calculate_integrality_gap(dfj_enum_int_obj, dfj_enum_relax_obj)
                
                row_dfj_enum = {
                    'Nom de l\'instance': instance_name,
                    'Formulation utilisée': 'DFJ_enum',
                    'Valeur objective de la solution entière': format_value(dfj_enum_int_obj),
                    'Temps de résolution du programme linéaire en nombres entiers (en secondes)': format_value(result['DFJ_Enum']['time']),
                    'Valeur objective de la relaxation continue (si applicable)': format_value(dfj_enum_relax_obj),
                    'Temps de résolution de la relaxation continue (en secondes, si applicable)': format_value(result['DFJ_Enum_Relax']['time']),
                    'Integrality gap (pour MTZ et DFJ_enum)': dfj_enum_gap,
                    'Nombre de variables': result['DFJ_Enum']['variables'],
                    'Nombre de contraintes (pour DFJ itérative: nombre final)': result['DFJ_Enum']['constraints']
                }
                writer.writerow(row_dfj_enum)
            
            # Ligne pour DFJ_iter (pas de relaxation ni gap)
            row_dfj_iter = {
                'Nom de l\'instance': instance_name,
                'Formulation utilisée': 'DFJ_iter',
                'Valeur objective de la solution entière': format_value(result['DFJ_Iter']['objective']),
                'Temps de résolution du programme linéaire en nombres entiers (en secondes)': format_value(result['DFJ_Iter']['time']),
                'Valeur objective de la relaxation continue (si applicable)': '',  # Pas applicable pour DFJ_iter
                'Temps de résolution de la relaxation continue (en secondes, si applicable)': '',  # Pas applicable
                'Integrality gap (pour MTZ et DFJ_enum)': '',  # Pas applicable pour DFJ_iter
                'Nombre de variables': result['DFJ_Iter']['variables'],
                'Nombre de contraintes (pour DFJ itérative: nombre final)': result['DFJ_Iter']['constraints']
            }
            writer.writerow(row_dfj_iter)
    
    print(f"\n{'='*60}")
    print(f"Résultats écrits dans {filename}")
    print(f"{'='*60}")


def main():
    """Fonction principale du benchmark."""
    print("="*60)
    print("BENCHMARK TSP SOLVER")
    print("="*60)
    
    # Liste des fichiers dans instances/
    instances_dir = "instances"
    if not os.path.exists(instances_dir):
        print(f"Erreur: Le dossier {instances_dir} n'existe pas.")
        return
    
    # Récupérer tous les fichiers .txt
    instance_files = [f for f in os.listdir(instances_dir) if f.endswith('.txt')]
    instance_files.sort()  # Trier pour un ordre cohérent
    
    if not instance_files:
        print(f"Aucun fichier d'instance trouvé dans {instances_dir}")
        return
    
    print(f"\n{len(instance_files)} instance(s) trouvée(s):")
    for f in instance_files:
        print(f"  - {f}")
    
    # Exécuter le benchmark sur chaque instance
    all_results = []
    
    for instance_file in instance_files:
        result = run_benchmark_on_instance(instance_file)
        if result:
            all_results.append(result)
    
    # Écrire les résultats dans le CSV
    if all_results:
        write_results_to_csv(all_results)
        print(f"\nBenchmark terminé! {len(all_results)} instance(s) traitée(s).")
    else:
        print("\nAucun résultat à sauvegarder.")


if __name__ == "__main__":
    main()

