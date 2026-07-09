#ifndef GA_H
#define GA_H

#include "fjsp.h"

/*
 * NOTA: estos operadores son una implementacion PRAGMATICA de lo descrito/inferido del paper:
 *   - Cruce OS: POX (Precedence Operation Crossover)
 *   - Cruce MS: dos puntos, por posicion (segun texto del paper)
 *   - Mutacion: swap en OS, reasignacion aleatoria valida en MS
 *   - Seleccion: torneo binario
 *   - Reemplazo: elitismo simple (se conserva el mejor individuo)
 *
 * IMPORTANTE - simplificacion pragmatica: como el cruce POX reordena el OS,
 * el cruce de MS por posicion puede dejar una maquina invalida para la
 * operacion que terminó en esa posicion. Se resuelve con repair_chromosome(),
 * que reemplaza cualquier gen de MS invalido por una maquina valida al azar.
 * Esto es una desviacion pragmatica del texto original del paper (que no
 * detalla como resuelve esta inconsistencia), adoptada por restriccion de
 * tiempo del proyecto.
 */

/* Seleccion por torneo binario: elige 2 individuos al azar de la poblacion
 * y devuelve el indice del que tiene mayor fitness. */
int tournament_select(const double *fitness, int pop_size);

/*
 * Cruce OS (POX) + cruce MS (dos puntos), combinados. p1, p2 son los padres;
 * c1, c2 reciben los hijos (deben estar SIN inicializar, se inicializan aca).
 */
void crossover(const FJSPInstance *inst, const Chromosome *p1, const Chromosome *p2, Chromosome *c1, Chromosome *c2);

/* Mutacion in-place: con probabilidad pm, aplica swap en OS y/o reasignacion en MS */
void mutate(const FJSPInstance *inst, Chromosome *c, double pm);

/* Corrige genes de MS invalidos (ver nota de arriba). Se llama despues de crossover/mutate. */
void repair_chromosome(const FJSPInstance *inst, Chromosome *c);

long repair_get_total_genes_checked(void);
long repair_get_total_genes_repaired(void);
void repair_reset_counters(void);
 
/*
 * Mutacion ALTERNATIVA: Pm se evalua UNA sola vez por cromosoma (no por
 * gen). Si ocurre, hace UN swap en OS y UN intento de swap Lei-MS.
 * Para comparar contra mutate() (Pm por gen)
 */
// Experimental
void mutate_once_per_chromosome(const FJSPInstance *inst, Chromosome *c, double pm);

#endif /* GA_H */