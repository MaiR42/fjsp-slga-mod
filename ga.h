#ifndef GA_H
#define GA_H

#include "fjsp.h"


/*
 * NOTA: estos operadores son una implementacion PRAGMATICA de lo descrito/inferido del paper:
 *   - Cruce OS+MS: POX (Precedence Operation Crossover) con MS heredado
 *     ACOPLADO al mismo mecanismo (ver nota detallada en pox_build_child,
 *     ga.c). El paper describe el cruce de MS como "dos puntos"
 *     independiente del OS (Fig. 7, Zhang et al. 2011), pero eso genera
 *     inconsistencias posicionales tras el reordenamiento de POX (medido
 *     empiricamente: hasta ~40% de los genes MS heredados se perdian en
 *     reparacion aleatoria). La version acoplada implementada aca es una
 *     mejora practica sobre lo que describe el paper, no una desviacion
 *     por descuido: elimina la perdida de informacion genetica sin
 *     cambiar el comportamiento en instancias donde nunca se necesitaba
 *     reparar (ej. Kacem, flexibilidad total).
 *   - Mutacion: swap en OS (confirmado, Lei 2012), swap validity-preserving
 *     en MS (confirmado, Lei 2012)
 *   - Seleccion: torneo binario para elegir padres (no confirmado del paper)
 *   - Reemplazo: elite retention strategy (confirmado, Sec 3.1.4)
 *
 * repair_chromosome() se conserva como red de seguridad para casos borde,
 * pero con el cruce acoplado ya no destruye informacion genetica en el
 * caso normal (medido: 0% de reparaciones tras el cambio)
 */

/* Seleccion por torneo binario: elige 2 individuos al azar de la poblacion
 * y devuelve el indice del que tiene mayor fitness. */
int tournament_select(const double *fitness, int pop_size);
int tournament_select_k(const double *fitness, int pop_size, int k);
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