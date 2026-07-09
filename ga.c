#include <stdlib.h>
#include "ga.h"
// instrumentacion de repair_chromosome  
static long g_repair_total_checked = 0;
static long g_repair_total_repaired = 0;

int tournament_select(const double *fitness, int pop_size)
{
    int a = rand() % pop_size;
    int b = rand() % pop_size;
    return (fitness[a] >= fitness[b]) ? a : b;
}

/* Auxiliar POX (VERSION MEJORADA): arma un hijo copiando de keep_parent
 * las posiciones cuyo job pertenece al conjunto "kept", y llena las
 * posiciones restantes con los jobs del conjunto complementario, en el
 * orden en que aparecen en fill_parent. */
static void pox_build_child(const int *in_j1, int num_jobs, int keep_is_j1, const Chromosome *keep_parent, const Chromosome *fill_parent, Chromosome *child)
{
    (void)num_jobs;
    int len = keep_parent->length;

    for (int i = 0; i < len; i++) {
        int job = keep_parent->os[i];
        int job_in_j1 = in_j1[job];
        int keep = keep_is_j1 ? job_in_j1 : !job_in_j1;
        if (keep) {
            child->os[i] = job;
            child->ms[i] = keep_parent->ms[i]; /* <-- NUEVO: MS acoplado (antes: cruce de 2 puntos aparte) */
        } else {
            child->os[i] = -1; /* -1 = vacante, se llena despues */
        }
    }

    int idx = 0;
    for (int i = 0; i < len; i++) {
        if (child->os[i] != -1) continue;
        while (idx < len) {
            int job = fill_parent->os[idx];
            int job_in_j1 = in_j1[job];
            int should_fill = keep_is_j1 ? !job_in_j1 : job_in_j1; /* complemento */
            if (should_fill) {
                child->os[i] = job;
                child->ms[i] = fill_parent->ms[idx]; /* <-- NUEVO: MS acoplado (antes: cruce de 2 puntos aparte) */
                idx++;
                break;
            }
            idx++;
        }
    }
}

void crossover(const FJSPInstance *inst, const Chromosome *p1, const Chromosome *p2, Chromosome *c1, Chromosome *c2)
{
    int len = p1->length;
    chromosome_init(c1, len);
    chromosome_init(c2, len);

    /* --- Cruce OS: POX --- */
    int *in_j1 = (int *)malloc(sizeof(int) * inst->num_jobs);
    for (int j = 0; j < inst->num_jobs; j++) in_j1[j] = rand() % 2;

    pox_build_child(in_j1, inst->num_jobs, 1, p1, p2, c1); /* c1: kept=J1 de p1, relleno=J2 de p2 */
    pox_build_child(in_j1, inst->num_jobs, 0, p2, p1, c2); /* c2: kept=J2 de p2, relleno=J1 de p1 */
    free(in_j1);

    /* repair_chromosome ahora es solo una red de seguridad: con MS acoplado
     * al POX, cada gen heredado ya es valido por construccion*/
    repair_chromosome(inst, c1);
    repair_chromosome(inst, c2);
}

/*
 * Mutacion MS segun Lei (2012), CONFIRMADO por texto (citado tambien en
 * el paper principal para "swap mutation"): intercambia las maquinas
 * asignadas entre dos operaciones i y k, pero SOLO si el intercambio es
 * valido en ambos sentidos (la maquina de i sirve para la operacion de k,
 * y la maquina de k sirve para la operacion de i). Garantiza validez por
 * construccion, sin necesitar reparacion posterior para este paso.
 * Requiere que la operacion tenga mas de una maquina posible (|O_ij|>1).
 */
static int try_ms_swap_at(const FJSPInstance *inst, Chromosome *c, int i)
{
    int job_i = c->os[i];
    int op_idx_i = fjsp_get_op_index(c->os, i);
    Operation *op_i = &inst->jobs[job_i].operations[op_idx_i];

    if (op_i->num_options <= 1) return 0; /* no mutable: una sola opcion de maquina */

    int m_ij = c->ms[i];
    int *theta = (int *)malloc(sizeof(int) * c->length);
    int theta_size = 0;

    for (int k = 0; k < c->length; k++) {
        if (k == i) continue;
        int job_k = c->os[k];
        int op_idx_k = fjsp_get_op_index(c->os, k);
        Operation *op_k = &inst->jobs[job_k].operations[op_idx_k];

        int m_ij_valid_for_k = 0;
        for (int a = 0; a < op_k->num_options; a++)
            if (op_k->options[a].machine == m_ij) { m_ij_valid_for_k = 1; break; }
        if (!m_ij_valid_for_k) continue;

        int m_gh = c->ms[k];
        int m_gh_valid_for_i = 0;
        for (int a = 0; a < op_i->num_options; a++)
            if (op_i->options[a].machine == m_gh) { m_gh_valid_for_i = 1; break; }
        if (!m_gh_valid_for_i) continue;

        theta[theta_size++] = k;
    }

    int did_swap = 0;
    if (theta_size > 0) {
        int choice = theta[rand() % theta_size];
        int tmp = c->ms[i];
        c->ms[i] = c->ms[choice];
        c->ms[choice] = tmp;
        did_swap = 1;
    }
    free(theta);
    return did_swap;
}

void mutate(const FJSPInstance *inst, Chromosome *c, double pm)
{
    int len = c->length;

    /* Mutacion OS: swap mutation, gen a gen con probabilidad pm.
     * (Puede desalinear MS/OS en las posiciones intercambiadas -> se
     * repara antes de aplicar la mutacion de MS, para que el swap de
     * Lei (2012) de abajo parta de un cromosoma valido.) */
    for (int i = 0; i < len; i++) {
        double r = (double)rand() / ((double)RAND_MAX + 1.0);
        if (r < pm) {
            int j = rand() % len;
            int tmp = c->os[i];
            c->os[i] = c->os[j];
            c->os[j] = tmp;
        }
    }
    repair_chromosome(inst, c);

    /* Mutacion MS: swap mutation de Lei (2012), gen a gen con probabilidad pm */
    for (int i = 0; i < len; i++) {
        double r = (double)rand() / ((double)RAND_MAX + 1.0);
        if (r < pm) {
            try_ms_swap_at(inst, c, i);
        }
    }
}

void repair_chromosome(const FJSPInstance *inst, Chromosome *c)
{
    for (int i = 0; i < c->length; i++) {
        int job = c->os[i];
        int op_idx = fjsp_get_op_index(c->os, i);
        Operation *op = &inst->jobs[job].operations[op_idx];

        int valid = 0;
        for (int k = 0; k < op->num_options; k++) {
            if (op->options[k].machine == c->ms[i]) { valid = 1; break; }
        }
        g_repair_total_checked++;
        if (!valid) {
            g_repair_total_repaired++;
            int choice = rand() % op->num_options;
            c->ms[i] = op->options[choice].machine;
        }
    }
}

/* getters/reset de la instrumentacion de arriba */
long repair_get_total_genes_checked(void) { return g_repair_total_checked; }
long repair_get_total_genes_repaired(void) { return g_repair_total_repaired; }
void repair_reset_counters(void) { g_repair_total_checked = 0; g_repair_total_repaired = 0; }

// Experimental DEBUG
void mutate_once_per_chromosome(const FJSPInstance *inst, Chromosome *c, double pm)
{
    int len = c->length;
    double r = (double)rand() / ((double)RAND_MAX + 1.0);
    if (r >= pm) return; /* no muta este cromosoma */

    /* Un swap en OS */
    int i = rand() % len;
    int j = rand() % len;
    int tmp = c->os[i];
    c->os[i] = c->os[j];
    c->os[j] = tmp;
    repair_chromosome(inst, c); /* el swap de OS puede desalinear MS */

    /* Un intento de swap Lei-MS (hasta 5 intentos en posiciones distintas) */
    for (int attempt = 0; attempt < 5; attempt++) {
        int pos = rand() % len;
        if (try_ms_swap_at(inst, c, pos)) break;
    }
}