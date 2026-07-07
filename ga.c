#include <stdlib.h>
#include "ga.h"

int tournament_select(const double *fitness, int pop_size)
{
    int a = rand() % pop_size;
    int b = rand() % pop_size;
    return (fitness[a] >= fitness[b]) ? a : b;
}

/* Auxiliar POX: arma un hijo copiando de keep_parent las posiciones cuyo
 * job pertenece al conjunto "kept" (J1 si keep_is_j1, J2 si no), y llena
 * las posiciones restantes con los jobs del conjunto complementario, en
 * el orden en que aparecen en fill_parent. */
static void pox_build_child(const int *in_j1, int num_jobs, int keep_is_j1,
                             const Chromosome *keep_parent, const Chromosome *fill_parent,
                             Chromosome *child)
{
    (void)num_jobs;
    int len = keep_parent->length;

    for (int i = 0; i < len; i++) {
        int job = keep_parent->os[i];
        int job_in_j1 = in_j1[job];
        int keep = keep_is_j1 ? job_in_j1 : !job_in_j1;
        child->os[i] = keep ? job : -1; /* -1 = vacante, se llena despues */
    }

    int idx = 0;
    for (int i = 0; i < len; i++) {
        if (child->os[i] != -1) continue;
        while (idx < len) {
            int job = fill_parent->os[idx];
            int job_in_j1 = in_j1[job];
            int should_fill = keep_is_j1 ? !job_in_j1 : job_in_j1; /* complemento */
            idx++;
            if (should_fill) {
                child->os[i] = job;
                break;
            }
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

    /* --- Cruce MS: dos puntos, por posicion --- */
    int cut1 = rand() % len;
    int cut2 = rand() % len;
    if (cut1 > cut2) { int t = cut1; cut1 = cut2; cut2 = t; }

    for (int i = 0; i < len; i++) {
        if (i >= cut1 && i <= cut2) {
            c1->ms[i] = p2->ms[i];
            c2->ms[i] = p1->ms[i];
        } else {
            c1->ms[i] = p1->ms[i];
            c2->ms[i] = p2->ms[i];
        }
    }

    /* El cruce de OS (POX) reordena las operaciones -> el MS heredado por
     * posicion puede quedar invalido para la operacion que cayo en esa
     * posicion. Se corrige a continuacion (ver nota en ga.h). */
    repair_chromosome(inst, c1);
    repair_chromosome(inst, c2);
}

void mutate(const FJSPInstance *inst, Chromosome *c, double pm)
{
    int len = c->length;

    /* Mutacion OS: swap mutation, gen a gen con probabilidad pm */
    for (int i = 0; i < len; i++) {
        double r = (double)rand() / ((double)RAND_MAX + 1.0);
        if (r < pm) {
            int j = rand() % len;
            int tmp = c->os[i];
            c->os[i] = c->os[j];
            c->os[j] = tmp;
        }
    }

    /* Mutacion MS: reasignacion a maquina valida aleatoria, gen a gen con probabilidad pm */
    for (int i = 0; i < len; i++) {
        double r = (double)rand() / ((double)RAND_MAX + 1.0);
        if (r < pm) {
            int job = c->os[i];
            int op_idx = fjsp_get_op_index(c->os, i);
            Operation *op = &inst->jobs[job].operations[op_idx];
            int choice = rand() % op->num_options;
            c->ms[i] = op->options[choice].machine;
        }
    }

    /* La mutacion de OS (swap) tambien puede invalidar el alineamiento
     * MS/OS en las posiciones intercambiadas (mismo motivo que en crossover). */
    repair_chromosome(inst, c);
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
        if (!valid) {
            int choice = rand() % op->num_options;
            c->ms[i] = op->options[choice].machine;
        }
    }
}