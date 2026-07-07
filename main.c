#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "fjsp.h"
#include "ga.h"
#include "rl.h"

/*
 * main.c - loop principal de SLGA + benchmark runner.
 *
 * Implementa:
 *   - SLGA original (fiel al paper, con las asunciones documentadas en
 *     fjsp.h / ga.h / rl.h)
 *   - Version MODIFICADA con las dos propuestas del usuario:
 *       (1) Mutacion adaptativa segun diversidad poblacional (usa d_star,
 *           Eq. 7, ya calculada en rl.c/rl.h)
 *       (2) Reintroduccion de individuos si no hay mejora significativa
 *           en las ultimas K generaciones
 *
 * NOTA sobre tiempo de desarrollo: dado el plazo del usuario, este archivo
 * prioriza tener resultados numericos funcionando por sobre una validacion
 * exhaustiva de cada pieza por separado (a diferencia de rl.c, que si se
 * valido con casos a mano paso a paso).
 */

#define POP_SIZE 30
#define MAX_GENERATIONS 200

/* --- Parametros de la modificacion propuesta --- */
#define STAGNATION_WINDOW 15        /* K generaciones para juzgar estancamiento */
#define STAGNATION_THRESHOLD 0.001  /* mejora relativa minima para NO considerar estancado */
#define REINTRO_FRACTION 0.20       /* fraccion de la poblacion a reintroducir si hay estancamiento */
#define ADAPTIVE_MUT_MAX_MULT 2.0   /* multiplicador maximo de Pm cuando diversidad es baja */

typedef struct {
    FJSPInstance *inst;
    Chromosome *pop;
    double *fitness;
    int pop_size;
} Population;

static void population_init(Population *p, FJSPInstance *inst, int pop_size)
{
    p->inst = inst;
    p->pop_size = pop_size;
    p->pop = (Chromosome *)malloc(sizeof(Chromosome) * pop_size);
    p->fitness = (double *)malloc(sizeof(double) * pop_size);

    for (int i = 0; i < pop_size; i++) {
        chromosome_random(inst, &p->pop[i]);
        p->fitness[i] = chromosome_fitness(inst, &p->pop[i]);
    }
}

static void population_free(Population *p)
{
    for (int i = 0; i < p->pop_size; i++) chromosome_free(&p->pop[i]);
    free(p->pop);
    free(p->fitness);
}

static int best_index(const double *fitness, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) if (fitness[i] > fitness[best]) best = i;
    return best;
}

static int worst_index_excluding(const double *fitness, int n, const int *excluded, int n_excluded)
{
    int worst = -1;
    for (int i = 0; i < n; i++) {
        int is_excluded = 0;
        for (int e = 0; e < n_excluded; e++) if (excluded[e] == i) { is_excluded = 1; break; }
        if (is_excluded) continue;
        if (worst == -1 || fitness[i] < fitness[worst]) worst = i;
    }
    return worst;
}

/*
 * Ejecuta SLGA sobre una instancia. use_modification activa las dos
 * mejoras propuestas por el usuario. Devuelve el mejor Cmax encontrado.
 */

static int run_slga(FJSPInstance *inst, int use_modification, unsigned int seed, int verbose)
{
    srand(seed);

    Population pop;
    population_init(&pop, inst, POP_SIZE);

    RLState rl;
    rl_init(&rl, pop.fitness, POP_SIZE, REWARD_USE_RC);

    int best_cmax_ever = decode_makespan(inst, &pop.pop[best_index(pop.fitness, POP_SIZE)]);
    double best_fitness_history[STAGNATION_WINDOW];
    for (int i = 0; i < STAGNATION_WINDOW; i++) best_fitness_history[i] = pop.fitness[best_index(pop.fitness, POP_SIZE)];

    for (int gen = 0; gen < MAX_GENERATIONS; gen++) {
        double pc, pm;
        rl_step(&rl, pop.fitness, &pc, &pm);

        /* --- Modificacion 1: mutacion adaptativa por diversidad --- */
        double pm_effective = pm;
        if (use_modification) {
            double d_star = compute_d_star(pop.fitness, rl.fit_gen1, POP_SIZE);
            /* d_star bajo (poblacion convergida/poco diversa) -> multiplicador alto (mas exploracion)
             * d_star ~1.0 o mayor (poblacion diversa) -> multiplicador ~1.0 (sin cambio) */
            double d_clamped = d_star > 1.0 ? 1.0 : (d_star < 0.0 ? 0.0 : d_star);
            double mult = 1.0 + (ADAPTIVE_MUT_MAX_MULT - 1.0) * (1.0 - d_clamped);
            pm_effective = pm * mult;
            if (pm_effective > 1.0) pm_effective = 1.0;
        }

        /* --- Generar nueva poblacion via seleccion + cruce + mutacion --- */
        Chromosome *new_pop = (Chromosome *)malloc(sizeof(Chromosome) * POP_SIZE);
        int filled = 0;
        while (filled < POP_SIZE) {
            int i1 = tournament_select(pop.fitness, POP_SIZE);
            int i2 = tournament_select(pop.fitness, POP_SIZE);

            double r = (double)rand() / ((double)RAND_MAX + 1.0);
            if (r < pc) {
                Chromosome c1, c2;
                crossover(inst, &pop.pop[i1], &pop.pop[i2], &c1, &c2);
                mutate(inst, &c1, pm_effective);
                if (filled < POP_SIZE) new_pop[filled++] = c1; else chromosome_free(&c1);
                if (filled < POP_SIZE) {
                    mutate(inst, &c2, pm_effective);
                    new_pop[filled++] = c2;
                } else {
                    chromosome_free(&c2);
                }
            } else {
                /* sin cruce: copiar padres directamente (con posible mutacion) */
                Chromosome c1, c2;
                chromosome_init(&c1, pop.pop[i1].length);
                chromosome_init(&c2, pop.pop[i2].length);
                memcpy(c1.os, pop.pop[i1].os, sizeof(int) * c1.length);
                memcpy(c1.ms, pop.pop[i1].ms, sizeof(int) * c1.length);
                memcpy(c2.os, pop.pop[i2].os, sizeof(int) * c2.length);
                memcpy(c2.ms, pop.pop[i2].ms, sizeof(int) * c2.length);
                mutate(inst, &c1, pm_effective);
                mutate(inst, &c2, pm_effective);
                if (filled < POP_SIZE) new_pop[filled++] = c1; else chromosome_free(&c1);
                if (filled < POP_SIZE) new_pop[filled++] = c2; else chromosome_free(&c2);
            }
        }

        /* Elitismo: si el mejor individuo anterior es mejor que todo lo nuevo, lo conservamos */
        int old_best = best_index(pop.fitness, POP_SIZE);
        double new_fitness[POP_SIZE];
        for (int i = 0; i < POP_SIZE; i++) new_fitness[i] = chromosome_fitness(inst, &new_pop[i]);
        int new_best = best_index(new_fitness, POP_SIZE);
        if (pop.fitness[old_best] > new_fitness[new_best]) {
            chromosome_free(&new_pop[new_best]);
            chromosome_init(&new_pop[new_best], pop.pop[old_best].length);
            memcpy(new_pop[new_best].os, pop.pop[old_best].os, sizeof(int) * new_pop[new_best].length);
            memcpy(new_pop[new_best].ms, pop.pop[old_best].ms, sizeof(int) * new_pop[new_best].length);
            new_fitness[new_best] = pop.fitness[old_best];
        }

        /* reemplazar poblacion vieja */
        for (int i = 0; i < POP_SIZE; i++) chromosome_free(&pop.pop[i]);
        free(pop.pop);
        free(pop.fitness);
        pop.pop = new_pop;
        pop.fitness = (double *)malloc(sizeof(double) * POP_SIZE);
        memcpy(pop.fitness, new_fitness, sizeof(double) * POP_SIZE);

        int cur_best_idx = best_index(pop.fitness, POP_SIZE);
        int cur_cmax = decode_makespan(inst, &pop.pop[cur_best_idx]);
        if (cur_cmax < best_cmax_ever) best_cmax_ever = cur_cmax;

        /* --- Modificacion 2: reintroduccion si hay estancamiento --- */
        if (use_modification) {
            int hist_idx = gen % STAGNATION_WINDOW;
            double improvement = (pop.fitness[cur_best_idx] - best_fitness_history[hist_idx]) /
                                  (best_fitness_history[hist_idx] > 0 ? best_fitness_history[hist_idx] : 1.0);
            if (gen >= STAGNATION_WINDOW && improvement < STAGNATION_THRESHOLD) {
                int n_reintro = (int)(POP_SIZE * REINTRO_FRACTION);
                for (int r = 0; r < n_reintro; r++) {
                    /* elegir el peor individuo actual (excluyendo el mejor, por elitismo) */
                    int excluded[1] = { best_index(pop.fitness, POP_SIZE) };
                    int worst = worst_index_excluding(pop.fitness, POP_SIZE, excluded, 1);
                    if (worst == -1) break;
                    chromosome_free(&pop.pop[worst]);
                    chromosome_random(inst, &pop.pop[worst]);
                    pop.fitness[worst] = chromosome_fitness(inst, &pop.pop[worst]);
                }
            }
            best_fitness_history[hist_idx] = pop.fitness[cur_best_idx];
        }

        if (verbose && (gen % 20 == 0 || gen == MAX_GENERATIONS - 1)) {
            printf("  gen %3d: best Cmax so far = %d\n", gen, best_cmax_ever);
        }
    }

    population_free(&pop);
    rl_free(&rl);
    return best_cmax_ever;
}

static void run_benchmark(const char *filepath, const char *label)
{
    FJSPInstance inst;
    if (fjsp_load_from_file(filepath, &inst) != 0) {
        printf("No se pudo cargar %s\n", filepath);
        return;
    }

    printf("=== %s (%s) ===\n", label, filepath);

    int n_seeds = 5;
    int baseline_results[5], modified_results[5];

    for (int s = 0; s < n_seeds; s++) {
        unsigned int seed = 1000 + s;
        baseline_results[s] = run_slga(&inst, 0, seed, 0);
        modified_results[s] = run_slga(&inst, 1, seed, 0);
    }

    double base_avg = 0, mod_avg = 0;
    int base_best = baseline_results[0], mod_best = modified_results[0];
    for (int s = 0; s < n_seeds; s++) {
        base_avg += baseline_results[s];
        mod_avg += modified_results[s];
        if (baseline_results[s] < base_best) base_best = baseline_results[s];
        if (modified_results[s] < mod_best) mod_best = modified_results[s];
    }
    base_avg /= n_seeds;
    mod_avg /= n_seeds;

    printf("  SLGA original : mejor Cmax=%d, promedio=%.1f (%d corridas)\n", base_best, base_avg, n_seeds);
    printf("  SLGA modificado: mejor Cmax=%d, promedio=%.1f (%d corridas)\n", mod_best, mod_avg, n_seeds);
    double improvement_pct = 100.0 * (base_avg - mod_avg) / base_avg;
    printf("  Diferencia promedio: %.2f%% %s\n\n", improvement_pct >= 0 ? improvement_pct : -improvement_pct,
           improvement_pct >= 0 ? "(modificado mejor)" : "(modificado peor)");

    fjsp_free(&inst);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Uso: %s <archivo1.txt> [archivo2.txt] ...\n", argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        run_benchmark(argv[i], argv[i]);
    }

    return 0;
}