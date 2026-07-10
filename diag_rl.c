#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fjsp.h"
#include "ga.h"
#include "rl.h"

/*
 * diag_rl.c - Herramienta de diagnostico del modulo RL (SEPARADA de main.c,
 * no se usa en las corridas normales). Corre una simulacion real de SLGA
 * y logea, generacion por generacion, todo lo necesario para diagnosticar
 * si el agente RL esta aprendiendo algo util:
 *
 *   1. Estado: f*, d*, m*, S*, estado discretizado (0-19)
 *   2. Accion elegida, Pc, Pm resultantes + histograma final de acciones
 *   3. Reward: rc, rm, reward final usado para actualizar Q
 *   4. Fila de la tabla Q del estado actual cada 50 generaciones,
 *      y aviso explicito cuando cambia SARSA -> Q-learning
 *
 * Uso: ./diag_rl <archivo.txt> [max_generaciones] [pop_size]
 * Salida: diagnostico_rl.txt (formato tabulado, facil de parsear)
 */

#define DEFAULT_MAX_GEN 500
#define DEFAULT_POP_SIZE 50

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "mk06.txt";
    int max_gen = argc > 2 ? atoi(argv[2]) : DEFAULT_MAX_GEN;
    int pop_size = argc > 3 ? atoi(argv[3]) : DEFAULT_POP_SIZE;

    FJSPInstance inst;
    if (fjsp_load_from_file(path, &inst) != 0) {
        printf("Error cargando %s\n", path);
        return 1;
    }

    srand(1000);

    Chromosome *pop = (Chromosome *)malloc(sizeof(Chromosome) * pop_size);
    double *fitness = (double *)malloc(sizeof(double) * pop_size);
    for (int i = 0; i < pop_size; i++) {
        chromosome_random(&inst, &pop[i]);
        fitness[i] = chromosome_fitness_active(&inst, &pop[i]);
    }

    RLState rl;
    rl_init(&rl, fitness, pop_size, REWARD_USE_RC_PLUS_RM);

    long action_histogram[N_ACTIONS] = {0};
    action_histogram[rl.a_t]++;

    RLMode last_mode = rl_get_mode(rl.n_ti);

    FILE *log = fopen("diagnostico_rl.txt", "w");
    fprintf(log, "# Instancia: %s | pop_size=%d | max_gen=%d\n", path, pop_size, max_gen);
    fprintf(log, "# reward_mode = REWARD_USE_RC_PLUS_RM (rc+rm)\n\n");
    fprintf(log, "gen\tf*\td*\tm*\tS*\tstate\taction\tPc\tPm\trc\trm\treward\tmode\n");

    for (int gen = 0; gen < max_gen; gen++) {
        /* --- Replicamos rl_step manualmente para poder loguear cada paso interno --- */
        double f_star = compute_f_star(fitness, rl.fit_gen1, pop_size);
        double d_star = compute_d_star(fitness, rl.fit_gen1, pop_size);
        double m_star = compute_m_star(fitness, rl.fit_gen1, pop_size);
        double S_star = compute_S_star(f_star, d_star, m_star);
        int s_next = discretize_state(S_star);

        double rc = compute_reward_rc(fitness, rl.fit_prev, pop_size);
        double rm = compute_reward_rm(fitness, rl.fit_prev, pop_size);
        double reward = rc + rm;

        RLMode mode = rl_get_mode(rl.n_ti);
        if (mode != last_mode) {
            fprintf(log, "### Generacion %d: Modo %s -> %s\n", gen,
                    last_mode == RL_MODE_SARSA ? "SARSA" : "QLEAR",
                    mode == RL_MODE_SARSA ? "SARSA" : "QLEAR");
            last_mode = mode;
        }

        int a_next;
        if (mode == RL_MODE_SARSA) {
            a_next = policy_epsilon_greedy_select(s_next);
            qtable_update_sarsa(rl.s_t, rl.a_t, reward, s_next, a_next);
        } else {
            qtable_update_qlearning(rl.s_t, rl.a_t, reward, s_next);
            a_next = policy_epsilon_greedy_select(s_next);
        }

        double pc, pm;
        action_get_params(a_next, &pc, &pm);

        rl.s_t = s_next;
        rl.a_t = a_next;
        rl.n_ti += 1;
        memcpy(rl.fit_prev, fitness, sizeof(double) * pop_size);

        action_histogram[a_next]++;

        fprintf(log, "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s\n",
                gen, f_star, d_star, m_star, S_star, s_next, a_next, pc, pm, rc, rm, reward,
                mode == RL_MODE_SARSA ? "SARSA" : "QLEAR");

        if (gen % 50 == 0) {
            fprintf(log, "### Q[estado=%d] = ", s_next);
            for (int a = 0; a < N_ACTIONS; a++) fprintf(log, "%.5f ", Q[s_next][a]);
            fprintf(log, "\n");
        }

        /* --- Avanzar la poblacion del GA con el Pc/Pm obtenido (misma logica que main.c) --- */
        Chromosome *newpop = (Chromosome *)malloc(sizeof(Chromosome) * pop_size);
        int filled = 0;
        while (filled < pop_size) {
            int i1 = tournament_select(fitness, pop_size);
            int i2 = tournament_select(fitness, pop_size);
            double r = (double)rand() / ((double)RAND_MAX + 1.0);
            Chromosome c1, c2;
            if (r < pc) {
                crossover(&inst, &pop[i1], &pop[i2], &c1, &c2);
            } else {
                chromosome_init(&c1, pop[i1].length);
                chromosome_init(&c2, pop[i2].length);
                memcpy(c1.os, pop[i1].os, sizeof(int) * c1.length);
                memcpy(c1.ms, pop[i1].ms, sizeof(int) * c1.length);
                memcpy(c2.os, pop[i2].os, sizeof(int) * c2.length);
                memcpy(c2.ms, pop[i2].ms, sizeof(int) * c2.length);
            }
            mutate(&inst, &c1, pm);
            mutate(&inst, &c2, pm);
            newpop[filled++] = c1;
            if (filled < pop_size) newpop[filled++] = c2; else chromosome_free(&c2);
        }

        /* elite retention (misma logica que main.c) */
        int combined_size = pop_size * 2;
        Chromosome *combined_pop = (Chromosome *)malloc(sizeof(Chromosome) * combined_size);
        double *combined_fit = (double *)malloc(sizeof(double) * combined_size);
        for (int i = 0; i < pop_size; i++) { combined_pop[i] = pop[i]; combined_fit[i] = fitness[i]; }
        for (int i = 0; i < pop_size; i++) {
            combined_pop[pop_size + i] = newpop[i];
            combined_fit[pop_size + i] = chromosome_fitness_active(&inst, &newpop[i]);
        }
        free(newpop);

        int *selected = (int *)malloc(sizeof(int) * pop_size);
        int *taken = (int *)calloc(combined_size, sizeof(int));
        for (int s = 0; s < pop_size; s++) {
            int best = -1;
            for (int i = 0; i < combined_size; i++) {
                if (taken[i]) continue;
                if (best == -1 || combined_fit[i] > combined_fit[best]) best = i;
            }
            selected[s] = best;
            taken[best] = 1;
        }
        Chromosome *survivors = (Chromosome *)malloc(sizeof(Chromosome) * pop_size);
        double *survivor_fit = (double *)malloc(sizeof(double) * pop_size);
        for (int s = 0; s < pop_size; s++) {
            survivors[s] = combined_pop[selected[s]];
            survivor_fit[s] = combined_fit[selected[s]];
        }
        for (int i = 0; i < combined_size; i++) if (!taken[i]) chromosome_free(&combined_pop[i]);
        free(combined_pop); free(combined_fit); free(selected); free(taken);

        /* OJO: no liberar los cromosomas de pop[] individualmente aca --
         * sus punteros .os/.ms ya fueron liberados arriba (si no fueron
         * seleccionados) o transferidos a survivors (si sí lo fueron).
         * Liberar de nuevo seria double-free. Solo liberamos el array
         * contenedor. */
        free(pop);
        free(fitness);
        pop = survivors;
        fitness = survivor_fit;
    }

    fprintf(log, "\n=== HISTOGRAMA DE ACCIONES (total %d generaciones) ===\n", max_gen);
    for (int a = 0; a < N_ACTIONS; a++) {
        fprintf(log, "Accion %d: %ld veces (%.1f%%)\n", a, action_histogram[a],
                100.0 * action_histogram[a] / max_gen);
    }

    fclose(log);
    printf("Diagnostico completo escrito en diagnostico_rl.txt\n");

    for (int i = 0; i < pop_size; i++) chromosome_free(&pop[i]);
    free(pop);
    free(fitness);
    rl_free(&rl);
    fjsp_free(&inst);
    return 0;
}