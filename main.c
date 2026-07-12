#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
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
 */

// ============================ Parametros ============================
#define POP_SIZE 30 // Tamaño de la poblacion
#define MAX_GENERATIONS 3000 // Maximo de generaciones/iteraciones

#define N_SEEDS 10 // Cantidad de veces que se repite la ejecucion, con distintas seeds. Para obtener ASL

#define TOURNAMENT_SIZE 2 // Definir en 2 para torneo binario

#define DIVERSITY_RESERVE_FRACTION 0.15 // Porcentaje de la poblacion (individuos aleatorios) que se mantiene en la siguiente generacion 

#define STAGNATION_THRESHOLD 0.02   // Mejora relativa minima para no considerar estancamiento
#define ADAPTIVE_MUT_MAX_MULT 1.2   // Multiplicador maximo de Pm cuando diversidad es baja



// Implementacion
static int compute_dynamic_stagnation_window(int total_ops)
{
    int w = 30 + total_ops / 3;
    if (w > 150) w = 150; // Tope temporal (para que ventana no sea exagerada o que sobrepase el MAX_GENERATIONS). Ajustado a operaciones de Brandimarte
    return w;
}

static double compute_dynamic_reintro_fraction(int total_ops)
{
    double base = 0.03;
    double scale = 100.0 / (double)total_ops;
    if (scale > 1.0) scale = 1.0;
    double frac = base * scale;
    if (frac < 0.01) frac = 0.01;
    return frac;
}

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
        p->fitness[i] = chromosome_fitness_active(inst, &p->pop[i]);
    }
}

static void population_free(Population *p)
{
    for (int i = 0; i < p->pop_size; i++) chromosome_free(&p->pop[i]);
    free(p->pop);
    free(p->fitness);
}

static int best_index(const double *fitness, int n) // Indice del mejor fitness
{
    int best = 0;
    for (int i = 1; i < n; i++) if (fitness[i] > fitness[best]) best = i;
    return best;
}

static int worst_index_excluding(const double *fitness, int n, const int *excluded, int n_excluded) // Indice del peor fitness
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
 * mejoras propuestas. Devuelve el mejor Cmax encontrado.
 */
static int run_slga(FJSPInstance *inst, int use_modification, unsigned int seed, int verbose)
{
    srand(seed);

    int total_ops = fjsp_total_operations(inst);
    int stagnation_window = compute_dynamic_stagnation_window(total_ops);
    double reintro_fraction = compute_dynamic_reintro_fraction(total_ops);

    Population pop;
    population_init(&pop, inst, POP_SIZE);

    RLState rl;
    rl_init(&rl, pop.fitness, POP_SIZE, REWARD_USE_RC_PLUS_RM);

    int best_cmax_ever = decode_makespan_active(inst, &pop.pop[best_index(pop.fitness, POP_SIZE)]);
    double *best_fitness_history = (double *)malloc(sizeof(double) * stagnation_window);
    for (int i = 0; i < stagnation_window; i++) best_fitness_history[i] = pop.fitness[best_index(pop.fitness, POP_SIZE)];

    for (int gen = 0; gen < MAX_GENERATIONS; gen++) {
        double pc, pm;
        rl_step(&rl, pop.fitness, &pc, &pm);

        /* --- Modificacion 1: mutacion adaptativa en base a diversidad --- */
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

        /* --- Generar nueva poblacion via: seleccion + cruce + mutacion --- */
        Chromosome *new_pop = (Chromosome *)malloc(sizeof(Chromosome) * POP_SIZE);
        int filled = 0;
        while (filled < POP_SIZE) {
            // Elegir padres para cruce
            int i1 = tournament_select_k(pop.fitness, POP_SIZE, TOURNAMENT_SIZE);
            int i2 = tournament_select_k(pop.fitness, POP_SIZE, TOURNAMENT_SIZE);

            double r = (double)rand() / ((double)RAND_MAX + 1.0);
            // Cruce (con posible mutacion)
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
                // Sin cruce: copiar padres directamente (con posible mutacion)
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

        /* --- Seleccion/Reemplazo: "elite retention strategy" + elitismo suavizado: se
         * reserva una fraccion de posiciones para diversidad (sobrevivientes
         * al azar), en vez de quedarse siempre con los N mejores estrictos. */
        int combined_size = POP_SIZE * 2;
        Chromosome *combined_pop = (Chromosome *)malloc(sizeof(Chromosome) * combined_size);
        double *combined_fit = (double *)malloc(sizeof(double) * combined_size);

        for (int i = 0; i < POP_SIZE; i++) {
            combined_pop[i] = pop.pop[i];
            combined_fit[i] = pop.fitness[i];
        }
        for (int i = 0; i < POP_SIZE; i++) {
            combined_pop[POP_SIZE + i] = new_pop[i];
            combined_fit[POP_SIZE + i] = chromosome_fitness_active(inst, &new_pop[i]);
        }
        free(new_pop); /* los Chromosome individuales ya se copiaron a combined_pop, no liberar sus arrays */

        int strict_count = (int)(POP_SIZE * (1.0 - DIVERSITY_RESERVE_FRACTION));
        int *selected = (int *)malloc(sizeof(int) * POP_SIZE);
        int *taken = (int *)calloc(combined_size, sizeof(int));

        for (int s = 0; s < strict_count; s++) {
            int best = -1;
            for (int i = 0; i < combined_size; i++) {
                if (taken[i]) continue;
                if (best == -1 || combined_fit[i] > combined_fit[best]) best = i;
            }
            selected[s] = best;
            taken[best] = 1;
        }
        /* Resto de las posiciones: al azar entre los NO tomados (diversidad) */
        for (int s = strict_count; s < POP_SIZE; s++) {
            int idx;
            do {
                idx = rand() % combined_size;
            } while (taken[idx]);
            selected[s] = idx;
            taken[idx] = 1;
        }

        Chromosome *survivors = (Chromosome *)malloc(sizeof(Chromosome) * POP_SIZE);
        double *survivor_fit = (double *)malloc(sizeof(double) * POP_SIZE);
        for (int s = 0; s < POP_SIZE; s++) {
            survivors[s] = combined_pop[selected[s]];
            survivor_fit[s] = combined_fit[selected[s]];
        }
        /* liberar los que NO sobrevivieron */
        for (int i = 0; i < combined_size; i++) {
            if (!taken[i]) chromosome_free(&combined_pop[i]);
        }
        free(combined_pop);
        free(combined_fit);
        free(selected);
        free(taken);

        free(pop.pop);
        free(pop.fitness);
        pop.pop = survivors;
        pop.fitness = survivor_fit;

        int cur_best_idx = best_index(pop.fitness, POP_SIZE);
        int cur_cmax = decode_makespan_active(inst, &pop.pop[cur_best_idx]);
        if (cur_cmax < best_cmax_ever) best_cmax_ever = cur_cmax;

        /* --- Modificacion 2: reintroduccion si hay estancamiento --- */
        if (use_modification) {
            int hist_idx = gen % stagnation_window;
            double improvement = (pop.fitness[cur_best_idx] - best_fitness_history[hist_idx]) /
                                  (best_fitness_history[hist_idx] > 0 ? best_fitness_history[hist_idx] : 1.0);
            if (gen >= stagnation_window && improvement < STAGNATION_THRESHOLD) {
                int n_reintro = (int)(POP_SIZE * reintro_fraction);
                for (int r = 0; r < n_reintro; r++) {
                    /* elegir el peor individuo actual (excluyendo el mejor, por elitismo) */
                    int excluded[1] = { best_index(pop.fitness, POP_SIZE) };
                    int worst = worst_index_excluding(pop.fitness, POP_SIZE, excluded, 1);
                    if (worst == -1) break;
                    chromosome_free(&pop.pop[worst]);
                    chromosome_random(inst, &pop.pop[worst]);
                    pop.fitness[worst] = chromosome_fitness_active(inst, &pop.pop[worst]);
                }
            }
            best_fitness_history[hist_idx] = pop.fitness[cur_best_idx];
        }

        if (verbose && (gen % 20 == 0 || gen == MAX_GENERATIONS - 1)) {
            printf("  gen %3d: best Cmax so far = %d\n", gen, best_cmax_ever); // DEBUG
        }
    }

    free(best_fitness_history);
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

    int n_seeds = N_SEEDS;
    int baseline_results[N_SEEDS], modified_results[N_SEEDS]; // Misma cantidad de iteraciones para solucion original y modificada

    clock_t inicio, fin;
    double time_baseline = 0, time_modified = 0;

    for (int s = 0; s < n_seeds; s++) {
        unsigned int seed = 1000 + s;

        inicio = clock();
        baseline_results[s] = run_slga(&inst, 0, seed, 0);
        fin = clock();
        time_baseline += ((double)(fin - inicio)) / CLOCKS_PER_SEC;

        inicio = clock();
        modified_results[s] = run_slga(&inst, 1, seed, 0);
        fin = clock();
        time_modified += ((double)(fin - inicio)) / CLOCKS_PER_SEC;
    }

    // Obtener BSL y ASL
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

    printf("  Tiempo SLGA original : %f segundos (%d corridas)\n", time_baseline, n_seeds);
    printf("  Tiempo SLGA modificado: %f segundos (%d corridas)\n\n", time_modified, n_seeds);

    fjsp_free(&inst);
}

// =========== Para buscar en path "instances/.../"===========

static int has_txt_extension(const char *name)
{
    size_t len = strlen(name);
    return len > 4 && strcmp(name + len - 4, ".txt") == 0;
}

static int compare_strings(const void *a, const void *b)
{
    return strcmp(*(const char **)a, *(const char **)b);
}

static const char *DEFAULT_INSTANCE_DIRS[] = {
    "instances/brandimarte",
    "instances/kacem",
};
#define NUM_DEFAULT_DIRS (int)(sizeof(DEFAULT_INSTANCE_DIRS) / sizeof(DEFAULT_INSTANCE_DIRS[0]))

static int resolve_instance_path(const char *arg, char *out, size_t out_size)
{
    struct stat st;
    size_t arg_len = strlen(arg);
    int already_has_txt = (arg_len > 4 && strcmp(arg + arg_len - 4, ".txt") == 0);

    /* 1. tal cual */
    if (stat(arg, &st) == 0) {
        snprintf(out, out_size, "%s", arg);
        return 1;
    }
    /* 2. agregando .txt */
    if (!already_has_txt) {
        snprintf(out, out_size, "%s.txt", arg);
        if (stat(out, &st) == 0) return 1;
    }
    /* 3-4-5-6: carpetas por defecto, con y sin .txt */
    for (int d = 0; d < NUM_DEFAULT_DIRS; d++) {
        snprintf(out, out_size, "%s/%s", DEFAULT_INSTANCE_DIRS[d], arg);
        if (stat(out, &st) == 0) return 1;

        if (!already_has_txt) {
            snprintf(out, out_size, "%s/%s.txt", DEFAULT_INSTANCE_DIRS[d], arg);
            if (stat(out, &st) == 0) return 1;
        }
    }

    return 0; // No encontrado
}

static void run_path(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) {
        printf("No se pudo acceder a '%s'\n", path);
        return;
    }

    if (S_ISDIR(st.st_mode)) {
        DIR *dir = opendir(path);
        if (!dir) {
            printf("No se pudo leer el directorio '%s'\n", path);
            return;
        }
        char **names = NULL;
        int count = 0, capacity = 0;
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (!has_txt_extension(entry->d_name)) continue;
            if (count >= capacity) {
                capacity = capacity == 0 ? 16 : capacity * 2;
                names = (char **)realloc(names, sizeof(char *) * capacity);
            }
            names[count] = strdup(entry->d_name);
            count++;
        }
        closedir(dir);

        qsort(names, count, sizeof(char *), compare_strings);

        for (int i = 0; i < count; i++) {
            char fullpath[1024];
            snprintf(fullpath, sizeof(fullpath), "%s/%s", path, names[i]);
            run_benchmark(fullpath, names[i]);
            free(names[i]);
        }
        free(names);
    } else {
        run_benchmark(path, path);
    }
}


int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Uso: %s <nombre_instancia | archivo.txt | carpeta> ...\n", argv[0]);
        printf("Ej:  %s mk01                    (busca en instances/brandimarte/mk01.txt)\n", argv[0]);
        printf("Ej:  %s mk01 mk02 k1             (varias instancias por nombre)\n", argv[0]);
        printf("Ej:  %s instances/brandimarte    (corre TODAS las .txt de la carpeta)\n", argv[0]);
        printf("Ej:  %s k1.txt mk01.txt          (rutas completas, como antes)\n", argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        struct stat st;
        if (stat(argv[i], &st) == 0 && S_ISDIR(st.st_mode)) {
            run_path(argv[i]);
            continue;
        }
        char resolved[1024];
        if (resolve_instance_path(argv[i], resolved, sizeof(resolved))) {
            run_benchmark(resolved, argv[i]);
        } else {
            printf("No se encontro la instancia '%s' (probe: tal cual, +.txt, "
                   "instances/brandimarte/, instances/kacem/)\n", argv[i]);
        }
    }

    // Imprimir algunos parametros
    printf("Maxima cantidad de generaciones: %d \n", MAX_GENERATIONS);
    printf("Tamaño de poblacion inicial: %d \n", POP_SIZE);

    printf("Numero de seeds/iteraciones %d \n", N_SEEDS);
    printf("Porcentaje de reserva de poblacion (DIVERSITY_RESERVE_FRACTION) %f%% \n", DIVERSITY_RESERVE_FRACTION*100);
    printf("Umbral de estancamiento %f \n", STAGNATION_THRESHOLD);
    printf("Multiplicador de mutacion maximo %f \n", ADAPTIVE_MUT_MAX_MULT);
    return 0;
}