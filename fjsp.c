#include <stdio.h>
#include <stdlib.h>
#include "fjsp.h"

/* Libera solo los primeros `count` jobs de un array (usado tanto por
 * fjsp_free como para limpiar parcialmente si el parser falla a mitad). */
static void free_jobs(Job *jobs, int count)
{
    if (!jobs) return;
    for (int j = 0; j < count; j++) {
        for (int o = 0; o < jobs[j].num_operations; o++) {
            free(jobs[j].operations[o].options);
        }
        free(jobs[j].operations);
    }
    free(jobs);
}

int fjsp_load_from_file(const char *path, FJSPInstance *inst)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "fjsp_load_from_file: no se pudo abrir '%s'\n", path);
        return -1;
    }

    if (fscanf(f, "%d %d", &inst->num_jobs, &inst->num_machines) != 2) {
        fprintf(stderr, "fjsp_load_from_file: error leyendo cabecera de '%s'\n", path);
        fclose(f);
        return -1;
    }

    inst->jobs = (Job *)malloc(sizeof(Job) * inst->num_jobs);

    for (int j = 0; j < inst->num_jobs; j++) {
        int num_ops;
        if (fscanf(f, "%d", &num_ops) != 1) {
            fprintf(stderr, "fjsp_load_from_file: error leyendo num_operaciones del job %d\n", j);
            free_jobs(inst->jobs, j); /* libera solo los jobs ya completados */
            fclose(f);
            return -1;
        }

        inst->jobs[j].num_operations = num_ops;
        inst->jobs[j].operations = (Operation *)malloc(sizeof(Operation) * num_ops);

        for (int o = 0; o < num_ops; o++) {
            int num_opts;
            if (fscanf(f, "%d", &num_opts) != 1) {
                fprintf(stderr, "fjsp_load_from_file: error leyendo num_opciones (job %d, op %d)\n", j, o);
                inst->jobs[j].num_operations = o; /* solo hay 'o' operaciones completas */
                free_jobs(inst->jobs, j + 1);
                fclose(f);
                return -1;
            }

            inst->jobs[j].operations[o].num_options = num_opts;
            inst->jobs[j].operations[o].options =
                (MachineOption *)malloc(sizeof(MachineOption) * num_opts);

            for (int k = 0; k < num_opts; k++) {
                int machine, time;
                if (fscanf(f, "%d %d", &machine, &time) != 2) {
                    fprintf(stderr, "fjsp_load_from_file: error leyendo opcion de maquina "
                                     "(job %d, op %d, opcion %d)\n", j, o, k);
                    free(inst->jobs[j].operations[o].options);
                    inst->jobs[j].num_operations = o;
                    free_jobs(inst->jobs, j + 1);
                    fclose(f);
                    return -1;
                }
                inst->jobs[j].operations[o].options[k].machine = machine;
                inst->jobs[j].operations[o].options[k].processing_time = time;
            }
        }
    }

    fclose(f);
    return 0;
}

void fjsp_free(FJSPInstance *inst)
{
    free_jobs(inst->jobs, inst->num_jobs);
    inst->jobs = NULL;
    inst->num_jobs = 0;
    inst->num_machines = 0;
}

void fjsp_print(const FJSPInstance *inst)
{
    printf("Instancia FJSP: %d jobs, %d maquinas\n", inst->num_jobs, inst->num_machines);
    for (int j = 0; j < inst->num_jobs; j++) {
        printf("  Job %d (%d operaciones):\n", j, inst->jobs[j].num_operations);
        for (int o = 0; o < inst->jobs[j].num_operations; o++) {
            printf("    Op %d: ", o);
            Operation *op = &inst->jobs[j].operations[o];
            for (int k = 0; k < op->num_options; k++) {
                printf("[M%d, t=%d] ", op->options[k].machine, op->options[k].processing_time);
            }
            printf("\n");
        }
    }
}

// ============================ Cromosoma (encoding OS + MS, Sec 3.1.1) ============================ 

int fjsp_total_operations(const FJSPInstance *inst)
{
    int total = 0;
    for (int j = 0; j < inst->num_jobs; j++) {
        total += inst->jobs[j].num_operations;
    }
    return total;
}

int fjsp_get_op_index(const int *os, int pos)
{
    int job = os[pos];
    int count = 0;
    for (int i = 0; i <= pos; i++) {
        if (os[i] == job) count++;
    }
    return count - 1; /* 0-indexado */
}

void chromosome_init(Chromosome *c, int length)
{
    c->length = length;
    c->os = (int *)malloc(sizeof(int) * length);
    c->ms = (int *)malloc(sizeof(int) * length);
}

void chromosome_free(Chromosome *c)
{
    free(c->os);
    free(c->ms);
    c->os = NULL;
    c->ms = NULL;
    c->length = 0;
}

/* Fisher-Yates shuffle sobre el array os */
static void shuffle_os(int *os, int length)
{
    for (int i = length - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = os[i];
        os[i] = os[j];
        os[j] = tmp;
    }
}

void chromosome_random(const FJSPInstance *inst, Chromosome *c)
{
    int total = fjsp_total_operations(inst);
    chromosome_init(c, total);

    /* 1. Armar OS en orden canonico: job 0 repetido num_operations[0] veces,
     *    luego job 1 repetido num_operations[1] veces, etc. */
    int idx = 0;
    for (int j = 0; j < inst->num_jobs; j++) {
        for (int o = 0; o < inst->jobs[j].num_operations; o++) {
            c->os[idx] = j;
            idx++;
        }
    }

    /* 2. Mezclar el OS al azar */
    shuffle_os(c->os, total);

    /* 3. Para cada posicion, sortear una maquina VALIDA para esa operacion especifica */
    for (int i = 0; i < total; i++) {
        int job = c->os[i];
        int op_idx = fjsp_get_op_index(c->os, i);
        Operation *op = &inst->jobs[job].operations[op_idx];

        int choice = rand() % op->num_options;
        c->ms[i] = op->options[choice].machine;
    }
}

void chromosome_print(const Chromosome *c)
{
    printf("OS: ");
    for (int i = 0; i < c->length; i++) printf("%d ", c->os[i]);
    printf("\nMS: ");
    for (int i = 0; i < c->length; i++) printf("%d ", c->ms[i]);
    printf("\n");
}


// ============================ Decoder (semi-activo) + Fitness ============================

// Decoder no indicado en el paper, por lo que se eligio uno semi-activo temporalmente
 
int decode_makespan(const FJSPInstance *inst, const Chromosome *c)
{
    int *job_end     = (int *)calloc(inst->num_jobs, sizeof(int));
    int *machine_free = (int *)calloc(inst->num_machines, sizeof(int));
    int makespan = 0;
 
    for (int i = 0; i < c->length; i++) {
        int job    = c->os[i];
        int op_idx = fjsp_get_op_index(c->os, i);
        Operation *op = &inst->jobs[job].operations[op_idx];
        int machine = c->ms[i];
 
        /* buscar tiempo de procesamiento para esta maquina */
        int proc_time = -1;
        for (int k = 0; k < op->num_options; k++) {
            if (op->options[k].machine == machine) {
                proc_time = op->options[k].processing_time;
                break;
            }
        }
        /* proc_time siempre deberia encontrarse si el cromosoma es valido */
 
        int start = job_end[job] > machine_free[machine] ? job_end[job] : machine_free[machine];
        int end = start + proc_time;
 
        job_end[job] = end;
        machine_free[machine] = end;
        if (end > makespan) makespan = end;
    }
 
    free(job_end);
    free(machine_free);
    return makespan;
}
 
double chromosome_fitness(const FJSPInstance *inst, const Chromosome *c)
{
    int cmax = decode_makespan(inst, c);
    if (cmax <= 0) return 0.0; // En caso de ser cero o negativo
    return 1.0 / (double)cmax;
}