#ifndef FJSP_H
#define FJSP_H

// ============================ Parser ============================

/*
 * Estructuras de datos para una instancia de FJSP, y parser del formato
 * .txt usado por los benchmarks de Brandimarte y Kacem (verificado
 * contra los .json de referencia: mk01.txt/mk01.json, k1.txt/k1.json).
 *
 * Formato del .txt:
 *   linea 1: <num_jobs> <num_machines>
 *   una linea por cada job, con esta secuencia de numeros:
 *     <num_operaciones>
 *     por cada operacion: <num_opciones_de_maquina> (maquina tiempo) x num_opciones
 *   Las maquinas ya vienen indexadas desde 0 (no hace falta restar 1). // DELETE
 */

typedef struct {
    int machine;          /* indice de maquina, 0-indexado */
    int processing_time;
} MachineOption;

typedef struct {
    int num_options;
    MachineOption *options;   /* array de tamano num_options */
} Operation;

typedef struct {
    int num_operations;
    Operation *operations;    /* array de tamano num_operations */
} Job;

typedef struct {
    int num_jobs;
    int num_machines;
    Job *jobs;                 /* array de tamano num_jobs */
} FJSPInstance;

int fjsp_load_from_file(const char *path, FJSPInstance *inst);

void fjsp_free(FJSPInstance *inst);

void fjsp_print(const FJSPInstance *inst); // DEGUB

// ============================ Cromosoma (encoding OS + MS, Sec 3.1.1) ============================ 

/*
 * os[i] (Operation Sequence): id de job (0-indexado). La k-esima vez que
 * aparece el job j en el array (leyendo de izquierda a derecha) representa
 * la operacion O_{j,k}.
 *
 * ms[i] (Machine Assignment): id de MAQUINA REAL (0-indexada, ej. "3" =
 * M3), emparejada por POSICION con os[i] (confirmado con Fig. 1 del paper:
 * misma posicion = misma operacion en ambos arrays).
 *
 * length = total de operaciones en toda la instancia = len(os) = len(ms).
 */
typedef struct {
    int length;
    int *os;
    int *ms;
} Chromosome;

/* Suma de operaciones de todos los jobs (longitud del cromosoma) */
int fjsp_total_operations(const FJSPInstance *inst);

/*
 * Dado un array OS y una posicion pos, devuelve el indice (0-indexado) de
 * la operacion dentro de su job, contando cuantas veces aparecio ese mismo
 * job desde el inicio del array hasta pos (inclusive).
 * Ej: os = [1,4,3,2,4,...], pos=4 (el segundo "4") -> devuelve 1 (segunda
 * operacion del job 4, es decir O_{4,2} en notacion 1-indexada del paper).
 */
int fjsp_get_op_index(const int *os, int pos);

/* Reserva memoria para os y ms (sin inicializar valores) */
void chromosome_init(Chromosome *c, int length);

/* Libera os y ms */
void chromosome_free(Chromosome *c);

/*
 * Genera un cromosoma aleatorio VALIDO para la instancia dada:
 *  - os: orden canonico de operaciones, mezclado al azar (shuffle)
 *  - ms: para cada posicion, una maquina sorteada entre las opciones
 *        validas de esa operacion especifica (respeta que cada operacion
 *        solo puede ir en sus maquinas permitidas)
 */
void chromosome_random(const FJSPInstance *inst, Chromosome *c);

void chromosome_print(const Chromosome *c); // DEGUB

// Para el decoder semi-activo
int decode_makespan(const FJSPInstance *inst, const Chromosome *c);

/* fitness = 1 / Cmax */
double chromosome_fitness(const FJSPInstance *inst, const Chromosome *c);

// Para el decoder activo
int decode_makespan_active(const FJSPInstance *inst, const Chromosome *c);
double chromosome_fitness_active(const FJSPInstance *inst, const Chromosome *c);


#endif /* FJSP_H */