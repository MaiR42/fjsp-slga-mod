#ifndef RL_H
#define RL_H

// ============== Constantes ============== 
#define EPSILON 0.85 // epsilon (del e-greedy)
#define ALPHA 0.75 // alpha
#define GAMMA 0.2 // gamma
#define R_INIT 1.0 // reward r

// Dimensiones de la tabla Q (4.3 / 4.4)
#define N_STATES  20   // Cantidad de estados s (s1 a s20)
#define N_ACTIONS 10   // Cantidad de acciones a (a1 a a10)

// Pesos de la formula de estado (Eq. 9)
#define W1 0.35
#define W2 0.35
#define W3 0.30

// Ancho del intervalo para discretizar S* en 20 estados (4.3)
#define STATE_INTERVAL 0.05

// ============== Action set (4.4) ============== 

/* Cada accion define un subrango de Pc y de Pm; el valor real se
 * sortea al azar dentro de ese rango cada vez que se usa la accion. */
typedef struct {
    double pc_min;
    double pc_max;
    double pm_min;
    double pm_max;
} Action;

// Tabla de las 10 acciones
extern const Action action_table[N_ACTIONS];

// Numero aleatorio uniforme en [0, 1)
double rl_uniform01(void);

/* Dado un indice de accion (0..N_ACTIONS-1), sortea Pc y Pm dentro del
 * sub-rango de esa accion y los devuelve por puntero. */
void action_get_params(int action_idx, double *pc, double *pm);

// Para las funciones * (Eq. 6->9)
double compute_d_star(const double *fit_t, const double *fit_1, int N); // Eq 7
double compute_f_star(const double *fit_t, const double *fit_1, int N); // Eq 6
double compute_m_star(const double *fit_t, const double *fit_1, int N); // Eq 8
double compute_S_star(double f_star, double d_star, double m_star);     // Eq 9
 
/* Discretiza S* en un estado entero [0, N_STATES-1] */
int discretize_state(double S_star);
 
/* Calcula f*, d*, m*, S* y devuelve directamente el estado discretizado */
int compute_state(const double *fit_t, const double *fit_1, int N);

// Q Table

extern double Q[N_STATES][N_ACTIONS];

void qtable_init(void);
int qtable_argmax_action(int s);
double qtable_max_value(int s);
void qtable_update_sarsa(int s_t, int a_t, double r, int s_t1, int a_t1);
void qtable_update_qlearning(int s_t, int a_t, double r, int s_t1);










#endif // RL_H