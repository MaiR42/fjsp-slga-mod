#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "rl.h"

// ============================ Action set (4.4) ============================ 

/* 
Rango: [px_min, px_max)
Paper menciona que la tabla se arma con la estructura ej:

px_min, px_max
x, y
y, z
z, ...

*/
const Action action_table[N_ACTIONS] = {
// pc_min  pc_max  pm_min  pm_max
    {0.40, 0.45, 0.01, 0.03},
    {0.45, 0.50, 0.03, 0.05},
    {0.50, 0.55, 0.05, 0.07},
    {0.55, 0.60, 0.07, 0.09},
    {0.60, 0.65, 0.09, 0.11},
    {0.65, 0.70, 0.11, 0.13},
    {0.70, 0.75, 0.13, 0.15},
    {0.75, 0.80, 0.15, 0.17},
    {0.80, 0.85, 0.17, 0.19},
    {0.85, 0.90, 0.19, 0.21},
};

double rl_uniform01(void)
{
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

void action_get_params(int action_idx, double *pc, double *pm)
{
    assert(action_idx >= 0 && action_idx < N_ACTIONS);
    const Action *a = &action_table[action_idx];

    *pc = a->pc_min + rl_uniform01() * (a->pc_max - a->pc_min);
    *pm = a->pm_min + rl_uniform01() * (a->pm_max - a->pm_min);
}

// Funciones aux (para las sumatorias)
static double sum_array(const double *arr, int N)
{
    double s = 0.0;
    for (int i = 0; i < N; i++) s += arr[i];
    return s;
}

static double mean_array(const double *arr, int N)
{
    return sum_array(arr, N) / (double)N;
}

static double max_array(const double *arr, int N)
{
    double m = arr[0];
    for (int i = 1; i < N; i++) if (arr[i] > m) m = arr[i];
    return m;
}

static double sum_abs_deviation(const double *arr, int N, double mean)
{
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double dev = arr[i] - mean;
        if (dev < 0) dev = -dev; /* valor absoluto */
        s += dev;
    }
    return s;
}


/* Eq (6): f_*
    razon entre suma de fitness actual y suma de fitness inicial */
double compute_f_star(const double *fit_t, const double *fit_1, int N)
{
    double num = sum_array(fit_t, N);
    double den = sum_array(fit_1, N);
    return num / den;
}

/* Eq (7): d_*
    razon entre diversidad actual (desviacion absoluta media) e inicial */
double compute_d_star(const double *fit_t, const double *fit_1, int N)
{
    double mean_t = mean_array(fit_t, N);
    double mean_1 = mean_array(fit_1, N);

    double num = sum_abs_deviation(fit_t, N, mean_t);
    double den = sum_abs_deviation(fit_1, N, mean_1);

    if (den == 0.0) return 0.0; // Evita dividir por cero

    return num / den;
}

/* Eq (8): m_*
    razon entre el mejor fitness actual y el mejor fitness inicial */
double compute_m_star(const double *fit_t, const double *fit_1, int N)
{
    double num = max_array(fit_t, N);
    double den = max_array(fit_1, N);
    return num / den;
}

// Eq (9): S_*
double compute_S_star(double f_star, double d_star, double m_star)
{
    double S_star = W1 * f_star + W2 * d_star + W3 * m_star;
    return S_star;
}

/* Discretiza S* (continuo) en un estado entero [0, N_STATES-1],
 * equivalente a s(1)..s(20) del paper, usando intervalos de 0.05. */
int discretize_state(double S_star)
{
    int state = (int)(S_star / STATE_INTERVAL);

    if (state < 0) state = 0;
    if (state > N_STATES - 1) state = N_STATES - 1;

    return state;
}

/* Junta las 4 ecuaciones y devuelve directamente el estado discretizado. */
int compute_state(const double *fit_t, const double *fit_1, int N)
{
    double f_star = compute_f_star(fit_t, fit_1, N);
    double d_star = compute_d_star(fit_t, fit_1, N);
    double m_star = compute_m_star(fit_t, fit_1, N);
    double S_star = compute_S_star(f_star, d_star, m_star);

    return discretize_state(S_star);
}


// ============================ Q TABLE ============================
double Q[N_STATES][N_ACTIONS];

void qtable_init(void)
{
    for (int s = 0; s < N_STATES; s++)
        for (int a = 0; a < N_ACTIONS; a++)
            Q[s][a] = 0.0;
}

int qtable_argmax_action(int s)
{
    int best_a = 0;
    double best_q = Q[s][0];

    for (int a = 1; a < N_ACTIONS; a++) {
        if (Q[s][a] > best_q) {
            best_q = Q[s][a];
            best_a = a;
        }
    }
    return best_a;
}

double qtable_max_value(int s)
{
    int a = qtable_argmax_action(s);
    return Q[s][a];
}

// Eq (3)
void qtable_update_sarsa(int s_t, int a_t, double r, int s_t1, int a_t1)
{
    double q_sa      = Q[s_t][a_t];
    double q_next_sa = Q[s_t1][a_t1];

    Q[s_t][a_t] = (1.0 - ALPHA) * q_sa + ALPHA * (r + GAMMA * q_next_sa);
}

// Eq (4)
void qtable_update_qlearning(int s_t, int a_t, double r, int s_t1)
{
    double q_sa  = Q[s_t][a_t];
    double q_max = qtable_max_value(s_t1);

    Q[s_t][a_t] = (1.0 - ALPHA) * q_sa + ALPHA * (r + GAMMA * q_max);
}


// ============================ Reward ============================

// Eq (10)
double compute_reward_rc(const double *fit_t, const double *fit_t_prev, int N)
{
    double max_t    = max_array(fit_t, N);
    double max_prev = max_array(fit_t_prev, N);
    if (max_prev == 0.0) return 0.0;

    return (max_t - max_prev) / max_prev;
}

// Eq (11)
double compute_reward_rm(const double *fit_t, const double *fit_t_prev, int N)
{
    double sum_t    = sum_array(fit_t, N);
    double sum_prev = sum_array(fit_t_prev, N);
    if (sum_prev == 0.0) return 0.0;

    return (sum_t - sum_prev) / sum_prev;
}

// ============================ e-greedy ============================

//  Conversion condition
RLMode rl_get_mode(int n_ti)
{
    double threshold = (N_STATES * N_ACTIONS) / 2.0; /* = 100 con 20x10 */

    if (n_ti < threshold)
        return RL_MODE_SARSA;
    else
        return RL_MODE_QLEARNING;
}

int policy_random_action(void)
{
    return rand() % N_ACTIONS;
}

int policy_epsilon_greedy_select(int state)
{
    double r = rl_uniform01(); // r_0-1 // Reward entre 0 y 1

    if (EPSILON >= r)
        return qtable_argmax_action(state);
    else
        return policy_random_action();
}

// ============================ Orquestador (Main) ============================

void rl_init(RLState *rl, const double *fit_initial, int pop_size, RewardMode mode)
{
    rl->pop_size = pop_size;
    rl->n_ti = 0;
    rl->reward_mode = mode;

    rl->fit_gen1 = (double *)malloc(sizeof(double) * pop_size);
    rl->fit_prev = (double *)malloc(sizeof(double) * pop_size);
    memcpy(rl->fit_gen1, fit_initial, sizeof(double) * pop_size);
    memcpy(rl->fit_prev, fit_initial, sizeof(double) * pop_size);

    qtable_init();

    /* "Choose a random action a_t, set a <- a_t" */
    rl->a_t = policy_random_action();

    /* "Calculate the State s_t of GA, set s <- s_t"
     * En t=0, fit_current == fit_gen1, por lo que f*=d*=m*=1.0 -> S*=1.0 */
    rl->s_t = compute_state(fit_initial, rl->fit_gen1, pop_size);
}


/* Auxiliar privada: calcula el reward segun el modo configurado */
static double compute_reward(RLState *rl, const double *fit_current)
{
    double rc, rm;
    switch (rl->reward_mode) {
        case REWARD_USE_RC:
            return compute_reward_rc(fit_current, rl->fit_prev, rl->pop_size);
        case REWARD_USE_RM:
            return compute_reward_rm(fit_current, rl->fit_prev, rl->pop_size);
        case REWARD_USE_RC_PLUS_RM:
        default:
            rc = compute_reward_rc(fit_current, rl->fit_prev, rl->pop_size);
            rm = compute_reward_rm(fit_current, rl->fit_prev, rl->pop_size);
            return rc + rm;
    }
}

// Mayor loop body of SLGA (Fig. 5)
void rl_step(RLState *rl, const double *fit_current, double *pc, double *pm)
{
    /* 1. Reward r_t+1 (Eq 10 / 11) */
    double r = compute_reward(rl, fit_current);

    /* 2. Nuevo estado s_t+1 (Eq 6-9) */
    int s_next = compute_state(fit_current, rl->fit_gen1, rl->pop_size);

    /* 3. Conversion condition (Eq 5) decide entre SARSA vs Q-learning */
    RLMode mode = rl_get_mode(rl->n_ti);
    int a_next;

    if (mode == RL_MODE_SARSA) {
        a_next = policy_epsilon_greedy_select(s_next);
        qtable_update_sarsa(rl->s_t, rl->a_t, r, s_next, a_next);
    } else {
        qtable_update_qlearning(rl->s_t, rl->a_t, r, s_next);
        a_next = policy_epsilon_greedy_select(s_next);
    }

    /* 4. Ejecutar accion a_t+1 y obtener nuevos Pc, Pm */
    action_get_params(a_next, pc, pm);

    /* 5. Avanzar estado interno para la proxima llamada */
    rl->s_t = s_next;
    rl->a_t = a_next;
    rl->n_ti += 1;
    memcpy(rl->fit_prev, fit_current, sizeof(double) * rl->pop_size);
}

void rl_free(RLState *rl)
{
    free(rl->fit_gen1);
    free(rl->fit_prev);
    rl->fit_gen1 = NULL;
    rl->fit_prev = NULL;
}

// The game
