#include <stdlib.h>
#include <assert.h>
#include "config.h"

// ============== Action set (4.4) ============== 

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

// No entendi cmo fukin funciona
void action_get_params(int action_idx, double *pc, double *pm)
{
    assert(action_idx >= 0 && action_idx < N_ACTIONS);
    const Action *a = &action_table[action_idx];

    *pc = a->pc_min + rl_uniform01() * (a->pc_max - a->pc_min);
    *pm = a->pm_min + rl_uniform01() * (a->pm_max - a->pm_min);
}