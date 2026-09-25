"""Integratore Runge-Kutta del quarto ordine (RK4) a passo fisso.

Il passo fisso rende la simulazione ripetibile e allinea l'integrazione al
controllo a tempo discreto. L'errore locale è proporzionale alla quinta potenza
del passo e quello globale alla quarta: dimezzando il passo, l'errore
accumulato si riduce di circa 16 volte.
"""

from collections.abc import Callable

from cubesat_sim.core.types import FloatArray


def rk4_step(
    derivative: Callable[[FloatArray], FloatArray], state: FloatArray, step: float
) -> FloatArray:
    """Un passo di RK4 per il sistema dy/dt = f(y).

    Gli ingressi esterni (comandi, ambiente) sono inclusi nella funzione
    derivative e restano costanti durante il passo.

    Args:
        derivative: funzione f che, dato lo stato, ne restituisce la derivata.
        state: stato all'inizio del passo.
        step: durata del passo [s].

    Returns:
        Stato alla fine del passo (un nuovo array).
    """
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * step * k1)
    k3 = derivative(state + 0.5 * step * k2)
    k4 = derivative(state + step * k3)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
