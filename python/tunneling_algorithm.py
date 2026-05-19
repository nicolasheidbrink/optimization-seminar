import numpy as np
from scipy.optimize import minimize
"""
This script implements the tunneling algorithm as described in "The Tunneling ALgorithm for the 
    Global Minimization of Functions" by Levy and Montalvo (1985).
Modifications to their algorithm are marked with `improved` and `annealing` flags.

The `TunnelingAlgorithm` class implements the algorithm for a given objective function, its gradient, and bounds.
The `apply_algorithm` method is the entry point.
"""
class TunnelingAlgorithm:
    def __init__(self, f, f_grad, bounds, verbose=False, improved=False, annealing=False):
        np.random.seed(42)
        
        self.f = f # Objective function
        self.f_grad = f_grad # Gradient of the objective function
        self.bounds = bounds # Bounds for the optimization variables; list of (lower, upper) tuples
        self.dim = len(bounds) # Dimensionality of the problem
        self.x_stars = [] # Stores all minimizers at the current f_star level
        self.lambda_list = [] # Pole strengths for each x_star
        self.f_star = np.inf # Best known minimum value of f

        self.eps1 = 1e-3 # Threshold for considering a new minimum better than f_star; the paper recommended 1e-9
        self.eps2 = 1e-5 # Transition width for the ramp function in eta
        self.eps3 = 1e-3 # Threshold for considering a point as a good enough to end the tunneling phase
        self.lambda_max = 5 # Maximum pole strength to consider during lambda search
        self.ns = 100 # Max iterations for the restoration algorithm during the tunneling phase
        self.nb = 20 # Max bisections for step-size
        self.n_eps_attempts = 2 * self.dim # Number of attempts starting at the previous minimum in the tunneling phase
        self.n_random_starts = 2 * self.dim # Number of random starts when local perturbations fail in the tunneling phase
        self.max_cycles = 1000 # Maximum number of cycles of minimization and tunneling phases to perform

        self.minimization_phase_counter = 0 # Counter to track how many times the minimization phase is called
        self.success = 0 # Counter for successful steps in the restoration algorithm
        self.failure = 0 # Counter for failed steps in the restoration algorithm

        self.verbose = verbose # If True, print detailed logs during execution
        self.improved = improved # If True, distance between xm and x_new is independent of x_hat
        self.annealing = annealing # If True, allow annealing steps during tunneling

        self.temperature_0 = 1000

    '''
    Input: Initial point `x_0` for the tunneling algorithm, and maximum number of cycles to perform.
    Output: A tuple containing the list of minimizers `x_stars` found at the best known minimum level `f_star`.

    The algorithm alternates between a minimization phase and a tunneling phase.
    '''
    def apply_algorithm(self, x_0):
        current_x = x_0
        for cycle_counter in range(self.max_cycles):
            self.k = cycle_counter+1 if self.annealing else None
            # 1. Minimization Phase
            print(f"starting minimization phase at {current_x} with f={self.f(current_x)}") if self.verbose else None
            x_star, f_val = self.minimization_phase(current_x)
            print(f"Minimization phase completed at {x_star} with f={f_val}") if self.verbose else None

            
            # If a new unique global minimizer was found, update global minimum and manage poles
            if f_val < self.f_star - self.eps1:
                print(f"New unique global minimizer found at {x_star} with f={f_val}") if self.verbose else None
                self.f_star = f_val
                self.x_stars = [x_star]
                self.lambda_list = [1.0] # Reset to 1.0 for the new minimum

            # If a new minimizer at the same level is found, add it to the list of minimizers
            elif abs(f_val - self.f_star) <= self.eps1:
                # (Double-check to avoid duplicates)
                if not any(np.linalg.norm(x_star - prev) < 1e-3 for prev in self.x_stars):
                    print(f"New minimizer found at {x_star} with f={f_val}") if self.verbose else None
                    self.x_stars.append(x_star)
                    self.lambda_list.append(1.0) # Add a new pole for the new minimum
                else:
                    print(f"Duplicate minimizer found at {x_star} with f={f_val}, ignoring.") if self.verbose else None
            else:
                print(f"No improvement found at {x_star} with f={f_val}, current best is {self.f_star} at {self.x_stars}") if self.verbose else None
            
            # 2. Tunneling Phase
            print(f"Starting tunneling phase from {x_star} with f={f_val}") if self.verbose else None
            next_start = self.tunneling_phase(x_star)
            print(f"Tunneling phase completed, next start point: {next_start}") if self.verbose else None
            
            if next_start is None:
                break 
            current_x = next_start
        
        print(f"Algorithm completed. Best minimum value found: {self.f_star} at {len(self.x_stars)} points: {self.x_stars}") if self.verbose else None
        return self.x_stars, self.f_star

    '''
    Input: Initial point `x_0` for the minimization phase.
    Output: A tuple containing the minimizer `x_star` and its function value `f_val`.

    The library function `scipy.optimize.minimize` is used for the minimization phase, 
    and any local optimization method that supports bounds can be selected as the method.  
    '''
    def minimization_phase(self, x_0):
        self.minimization_phase_counter += 1
        res = minimize(self.f, x_0, jac=self.f_grad, bounds=self.bounds, method='L-BFGS-B')
        return res.x, res.fun


    '''
    Input: The last found local minimizer `x_star`.
    Output: A new starting point for the next minimization phase, or `None` if no valid point is found.

    The tunneling phase first attempts local perturbations around the last found minimizer to escape its basin of attraction.
    If local perturbations fail, it performs a global search by randomly sampling points within the bounds 
        and applying the restoration algorithm.
    '''
    def tunneling_phase(self, x_star):
        # 1. Local search: Try random perturbations around the last found minimizer
        epsilons = []
        if self.improved:
            for direction in [-0.1, 0.1]:
                for i in range(self.dim):
                    ep = np.zeros(self.dim)
                    ep[i] = direction + np.random.uniform(-0.05, 0.05) 
                    epsilons.append(ep)
        else:
            for _ in range(self.n_random_starts):
                epsilons.append(np.random.uniform(-0.1, 0.1, self.dim))

        for nr_eps, epsilon in enumerate(epsilons):
            print(f"Attempting local perturbation around {x_star} with epsilon={epsilon} (attempt {nr_eps+1})") if self.verbose else None

            # Determine the pole strength in the tunneling function for this minimizer:
            print(f"    looking for appropriate lambda for minimizer") if self.verbose else None
            self.lambda_list[-1] = self.find_iterative_lambda(x_star, epsilon)
            print(f"    Found lambda={self.lambda_list[-1]} for minimizer {x_star}") if self.verbose else None

            # Search for a point x with T(x) <= eps3, starting the search from x_star + epsilon:
            print(f"    Running restoration algorithm from {x_star + epsilon}") if self.verbose else None
            next_start = self.run_restoration(x_star + epsilon)
            if next_start is not None:
                print(f"Found valid start point at {next_start}") if self.verbose else None
                return next_start
            else:
                print(f"Local perturbation {nr_eps+1} around {x_star} failed to find a valid start point.") if self.verbose else None

        # 2. Global search: If local search fails, try random points in Omega
        print(f"Global search initiated from minimizer {x_star}") if self.verbose else None
        return self.random_tunnel_search()

    '''
    Input: A new minimizer `x_star` and and a perturbation vector `epsilon`.
    Output: A pole strength `lambda` for the pole at `x_star` in the tunneling function.

    This function iteratively tests increasing values of `lambda` for the pole being added at `x_star` to ensure that the 
        tunneling function's gradient points in a direction that allows escape from the current basin of attraction.

    The descent check is an implementation of condition (A.2) from the paper, checking that `delta_x` is a descent direction.
    The outward check is an implementation of condition (A.2a) from the paper, checking that `delta_x` points away from the pole. 
    
    '''
    def find_iterative_lambda(self, x_star, epsilon):
        l_trial = 1.0
        x_test = x_star + epsilon
        while l_trial <= self.lambda_max:
            # Temporarily apply l_trial
            self.lambda_list[-1] = l_trial
            
            # Get the displacement vector delta_x (from section 2.3.4)
            print(f"    Calculating displacement to find lambda") if self.verbose else None
            delta_x = self.get_displacement(x_test, x_star, 0.0)
            t_grad = self.get_t_and_grad(x_test, x_star, 0.0)[1]

            descent_check = np.dot(t_grad, delta_x) < 0
            outward_check = np.dot(epsilon, delta_x) > 0
            
            if descent_check and outward_check:
                return l_trial
                    
            l_trial += 1
        
        return self.lambda_max

    '''
    Input: An iterate `x_start` that serves as the initial point for the restoration algorithm.
    Output: A new starting point for the next minimization phase, or `None` if no valid point is found.

    This function implements the restoration algorithm as described in section 2.3.4 of the paper, which is used 
        during the tunneling phase to find a new point that satisfies T(x) < eps3.
    '''
    def run_restoration(self, x_start):
        x_hat = self.apply_bounds(x_start) # The point that we move away from
        x_m = np.copy(x_start) # Initial movable pole position
        lambda_0 = 0.0         # Initial movable pole strength
        
        for _ in range(self.ns):
            t_val, t_grad = self.get_t_and_grad(x_hat, x_m, lambda_0)
            
            # 1. Check for Tunnel Exit
            if t_val <= self.eps3:
                # Must be far from ALL found minima to be a valid new start
                if all(np.linalg.norm(x_hat - prev) > 1e-2 for prev in self.x_stars):
                    return x_hat
                
            elif self.annealing and np.exp((self.f_star-self.f(x_hat))/(self.temperature_0/self.k)) > np.random.uniform(0,1):
                print(f"    Trying annealing step from {x_hat} with T={t_val}") if self.verbose else None
                if self.minimization_phase(x_hat)[1] < self.f_star + self.eps1:
                    print(f"    Annealing step successful from {x_hat} with T={t_val}") if self.verbose else None
                    return x_hat


            # 2. Calculate Displacement (Section 2.3.4)
            print(f"        Calculating displacement to find next restoration iterate") if self.verbose else None
            delta_x = self.get_displacement(x_hat, x_m, lambda_0)

            # 3. Step-Size Bisection
            alpha = 1.0
            x_new = x_hat # Default
            success = False
            for _ in range(self.nb):
                x_try = self.apply_bounds(x_hat + alpha * delta_x)
                if self.get_t_and_grad(x_try, x_m, lambda_0)[0] < t_val:
                    print(f"    Next iterate of restoration algorithm: {x_try} with T={self.get_t_and_grad(x_try, x_m, lambda_0)[0]}. Previous t_val was {t_val}. alpha = {alpha} and delta_x = {delta_x}") if self.verbose else None
                    x_new = x_try
                    success = True
                    self.success += 1
                    break
                alpha /= 2.0
            
            if not success:
                print(f"    Bisection failed to find a better point from {x_hat} with T={t_val}") if self.verbose else None
                self.failure += 1
                break

            # 4. Handle Movable Pole (Appendix I, Step 3)
            x_m = self.determine_xm(x_hat, x_new)
            print(f"        Updated movable pole position to x_m={x_m}") if self.verbose else None
            print(f"        Calculating displacement to see if lambda_0 has to be increased") if self.verbose else None
            delta_x_from_x_new_to_x_tilde = self.get_displacement(x_new, x_m, lambda_0)
            u = np.dot(x_new - x_hat, delta_x_from_x_new_to_x_tilde) 
            if u <= 0:
                # landed in undesirable local minimum of T(x); update pole[cite: 1]
                lambda_0 = self.find_iterative_lambda_0(x_hat, x_new, x_m)
                print(f"        Updating lambda_0 to {lambda_0} for movable pole at x_m={x_m}") if self.verbose else None
            else:
                # Heuristic reset rule (A.9): return to simpler geometry[cite: 1]
                # Compute displacement for lambda_0 = 0[cite: 1]
                print(f"        Calculating displacement to see if lambda_0 can be reset to 0") if self.verbose else None
                delta_x_0 = self.get_displacement(x_new, x_m, 0.0)
                if np.dot(delta_x_0, delta_x_from_x_new_to_x_tilde) > 0:
                    print(f"        Resetting lambda_0 to 0 for movable pole at x_m={x_m}") if self.verbose else None
                    lambda_0 = 0.0
                else:
                    print(f"        Keeping lambda_0 at {lambda_0} for movable pole at x_m={x_m}") if self.verbose else None

            x_hat = x_new
            
        return None
    
    '''
    Input: An iterate `x_hat`, the following iterate `x_new`, and the movable pole position `x_m`.
    Output: A pole strength `lambda_0` for the movable pole in the tunneling function.

    This function iteratively tests increasing values of `lambda_0` for the movable pole to ensure that the
        displacement vector from section 2.3.4 points in a direction that allows escape from the current basin of attraction.
    
    '''
    def find_iterative_lambda_0(self, x_hat, x_new, x_m):
        l0 = 1.0
        while l0 <= self.lambda_max:
            print(f"        Calculating displacement to see if lambda_0 increase to {l0} is sufficient") if self.verbose else None
            delta_x_from_x_new_to_x_tilde = self.get_displacement(x_new, x_m, l0)
            # If the step taken away from x_new is in the same general direction (acute angle) as the step taken from 
            # x_hat to x_new, then the lambda_0 is satisfactory. Otherwise, increase lambda_0 and try again.
            if np.dot(x_new - x_hat, delta_x_from_x_new_to_x_tilde) > 0:
                print(  f"      Found lambda_0={l0} for movable pole at x_m={x_m}") if self.verbose else None
                return l0
            l0 += 1.0
        print(f"        Warning: lambda_0 reached maximum value of {self.lambda_max} without satisfying condition.") if self.verbose else None
        return self.lambda_max


    '''
    Input: An iterate `x`, the movable pole position `x_m`, and the movable pole strength `lambda_0`.
    Output: The displacement vector for the restoration algorithm in the tunneling phase

    This function calculates the displacement vector as described in section 2.3.4 of the paper, which 
        is used to guide the restoration algorithm during the tunneling phase.
    '''
    def get_displacement(self, x, x_m, lambda_0):
        t_val, t_grad = self.get_t_and_grad(x, x_m, lambda_0)
        denom = np.dot(t_grad, t_grad)
        # Eq. in section 2.3.4[cite: 1]
        displacement = -(t_val / max(denom, 1e-20)) * t_grad
        print(f"        Displacement calculated as {displacement} using t_val={t_val}, t_grad={t_grad}") if self.verbose else None
        
        
        return displacement

    '''
    Input: Two iterates `x_hat` (an iterate from the tunneling phase) and `x_new` (the following iterate).
    Output: A new position `x_m` for the movable pole.

    This function determines the new position for the movable pole when the tunneling algorithm lands in an undesirable 
        local minimum of the tunneling function.

    The calculation of `x_m` is based on equation (A.7) from the paper, with xi = 0.9 * dist.
    '''
    def determine_xm(self, x_hat, x_new):

        if self.improved:
            if np.linalg.norm(x_hat - x_new) != 0:
                return x_new + (x_hat - x_new) * 0.9 / np.linalg.norm(x_hat - x_new) #here222
            else:
                return x_hat

        dist = np.linalg.norm(x_hat - x_new)
        if dist < 1: return np.copy(x_hat)
        xi = 0.9 / dist # 1.5 seemed a little better in a small test
        print(f"        Updating movable pole position to xm={xi*x_hat + (1 - xi) * x_new}") if self.verbose else None
        return xi * x_hat + (1 - xi) * x_new

    '''
    Input: An iterate `x`, the movable pole position `x_m`, and the movable pole strength `lambda_0`.
    Output: The value of the tunneling function T(x) and its gradient at `x`.

    This function computes the tunneling function T(x) and its gradient based on the current known minima (fixed poles), which are
        stored as `self.x_stars`, the corresponding strengths stored in `self.lambda_list`, and the movable pole with its strength.
    
    The function T(x) is defined as equation (A.0) in the paper, and the gradient is derived using the product and chain rules.
    
    This method was rewritten by Google Gemini to speed up the code.
    '''
    def get_t_and_grad(self, x, x_m, lambda_0):
        f_val = self.f(x)
        f_diff = f_val - self.f_star
        grad_f = self.f_grad(x)
        
        denom = 1.0
        # This vector will store grad(D)/D
        grad_D_over_D = np.zeros_like(x)
        
        # 1. Process Fixed Poles
        for i, x_star_i in enumerate(self.x_stars):
            diff = x - x_star_i
            dist = np.linalg.norm(diff)
            dist_sq = max(dist**2, 1e-20)
            
            # Calculate eta and its derivative (Ramp Logic from equation (A.5) in the paper)
            if dist >= 1 + self.eps2:
                eta = 0.0
                grad_eta = np.zeros_like(x)
            elif dist <= 1 - self.eps2:
                eta = self.lambda_list[i]
                grad_eta = np.zeros_like(x)
            else:
                # Transition zone
                eta = (self.lambda_list[i] / 2) * (1 + (1 - dist) / self.eps2)
                grad_eta = -(self.lambda_list[i] / (2 * self.eps2)) * (diff / (dist + 1e-20))
            
            denom *= (dist_sq ** eta)
            
            # d/dx [eta(x) * ln(dist_sq)] = grad_eta * ln(dist_sq) + eta * (2*diff / dist_sq)
            grad_D_over_D += grad_eta * np.log(dist_sq) + (eta * 2 * diff) / dist_sq

        # 2. Process Movable Pole (lambda_0 is constant during the step)
        diff_m = x - x_m
        dist_sq_m = max(np.sum(diff_m**2), 1e-20)
        denom *= (dist_sq_m ** lambda_0)
        grad_D_over_D += (lambda_0 * 2 * diff_m) / dist_sq_m

        # 3. Final T and grad(T)
        t_val = f_diff / max(denom, 1e-20)
        # grad(T) = (grad_f / denom) - T * (grad_D / D)
        t_grad = (grad_f / max(denom, 1e-20)) - (t_val * grad_D_over_D)
        
        return t_val, t_grad

    '''
    Input: An iterate `x` that may or may not be within the bounds.
    Output: The iterate `x` clamped to the bounds.

    This function ensures that the iterate `x` remains within the specified bounds.
    '''
    def apply_bounds(self, x):
        return np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])


    '''
    Input: None (uses class attributes for bounds and dimensions).
    Output: A new starting point for the next minimization phase, or `None` if no valid point is found.

    This function implements the global search strategy of the tunneling phase by randomly sampling points within the bounds 
        and applying the restoration algorithm to find a valid starting point for the next minimization phase.
    It is used when local perturbations around the last found minimizer fail to find a new valid minimizer.
    
    '''
    def random_tunnel_search(self):
        
        for i_random_start in range(self.n_random_starts):
            # Generate a random point within the bounds
            x_rand = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
            print(f"Global search: trying random point {x_rand} (attempt {i_random_start + 1})") if self.verbose else None
            # Initiate the restoration algorithm from this random point[cite: 1]
            # For random starts, the paper assumes lambda_0 = 0[cite: 1]
            res = self.run_restoration(x_rand)
            print(f"Global search: restoration algorithm returned {res} for random point {x_rand}") if self.verbose else None
            if res is not None:
                return res
        return None