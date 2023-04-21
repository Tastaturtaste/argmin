use crate::core::{
    ArgminFloat, CostFunction, DeserializeOwnedAlias, Executor, IterState, LineSearch,
    OptimizationResult, Problem, SerializeAlias, Solver, State,
};
use crate::solver::brent::BrentOpt;
use argmin_math::{ArgminAdd, ArgminDot, ArgminScaledAdd, ArgminSub, ArgminZeroLike};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::mem;
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
/// Implementation of the powell method for the optimization of mulitvariate functions without the need for
/// derivative information.
pub struct Powell<P, F> {
    search_vectors: Vec<P>,
    param_tol: P,
    func_tol: F,
}

impl<P, F> Powell<P, F> {
    /// Initialize the Powell optimizer
    pub fn new(initial_search_vectors: Vec<P>, param_tol: P, func_tol: F) -> Self {
        Powell {
            search_vectors: initial_search_vectors,
            param_tol,
            func_tol,
        }
    }
}

impl<O, P, F> Solver<O, IterState<P, (), (), (), F>> for Powell<P, F>
where
    O: CostFunction<Param = P, Output = F>,
    P: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminAdd<P, P>
        + ArgminZeroLike
        + ArgminSub<P, P>
        + ArgminDot<P, F>
        + ArgminScaledAdd<P, F, P> + std::fmt::Debug,
    F: ArgminFloat,
{
    const NAME: &'static str = "Powell";
    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, (), (), (), F>,
    ) -> Result<(IterState<P, (), (), (), F>, Option<crate::core::KV>), anyhow::Error> {
        let init_param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`Powell` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let cost = if state.get_cost().is_infinite() {
            problem.cost(&init_param)?
        } else {
            state.cost
        };
        Ok((state.param(init_param).cost(cost), None))
    }
    /// An iteration of the powell method as implemented in scipy
    fn next_iter(
        &mut self,
        problem: &mut crate::core::Problem<O>,
        state: IterState<P, (), (), (), F>,
    ) -> Result<(IterState<P, (), (), (), F>, Option<crate::core::KV>), anyhow::Error> {
        let mut param = state
            .get_param()
            .cloned()
            .ok_or_else(argmin_error_closure!(NotInitialized, "not initialized"))?; // TODO add Error message
        let mut cost = state.get_cost();

        // new displacement vector created from displacement vectors of line searches
        let mut best_direction: (usize, F) = (0, float!(0.0));

        // Loop over all search vectors and perform line optimization along each search direction
        for (i, search_vector) in self.search_vectors.iter().enumerate() {
            let inner_cost = problem.take_problem().unwrap();
            let BracketResult {
                xa,
                xc,
                func_eval_count,
            } = bracket(
                |step| inner_cost.cost(&param.scaled_add(step, search_vector)),
                Default::default(),
            )?;
            // TODO: Find way to add func_eval_counts from bracket to the overall counter
            let mut brent_line_search = BrentOptLineSearch::new(xa, xc);
            brent_line_search.search_direction(search_vector.to_owned());
            brent_line_search.initial_step_length(float!(1e-3))?;

            // Run solver
            let OptimizationResult {
                problem: line_problem,
                state: line_state,
                ..
            } = Executor::new(inner_cost, brent_line_search)
                .configure(|state| state.param(param.clone()).cost(cost))
                .ctrlc(false)
                .run()?;
            problem.consume_problem(line_problem);

            // update new displacement vector
            let IterState {
                best_param,
                best_cost,
                ..
            } = line_state;
            let best_param = best_param.expect(
                "The best parameters should always be set and at least correspond 
            to the initial parameters passed in.",
            );

            //store index of displacement vector with greatest cost improvement
            let cost_improvement = cost - best_cost;
            if best_direction.1 < cost_improvement {
                best_direction.0 = i;
                best_direction.1 = cost_improvement;
            }

            // Update parameters and cost
            param = best_param;
            cost = best_cost;
        }
        // replace best performing search direction with new search direction
        let _ = mem::replace(
            &mut self.search_vectors[best_direction.0],
            param.sub(state.get_param().unwrap()),
        );

        // Update state
        let mut state = state;
        if float!(2.) * (state.cost - cost)
            <= (self.func_tol * (state.cost.abs() + cost.abs()) + F::epsilon())
        {
            state = state.terminate_with(crate::core::TerminationReason::SolverExit(
                "Improvement of cost per iteration below specified tolerance.".to_owned(),
            ));
        }
        let state = state.param(param).cost(cost);
        Ok((state, None))
    }
}

#[derive(Clone, Copy, Debug)]
struct BracketOptions<F> {
    xa: F,
    xb: F,
    grow_limit: F,
    max_iter: usize,
}
impl<F: ArgminFloat> Default for BracketOptions<F> {
    fn default() -> Self {
        Self {
            xa: float!(0.),
            xb: float!(1.),
            grow_limit: float!(110.),
            max_iter: 1000,
        }
    }
}

struct BracketResult<F> {
    xa: F,
    xc: F,
    func_eval_count: usize,
}

// Implementation follows scipy
// https://github.com/scipy/scipy/blob/c1ed5ece8ffbf05356a22a8106affcd11bd3aee0/scipy/optimize/_optimize.py#L2811
fn bracket<F: ArgminFloat>(
    f: impl Fn(&F) -> Result<F, crate::core::Error>,
    options: BracketOptions<F>,
) -> Result<BracketResult<F>, crate::core::Error> {
    let golden_ratio: F = float!(1.618_033_988_749_895);
    let BracketOptions {
        mut xa,
        mut xb,
        grow_limit,
        max_iter,
    } = options;
    let mut func_eval_count = 0;
    macro_rules! f_counted {
        ($x:expr) => {{
            func_eval_count += 1;
            f($x)
        }};
    }
    let mut fa = f_counted!(&xa)?;
    let mut fb = f_counted!(&xb)?;
    if fa < fb {
        std::mem::swap(&mut xa, &mut xb);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut xc = xb + golden_ratio * (xb - xa);
    let mut fc = f_counted!(&xc)?;
    let mut iter = 0;
    while fc < fb {
        let tmp1 = (xb - xa) * (fb - fc);
        let tmp2 = (xb - xc) * (fb - fa);
        let val = tmp2 - tmp1;
        let denom = if val.abs() < F::epsilon() {
            float!(2.) * F::epsilon()
        } else {
            float!(2.) * val
        };
        let mut w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
        let wlim = xb + grow_limit * (xc - xb);
        if iter > max_iter {
            return Err(crate::core::Error::msg("Too many iterations"));
        }
        iter += 1;
        let mut fw;
        if (w - xc) * (xb - w) > float!(0.0) {
            fw = f_counted!(&w)?;
            if fw < fc {
                xa = xb;

                return Ok(BracketResult {
                    xa,
                    xc,
                    func_eval_count,
                });
            } else if fw > fb {
                xc = w;

                return Ok(BracketResult {
                    xa,
                    xc,
                    func_eval_count,
                });
            }
            w = xc + golden_ratio * (xc - xb);
            fw = f_counted!(&w)?;
        } else if (w - wlim) * (wlim - xc) >= float!(0.0) {
            w = wlim;
            fw = f_counted!(&w)?;
        } else if (w - wlim) * (xc - w) > float!(0.0) {
            fw = f_counted!(&w)?;
            if fw < fc {
                xb = xc;
                xc = w;
                w = xc + golden_ratio * (xc - xb);
                fb = fc;
                fc = fw;
                fw = f_counted!(&w)?
            }
        } else {
            w = xc + golden_ratio * (xc - xb);
            fw = f_counted!(&w)?;
        }
        xa = xb;
        xb = xc;
        xc = w;
        fa = fb;
        fb = fc;
        fc = fw;
    }
    Ok(BracketResult {
        xa,
        xc,
        func_eval_count,
    })
}

struct ScalarizedObjective<O, P> {
    cost_fn: O,
    init_point: P,
    direction: P,
}
impl<O, F, P> CostFunction for ScalarizedObjective<O, P>
where
    O: CostFunction<Param = P, Output = F>,
    P: ArgminScaledAdd<P, F, P>,
    F: ArgminFloat,
{
    type Param = F;

    type Output = F;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, anyhow::Error> {
        self.cost_fn
            .cost(&self.init_point.scaled_add(param, &self.direction))
    }
}

struct BrentOptLineSearch<F, P> {
    brent: BrentOpt<F>,
    brent_state: IterState<F, (), (), (), F>,
    init_param: Option<P>,
    init_cost: F,
    search_direction: Option<P>,
    step_length: F,
}
impl<F: ArgminFloat, P> LineSearch<P, F> for BrentOptLineSearch<F, P> {
    fn search_direction(&mut self, direction: P) {
        self.search_direction = Some(direction);
    }

    fn initial_step_length(&mut self, step_length: F) -> Result<(), crate::core::Error> {
        if !step_length.is_normal() {
            return Err(crate::core::Error::msg("Initial step length not normal"));
        } else if step_length.is_zero() {
            return Err(crate::core::Error::msg("Initial step length of zero"));
        }
        self.step_length = step_length;
        Ok(())
    }
}
impl<F: ArgminFloat, P> BrentOptLineSearch<F, P> {
    fn new(min: F, max: F) -> Self {
        Self {
            brent: BrentOpt::new(min, max),
            brent_state: IterState::new(),
            init_param: None,
            init_cost: F::infinity(),
            search_direction: None,
            step_length: F::epsilon(),
        }
    }
}

impl<O, F, P> Solver<O, IterState<P, (), (), (), F>> for BrentOptLineSearch<F, P>
where
    O: CostFunction<Param = P, Output = F>,
    P: Clone + ArgminScaledAdd<P, F, P>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Brent-LS";

    fn init(
        &mut self,
        problem: &mut crate::core::Problem<O>,
        mut state: IterState<P, (), (), (), F>,
    ) -> Result<(IterState<P, (), (), (), F>, Option<crate::core::KV>), anyhow::Error> {
        check_param!(
            self.search_direction,
            concat!(
                "`BrentRootLineSearch`: Search direction not initialized. ",
                "Call `search_direction` before executing the solver."
            )
        );

        let init_param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`BrentRootLineSearch` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let cost = state.get_cost();
        self.init_cost = if cost.is_finite() {
            cost
        } else {
            problem.cost(&init_param)?
        };
        let mut scalarized_problem = Problem::new(ScalarizedObjective {
            cost_fn: problem.take_problem().unwrap(),
            init_point: init_param,
            direction: self.search_direction.take().unwrap(),
        });
        let (brent_state, brent_kv) = self.brent.init(
            &mut scalarized_problem,
            std::mem::replace(&mut self.brent_state, IterState::new()),
        )?;
        self.brent_state = brent_state;
        {
            let ScalarizedObjective {
                cost_fn,
                init_point,
                direction,
            } = scalarized_problem.take_problem().unwrap();

            problem.problem = Some(cost_fn);
            self.init_param = Some(init_point);
            self.search_direction = Some(direction);
        }
        problem.consume_func_counts(scalarized_problem);
        Ok((
            state.param(self.init_param.clone().unwrap()).cost(cost),
            brent_kv,
        ))
    }
    fn next_iter(
        &mut self,
        problem: &mut crate::core::Problem<O>,
        state: IterState<P, (), (), (), F>,
    ) -> Result<(IterState<P, (), (), (), F>, Option<crate::core::KV>), anyhow::Error> {
        let scalarized_objective = ScalarizedObjective {
            cost_fn: problem.take_problem().unwrap(),
            init_point: self.init_param.take().unwrap(),
            direction: self.search_direction.take().unwrap(),
        };
        let mut scalarized_problem = Problem::new(scalarized_objective);
        let (brent_state, brent_kv) = self.brent.next_iter(
            &mut scalarized_problem,
            std::mem::replace(&mut self.brent_state, IterState::new()),
        )?;
        self.brent_state = brent_state;
        {
            let ScalarizedObjective {
                cost_fn,
                init_point,
                direction,
            } = scalarized_problem.take_problem().unwrap();

            problem.problem = Some(cost_fn);
            self.init_param = Some(init_point);
            self.search_direction = Some(direction);
        }

        let state = state
            .param(self.init_param.as_ref().unwrap().scaled_add(
                self.brent_state.param.as_ref().unwrap(),
                self.search_direction.as_ref().unwrap(),
            ))
            .cost(self.brent_state.cost);
        let state = if let Some(termination_reason) = self.brent_state.get_termination_reason() {
            state.terminate_with(termination_reason.to_owned())
        } else {
            state
        };
        Ok((state, brent_kv))
    }
}
