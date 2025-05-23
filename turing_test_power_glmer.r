# Power calculations for Clinician Turing Test
# We are going to use a mixed-effects logistic regression model to analyze the results with crossed random effects for vignette and participant.

library(data.table)
library(ggplot2)
library(ggsci)
library(glue)
library(TOSTER)
library(lme4)
library(Hmisc)
library(furrr)
library(doFuture)
registerDoFuture()
plan(multisession)
set.seed(24601)
```

# Mixed effects logistic regression model with Equivalence design
# Summary of important variables in estimating power through a simulation approach:

# `N_participants`: a vector of the candidate number of participants who will complete the survey instrument
# `N_vignettes`: a vector of the candidate number of vignettes that each participant will complete (not including the one internal control vignette)
# `p_guess_range`: a vector of the candidate baseline guess proportions for correctly guessing a "human" treatment profile
# `equiv_margin_probs`: the +/- equivalence margin in the design (delta)
# `fixed_effect_sizes_probs`: a vector of the candidate effect sizes as the difference from the baseline guess probability for guessing correctly when presented with an "AI" treatment profile. 
# `alpha`: the Type I error rate = 0.05
# `N_SIMS`: the number of simulations to run for each combination of parameters, default = 1000
# `Vignette_RE_SD`: the standard deviation of the vignette-level random effect standard error
# `Part_RE_SD`: the standard deviation of the participant-level random effect standard error
# `ResidSD`: residual error term

N_participants <- seq(50, 500, by = 50)
N_vignettes <- c(6, 8, 10) # make even numbers for balanced randomization (NB does not include the internal control vignette which is not randomized)
p_guess_range <- c(0.5) # this is the baseline probability of guessing the right answer in the data generation process
fixed_effect_sizes_probs <- seq(0.01, 0.10, by = 0.01) # the possible change in probability of guessing correctly when assigned to treatment arm (i.e. get AI suggestions)
equiv_margin_probs <- seq(0.05, .20, by = 0.05) # the equivalence margin for the equivalence test, i.e +/- this %
alpha <- 0.05       # significance level
N_SIMS <- 1000 # number of simulations for each combination of study features
Vignette_RE_SD <- 0.03
Part_RE_SD <- 0.03
ResidSD <- 0.03

combos_dt <- expand.grid(npar = N_participants, 
                      nvig = N_vignettes, 
                      pguess = p_guess_range,
                      feffsize = fixed_effect_sizes_probs,
                      equiv_margin = equiv_margin_probs) |> as.data.table()
combos_dt[, alpha := alpha]

combos_dt[, index := .I]

make_sim_data <- function(npar, nvig, pguess, feffsize) {
  # Create a balanced set for each participant
  assignments_per_participant <- c(rep(0, ceiling(nvig/2)), rep(1, ceiling(nvig/2)))
  # Randomize the assignment order for each participant
  assignment <- unlist(lapply(1:npar, function(x) sample(assignments_per_participant)[1:nvig]))

  dd <- data.table(
    participant_id = rep(1:npar, each = nvig),
    vignette_id = rep(1:nvig, times = npar),
    assignment = as.factor(assignment)
  )
    # Adjust probabilities for vignettes and participants random effects
    vign_level_re <- rnorm(n = nvig, mean = 0, sd = Vignette_RE_SD)
    part_level_re <- rnorm(n = npar, mean = 0, sd = Part_RE_SD)
    
    dd[, idx := .I]
    
    # Adapted from Supplement from: Arnold et al.: Simulation methods to estimate design power: an overview for applied research. BMC Medical Research Methodology 2011 11:94.
    
    dd[, prob := pmax(pmin(pguess + 
                        feffsize * as.numeric(as.character(assignment)) + 
                        vign_level_re[vignette_id] + 
                        part_level_re[participant_id] + 
			rnorm(n = 1, mean = 0, sd = ResidSD), 0.999), 0.001), by = idx]

  dd[, guess_correct := rbinom(n = 1, size = 1, prob = prob), by = idx]
  
  dd[, c('prob', 'idx') := NULL] # clean up

  return(dd)
}

get_power <- function(npar, nvig, pguess, feffsize, alpha, equiv_margin) {
  pval_list <- sapply(1:N_SIMS, \(x) {
      sim_data_dt <- make_sim_data(npar, nvig, pguess, feffsize)
      stopifnot(length(unique(sim_data_dt$assignment)) == 2)
      if (length(unique(sim_data_dt$guess_correct)) != 2) {
        message(glue('\nError: Outcomes does not have 2 unique values for {feffsize} effect size, {npar} participants, {nvig} vignettes, and {pguess} probability of a correct guess. Returning p-value = 1.'))
        return(1)
      }
  #Need try/catch here in case the model doesn't converge
      tryCatch({
          design <- glmer(guess_correct ~ assignment + (1|vignette_id) + (1|participant_id), 
                           data = sim_data_dt, 
                           family = 'binomial',
                          control = glmerControl(check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4))) # ignore fit warnings
        model_res <- summary(design)$coefficients['assignment1',]
        tost_res <- tsum_TOST(m1 = model_res[1], 
                              sd1 = model_res[2] * sqrt(nrow(sim_data_dt)),
                              n1 = nrow(sim_data_dt),
                              eqb = c(qlogis(max(0.0001, pguess - equiv_margin)),
                                       qlogis(min(0.9999, pguess + equiv_margin))))
        tost_p_lower <- tost_res$TOST$p.value[2]
        tost_p_upper <- tost_res$TOST$p.value[3]
        this_pval <- max(tost_p_lower, tost_p_upper) # take the max p-value for both tests as the TOST p-value, which is convention for TOSTs per Lakens
        return(this_pval)},
        error = function(err) {
          message(glue("Error: Model didn't converge for feffsize: {feffsize}, npar: {npar}, nvig: {nvig}, pguess: {pguess}, so returning p-value = 1"))
          return(1)
        })
  })
  # If the p-value is NA for non-convergence, make it 1
  pval_list <- sapply(pval_list, \(x) ifelse(is.na(x) || is.null(x), 1, x))
  res <- binconf(sum(pval_list < alpha), length(pval_list))
  return(as.data.frame(res))
}
  
# NB there are some issues with scope in using future: 
# https://cran.r-project.org/web/packages/future/vignettes/future-4-issues.html  
res_test <- future_pmap(combos_dt, 
                   \(npar, nvig, pguess, feffsize, alpha, equiv_margin, ...) get_power(npar, nvig, pguess, feffsize, alpha, equiv_margin),
                   .progress = TRUE, seed = TRUE,
                   .options = furrr_options(seed = 24601, packages = c('TOSTER', 'glue'))) |> 
  rbindlist()

combos_dt[, c('calc_power', 'ci_lo', 'ci_hi') := res_test]
combos_dt[, index := NULL]

fwrite(combos_dt, 'mer_dt.csv') 

# which design combinations get us to sufficient power?

ggplot(combos_dt, aes(npar, calc_power, color = as.factor(equiv_margin))) + 
         geom_line() +
         geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.5) +
	 geom_hline(yintercept = 0.8, color = 'black', linewidth = 0.5, linetype = 'dotdash') + 
         facet_grid(feffsize ~ nvig) + 
         scale_y_continuous('Power') +
         scale_x_continuous('Participants') +
         scale_color_aaas(name = 'Equivalence margin\n(+/- probability)') +
         theme_bw()



# Clean up
plan(sequential)
sessionInfo()

