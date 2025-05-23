# Power calculations for Turing test as fitting of binary classifier

# NB install packages with:
# using Pkg
# Pkg.add("PackageName")
using StatsBase, StatsModels, GLM, Distributions, DataFrames, HypothesisTests, CSV, ProgressMeter, Random

Random.seed!(24601)

# Set variable definitions for simulation
N_SIMS = 1000 # Number of simulations to run for each combination of parameters
C_STAT_BASE_V = [0.5] # Baseline c-statistics of correct guess
C_STAT_DIFF_V = [0.05, 0.10, 0.15, 0.20] # Difference in C-statistic, +/- the equivalence margin
EFF_SIZE_PROBS_V = [0.01:0.01:0.10;] # the possible change in probability of guessing correctly when assigned to treatment arm (i.e. get AI suggestions)
N_PAR_V = [50:50:500;] # Number of participants
N_VIG_V = [6, 8, 10] # Number of vignettes each participant will see (NB does _not_ include the internal control)
VIG_RE_SD = 0.03 # the standard deviation of the vignette-level random effect standard error
PART_RE_SD = 0.03 # the standard deviation of the participant-level random effect standard error
RESID_SD = 0.03 # Residual error
ALPHA =  0.05 # the Type I error rate = 0.05
CSTAT_TOL = 0.01 # the tolerance of the performance of the model to desired c-statistic in the simulated data

# --------- Function to calculate the c-stat --------
function cstat(preds, obs)
    o = preds[obs .== 1] .- preds[obs .==0]'
    cs = mean((o .> 0) .+ .5 .* (o .== 0))
    return(cs)
end

# --------- Function to calculate the bootstrapped c-stat and return max empiric p-value on both sides  --------
function boot_cstat_pval(preds, obs, cstat_val, cstat_diff, reps = 1000)
    cvals = []
    for n in 1:reps
        idxs = sample(1:length(preds), length(preds), replace = true)
        push!(cvals, cstat(preds[idxs], obs[idxs]))
    end
    p_test_lo = mean(cvals .< (cstat_val - cstat_diff))
    p_test_hi = mean(cvals .> (cstat_val + cstat_diff))
    return(max(p_test_lo, p_test_hi))
end

# --------- Function to make simulated data --------
function make_sim_data(npar, nvig, cstat_base, cstat_diff, eff_size, resid_sd)
    assignments_per_participant = [repeat([0], outer = Int(nvig / 2)); repeat([1], outer = Int(nvig / 2))]
    assignment = map(N -> sample(assignments_per_participant,  nvig), 1:npar)
    assignment = reduce(vcat, assignment)

    dd = DataFrame(participant_id = repeat(1:npar, inner = nvig),
                    vignette_id = repeat(1:nvig, outer = npar),
                    assignment = assignment) # should be categorical
  
    # Loosely based on Riley et al 2021. Adapted from the R code.
    # Create the outcome which is sensitive to the treatment assignment
    dd.prob = map(assign -> 0.5 + eff_size * assign + rand(Normal(0, resid_sd), 1)[1], dd.assignment)
    # Ensure probability is within appropriate bounds
    dd.prob = map(newp -> min(max(newp, 0.001), 0.999), dd.prob)
    dd.guess_ai = map(p -> rand(Binomial(1, p), 1)[1], dd.prob)
    return(dd)
end


# --------- Function to fit model and return pvalue --------
# Use all data without a train/test split
# Overfitting here would bias us toward a "better" model and thus away from equivalence 
function one_iteration(npar, nvig, cstat_base, cstat_diff, alpha, eff_size, resid_sd)
    thisdata = make_sim_data(npar, nvig, cstat_base, cstat_diff, eff_size, resid_sd)
    thispreds = fit_and_predict(thisdata)
    pv = boot_cstat_pval(thispreds, 
            thisdata[:, "assignment"], # thisdata[thisdata.split .== "test", "assignment"]
            cstat_base, cstat_diff)
    return(pv < alpha)
end

# --------- Function to fit model and return pvalue --------
function fit_and_predict(data)
    # Train logistic regression model
    fm = @formula(assignment ~ guess_ai)
    logit_model = glm(fm, data, Binomial(), LogitLink())
    all_preds = predict(logit_model, data)
    return(all_preds)
end


# --------- Function to estimate power --------
function get_power(npar, nvig, n_sims, cstat_base, cstat_diff, alpha, eff_size, resid_sd)
    psig = map(N -> one_iteration(npar, nvig, cstat_base, cstat_diff, alpha, eff_size, resid_sd),
                1:n_sims)
    power = mean(psig)
    cilo, cihi = confint(BinomialTest(sum(psig), length(psig)))
    return((power, cilo, cihi))
end

# --------- Now run it all
combs_df = allcombinations(DataFrame, 
                    cstat_base = C_STAT_BASE_V, 
                    cstat_diff = C_STAT_DIFF_V,
                    nvig = N_VIG_V,
                    npar = N_PAR_V,
                    eff_size = EFF_SIZE_PROBS_V)

power_list = []

@showprogress Threads.@threads for ii in 1:nrow(combs_df)
    res = get_power(combs_df.npar[ii], combs_df.nvig[ii], N_SIMS,
            combs_df.cstat_base[ii], combs_df.cstat_diff[ii],
            ALPHA, combs_df.eff_size[ii], RESID_SD)
    push!(power_list, res)
end


final_power_res = DataFrame(power_list, ["power", "ci_lo", "ci_hi"])
final_power_res = hcat(combs_df, final_power_res)
CSV.write("final_power_results.csv", final_power_res)


