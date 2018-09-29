def update_probability(prior_probability, test_name, distribution, test_result, debug):

    # Account for evidences

    if test_name in distribution.keys(): ## First, let's check if we have probabilities for the requested test_name
        ## If so, we use these probabilities instead of the default ones
        likelihood = distribution[test_name][test_result]['Positive']
        non_likelihood = distribution[test_name][test_result]['Negative']
    else:  ## if not, we go for generic values
        likelihood = 0.5
        non_likelihood = 0.5
        test_result = 'unknown'

    # print(test_name, likelihood, non_likelihood)
    numerator = likelihood * prior_probability
    denominator = (likelihood * prior_probability) + (non_likelihood * (1 - prior_probability))
    conditional_probability = numerator / denominator

    if debug:
        print('\t* "{}" is {} ({:.2f}). Updating {:.2f} to {:.2f}'.format(test_name, test_result, likelihood, prior_probability, conditional_probability))

    return conditional_probability

def analyse_events(prior, events, probabilities, debug):
    posterior = prior
    for ll in probabilities.keys():
        if ll not in events:
            posterior = update_probability(posterior, ll, probabilities, 'False')
    for ee in events:
        posterior = update_probability(posterior, ee, probabilities, 'True')

    return posterior
