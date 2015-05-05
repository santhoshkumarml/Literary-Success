__author__ = 'santhosh'

def normalize_dist(feature_dict_for_file, diff_feature_vals, doNomalize = False):
    sum_of_production_rules = sum(feature_dict_for_file.values())
    if doNomalize:
        feature_dict_for_file = {k:(feature_dict_for_file[k]/sum_of_production_rules)\
            if k in feature_dict_for_file else 0.0 for k in diff_feature_vals}
    else:
        feature_dict_for_file = {k: feature_dict_for_file[k]\
            if k in feature_dict_for_file else 0.0 for k in diff_feature_vals}
    return feature_dict_for_file