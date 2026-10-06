from typing import Callable, Iterable

import optuna

def to_unique_list[V](original : Iterable[V], *, key : Callable[[V], str] | None = None):
    _tmp = set()
    _list : list[V] = []

    for it in original:
        _k = it if key is None else key(it)
        if _k not in _tmp:
            _tmp.add(_k)
            _list.append(it)

    return _list

def common_checks(names, src, tgt):
    if src == tgt:
        print('Warning, source name is same with target name!')
        return False
    elif src not in names:
        print("Warning, trying to copy study that not exist: {}".format(src))
        return False
    elif tgt in names:
        print("Warning, trying to create study that already exist: {}".format(tgt))
        return False
    
    return True

def copy_stage1_stage2(storage, src : str, tgt : str):
    names = optuna.get_all_study_names(storage = storage)

    if not common_checks(names, src, tgt):
        return
    
    # now we are creating stage 2 study basing on stage 1, shrunked range of m0,m1 and the number of stage 1 trials should be provided.
    study1 = optuna.load_study(storage=storage, study_name=src)

    # now trials are unique, therefore we can safely sort them
    trials = to_unique_list(
        filter(lambda it: it.state == optuna.trial.TrialState.COMPLETE, study1.trials),
        key=lambda it: "{m0_m1},{pivot_factor}".format_map(it.params)
    )

    from collections import defaultdict
    mapping : defaultdict[str, list[optuna.trial.FrozenTrial]] = defaultdict(list)
    for trial in trials:
        mapping[trial.params['m0_m1']].append(trial)

    smapping1 = { k:sorted(v, key=lambda t:t.values[-1], reverse=True) for k,v in mapping.items() }
    slist1 = sorted([ v[0] for v in smapping1.values() ], key=lambda t:t.values[-1], reverse=True)
    # slist1 = [ (k, v[0].params['pivot_factor']) for k,v in smapping1.items() ]

    smapping2 = { k:sorted(v, key=lambda t:t.values[-2], reverse=True) for k,v in mapping.items() }
    # slist2 = [ (k, v[0].params['pivot_factor']) for k,v in smapping2.items() ]
    slist2 = sorted([ v[0] for v in smapping2.values() ], key=lambda t:t.values[-2], reverse=True)

    # plist1 = {
    #     k: key=max(v, key=lambda it : it.values[-1]) for k,v in mapping.items()
    # }
    # plist1 = sorted(mapping.items(), key=lambda pair: max([t.values[-1] for t in pair[1]]))
    # plist2 = sorted(mapping.items(), key=lambda pair: max([t.values[-2] for t in pair[1]]))
    # plist2 = {
    #     k: max(v, key=lambda it : it.values[-2]) for k,v in mapping.items()
    # }

    # plist1 : list[tuple[str, float, float]] = list()
    # plist2 : list[tuple[str, float, float]] = list()
    # for k,v in mapping.items():
    #     # plist.append((k, v[0].values[-1]))
    #     v1 = sorted(v, key=lambda key: key.values[-1], reverse=True)
    #     v2 = sorted(v, key=lambda key: key.values[-2], reverse=True)

    #     plist1.append((k, v1[0].values[-1], v1[0].params['pivot_factor']))
    #     plist2.append((k, v2[0].values[-2], v2[0].params['pivot_factor']))

    # plist1.sort(key = lambda it: it[1], reverse=True)
    # plist2.sort(key = lambda it: it[1], reverse=True)
    # _tmp = plist1[:2] + plist2[:2]
    # _tmp = map(lambda pair: "{},{}".format(pair[0], _tmp)

    m0_m1_fp = slist1[:2] + slist2[:2]
    m0_m1_fp_list = []
    for t in m0_m1_fp:
        m0_m1 = t.params['m0_m1']
        fp = t.params['pivot_factor']

        lb = 0 if fp <= 0.1 else (8 if fp >= 0.8 else int(fp * 10) - 1)
        for it in range(3):
            m0_m1_fp_list.append('{},{}'.format(m0_m1, lb + it))

    m0_m1_fp_list = to_unique_list(m0_m1_fp_list)

    def make_new_param(trial: optuna.trial.FrozenTrial):
        params = trial.params
        return '{},{}'.format(params['m0_m1'], int(params['pivot_factor'] * 10))

    def convert_trials(trial : optuna.trial.FrozenTrial):
        distributions = dict(trial.distributions)
        del distributions['m0_m1']
        del distributions['pivot_factor']
        distributions['m0_m1_fp'] = optuna.distributions.CategoricalDistribution(m0_m1_fp_list)

        params = dict(trial.params)
        params['m0_m1_fp'] = make_new_param(trial)
        del params['m0_m1']
        del params['pivot_factor']

        return optuna.trial.create_trial(
            values = trial.values,
            params = params,
            user_attrs = trial.user_attrs,
            system_attrs = trial.system_attrs,
            distributions = distributions
        )

    study2 = optuna.create_study(
        storage=storage, 
        study_name=tgt,
        directions=['maximize'] * 3,
        sampler=None
    )

    print(m0_m1_fp_list)

    # sub_trials = [convert_trials(t) for t in filter(lambda it: '{},{}'.format(it.params['m0_m1'], int(it.params['pivot_factor'] * 10)) in m0_m1_fp_list, trials)]
    # sub_trials = list(filter(lambda t : t.params['m0_m1_fp'] in m0_m1_fp_list, map(convert_trials, trials)))
    # sub_trials = [convert_trials(t) for t in trials if make_new_param(t) in m0_m1_fp_list]
    # study2.add_trials(sub_trials)
    study2.set_user_attr('m0_m1_fp', '/'.join(m0_m1_fp_list))
    # study2.set_user_attr('num_of_copied', len(sub_trials))

def copy_stage1_stage2_by_mean(storage, src : str, tgt : str):
    names = optuna.get_all_study_names(storage = storage)

    if not common_checks(names, src, tgt):
        return
    
    # now we are creating stage 2 study basing on stage 1, shrunked range of m0,m1 and the number of stage 1 trials should be provided.
    study1 = optuna.load_study(storage=storage, study_name=src)

    # now trials are unique, therefore we can safely sort them
    trials = to_unique_list(
        filter(lambda it: it.state == optuna.trial.TrialState.COMPLETE, study1.trials),
        key=lambda it: "{m0_m1},{pivot_factor}".format_map(it.params)
    )

    from collections import defaultdict
    mapping : defaultdict[str, list[optuna.trial.FrozenTrial]] = defaultdict(list)
    for trial in trials:
        mapping[trial.params['m0_m1']].append(trial)

    smapping1 = { k:sorted(v, key=lambda t:t.values[-1], reverse=True) for k,v in mapping.items() }
    # slist1 = sorted([ v[0] for v in smapping1.values() ], key=lambda t:t.values[-1], reverse=True)
    slist1 = sorted(list(smapping1.values()), key=lambda ts: sum([t.values[-1] for t in ts]) / len(ts), reverse=True)

    smapping2 = { k:sorted(v, key=lambda t:t.values[-2], reverse=True) for k,v in mapping.items() }
    slist2 = sorted(list(smapping2.values()), key=lambda ts: sum([t.values[-2] for t in ts]) / len(ts), reverse=True)

    m0_m1_fp = slist1[:2] + slist2[:2]
    m0_m1_fp_list = []
    for vs in m0_m1_fp:
        t = vs[0]
        m0_m1 = t.params['m0_m1']
        fp = t.params['pivot_factor']

        lb = 0 if fp <= 0.1 else (8 if fp >= 0.8 else int(fp * 10) - 1)
        for it in range(3):
            m0_m1_fp_list.append('{},{}'.format(m0_m1, lb + it))

    m0_m1_fp_list = to_unique_list(m0_m1_fp_list)

    def make_new_param(trial: optuna.trial.FrozenTrial):
        params = trial.params
        return '{},{}'.format(params['m0_m1'], int(params['pivot_factor'] * 10))

    def convert_trials(trial : optuna.trial.FrozenTrial):
        distributions = dict(trial.distributions)
        del distributions['m0_m1']
        del distributions['pivot_factor']
        distributions['m0_m1_fp'] = optuna.distributions.CategoricalDistribution(m0_m1_fp_list)

        params = dict(trial.params)
        params['m0_m1_fp'] = make_new_param(trial)
        del params['m0_m1']
        del params['pivot_factor']

        return optuna.trial.create_trial(
            values = trial.values,
            params = params,
            user_attrs = trial.user_attrs,
            system_attrs = trial.system_attrs,
            distributions = distributions
        )

    study2 = optuna.create_study(
        storage=storage, 
        study_name=tgt,
        directions=['maximize'] * 3,
        sampler=None
    )

    print(m0_m1_fp_list)

    # sub_trials = [convert_trials(t) for t in filter(lambda it: '{},{}'.format(it.params['m0_m1'], int(it.params['pivot_factor'] * 10)) in m0_m1_fp_list, trials)]
    # sub_trials = list(filter(lambda t : t.params['m0_m1_fp'] in m0_m1_fp_list, map(convert_trials, trials)))
    sub_trials = [convert_trials(t) for t in trials if make_new_param(t) in m0_m1_fp_list]
    study2.add_trials(sub_trials)
    study2.set_user_attr('m0_m1_fp', '/'.join(m0_m1_fp_list))
    study2.set_user_attr('num_of_copied', len(sub_trials))

def copy_stage2_stage3(storage, src : str, tgt: str):
    names = optuna.get_all_study_names(storage = storage)

    if not common_checks(names, src, tgt):
        return
    
    study2 = optuna.load_study(storage=storage, study_name=src)
    # m0_m1_sublist = study2.user_attrs['m0_m1_sublist'].splite('/')
    # num_of_copied = study2.user_attrs['num_of_copied']

    trials = to_unique_list(
        filter(lambda it : it.state == optuna.trial.TrialState.COMPLETE, study2.trials),
        key=lambda t: "{m0_m1_fp},{m2}".format_map(t.params)
    )
    
    ordered1 = sorted(trials, key=lambda it : it.values[-1], reverse=True)
    ordered2 = sorted(trials, key=lambda it : it.values[-2], reverse=True)

    m0_m1_fp_m2_list1 = to_unique_list(map(lambda t: "{m0_m1_fp},{m2}".format_map(t.params), ordered1))[:1]
    m0_m1_fp_m2_list2 = to_unique_list(map(lambda t: "{m0_m1_fp},{m2}".format_map(t.params), ordered2))[:1]

    m0_m1_fp_m2_list = to_unique_list(m0_m1_fp_m2_list1 + m0_m1_fp_m2_list2)
    print(m0_m1_fp_m2_list1)
    print(m0_m1_fp_m2_list2)
    print(m0_m1_fp_m2_list)

    def make_new_param(trial : optuna.trial.FrozenTrial):
        return "{m0_m1_fp},{m2}".format_map(trial.params)
    
    def convert_trial(trial : optuna.trial.FrozenTrial):
        distributions = dict(trial.distributions)
        del distributions['m0_m1_fp']
        del distributions['m2']
        distributions['m0_m1_fp_m2'] = optuna.distributions.CategoricalDistribution(m0_m1_fp_m2_list)
        distributions['imp_factor'] = optuna.distributions.FloatDistribution(0, 1, False, 0.1)

        params = dict(trial.params)
        params['m0_m1_fp_m2'] = make_new_param(trial)
        del params['m0_m1_fp']
        del params['m2']

        return optuna.trial.create_trial(
            values = trial.values,
            params = params,
            user_attrs = trial.user_attrs,
            system_attrs = trial.system_attrs,
            distributions = distributions
        )

    study3 = optuna.create_study(
        storage=storage,
        study_name=tgt,
        directions = study2.directions
    )
    # sub_trials = map(convert_trial, trials)
    # sub_trials = list(filter(lambda t: t.params['m0_m1_fp_m2'] in m0_m1_fp_m2_list, sub_trials))
    # study3.add_trials([convert_trial(it) for it in trials if "{},{}".format(it.params['m0_m1_fp'], it.params['m2']) in m0_m1_fp_m2_list])
    sub_trials = [convert_trial(t) for t in trials if make_new_param(t) in m0_m1_fp_m2_list]
    study3.add_trials(sub_trials)
    study3.set_user_attr('m0_m1_fp_m2', '/'.join(m0_m1_fp_m2_list))

def copy_stage3_stage4(storage, src : str, tgt : str):
    names = optuna.get_all_study_names(storage = storage)

    if not common_checks(names, src, tgt):
        return
    
    study3 = optuna.load_study(storage=storage, study_name=src)
    # m0_m1_fp_m2_list = study3.user_attrs['m0_m1_fp_m2_list'].split('/')
    
    trials = to_unique_list(
        filter(lambda it : it.state == optuna.trial.TrialState.COMPLETE, study3.trials),
        key=lambda t: "{m0_m1_fp_m2},{m3_m4},{imp_factor}".format_map(t.params)
    )

    ordered1 = sorted(trials, key=lambda it : it.values[-1], reverse=True)
    ordered2 = sorted(trials, key=lambda it : it.values[-2], reverse=True)

    sub_trials = [ordered1[0], ordered2[0]]
    sub_ms = []

    def make_key(t : optuna.trial.FrozenTrial):
        m0, m1, _, m2 = t.params['m0_m1_fp_m2'].split(',')
        m3_m4 = t.params['m3_m4']

        return "{},{},{},{}".format(m0, m1, m2, m3_m4)

    sub_ms = to_unique_list(map(make_key, sub_trials))

    def convert_trials(t : optuna.trial.FrozenTrial):
        m0, m1, fp, m2 = t.params['m0_m1_fp_m2'].split(',')
        m3_m4 = t.params['m3_m4']

        params = dict(t.params)
        dist = dict(t.distributions)

        params = {
            'ms': "{},{},{},{}".format(m0, m1, m2, m3_m4),
            'pivot_factor': int(fp) / 10,
            'imp_factor': params['imp_factor']
        }

        dist = {
            'ms': optuna.distributions.CategoricalDistribution(sub_ms),
            'pivot_factor': optuna.distributions.FloatDistribution(0., 1., step=0.1),
            'imp_factor': optuna.distributions.FloatDistribution(0., 1., step=0.1),
        }

        return optuna.create_trial(
            values=t.values,
            params = params,
            distributions=dist,
            user_attrs=t.user_attrs,
            system_attrs=t.system_attrs,
        )
    
    study4 = optuna.create_study(
        storage=storage,
        study_name=tgt,
        directions=study3.directions
    )

    sub_trials = map(convert_trials, filter(lambda t: make_key(t) in sub_ms, trials))
    study4.add_trials(sub_trials)
    study4.set_user_attr('ms', '/'.join(sub_ms))