"""Unseal outcomes only after all geometry-only test decisions are frozen."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments.activation_drift import pilot


def risk_matched_expectation(f1,choices,risk,bins=10):
    n = len(risk)
    labels = np.empty(n,dtype=int)
    labels[np.argsort(risk,kind='stable')] = np.arange(n)*bins//n
    expected = np.empty(n)
    for b in np.unique(labels):
        mask = labels==b
        frequencies = np.bincount(choices[mask],minlength=f1.shape[1])/mask.sum()
        expected[mask] = f1[mask]@frequencies
    return expected, labels


def article_components(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    seen = {}
    for i,row in enumerate(rows):
        for title in row['supporting_facts']['title']:
            if title in seen:
                parent[find(i)] = find(seen[title])
            else:
                seen[title] = i
    roots = [find(i) for i in range(len(rows))]
    unique = {r:j for j,r in enumerate(sorted(set(roots)))}
    return np.array([unique[r] for r in roots])


def risk_matched_interval(f1,choices,risk,bins=10,draws=5000):
    expected,_ = risk_matched_expectation(f1,choices,risk,bins)
    point = float(np.mean(f1[np.arange(len(f1)),choices]-expected))
    rng=np.random.default_rng(pilot.SEED); values=[]
    for _ in range(draws):
        idx=rng.integers(len(f1),size=len(f1))
        ff,cc,rr=f1[idx],choices[idx],risk[idx]
        ee,_=risk_matched_expectation(ff,cc,rr,bins)
        values.append(float(np.mean(ff[np.arange(len(ff)),cc]-ee)))
    return {'mean':point,'ci95':np.quantile(values,[.025,.975]).tolist()}


def interval(difference,draws=5000,groups=None):
    rng = np.random.default_rng(pilot.SEED)
    difference = np.asarray(difference,dtype=float)
    if groups is None:
        totals,weights = difference,np.ones(len(difference))
    else:
        totals = np.bincount(groups,weights=difference)
        weights = np.bincount(groups)
    values=[]
    for start in range(0,draws,250):
        idx = rng.integers(len(totals),size=(min(250,draws-start),len(totals)))
        values.extend((totals[idx].sum(1)/weights[idx].sum(1)).tolist())
    return {'mean':float(difference.mean()),'ci95':np.quantile(values,[.025,.975]).tolist()}


def versus_maximum_fixed(f1,quality,draws=5000):
    rng = np.random.default_rng(pilot.SEED)
    values=[]
    for start in range(0,draws,250):
        idx = rng.integers(len(f1),size=(min(250,draws-start),len(f1)))
        values.extend((quality[idx].mean(1)-f1[idx].mean(1).max(1)).tolist())
    return {'mean':float(quality.mean()-f1.mean(0).max()),'ci95':np.quantile(values,[.025,.975]).tolist()}


def policy_stats(f1,em,choices,tokens):
    row = np.arange(len(choices)); active = choices>0
    return {'f1':float(f1[row,choices].mean()),'em':float(em[row,choices].mean()),
        'rescued':int(((em[:,0]==0)&(em[row,choices]==1)).sum()),
        'harmed':int(((em[:,0]==1)&(em[row,choices]==0)).sum()),
        'extra_calls':float(active.mean()),
        'extra_output_tokens':float((tokens[row,choices]*active).mean()),
        'counts':{pilot.ACTIONS[a]:int((choices==a).sum()) for a in range(f1.shape[1])}}


def correlation(a,b):
    if np.std(a)==0 or np.std(b)==0:
        return None
    return float(spearmanr(a,b).statistic)


def audit_predictions(key,model):
    directory = pilot.ROOT/key/'test'
    predictions = {}; n = 0; max_diff = 0.
    for path in sorted(directory.glob('batch_*.json')):
        payload = json.loads(path.read_text()); start = n
        states = np.load(path.with_suffix('.npz'))['states']
        actual = model.predict(states[:,0])
        cached_path = directory/f'predictions_{start:05d}.npz'
        meta = json.loads(cached_path.with_suffix('.json').read_text())
        if meta['sha256']!=pilot.sha(cached_path) or meta['geometry']!=pilot.sha(pilot.ROOT/key/'geometry.joblib'):
            raise ValueError('Changed policy artifact')
        if meta['time_unix']>=payload['completed_time_unix']:
            raise ValueError('Policy was not frozen before actions completed')
        if meta['ids']!=[r['id'] for r in payload['rows']]:
            raise ValueError('Prediction IDs differ')
        if payload['fingerprint']['source']!=pilot.sha(pilot.CODE/'pilot.py'):
            raise ValueError('Changed frozen collection/prediction source')
        saved = np.load(cached_path)
        for name,value in actual.items():
            d = float(np.max(np.abs(saved[name]-value)))
            max_diff = max(max_diff,d)
            np.testing.assert_allclose(saved[name],value,atol=1e-5,rtol=1e-6)
            if value.ndim==2:
                np.testing.assert_array_equal(saved[name].argmax(1),value.argmax(1))
            predictions.setdefault(name,[]).append(saved[name])
        n += len(states)
    return {k:np.concatenate(v) for k,v in predictions.items()}, {'n':n,'max_reload_difference':max_diff}


def evaluate(key):
    from experiments.recovery_help.task import score_answer
    states,rows = pilot.load_arrays(key,'test')
    model = joblib.load(pilot.ROOT/key/'geometry.joblib')
    predictions,audit = audit_predictions(key,model)
    raw = json.loads((pilot.ROOT/'data'/'test.json').read_text())
    if [r['id'] for r in raw]!=[r['id'] for r in rows]:
        raise ValueError('Gold/scoring alignment differs')
    f1 = np.array([[score_answer(a['answer'],gold['answer'])['f1'] for a in r['actions']]
                   for r,gold in zip(rows,raw)])
    em = np.array([[score_answer(a['answer'],gold['answer'])['em'] for a in r['actions']]
                   for r,gold in zip(rows,raw)])
    tokens = np.array([[a['output_tokens'] for a in r['actions']] for r in rows])
    input_tokens = np.array([[a['input_tokens'] for a in r['actions']] for r in rows])
    groups = article_components(raw)
    row = np.arange(len(f1)); best = int(f1.mean(0).argmax())
    result = {'model':key,'n':len(f1),'audit':audit,'best_fixed':pilot.ACTIONS[best],
        'article_components':int(groups.max()+1),'max_component':int(np.bincount(groups).max()),
        'initial_errors':int((em[:,0]==0).sum()),'policies':{},'fixed':{},'diagnostics':{}}
    for a,name in enumerate(pilot.ACTIONS):
        choices = np.full(len(f1),a)
        result['fixed'][name] = policy_stats(f1,em,choices,tokens)
        result['fixed'][name].update({'mean_input_tokens':float(input_tokens[:,a].mean()),
            'mean_output_tokens':float(tokens[:,a].mean()),
            'empty':float(np.mean([not r['actions'][a]['answer'] for r in rows])),
            'at_cap':float(np.mean([r['actions'][a]['at_cap'] for r in rows]))})
    risk = predictions['risk']
    rng = np.random.default_rng(pilot.SEED)
    for name,scores in predictions.items():
        if name=='risk':
            continue
        choices = scores.argmax(1); quality = f1[row,choices]
        stats = policy_stats(f1,em,choices,tokens)
        stats['vs_keep'] = interval(quality-f1[:,0])
        stats['vs_best_fixed'] = interval(quality-f1[:,best])
        stats['vs_best_fixed_cluster'] = interval(quality-f1[:,best],groups=groups)
        stats['vs_resampled_max_fixed'] = versus_maximum_fixed(f1,quality)
        expected,bins = risk_matched_expectation(f1,choices,risk)
        stats['vs_risk_matched_shuffle'] = risk_matched_interval(f1,choices,risk)
        stats['shuffle_expected_f1'] = float(expected.mean())
        perm_values=[]
        positions = [np.flatnonzero(bins==b) for b in np.unique(bins)]
        for _ in range(2000):
            perm = choices.copy()
            for pos in positions:
                perm[pos] = rng.permutation(perm[pos])
            perm_values.append(float(f1[row,perm].mean()))
        stats['routing_permutation_p_one_sided'] = float((1+np.sum(np.array(perm_values)>=quality.mean()))/2001)
        active = int((choices>0).sum())
        risky = np.argsort(risk,kind='stable')[::-1][:active]
        gated = np.repeat(f1[:,0,None],len(pilot.ACTIONS),axis=1)
        gated[risky] = f1[risky]
        gb = int(gated.mean(0).argmax())
        stats['risk_gated_best_action_at_same_calls'] = pilot.ACTIONS[gb]
        stats['vs_risk_gated_best_at_same_calls'] = interval(quality-gated[:,gb])
        result['policies'][name] = stats
    p,e = model.encode(states[:,0]); r=e-model.references['correct'].predict(p)
    x = np.concatenate([p,e],axis=1)
    actual_moves = np.stack([model.encode(states[:,a])[1]-e for a in range(len(pilot.ACTIONS))],axis=1)
    actual_scores = pilot.drift_scores(r,actual_moves)
    actual_choices = actual_scores.argmax(1)
    diagnostics = result['diagnostics']
    diagnostics['post_action_geometry_only_not_deployable'] = policy_stats(f1,em,actual_choices,tokens)
    diagnostics['post_action_geometry_vs_best_fixed'] = interval(f1[row,actual_choices]-f1[:,best])
    diagnostics['baseline_drift_error_auroc'] = float(roc_auc_score(1-em[:,0],risk))
    diagnostics['risk_mean_correct'] = float(risk[em[:,0]==1].mean())
    diagnostics['risk_mean_incorrect'] = float(risk[em[:,0]==0].mean())
    diagnostics['by_action'] = {}
    for a,name in enumerate(pilot.ACTIONS[1:],1):
        delta = actual_moves[:,a]
        mean_mse = float(np.mean((delta-model.means[a])**2))
        op_mse = float(np.mean((delta-model.operators[a].predict(x))**2))
        gains = f1[:,a]-f1[:,0]
        diagnostics['by_action'][name] = {'mean_displacement_mse':mean_mse,'operator_displacement_mse':op_mse,
            'operator_mse_reduction_fraction':1-op_mse/mean_mse,
            'predicted_vs_actual_reduction_spearman':correlation(predictions['operator'][:,a],actual_scores[:,a]),
            'predicted_reduction_vs_f1_gain_spearman':correlation(predictions['operator'][:,a],gains),
            'actual_reduction_vs_f1_gain_spearman':correlation(actual_scores[:,a],gains),
            'split_half_cosine':model.direction_diagnostics[a-1]['split_half_cosine'],
            'individual_alignment':model.direction_diagnostics[a-1]['individual_alignment'],
            'mean_energy_fraction':model.direction_diagnostics[a-1]['mean_energy_fraction']}
    diagnostics['oracle_f1'] = float(f1.max(1).mean())
    diagnostics['oracle_exact_rescues'] = int(((em[:,0]==0)&(em.max(1)==1)).sum())
    mean = model.means[1:]
    unit = mean/np.maximum(np.linalg.norm(mean,axis=1,keepdims=True),1e-9)
    diagnostics['direction_cosine_matrix'] = (unit@unit.T).tolist()
    diagnostics['frozen_parameters'] = model.details
    result['resources'] = {'generation_calls':int(len(rows)*len(pilot.ACTIONS)),
        'input_tokens':int(input_tokens.sum()),'nonterminal_output_tokens':int(tokens.sum()),
        'activation_replay_passes':int(len(rows)*len(pilot.ACTIONS))}
    primary=result['policies']['operator']
    result['primary_criterion_passed'] = bool(primary['vs_resampled_max_fixed']['ci95'][0]>0 and
        primary['vs_best_fixed_cluster']['ci95'][0]>0 and primary['vs_risk_matched_shuffle']['ci95'][0]>0)
    pilot.save(pilot.ROOT/key/'evaluation.json',result)
    np.savez_compressed(pilot.ROOT/key/'scored_test.npz',f1=f1,em=em,actual_geometry_scores=actual_scores,
                        baseline_risk=risk,groups=groups)
    print(json.dumps({'model':key,'fixed':{a:d['f1'] for a,d in result['fixed'].items()},
        'policies':{a:{k:d[k] for k in ('f1','extra_calls','vs_best_fixed','vs_risk_matched_shuffle')}
                    for a,d in result['policies'].items()},'diagnostics':diagnostics,
        'primary_criterion_passed':result['primary_criterion_passed']},indent=2),flush=True)
    return result


def validate_data():
    from experiments.recovery_scale import data
    from experiments.recovery_help import data as old
    manifest=pilot.prepare()
    splits={s:json.loads((pilot.ROOT/'data'/f'{s}.json').read_text()) for s in manifest['ids']}
    seen=set(); questions=set()
    previous=[r for root in (old.OUT,data.OUT) for p in (root/'data').glob('*.json') for r in json.loads(p.read_text())]
    oldids={r['id'] for r in previous}; oldq={old.normalize(r['question']) for r in previous}
    for name,rows in splits.items():
        ids=[r['id'] for r in rows]; qq=[old.normalize(r['question']) for r in rows]
        assert not set(ids)&(seen|oldids) and len(ids)==len(set(ids))
        assert not set(qq)&(questions|oldq) and len(qq)==len(set(qq))
        seen.update(ids); questions.update(qq)
    titles={s:{t for r in rr for t in r['supporting_facts']['title']} for s,rr in splits.items()}
    assert not titles['train']&(titles['test']|titles['validation'])
    assert not titles['validation']&titles['test']
    source=json.loads((pilot.ROOT/'source_manifest.json').read_text())
    for path,value in source['files'].items():
        assert pilot.sha(Path(__file__).resolve().parents[2]/path)==value, path
    analysis=json.loads((pilot.ROOT/'analysis_frozen.json').read_text())
    for name,value in analysis['files'].items():
        assert pilot.sha(pilot.CODE/name)==value, name
    result={'all_unique':True,'no_prior_recovery_question_overlap':True,
            'training_validation_test_support_title_embargo':True,'frozen_sources_unchanged':True,
            'frozen_analysis_unchanged':True,
            'counts':{s:len(rr) for s,rr in splits.items()}}
    pilot.save(pilot.ROOT/'data_audit.json',result)
    return result


def report():
    results=[json.loads((pilot.ROOT/k/'evaluation.json').read_text()) for k in ('qwen7b','phi35')]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for r,axrow in zip(results,axes):
        policies=r['policies']; best=r['best_fixed']
        values=[r['fixed']['keep']['f1'],r['fixed'][best]['f1'],policies['mean']['f1'],policies['operator']['f1']]
        bars=axrow[0].bar(['Keep','Best fixed\n'+best,'Mean\ndirections','Conditional\ndirections'],np.array(values)*100,
                      color=['#8996a2','#537a5a','#73a2c6','#205c8e'])
        for bar,value in zip(bars,values):
            axrow[0].text(bar.get_x()+bar.get_width()/2,100*value+.5,f'{100*value:.1f}',ha='center')
        axrow[0].set_ylabel('Answer F1 (%)'); axrow[0].set_ylim(0,100)
        axrow[0].set_title(r['model']+' · fresh test n=1,000')
        im=axrow[1].imshow(r['diagnostics']['direction_cosine_matrix'],vmin=-1,vmax=1,cmap='coolwarm')
        axrow[1].set_xticks(range(6),pilot.ACTIONS[1:],rotation=35,ha='right')
        axrow[1].set_yticks(range(6),pilot.ACTIONS[1:])
        axrow[1].set_title('Cosine between mean mitigation displacements')
    fig.colorbar(im,ax=axes[:,1],shrink=.75,label='Cosine similarity')
    fig.savefig(pilot.ROOT/'geometry_results.png',dpi=180)
    fig.savefig(pilot.ROOT/'geometry_results.pdf')
    lines=['# Activation drift and mitigation directions: exploratory pilot','',
        'This is a completed feasibility test, with no mitigation-success labels used in fitting or selecting the geometry policies. '
        'Baseline correctness labels define a reference trajectory. The collector and policy were frozen before held-out repair outcomes were scored.','',
        '## Method','',
        'For each backbone, 800 fresh calibration questions estimate activation coordinates and directions; '
        '200 separate questions select regularization by activation-prediction error only. '
        'The independent test contains 1,000 fresh questions. Six non-keep strategies each cost one extra generation; '
        'reasoning and direct answers have different token caps.','',
        '1. Compress six layers of prompt-end and final-token decision states with training-only PCA (16 dimensions per layer/time).',
        '2. Learn a typical correct endpoint F(p) from correct ordinary baseline answers. Define drift r=e−F(p).',
        '3. Estimate each strategy’s mean displacement and a ridge field Δa(p,e), using activation pairs without scoring their success.',
        '4. Choose the action maximizing (||r||²−||r+Δa||²)/96, with keep fixed at zero.','',
        'The estimated endpoint is not a ground-truth correct state for a new question. '
        'The entire method depends on this reference geometry having useful meaning.','',
        '## Primary results','',
        '| Backbone | Keep F1 | Best fixed F1 | Mean directions | Conditional directions | Conditional − best fixed, 95% CI |',
        '|---|---:|---:|---:|---:|---|']
    def fmt(stat):
        return f"{100*stat['mean']:+.2f} [{100*stat['ci95'][0]:+.2f}, {100*stat['ci95'][1]:+.2f}]"
    for r in results:
        pp=r['policies']; fixed=r['fixed']; best=r['best_fixed']
        lines.append(f"| {r['model']} | {100*fixed['keep']['f1']:.2f} | {100*fixed[best]['f1']:.2f} ({best}) | "
                     f"{100*pp['mean']['f1']:.2f} | {100*pp['operator']['f1']:.2f} | {fmt(pp['operator']['vs_best_fixed'])} |")
    passed=all(r['primary_criterion_passed'] for r in results)
    lines += ['',f"The prespecified two-backbone criterion **{'passed' if passed else 'did not pass'}**. "
        'The best fixed comparator is a hindsight benchmark across seven fixed choices; it is not a tuned deployment policy. '
        'The full evaluation also resamples the maximum across fixed strategies within each bootstrap.','',
        '## What the geometry learned','',
        '| Backbone | Mean-direction split-half cosine (range) | Drift predicts initial error (AUROC) | Operator vs risk-matched shuffled routing, 95% CI |',
        '|---|---:|---:|---|']
    for r in results:
        d=r['diagnostics']; cos=[x['split_half_cosine'] for x in d['by_action'].values()]
        lines.append(f"| {r['model']} | {min(cos):.3f}–{max(cos):.3f} | {d['baseline_drift_error_auroc']:.3f} | "
                     f"{fmt(r['policies']['operator']['vs_risk_matched_shuffle'])} |")
    lines += ['', 'Stable directions establish repeatable activation changes. They do not establish that those '
        'changes repair errors. Prompt wording, answer anchoring, evidence length and answer format can all contribute to them.','',
        '## Per-model diagnostics and controls','']
    for r in results:
        lines += [f"### {r['model']}",'',
            '| Policy | F1 | Extra calls/question | Rescued exact errors | Damaged initially exact answers |',
            '|---|---:|---:|---:|---:|']
        for name,p in r['policies'].items():
            lines.append(f"| {name} | {100*p['f1']:.2f} | {p['extra_calls']:.3f} | {p['rescued']} | {p['harmed']} |")
        d=r['diagnostics']; post=d['post_action_geometry_only_not_deployable']; op=r['policies']['operator']
        lines += ['',f"Using actual post-action states with the same geometric score gives F1 {100*post['f1']:.2f}. "
            'This requires running every action and is only a diagnostic of the reference metric; it is not a deployable selector.',
            f"Question bootstrap conditional-minus-best-fixed: {fmt(op['vs_best_fixed'])} F1 points. "
            f"Article-component bootstrap: {fmt(op['vs_best_fixed_cluster'])}; "
            f"resampled best-fixed maximum: {fmt(op['vs_resampled_max_fixed'])}. "
            f"There are {r['article_components']} article components; the largest has {r['max_component']} questions.",'',
            '| Action | Operator MSE reduction vs mean | Actual geometric improvement vs F1 gain (Spearman) |',
            '|---|---:|---:|']
        for name,a in d['by_action'].items():
            rho=a['actual_reduction_vs_f1_gain_spearman']
            rho_text='undefined (constant values)' if rho is None else f'{rho:.3f}'
            lines.append(f"| {name} | {100*a['operator_mse_reduction_fraction']:.1f}% | {rho_text} |")
        lines += ['', 'Conditional policy action counts: '+json.dumps(op['counts'])+'.','']
    lines += ['## Interpretation and limitations','',
        '- The study tests one precise drift definition: deviation from a ridge-predicted typical correct endpoint. It does not rule out other drift representations.',
        '- Geometry fitting uses baseline correctness; it is not an entirely unsupervised method. No action-success target, action-quality tuning, text feature, or confidence feature enters the selector.',
        '- A geometric move toward a reference region does not necessarily add missing facts or repair reasoning. An error-predictive feature need not be a useful repair objective.',
        '- The mean displacement is not a proven causal steering vector. The actions change prompts and evidence; their differences include those effects. No activation injection was performed.',
        '- The end measurement precedes the last nonterminal generated token, often closing answer markup. It is not necessarily the decision that selected the semantic answer.',
        '- The primary score evaluates a predicted mean displacement and omits conditional action variance. Expected squared residual would also contain the trace of the displacement covariance. A separately frozen pre-scoring supplement tests a calibration-error penalty; it does not replace the primary policy.',
        '- Risk-decile permutations control a coarse baseline drift ordering; they do not rule out every finer risk-only decision rule.',
        '- States are measured by Transformers replay of exact vLLM token sequences. Same capture procedure for all actions; cross-engine activation equality is not claimed. Native-generation smoke checks are in each model folder.',
        '- Each action costs one additional generation, but generated tokens and input lengths differ. Replay incurs measurement overhead; no end-to-end speedup is claimed.',
        '- Intervals condition on these fitted models and recorded outputs. This is two backbones, one task, one seed, and one trajectory reference. No repeat-generation or task-transfer claim.',
        '- A retry can preserve the original answer and yield little change; a more useful retrieval can move far away. Penalizing movement may therefore prefer ineffective but small displacements.',
        '- Correctness probes plus a constant additive direction have a simple limitation: for linear score wᵀh+b, its change is wᵀv, independent of the question. State-dependent routing requires additional structure, such as the tested conditional displacement and nonlinear squared-distance objective.','',
        '## Related work and potential contribution','',
        '[Function Vectors](https://arxiv.org/abs/2310.15213) studies internal task vectors. '
        '[Activation Transport](https://arxiv.org/abs/2410.23054) studies transport of activations. '
        '[Understanding (Un)Reliability of Steering Vectors](https://arxiv.org/abs/2505.22637) relates directional coherence to steering. '
        '[Predicting Where Steering Vectors Succeed](https://arxiv.org/abs/2604.15557) studies geometric diagnostics for steering effectiveness. '
        'A direction dictionary alone would not establish novelty. The proposed contribution would require showing that drift/action geometry predicts external repair choice without action-outcome supervision, and eventually transfers to a new strategy with unlabelled calibration. That claim is not established by merely finding repeatable directions.','',
        '## Artifacts and reproduction','',
        '- `experiments/activation_drift/README.md`: method and supplementary design.',
        '- `pilot.py`: calibration, exact-token state capture, geometry fitting, and choice freezing.',
        '- `analyze.py`: audited label unsealing, comparisons, controls, and plots.',
        '- `data_manifest.json`, `source_manifest.json`, `data_audit.json`: provenance and checks.',
        '- `<model>/geometry.joblib`, `geometry_fit.json`, `evaluation.json`: saved method and evidence.',
        '- `<model>/{train,validation,test}/`: raw token sequences, states, and test predictions saved before mitigation generation.',
        '- `geometry_results.png` / `.pdf`: comparison and direction similarity plot.','',
        'Run the module commands in REPRODUCE.md from the repository root. '
        'Setup freezes the packaged sources before collection. Results are written under the configured '
        'ACTIVATION_RUNS directory; original recorded summaries remain under results/.','']
    (pilot.ROOT/'REPORT.md').write_text('\n'.join(lines))
    pilot.save(pilot.ROOT/'summary.json',{'models':results,'two_backbone_criterion_passed':passed})
    print('Report and plots written to '+str(pilot.ROOT))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--model',choices=('qwen7b','phi35'))
    parser.add_argument('--report',action='store_true'); args=parser.parse_args()
    validate_data()
    if args.report:
        report()
    else:
        evaluate(args.model)
