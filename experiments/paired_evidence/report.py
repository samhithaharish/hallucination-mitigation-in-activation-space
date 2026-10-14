"""Publish compact results after both frozen test evaluations exist."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

from experiments.paths import REPO
from experiments.recovery_help.data import dump, file_digest
from .data import OUT
from .evaluate import replicated_acceptance

MODELS = ('qwen7b', 'phi35')
NAMES = {'choices':'Learned strategy changes', 'class_rule':'Condition rule',
         'observable_rule':'Public input', 'confidence_rule':'Confidence'}


def percentage(x):
    return f'{100*x:.2f}'


def difference(value):
    lo, hi = value['ci95']
    return f"{100*value['mean']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]"


def report():
    results = {key:json.loads((OUT/key/'test_evaluation.json').read_text()) for key in MODELS}
    acceptance = replicated_acceptance(results)
    root = REPO/'results/paired_evidence'
    root.mkdir(exist_ok=True)
    for key in MODELS:
        shutil.copy2(OUT/key/'test_evaluation.json', root/f'{key}_test.json')
        shutil.copy2(OUT/key/'pilot_evaluation.json', root/f'{key}_pilot.json')
    shutil.copy2(OUT/'data_manifest.json', root/'cohorts.json')
    loop = REPO/'autoresearch/loop-261013-paired'
    search = json.loads((loop/'summary.json').read_text())
    dump(root/'search.json', search)
    frozen = {k:json.loads((OUT/k/'fitted/best.json').read_text()) for k in MODELS}
    coordinates = {k:json.loads((OUT/k/'fitted/coordinates.json').read_text()) for k in MODELS}
    dump(root/'provenance.json', dict(frozen=frozen, coordinates=coordinates,
         source_hashes={p.name:file_digest(p) for p in Path(__file__).parent.glob('*.py')},
         data_manifest_sha256=file_digest(OUT/'data_manifest.json'), acceptance=acceptance))
    passed = [NAMES[k] for k,v in acceptance['routing'].items() if v]
    lines = ['# Paired evidence repair: heldout results', '',
        ('Controlled routing criterion met by: '+', '.join(passed)+'.') if passed else
        'Neither activation policy met the prespecified routing criterion on both backbones.', '',
        'Learning strategy changes '+('also added value over the condition rule.' if
          acceptance['strategy_fields_add_value'] else 'did not establish added value over the condition rule on both backbones.'), '',
        'Each backbone: 400 training, 100 validation, 800 fresh test base questions; '
        'three paired versions per base (2,400 test inputs). The 200-base pilot is separate. '
        'The benchmark weights the three conditions equally, not by natural failure prevalence. All test decisions were saved before any mitigation preprocessing. No mitigation-success '
        'targets were used for learned parameter fitting or activation-configuration selection. '
        'Independent pilot outcomes informed the action definitions and fixed missing-only condition map.', '',
        '## Test answer F1 (%)', '',
        '| Backbone | Keep | Best always-on action | Strategy changes | Condition rule | Public input | Confidence |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,r in results.items():
        baseline=np.mean([r['fixed'][c]['keep']['f1'] for c in ('clean','missing','distractor')])
        fixed=max(np.mean([r['fixed'][c][a]['f1'] for c in ('clean','missing','distractor')])
                  for a in ('keep','retrieve','filter','focus','sham','empty'))
        values=[baseline,fixed]+[r['policies'][p]['f1'] for p in NAMES]
        lines.append('| '+key+' | '+' | '.join(percentage(v) for v in values)+' |')
    lines += ['', 'Best always-on action is chosen in hindsight. Matched-risk fixed controls below '
        'optimize action and risk threshold in hindsight within the same extra-generation-call budget, '
        'including cheaper sham and empty controls and unused calls; '
        'this does not match tokens or latency.', '',
              '## Paired F1 differences (percentage points, 95% intervals)', '',
              'Intervals resample 800 base questions, retaining all three variants. Positive lower bounds '
              'are required against **every** listed control and on both backbones for the same policy.', '',
              '| Backbone / policy | Keep | Matched-risk fixed | Risk-bin shuffle | Public input | Confidence |',
              '|---|---:|---:|---:|---:|---:|']
    controls=('vs_keep','vs_matched_risk_fixed','vs_risk_matched_shuffle','vs_observable_rule','vs_confidence_rule')
    for key,r in results.items():
        for policy in ('choices','class_rule'):
            p=r['policies'][policy]
            lines.append('| '+key+' / '+NAMES[policy]+' | '+' | '.join(difference(p[c]) for c in controls)+' |')
    lines += ['', 'Strategy changes versus the condition rule: '+ '; '.join(
        k+' '+difference(r['policies']['choices']['vs_class_rule']) for k,r in results.items())+'.', '',
        '## Activation-only search', '',
        f"Twelve ridge configurations were attempted plus the unconditional mean baseline; "
        f"{sum(r['status']=='crash' for r in search['trials'])} trials crashed. "
        f"Mean validation activation NMSE: {search['initial_metric']:.4f} → {search['final_metric']:.4f}. "
        f"Selected {search['selected_trial']}; configuration: `{json.dumps(next(iter(frozen.values()))['config'],sort_keys=True)}`. "
        'This score measures predicted activation changes, not answer improvement.', '',
        '| Backbone | Validation condition accuracy | Test condition accuracy | Mean extra calls: strategy / condition |',
        '|---|---:|---:|---:|']
    for key,r in results.items():
        lines.append(f"| {key} | {percentage(coordinates[key]['coordinates']['validation_condition_accuracy'])} | "
                     f"{percentage(r['condition_accuracy'])} | {r['policies']['choices']['extra_calls']:.3f} / "
                     f"{r['policies']['class_rule']['extra_calls']:.3f} |")
    lines += ['', '## A failed geometry criterion', '',
        'Transforming both a corrupted and a matched clean prompt with the same action, then minimizing '
        'their activation distance, has a deletion-collapse degeneracy. Emptying the evidence makes '
        'both prompts identical and therefore maximally closes that distance, regardless of answer truth. '
        'Counterexamples start with a correct clean answer and incorrect corrupted answer, yet emptying '
        'makes both answers wrong while maximizing the gap score. This is a failure of this specific '
        'score, not a proof that all activation geometry is useless.', '',
        '| Backbone | Empty is a gap-score maximizer (%) | Empty F1 | Original corrupted-input F1 | Both wrong counterexamples |',
        '|---|---:|---:|---:|---:|']
    for key,r in results.items():
        g=r['oracle_same_action_gap']
        lines.append('| '+key+' | '+' | '.join(percentage(g[k]) for k in
                     ('empty_maximizer_rate','empty_f1','original_bad_keep_f1'))+' | '+
                     str(g['empty_maximizer_both_answers_wrong_counterexamples'])+' |')
    lines += ['', 'The learned score with sham/empty controls allowed (diagnostic only): '+ '; '.join(
        k+' empty selections '+str(r['learned_control_policy']['counts']['empty'])+' / '+str(r['n_versions'])
        +', F1 '+percentage(r['learned_control_policy']['f1']) for k,r in results.items())+'.', '',
        '## What these results establish', '',
        'The benchmark controls evidence interventions; its labels describe input conditions. '
        'They do not identify a defective layer or prove an internal failure mechanism. '
        'Natural-error transfer, other tasks, other backbones, semantic controls and stronger '
        'representation-based observable baselines remain necessary. An unrepaired output is '
        'not proven irrecoverable.', '',
        'Activation-based retrieval and evidence-condition classification have prior work; see '
        '[method and related work](../../experiments/paired_evidence/README.md). '
        'The two-model acceptance result above is the decision for this experiment; no test outcomes '
        'were used to refit a winning policy.', '',
        'Files: *_test.json contains all answer metrics, paired controls, call/token costs and timing audits; '
        '*_pilot.json records separate feasibility checks; cohorts.json and qualification_cohorts.json '
        'identify fresh and excluded exploratory cohorts; search.json and provenance.json record '
        'activation-only selection and frozen source/model hashes. Raw states, generated answers and '
        'fitted objects stay in the local ignored run root.', '']
    if search.get('aborted_queued_trials'):
        lines += ['An earlier queued search loaded a stale trial grid and was stopped before '
                  'heldout collection. Its attempted trials were preserved; the committed grid '
                  'was then run afresh. Both records are included in search.json.', '']
    (root/'README.md').write_text('\n'.join(lines))
    figure(results, root)
    print(json.dumps(acceptance, indent=2))
    print(root/'README.md')


def figure(results, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
    controls=('vs_keep','vs_matched_risk_fixed','vs_risk_matched_shuffle','vs_observable_rule','vs_confidence_rule')
    labels=('Keep','Matched calls + risk','Shuffle within risk','Public input','Confidence')
    for ax,(key,r) in zip(axes,results.items()):
        for policy,offset,color in (('choices',-.13,'#1764a3'),('class_rule',.13,'#bd5b23')):
            values=[r['policies'][policy][c] for c in controls]
            x=np.array([100*v['mean'] for v in values]);bounds=np.array([v['ci95'] for v in values])*100
            # Bootstrap bounds need not contain the point estimate in unusual cases.
            y=np.arange(len(controls))+offset
            ax.hlines(y,bounds[:,0],bounds[:,1],color=color,lw=1.5)
            ax.scatter(x,y,s=23,color=color,label=NAMES[policy],zorder=3)
        ax.axvline(0,color='.45',lw=.8);ax.set_title(key);ax.set_xlabel('F1 gain (percentage points)')
        ax.set_yticks(np.arange(len(controls)),labels);ax.grid(axis='x',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis();axes[1].legend(frameon=False,fontsize=8,loc='best')
    fig.tight_layout();fig.savefig(root/'routing_comparison.png',dpi=180)
    fig.savefig(root/'routing_comparison.pdf');plt.close(fig)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    report()
