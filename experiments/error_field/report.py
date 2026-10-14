"""Export the exploratory error-field result with its supervision boundaries."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from experiments.paths import REPO
from experiments.paired_evidence.data import OUT
from experiments.recovery_help.data import dump

NAMES={'choices':'Error-logit transitions','class_rule':'Activation condition rule',
       'observable_rule':'Public-feature transitions','confidence_rule':'Confidence transitions',
       'condition_observable_rule':'Public condition rule','condition_confidence_rule':'Confidence condition rule',
       'condition_field_choices':'Previous condition transitions'}


def diff(v):
    return f"{100*v['mean']:+.2f} [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]"


def report():
    results={k:json.loads((OUT/k/'error_evaluation.json').read_text()) for k in ('qwen7b','phi35')}
    success=all(r['exploratory_all_controls_passed'] for r in results.values())
    selection=json.loads((OUT.parent/'error_field/selection.json').read_text())
    root=REPO/'results/error_field';root.mkdir(parents=True,exist_ok=True)
    dump(root/'selection.json',selection);shutil.copy2(OUT/'data_manifest.json',root/'cohorts.json')
    for k,r in results.items():dump(root/f'{k}_test.json',r)
    lines=['# Forecasting baseline error-probe changes', '',
        ('The exploratory error-field policy exceeded every prespecified control on both backbones.' if success else
         'The exploratory error-field policy did not exceed every control on both backbones.'), '',
        'This is a follow-up method-development experiment with new heldout questions. '
        'It is not a confirmed ICML contribution. Its design and models were frozen before inspecting '
        'the preceding paired-evidence study\'s heldout answer-quality results. The development '
        'executions are reused; all 800 new test question IDs are excluded from preceding cohorts. '
        'Their supporting/filler article titles are absent from the source v3 candidate pools; '
        'article-title disjointness from earlier qualification cohorts is not guaranteed. All three input variants stay together '
        'in resampling. The benchmark weights conditions equally.', '',
        '**Supervision:** the error probe uses baseline correctness labels. Strategy-response fitting '
        'and configuration selection use activation-derived logit changes, without mitigation-success '
        'targets. Independent pilot outcomes informed the action bank and the fixed missing-only '
        'condition map. Public/confidence transition controls receive the same logit-change targets. '
        'Every test decision is saved before any mitigation preprocessing.', '',
        '## Test F1 (%) and extra calls', '',
        '| Backbone | Policy | F1 | Extra calls | Rescued / harmed (EM) |',
        '|---|---|---:|---:|---:|']
    for key,r in results.items():
        baseline=np.mean([r['fixed'][c]['keep']['f1'] for c in ('clean','missing','distractor')])
        lines.append(f'| {key} | Keep | {100*baseline:.2f} | 0 | — |')
        for name,label in NAMES.items():
            p=r['policies'][name]
            lines.append(f"| {key} | {label} | {100*p['f1']:.2f} | {p['extra_calls']:.3f} | {p['rescued']} / {p['harmed']} |")
    controls=('vs_keep','vs_matched_risk_fixed','vs_risk_matched_shuffle')+tuple('vs_'+n for n in list(NAMES)[1:])
    labels=('Keep','Best risk-gated fixed within call budget','Shuffle within baseline-error risk bins')+tuple(list(NAMES.values())[1:])
    lines += ['', '## Error-field F1 differences (pp, paired 95% intervals)', '',
        '| Control | Qwen | Phi |','|---|---:|---:|']
    for c,label in zip(controls,labels):
        lines.append('| '+label+' | '+' | '.join(diff(r['policies']['choices'][c]) for r in results.values())+' |')
    lines += ['', 'The fixed control optimizes action and risk threshold in hindsight, including unused '
        'calls, within the policy\'s extra-generation-call budget. It also includes one-call sham/empty '
        'controls. Matching a call-budget cap does not match tokens, latency or activation-extraction '
        'overhead. JSON files include token costs, EM, rescue/harm and prediction timing audits.', '',
        '## Latent model and transfer assumption', '',
        f"Selected configuration: `{json.dumps(selection['best']['config'],sort_keys=True)}`. "
        f"Shared validation transition NMSE {selection['trials'][0]['metric']:.4f} → {selection['best']['metric']:.4f}. "
        'This predicts an error-probe response, not true strategy success.', '',
        '| Backbone | Validation baseline-error AUROC | Test baseline-error AUROC | Oracle action F1 | Empty selections if allowed |',
        '|---|---:|---:|---:|---:|']
    for key,r in results.items():
        lines.append(f"| {key} | {r['baseline_error_probe']['validation_baseline_error_auroc']:.3f} | "
            f"{r['baseline_probe_transfer']['keep']['baseline_probe_applied_to_action_error_auroc']:.3f} | "
            f"{100*r['oracle_f1']:.2f} | {r['learned_control_policy']['counts']['empty']} / {r['n_versions']} |")
    lines += ['', 'A probe trained on original prompts may be wrong on modified prompts. The per-action '
        'probe AUROC/log loss and empty-evidence diagnostic in the JSON files test this assumption; '
        'low transition MSE alone cannot validate it. The one-standard-deviation residual penalty '
        'is a heuristic and has no calibrated risk guarantee.', '',
        'Recoverability is restricted to the available action bank. These data do not identify '
        'causal layer pathways, prove universal irrecoverability, establish natural-error transfer, '
        'or establish novelty. See [method](../../experiments/error_field/README.md) and the '
        '[preceding evidence experiment](../paired_evidence/README.md). Raw states and model objects '
        'remain in the ignored run root; selection.json contains all configurations and frozen hashes.', '']
    (root/'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,5.2),sharey=True)
    for ax,(key,r) in zip(axes,results.items()):
        for i,c in enumerate(controls):
            v=r['policies']['choices'][c];lo,hi=np.array(v['ci95'])*100
            ax.hlines(i,lo,hi,color='#1764a3',lw=1.5)
            ax.scatter(100*v['mean'],i,color='#1764a3',s=22,zorder=3)
        ax.axvline(0,color='.4',lw=.8);ax.set_title(key)
        ax.set_yticks(np.arange(len(labels)),labels);ax.set_xlabel('Error-field F1 gain (pp)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis();fig.tight_layout();fig.savefig(root/'comparison.png',dpi=180)
    fig.savefig(root/'comparison.pdf');plt.close(fig)
    print(json.dumps({'exploratory_passed_both_backbones':success,'report':str(root/'README.md')},indent=2))


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();report()
