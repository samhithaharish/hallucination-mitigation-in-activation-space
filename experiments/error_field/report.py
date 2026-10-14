"""Export the exploratory error-field result with its supervision boundaries."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from experiments.paths import REPO
from experiments.paired_evidence.data import OUT
from experiments.recovery_help.data import dump

PRIMARY=('choices','reference_choices')
NAMES={'choices':'Error-logit transitions','reference_choices':'Paired clean-state reference',
       'class_rule':'Activation condition rule',
       'observable_rule':'Public-feature transitions','confidence_rule':'Confidence transitions',
       'condition_observable_rule':'Public condition rule','condition_confidence_rule':'Confidence condition rule',
       'condition_field_choices':'Previous condition transitions',
       'reference_observable_rule':'Public clean-reference transitions',
       'reference_confidence_rule':'Confidence clean-reference transitions'}


def diff(v):
    return f"{100*v['mean']:+.2f} [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]"


def report():
    results={k:json.loads((OUT/k/'error_evaluation.json').read_text()) for k in ('qwen7b','phi35')}
    success={p:all(r['exploratory_policy_passes'][p] for r in results.values()) for p in PRIMARY}
    selection=json.loads((OUT.parent/'error_field/selection.json').read_text())
    reference=json.loads((OUT.parent/'error_field/reference_selection.json').read_text())
    root=REPO/'results/error_field';root.mkdir(parents=True,exist_ok=True)
    dump(root/'selection.json',selection);shutil.copy2(OUT/'data_manifest.json',root/'cohorts.json')
    dump(root/'reference_selection.json',reference)
    for k,r in results.items():dump(root/f'{k}_test.json',r)
    passed=[NAMES[p] for p,v in success.items() if v]
    lines=['# Two fixed-reference response policies', '',
        ('Exploratory policies exceeding every control on both backbones: '+', '.join(passed)+'.' if passed else
         'Neither exploratory policy exceeded every control on both backbones.'), '',
        'This is a follow-up method-development experiment with new heldout questions. '
        'It is not a confirmed ICML contribution. Its design and models were frozen before inspecting '
        'the preceding paired-evidence study\'s heldout answer-quality results. The development '
        'executions are reused; all 800 new test question IDs are excluded from preceding cohorts. '
        'Their supporting/filler article titles are absent from the source v3 candidate pools; '
        'article-title disjointness from earlier qualification cohorts is not guaranteed. All three input variants stay together '
        'in resampling. The benchmark weights conditions equally.', '',
        '**Supervision:** the error-logit policy uses baseline correctness labels for its probe. Its strategy-response fitting '
        'and configuration selection use activation-derived logit changes, without mitigation-success '
        'targets. Independent pilot outcomes informed the action bank and the fixed missing-only '
        'condition map. Public/confidence transition controls receive the same logit-change targets. '
        'The clean-state reference policy uses paired activation targets without correctness labels. '
        'Its public/confidence controls receive the same residual/state-change targets. Adding it leaves '
        'every error-logit policy parameter and validation prediction unchanged, checked before test. '
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
    baselines=[n for n in NAMES if n not in PRIMARY]
    controls=('vs_keep','vs_matched_risk_fixed','vs_risk_matched_shuffle')+tuple('vs_'+n for n in baselines)
    labels=('Keep','Best risk-gated fixed within call budget','Shuffle within policy risk bins')+tuple(NAMES[n] for n in baselines)
    for policy in PRIMARY:
        lines += ['', '## '+NAMES[policy]+' F1 differences (pp, paired 95% intervals)', '',
            '| Control | Qwen | Phi |','|---|---:|---:|']
        for c,label in zip(controls,labels):
            lines.append('| '+label+' | '+' | '.join(diff(r['policies'][policy][c]) for r in results.values())+' |')
    lines += ['', 'The fixed control optimizes action and risk threshold in hindsight, including unused '
        'calls, within the policy\'s extra-generation-call budget. It also includes one-call sham/empty '
        'controls. Matching a call-budget cap does not match tokens, latency or activation-extraction '
        'overhead. JSON files include token costs, EM, rescue/harm and prediction timing audits.', '',
        '## Latent model and transfer assumption', '',
        f"Selected configuration: `{json.dumps(selection['best']['config'],sort_keys=True)}`. "
        f"Shared validation transition NMSE {selection['trials'][0]['metric']:.4f} → {selection['best']['metric']:.4f}. "
        'This predicts an error-probe response, not true strategy success.', '',
        f"Clean-reference configuration: `{json.dumps(reference['best']['config'],sort_keys=True)}`; "
        f"validation state-transition NMSE {reference['trials'][0]['metric']:.4f} → {reference['best']['metric']:.4f}. "
        'The two NMSE objectives measure different quantities and should not be compared directly.', '',
        '| Backbone | Validation baseline-error AUROC | Test baseline-error AUROC | Oracle action F1 | Empty selections: error / reference |',
        '|---|---:|---:|---:|---:|']
    for key,r in results.items():
        lines.append(f"| {key} | {r['baseline_error_probe']['validation_baseline_error_auroc']:.3f} | "
            f"{r['baseline_probe_transfer']['keep']['baseline_probe_applied_to_action_error_auroc']:.3f} | "
            f"{100*r['oracle_f1']:.2f} | {r['learned_control_policy']['counts']['empty']} / "
            f"{r['reference_control_policy']['counts']['empty']} (of {r['n_versions']}) |")
    lines += ['', 'A probe trained on original prompts may be wrong on modified prompts. The per-action '
        'probe AUROC/log loss and empty-evidence diagnostic in the JSON files test this assumption; '
        'low transition MSE alone cannot validate it. The one-standard-deviation residual penalty '
        'is a heuristic and has no calibrated risk guarantee. The clean-reference policy instead '
        'requires predicted distance toward its denoised matched-question reference to track repair. '
        'Neither assumption follows from transition predictability.', '',
        'Two exploratory policies were fixed before this fresh cohort was executed. Each must beat '
        'every listed control with a positive lower 95% interval bound on both backbones for the same '
        'policy. Two-sided 95% bounds correspond to nominal one-sided 2.5% per policy; the '
        'Bonferroni family threshold for these two policy claims is 5%, subject to bootstrap '
        'approximation. The broader iterative project remains exploratory.', '',
        'Recoverability is restricted to the available action bank. These data do not identify '
        'causal layer pathways, prove universal irrecoverability, establish natural-error transfer, '
        'or establish novelty. See [method](../../experiments/error_field/README.md) and the '
        '[preceding evidence experiment](../paired_evidence/README.md). Raw states and model objects '
        'remain in the ignored run root; selection.json and reference_selection.json contain all configurations and frozen hashes.', '']
    (root/'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,10),sharey=True)
    for row,policy in enumerate(PRIMARY):
        for ax,(key,r) in zip(axes[row],results.items()):
            for i,c in enumerate(controls):
                v=r['policies'][policy][c];lo,hi=np.array(v['ci95'])*100
                ax.hlines(i,lo,hi,color='#1764a3',lw=1.5)
                ax.scatter(100*v['mean'],i,color='#1764a3',s=22,zorder=3)
            ax.axvline(0,color='.4',lw=.8);ax.set_title(key+' / '+NAMES[policy],fontsize=9)
            ax.set_yticks(np.arange(len(labels)),labels,fontsize=8);ax.set_xlabel('F1 gain (pp)')
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    axes[0,0].invert_yaxis();fig.tight_layout();fig.savefig(root/'comparison.png',dpi=180)
    fig.savefig(root/'comparison.pdf');plt.close(fig)
    print(json.dumps({'exploratory_passed_both_backbones':success,'report':str(root/'README.md')},indent=2))


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();report()
