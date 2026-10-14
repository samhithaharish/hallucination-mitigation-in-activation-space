"""Report discovery and fresh replication without revising primary decisions."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .evaluate import ROOT,MODELS,CONTROLS

LABELS=('Keep','Fixed action + activation risk','Fixed action + public margin',
        'Fixed action + confidence margin','Shuffle within activation risk')


def difference(v):
    return f"{100*v['mean']:+.2f} [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]"


def report():
    records={label:{key:json.loads((ROOT/f'{label}_{key}.json').read_text()) for key in MODELS}
             for label in ('exploration','replication')}
    passed=all(r['all_controls_passed'] for r in records['replication'].values())
    lines=['# Missing-evidence routing at a fixed call budget','',
        ('The unchanged activation condition rule passed every secondary control on both fresh backbones.' if passed else
         'The unchanged activation condition rule did not pass every secondary control on both fresh backbones.'),'',
        '**Secondary exploratory analysis.** The preceding paired study failed its original '
        'primary acceptance criterion. This narrower call-budget hypothesis arose after '
        'inspecting its results. Its code, controls, policy and fresh cohort/model hashes were '
        'registered during the new collection, before either collector completed and before '
        'fresh answer outcomes were inspected. It does not replace either preceding study\'s '
        'primary decision. Nominal intervals do not control multiplicity over the broader project.','',
        '## Detector and costs','',
        'The detector is the already frozen condition rule: retrieve for predicted missing '
        'evidence, otherwise keep. It is not trained on mitigation-success targets. The independent '
        'pilot informed the action bank and fixed map. The action bank contains keep, retrieve, '
        'filter, focus, sham and empty. All three input variants have equal benchmark weight.','',
        '| Cohort | Backbone | Keep F1 | Activation F1 | Public default F1 | Extra calls: activation / public default |',
        '|---|---|---:|---:|---:|---:|']
    for label,results in records.items():
        for key,r in results.items():
            lines.append(f"| {label} | {key} | {100*r['baseline_keep_f1']:.2f} | {100*r['policy']['f1']:.2f} | "
                f"{100*r['default_public']['f1']:.2f} | {r['policy']['extra_calls']:.3f} / {r['default_public']['extra_calls']:.3f} |")
    for label,results in records.items():
        lines += ['', '## '+label.capitalize()+' differences (F1 pp, paired 95% intervals)','',
            '| Control | Qwen | Phi |','|---|---:|---:|']
        for control,name in zip(CONTROLS,LABELS):
            lines.append('| '+name+' | '+' | '.join(difference(r['comparisons'][control]) for r in results.values())+' |')
    lines += ['', 'Each fixed control selects its action and best affordable prefix **in hindsight**. '
        'It can leave calls unused and can use one-call sham/empty actions on more inputs. '
        'Prefixes rank activation condition drift or frozen public/confidence missing-condition '
        'margins. The budget cap equals the activation detector\'s extra call count. Action, '
        'threshold and budget are recomputed in every base-question bootstrap. This is an '
        'evaluation control, not success-supervised fitting of the deployed rule.','',
        'Matching extra generation calls does not match tokens, latency or activation capture/replay '
        'overhead. The observable comparator is the recorded TF-IDF/SVD, confidence, length and '
        'public-pool model, not every possible text model. These controlled evidence interventions '
        'do not establish natural-error transfer or causal layer diagnosis. Activation-based '
        'retrieval and evidence-condition probing have prior work; this result cannot establish novelty.','',
        'The fresh analysis requires positive lower 95% bounds against every listed control on both '
        'models for the same frozen policy. See [registration](registration.json), '
        '[method and reproduction](../../experiments/retrieval_budget/README.md), '
        '[original paired results](../paired_evidence/README.md), and '
        '[separate response-policy results](../error_field/README.md).','']
    (ROOT/'README.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,(key,r) in zip(axes,records['replication'].items()):
        for i,c in enumerate(CONTROLS):
            v=r['comparisons'][c];lo,hi=np.array(v['ci95'])*100
            ax.hlines(i,lo,hi,color='#1764a3');ax.scatter(100*v['mean'],i,color='#1764a3',s=24)
        ax.axvline(0,color='.4',lw=.8);ax.set_title(key+' / fresh secondary test')
        ax.set_yticks(range(len(LABELS)),LABELS,fontsize=8);ax.set_xlabel('F1 difference (pp)')
        ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis();fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=180)
    fig.savefig(ROOT/'comparison.pdf');plt.close(fig)
    print(json.dumps(dict(secondary_replication_passed_both=passed,report=str(ROOT/'README.md'))))


if __name__=='__main__':report()
